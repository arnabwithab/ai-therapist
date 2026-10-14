"""Generate Qwen3-4B-Instruct-2507 responses for all 10,000 MentalBench-100k rows (vLLM, tensor parallel 2).

Replaces the two single-stream shard kernels: no sharding, one resumable output CSV. The 1,781 rows those
kernels already produced are attached as a seed dataset and copied in, so they are not regenerated.
"""
import csv, glob, io, logging, os, subprocess, sys, time, urllib.request, zipfile

OUT = "/kaggle/working/output_mbench.csv"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATA_URL = "https://raw.githubusercontent.com/abeerbadawi/MentalBench-Align/main/MentalBench-100k/MentalBench-100k.zip"
COL = "qwen3_4b_instruct_2507"
CHUNK = 256  # prompts per generate() call, so progress is flushed between chunks
MAX_TOKENS = 512
MAX_MODEL_LEN = 2048  # longest context is ~730 tokens, + 512 generated

PROMPT = (
    "You are a licensed psychiatrist responding to a user who has expressed mental health concerns.\n"
    "Respond in natural, supportive paragraphs \u2014 not as an email, letter, or bullet points.\n"
    "Your response should be supportive, informative, emotionally attuned, and offer clear, empathetic guidance "
    "while addressing the user's emotional state.\n"
    "Ensure the response is complete, thoughtful, and psychologically appropriate.\n\n"
    "User Concern:\n{user_concern}\n\n"
    "Provide a compassionate, professional, and helpful response."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("gen")


def load_contexts():
    log.info("downloading dataset (46MB)...")
    req = urllib.request.Request(DATA_URL, headers={"User-Agent": "ai-therapist"})
    with urllib.request.urlopen(req, timeout=120) as r:
        blob = r.read()
    log.info("downloaded %d bytes", len(blob))
    with zipfile.ZipFile(io.BytesIO(blob)) as zf, zf.open("MentalBench-100k.csv") as f:
        reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8-sig", errors="replace"))
        rows = [(i, r["context"]) for i, r in enumerate(reader)]
    log.info("total rows: %d", len(rows))
    return rows


def read_generated(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["idx"]): r[COL] for r in csv.DictReader(f) if r["idx"].isdigit()}


def seeded_rows():
    """Rows from the earlier single-stream shard kernels, attached as a dataset."""
    out = {}
    for p in sorted(glob.glob("/kaggle/input/datasets/*/*/output_shard_*.csv")):
        out.update(read_generated(p))
    return out


def main():
    # Install order matters: vLLM >= 0.28 pins torch==2.13.0, which only exists as cu130, while
    # Kaggle's torchaudio stays cu128 -> RuntimeError at import (v1/v2 died on exactly this).
    # vLLM 0.25.0 pins torch/torchaudio==2.11.0, which the cu128 index carries, so pre-install
    # that stack from cu128 and forbid upgrades: 2.11.0+cu128 satisfies ==2.11.0 (PEP 440
    # ignores the local segment), and pip then leaves the CUDA-consistent stack alone.
    cu128 = "https://download.pytorch.org/whl/cu128"
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision", "torchaudio",
         "--index-url", cu128],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "vllm==0.25.0",
         "--extra-index-url", cu128, "--upgrade-strategy", "only-if-needed"],
        check=True,
    )
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = load_contexts()
    done = read_generated(OUT)
    seed = {i: t for i, t in seeded_rows().items() if i not in done}
    log.info("resume: %d rows already in %s, %d seeded from earlier runs", len(done), OUT, len(seed))

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    fout = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=["idx", "context", COL])
    if not done:
        w.writeheader()
    ctx = dict(rows)
    for idx, text in sorted(seed.items()):
        w.writerow({"idx": idx, "context": ctx[idx], COL: text})
    fout.flush()

    todo = [(i, c) for i, c in rows if i not in done and i not in seed]
    log.info("todo=%d", len(todo))

    log.info("loading %s with tensor_parallel_size=2 ...", MODEL_ID)
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=2,
        dtype="float16",  # T4 is sm_75: no bf16
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.90,
        trust_remote_code=True,
    )
    sp = SamplingParams(temperature=0.7, top_p=0.8, max_tokens=MAX_TOKENS)

    t0 = time.time()
    n_new = 0
    for start in range(0, len(todo), CHUNK):
        batch = todo[start : start + CHUNK]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(user_concern=c)}],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=False,
            )
            for _, c in batch
        ]
        for (idx, c), out in zip(batch, llm.generate(prompts, sp)):
            w.writerow({"idx": idx, "context": c, COL: out.outputs[0].text.strip()})
            n_new += 1
        fout.flush()
        dt = time.time() - t0
        log.info("%d/%d new (%.0f rows/h, %.2fh elapsed)", n_new, len(todo), n_new / dt * 3600, dt / 3600)
    fout.close()
    log.info("DONE wrote %d new rows -> %s", n_new, OUT)


if __name__ == "__main__":
    main()
