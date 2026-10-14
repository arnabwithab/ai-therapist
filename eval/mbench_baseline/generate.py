"""Generate Qwen3-4B-Instruct-2507 responses for all 10,000 MentalBench-100k rows.

Plain HF batched generation (vLLM abandoned: v1/v2 died on torch-cu130 vs torchaudio-cu128,
v3 died on libcudart.so.13 missing from Kaggle's CUDA 12.8 image). One kernel drives both T4s:
the parent spawns two worker subprocesses, each confined to one GPU via CUDA_VISIBLE_DEVICES
and generating one contiguous half of the rows. The 1,781 rows from the earlier single-stream
shard kernels are attached as a seed dataset and copied in, not regenerated; attaching a prior
partial output_mbench.csv as a dataset resumes a cap-killed run the same way.
"""
import csv, glob, io, logging, os, subprocess, sys, time, urllib.request, zipfile

OUT = "/kaggle/working/output_mbench.csv"
MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
DATA_URL = "https://raw.githubusercontent.com/abeerbadawi/MentalBench-Align/main/MentalBench-100k/MentalBench-100k.zip"
COL = "qwen3_4b_instruct_2507"
BATCH = 8  # shortcut: fits T4-14.5GB with margin; raise to 16 only if a run logs VRAM headroom
MAX_TOKENS = 512  # keep in lockstep with the 1,781 existing rows (temp 0.7 / top_p 0.8 too)

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
        return {int(r["idx"]): (r["context"], r[COL]) for r in csv.DictReader(f) if r["idx"].isdigit()}


def seeded_rows():
    """Rows from attached datasets: the seed shards and any prior partial output (resume)."""
    out = {}
    for p in sorted(glob.glob("/kaggle/input/datasets/*/*/output_*.csv")):
        for idx, (c, t) in read_generated(p).items():
            out[idx] = (c, t)
    return out


def run_parent():
    done = read_generated(OUT)  # always empty: /kaggle/working starts fresh each run
    seed = {i: v for i, v in seeded_rows().items() if i not in done}
    log.info("seed: %d rows from attached datasets", len(seed))
    with open(OUT, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=["idx", "context", COL])
        w.writeheader()
        for idx in sorted(seed):
            c, t = seed[idx]
            w.writerow({"idx": idx, "context": c, COL: t})

    script = os.path.abspath(__file__)
    assert os.path.isfile(script), script
    procs = []
    for shard in (0, 1):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard))
        procs.append(subprocess.Popen([sys.executable, script, "--shard", str(shard)], env=env))
    rc = [p.wait() for p in procs]
    if any(rc):
        raise SystemExit(f"worker failed, returncodes={rc}")

    merged = read_generated(OUT)
    for shard in (0, 1):
        merged.update(read_generated(f"/kaggle/working/output_part_{shard}.csv"))
    with open(OUT, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=["idx", "context", COL])
        w.writeheader()
        for idx in sorted(merged):
            c, t = merged[idx]
            w.writerow({"idx": idx, "context": c, COL: t})
    log.info("DONE %d/10000 rows -> %s", len(merged), OUT)


def run_child(shard):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_device(0)  # only its own GPU is visible
    assert torch.cuda.device_count() == 1, torch.cuda.device_count()
    rows = load_contexts()
    half = [r for r in rows if (r[0] < 5000) == (shard == 0)]
    skip = set(read_generated(OUT)) | set(seeded_rows())
    todo = [(i, c) for i, c in half if i not in skip]
    log.info("[gpu%d] todo=%d", shard, len(todo))

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tok.padding_side = "left"  # decoder-only batching must pad left
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=torch.float16, device_map="auto"
    )
    model.eval()
    log.info("[gpu%d] model on %s", shard, model.device)

    part = f"/kaggle/working/output_part_{shard}.csv"
    new_file = not os.path.exists(part)
    fout = open(part, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=["idx", "context", COL])
    if new_file:
        w.writeheader()
        fout.flush()

    t0 = time.time()
    n_new = 0
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(user_concern=c)}],
                add_generation_prompt=True,
                enable_thinking=False,
                tokenize=False,
            )
            for _, c in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(model.device)
        attn = enc["attention_mask"].to(model.device)
        lens = attn.sum(dim=1).tolist()  # left-padded: real tokens end at lens[i]
        with torch.no_grad():
            out = model.generate(
                input_ids,
                attention_mask=attn,
                max_new_tokens=MAX_TOKENS,
                temperature=0.7,
                top_p=0.8,
                do_sample=True,
                pad_token_id=tok.eos_token_id,
            )
        for (idx, c), seq, ln in zip(batch, out, lens):
            w.writerow({"idx": idx, "context": c, COL: tok.decode(seq[ln:], skip_special_tokens=True).strip()})
            n_new += 1
        fout.flush()
        if n_new % 32 == 0:
            dt = time.time() - t0
            log.info("[gpu%d] %d/%d new (%.0f rows/h)", shard, n_new, len(todo), n_new / dt * 3600)
    fout.close()
    dt = time.time() - t0
    log.info("[gpu%d] DONE %d new (%.0f rows/h) -> %s", shard, n_new, n_new / max(dt, 1) * 3600, part)


def main():
    if "--shard" in sys.argv:
        run_child(int(sys.argv[sys.argv.index("--shard") + 1]))
    else:
        run_parent()


if __name__ == "__main__":
    main()
