"""Generate Phi-4-mini-instruct baseline responses for all 10,000 MentalBench-100k rows.

Base model + LoRA adapter (auto-discovered from attached datasets) via plain HF
batched generation (vLLM abandoned: v1/v2 died on torch-cu130 vs torchaudio-cu128,
v3 died on libcudart.so.13 missing from Kaggle's CUDA 12.8 image). One kernel drives both T4s:
the parent spawns two worker subprocesses, each confined to one GPU via CUDA_VISIBLE_DEVICES
and generating one contiguous half of the rows. Attaching a prior partial
output_mbench_sft5.csv as a dataset resumes a cap-killed run the same way
(base-model baseline: attach the seed shards + pass --col qwen3_4b_instruct_2507
--out /kaggle/working/output_mbench.csv).
"""
import csv, glob, io, logging, os, subprocess, sys, time, urllib.request, zipfile

OUT = "/kaggle/working/output_mbench_phi4.csv"
MODEL_ID = "microsoft/Phi-4-mini-instruct"
DATA_URL = "https://raw.githubusercontent.com/abeerbadawi/MentalBench-Align/main/MentalBench-100k/MentalBench-100k.zip"
COL = "phi4_mini_base"  # flags --col/--out/--model-id restore other runs
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


def read_generated(path, col=COL):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {int(r["idx"]): (r["context"], (r.get(col) or "")) for r in csv.DictReader(f) if r["idx"].isdigit()}


def completed(path, col=COL):
    """Idxs with a non-blank response under col (seed files may use another column)."""
    return {i for i, (_, t) in read_generated(path, col).items() if t.strip()}


def find_adapter(explicit=None, _roots=None):
    """LoRA adapter dir: explicit path, else first adapter_config.json under the inputs."""
    if explicit:
        return explicit if os.path.isdir(explicit) else None
    for root in _roots or ("/kaggle/input",):
        hits = sorted(glob.glob(os.path.join(root, "**", "adapter_config.json"), recursive=True))
        if hits:
            return os.path.dirname(hits[0])
    return None


def seeded_rows(col=COL):
    """Rows from attached datasets: the seed shards and any prior partial output (resume)."""
    out = {}
    for p in sorted(glob.glob("/kaggle/input/datasets/*/*/output_*.csv")):
        for idx, (c, t) in read_generated(p, col).items():
            out[idx] = (c, t)
    return out


def run_parent(args):
    col, out_path = args.col, args.out
    done = completed(out_path, col)  # always empty: /kaggle/working starts fresh each run
    seed = {i: v for i, v in seeded_rows(col).items() if i not in done}
    log.info("seed: %d rows from attached datasets", len(seed))
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=["idx", "context", col])
        w.writeheader()
        for idx in sorted(seed):
            c, t = seed[idx]
            w.writerow({"idx": idx, "context": c, col: t})

    if not transformers_has_loss_kwargs():
        # Phi-4-mini's remote modeling.py needs utils.LossKwargs (4.46-4.50 only).
        log.info("downgrading transformers to 4.50.3 for Phi modeling...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "transformers==4.50.3"]
        )

    if find_adapter(args.adapter) is not None:
        # Unconditional: the image preinstalls peft but pins torchao 0.10, which
        # peft rejects at PeftModel import (workers died at 88s/120s on v1/v2).
        log.info("ensuring peft + torchao>=0.16 for LoRA...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-q",
                "peft",
                "--no-deps",
                "--upgrade",
                "torchao>=0.16.0",
            ]
        )
        from peft import PeftModel  # noqa: F401 — fail fast here, before spawning GPU workers

    script = os.path.abspath(__file__)
    assert os.path.isfile(script), script
    procs = []
    for shard in (0, 1):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(shard))
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    script,
                    "--shard",
                    str(shard),
                    "--col",
                    col,
                    "--out",
                    out_path,
                    "--adapter",
                    args.adapter or "",
                ],
                env=env,
            )
        )
    rc = [p.wait() for p in procs]
    if any(rc):
        raise SystemExit(f"worker failed, returncodes={rc}")

    merged = read_generated(out_path, col)
    for shard in (0, 1):
        merged.update(read_generated(part_path(out_path, shard), col))
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=["idx", "context", col])
        w.writeheader()
        for idx in sorted(merged):
            c, t = merged[idx]
            w.writerow({"idx": idx, "context": c, col: t})
    log.info("DONE %d/10000 rows -> %s", len(merged), out_path)


def strip_to_generated(out, input_len):
    """Drop the (left-padded) prompt, keeping only newly generated tokens.

    Rows are padded to a common width, so generation always starts at input_len.
    Slicing at the per-row *real* length instead leaks prompt tokens for padded
    rows (the parrot bug: short rows re-emitted their prompt tail as the answer).
    """
    return out[:, input_len:]


def chat_kwargs(model_id=MODEL_ID):
    """Extra apply_chat_template kwargs. Only Qwen3 templates take enable_thinking."""
    return {"enable_thinking": False} if "qwen" in model_id.lower() else {}


def transformers_has_loss_kwargs():
    """Phi-4-mini's remote modeling imports LossKwargs, which only exists in
    transformers.utils up to 4.50.x (removed after; image ships 5.16.1)."""
    try:
        from transformers.utils import LossKwargs  # noqa: F401
        return True
    except ImportError:
        return False


def load_model(model_id, loader):
    """from_pretrained with dtype (new transformers) falling back to torch_dtype (<4.53)."""
    import torch  # lazy: keeps module importable without GPU deps

    try:
        return loader.from_pretrained(
            model_id, trust_remote_code=True, dtype=torch.float16, device_map="auto"
        )
    except TypeError:
        return loader.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"
        )


def part_path(out_path, shard):
    return out_path.replace(".csv", f"_part_{shard}.csv")


def run_child(args, shard):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    col, out_path, model_id = args.col, args.out, args.model_id
    torch.cuda.set_device(0)  # only its own GPU is visible
    assert torch.cuda.device_count() == 1, torch.cuda.device_count()
    rows = load_contexts()
    half = [r for r in rows if (r[0] < 5000) == (shard == 0)]
    skip = completed(out_path, col) | {
        i for i, (_, t) in seeded_rows(col).items() if t.strip()
    }
    todo = [(i, c) for i, c in half if i not in skip]
    log.info("[gpu%d] todo=%d", shard, len(todo))

    adapter = find_adapter(args.adapter or None)
    tok_src = adapter if adapter and os.path.exists(os.path.join(adapter, "tokenizer.json")) else model_id
    tok = AutoTokenizer.from_pretrained(tok_src, trust_remote_code=True)
    tok.padding_side = "left"  # decoder-only batching must pad left
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_model(model_id, AutoModelForCausalLM)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
        log.info("[gpu%d] LoRA adapter: %s", shard, adapter)
    model.eval()
    log.info("[gpu%d] model on %s", shard, model.device)

    part = part_path(out_path, shard)
    new_file = not os.path.exists(part)
    fout = open(part, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(fout, fieldnames=["idx", "context", col])
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
                tokenize=False,
                **chat_kwargs(getattr(args, "model_id", MODEL_ID)),
            )
            for _, c in batch
        ]
        enc = tok(texts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(model.device)
        attn = enc["attention_mask"].to(model.device)
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
        for (idx, c), gen in zip(batch, strip_to_generated(out, input_ids.shape[1])):
            w.writerow({"idx": idx, "context": c, col: tok.decode(gen, skip_special_tokens=True).strip()})
            n_new += 1
        fout.flush()
        if n_new % 32 == 0:
            dt = time.time() - t0
            log.info("[gpu%d] %d/%d new (%.0f rows/h)", shard, n_new, len(todo), n_new / dt * 3600)
    fout.close()
    dt = time.time() - t0
    log.info("[gpu%d] DONE %d new (%.0f rows/h) -> %s", shard, n_new, n_new / max(dt, 1) * 3600, part)


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=None)
    ap.add_argument("--col", default=COL)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--model-id", default=MODEL_ID)
    ap.add_argument("--adapter", default="")
    args = ap.parse_args(argv)
    if args.shard is None:
        run_parent(args)
    else:
        run_child(args, args.shard)


if __name__ == "__main__":
    main()
