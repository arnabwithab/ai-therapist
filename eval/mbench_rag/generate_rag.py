"""OARS + RAG eval pass over MentalBench-100k (model + RAG pipeline).

Same 10k contexts as the base runs, but generation goes through the serving
path: backend/core/chat.py's OARS prompt + src/rag/query.py retrieval, sent to
a vLLM OpenAI endpoint. Runs on the 24GB box (single GPU, no Kaggle sharding):

    make model-server-24g   # serve MODEL_ID at full precision
    make eval-rag           # this script (one shard; loop SHARD=0..N-1)

Model swap = restart the model server with another MODEL_ID; this script only
needs --model to match its --served-model-name. --no-rag reuses the same OARS
prompt with empty retrieval (ablation isolating the RAG effect from the base
prompt's psychiatrist style). Shard files push to a HF dataset incrementally,
so runs are resumable and remotely viewable.
"""

import argparse
import csv
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from backend.core.chat import build_messages
from eval.hf_push import push_file
from eval.mbench_baseline.generate import completed, load_contexts
from src.rag.query import retrieve

COL = "response"
OUT = "data/eval/mbench-rag/output_rag.csv"
MAX_TOKENS = 512  # lockstep with the base rows (temp 0.7 / top_p 0.8 too)
WORKERS = 16
PUSH_EVERY = 250

log = logging.getLogger("gen-rag")


def shard_rows(rows, shard, num_shards):
    """Rows for this shard: idx % num_shards == shard (full coverage, no overlap)."""
    return [r for r in rows if r[0] % num_shards == shard]


def part_path(out_path, shard):
    return out_path.replace(".csv", f"_shard_{shard}.csv")


def remote_path(col, filename):
    return f"{col}/{filename}"


def format_sources(passages):
    return json.dumps(
        [{"source": p["source"], "chapter": p["chapter"], "score": p["score"]} for p in passages]
    )


def request_body(messages, model, max_tokens=MAX_TOKENS):
    return {
        "model": model,
        "messages": messages,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repetition_penalty": 1.1,
        "max_tokens": max_tokens,
        "stream": False,
    }


def generate_one(client, messages, model, max_tokens=MAX_TOKENS, retries=3):
    last = None
    for attempt in range(retries):
        try:
            r = client.post("/v1/chat/completions", json=request_body(messages, model, max_tokens))
            r.raise_for_status()
            return r.json()["choices"][0]["message"].get("content", "").strip()
        except (httpx.HTTPError, ValueError, KeyError, IndexError) as e:
            last = e  # transient 5xx / truncated JSON: back off, retry
            time.sleep(2 * (attempt + 1))
    raise last


def answer_one(client, idx, context, model, max_tokens, rag_top_k, no_rag):
    """One eval row: retrieve (unless ablated), build the serving prompt, generate."""
    passages = [] if no_rag else retrieve(context, n=rag_top_k)
    messages = build_messages([{"role": "user", "content": context}], passages)
    text = generate_one(client, messages, model, max_tokens)
    return {"idx": idx, "context": context, "text": text, "sources": format_sources(passages)}


def merge_parts(out_path, col, num_shards):
    """Rewrite the merged OUT from all existing shard files (partial OK)."""
    merged = {}
    for shard in range(num_shards):
        part = part_path(out_path, shard)
        if not os.path.exists(part):
            continue
        with open(part, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("idx", "").isdigit():
                    merged[int(r["idx"])] = r
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=["idx", "context", col, "sources"])
        w.writeheader()
        for idx in sorted(merged):
            r = merged[idx]
            w.writerow(
                {
                    "idx": idx,
                    "context": r.get("context", ""),
                    col: r.get(col, ""),
                    "sources": r.get("sources", ""),
                }
            )
    return len(merged)


def run_shard(args):
    rows = load_contexts()
    if args.limit:
        rows = rows[: args.limit]
    mine = shard_rows(rows, args.shard, args.num_shards)
    part = part_path(args.out, args.shard)
    done = completed(part, args.col) | completed(args.out, args.col)
    todo = [(i, c) for i, c in mine if i not in done]
    log.info(
        "shard %d/%d: todo=%d already_done=%d",
        args.shard,
        args.num_shards,
        len(todo),
        len(mine) - len(todo),
    )

    parent = os.path.dirname(args.out)
    if parent:
        os.makedirs(parent, exist_ok=True)
    new_file = not os.path.exists(part)
    client = httpx.Client(base_url=args.vllm_url, timeout=300)
    n_new, t0 = 0, time.time()
    try:
        with open(part, "a", newline="", encoding="utf-8") as fout:
            w = csv.DictWriter(fout, fieldnames=["idx", "context", args.col, "sources"])
            if new_file:
                w.writeheader()
                fout.flush()

            def job(item):
                idx, context = item
                row = answer_one(
                    client,
                    idx,
                    context,
                    args.model,
                    args.max_tokens,
                    args.rag_top_k,
                    args.no_rag,
                )
                return {
                    "idx": idx,
                    "context": context,
                    args.col: row["text"],
                    "sources": row["sources"],
                }

            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for row in ex.map(job, todo):
                    w.writerow(row)
                    fout.flush()
                    n_new += 1
                    if n_new % args.push_every == 0:
                        push_file(part, args.hf_repo, remote_path(args.col, os.path.basename(part)))
                        dt = time.time() - t0
                        log.info(
                            "shard %d: %d new (%.0f rows/h)",
                            args.shard,
                            n_new,
                            n_new / max(dt, 1) * 3600,
                        )
    finally:
        client.close()
    push_file(part, args.hf_repo, remote_path(args.col, os.path.basename(part)))
    n_merged = merge_parts(args.out, args.col, args.num_shards)
    push_file(args.out, args.hf_repo, remote_path(args.col, os.path.basename(args.out)))
    log.info("shard %d DONE %d new, merged %d rows -> %s", args.shard, n_new, n_merged, args.out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("AI_THERAPIST_MODEL", "ai-therapist-qwen3-4b"))
    ap.add_argument("--vllm-url", default=os.getenv("VLLM_URL", "http://localhost:8000"))
    ap.add_argument("--col", default=COL)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--rag-top-k", type=int, default=int(os.getenv("RAG_TOP_K", "5")))
    ap.add_argument("--no-rag", action="store_true", help="same OARS prompt, empty retrieval")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test first N rows")
    ap.add_argument("--push-every", type=int, default=PUSH_EVERY)
    ap.add_argument("--hf-repo", default="", help="HF dataset repo for live shard pushes")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    run_shard(args)


if __name__ == "__main__":
    main()
