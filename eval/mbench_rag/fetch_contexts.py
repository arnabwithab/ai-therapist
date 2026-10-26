"""Precompute RAG passages for all 10k MentalBench contexts, once.

Writes data/eval/mbench-rag-contexts.jsonl (one line per idx:
{idx, passages: [{text, source, chapter, score}]}) plus a .meta.json sidecar
(embed model, collection size, top_k) so a stale corpus fails loudly on rebuild.
A second pass emits data/eval/mbench-rag-judge.csv [idx, context, rag_context]
where rag_context is byte-identical to the block serving injects — evals then
need no chroma, no embedder, no serving stack.

Usage: uv run python -m eval.mbench_rag.fetch_contexts [--top-k 5]
"""

import argparse
import csv
import json
import logging
import os

from backend.core.chat import RAG_TOP_K
from backend.core.chat import format_passages as build_rag_block
from src.rag import config as rag_config
from src.rag.query import retrieve

BASELINE = "data/eval/mbench-qwen3-4b-baseline/output_mbench.csv"
OUT_JSONL = "data/eval/mbench-rag-contexts.jsonl"
OUT_CSV = "data/eval/mbench-rag-judge.csv"

log = logging.getLogger("fetch-contexts")


# build_rag_block is backend.core.chat.format_passages (single source of truth,
# imported at module top). The wire format is pinned by test_exact_block_format.


def check_resume_meta(meta, top_k, embed_model, collection_count):
    """Refuse to append rows fetched under different settings (silent mix otherwise)."""
    problems = []
    if meta.get("top_k") != top_k:
        problems.append(f"top_k {meta.get('top_k')} != {top_k}")
    if meta.get("embed_model") != embed_model:
        problems.append(f"embed_model {meta.get('embed_model')} != {embed_model}")
    if meta.get("collection_count") != collection_count:
        problems.append(f"collection_count {meta.get('collection_count')} != {collection_count}")
    if problems:
        raise SystemExit(
            "refusing resume: " + "; ".join(problems) + " (delete the jsonl + meta to start over)"
        )


def _read_meta(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def append_record(path, idx, passages):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"idx": idx, "passages": passages}) + "\n")


def load_done(path):
    if not os.path.exists(path):
        return set()
    done = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(int(json.loads(line)["idx"]))
    return done


def load_contexts(path=BASELINE):
    with open(path, newline="", encoding="utf-8") as f:
        return [
            (int(r["idx"]), r["context"]) for r in csv.DictReader(f) if r.get("idx", "").isdigit()
        ]


def fetch_all(rows, out_path, retrieve_fn=retrieve, top_k=RAG_TOP_K):
    """Fetch passages for rows not already in out_path. Returns new row count."""
    done = load_done(out_path)
    fresh = 0
    for pos, (idx, ctx) in enumerate(rows):
        if idx in done:
            continue
        append_record(out_path, idx, retrieve_fn(ctx, n=top_k))
        fresh += 1
        if fresh % 500 == 0:
            log.info("fetched %d new (%d/%d rows)", fresh, pos + 1, len(rows))
    log.info("done: %d new, %d total", fresh, len(done) + fresh)
    return fresh


def collection_count():
    import chromadb

    client = chromadb.PersistentClient(path=rag_config.CHROMA_DIR)
    return client.get_collection(rag_config.COLLECTION).count()


def write_meta(jsonl_path, top_k):
    meta = {
        "embed_model": rag_config.EMBED_MODEL,
        "collection": rag_config.COLLECTION,
        "collection_count": collection_count(),
        "top_k": top_k,
    }
    with open(jsonl_path + ".meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
        f.write("\n")
    log.info("meta: %s", meta)


def emit_csv(jsonl_path, csv_path, baseline_path=BASELINE):
    contexts = dict(load_contexts(baseline_path))
    with open(jsonl_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if len(records) != len(contexts):
        missing = sorted(set(contexts) - {r["idx"] for r in records})[:5]
        raise SystemExit(
            f"refusing to emit: {len(records)}/{len(contexts)} rows (e.g. missing {missing}) — fetch is incomplete"
        )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "context", "rag_context"])
        w.writeheader()
        for r in sorted(records, key=lambda r: r["idx"]):
            ctx = contexts.get(r["idx"])
            if ctx is None:
                raise SystemExit(f"jsonl idx {r['idx']} not in baseline {baseline_path}")
            w.writerow(
                {"idx": r["idx"], "context": ctx, "rag_context": build_rag_block(r["passages"])}
            )
    log.info("wrote %d rows -> %s", len(records), csv_path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_JSONL)
    ap.add_argument("--csv", default=OUT_CSV)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--top-k", type=int, default=RAG_TOP_K)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rows = load_contexts(a.baseline)
    log.info("contexts: %d", len(rows))
    meta = _read_meta(a.out + ".meta.json")
    if meta is not None and load_done(a.out):
        check_resume_meta(
            meta,
            top_k=a.top_k,
            embed_model=rag_config.EMBED_MODEL,
            collection_count=collection_count(),
        )
    fetch_all(rows, a.out, retrieve_fn=retrieve, top_k=a.top_k)
    write_meta(a.out, a.top_k)
    emit_csv(a.out, a.csv, a.baseline)


if __name__ == "__main__":
    main()
