"""Strip <think> blocks, materializing judge-ready finals.

DPO adapters emit thinking traces (canonical 1.7B run: 10k/10k rows); the
judge scores post-</think> finals only, mirroring the _final.csv precedent.
Unclosed thoughts (truncated rows) become blank and the judge skips blanks.

Usage: uv run python -m eval.mbench_rag.strip_think --in IN.csv --col COL --out FINAL.csv
"""

import argparse
import csv
import logging

log = logging.getLogger("strip-think")


def strip_think(text):
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    if "<think>" in text:
        return ""  # truncated thought, no answer: let the judge skip it as blank
    return text.strip()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--col", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    with open(a.src, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    dropped = 0
    for r in rows:
        r[a.col] = strip_think(r.get(a.col) or "")
        dropped += not r[a.col]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else ["idx", "context", a.col])
        w.writeheader()
        w.writerows(rows)
    log.info("wrote %d rows (%d blank) -> %s", len(rows), dropped, a.out)
    return len(rows), dropped


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
