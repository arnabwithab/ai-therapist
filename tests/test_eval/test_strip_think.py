"""Tests for eval/mbench_rag/strip_think.py — pure string/CSV logic."""

import csv

from eval.mbench_rag import strip_think as S


def test_stripped_post_think_final():
    assert S.strip_think("<think>hmm</think>  Hello there  ") == "Hello there"


def test_no_think_passthrough():
    assert S.strip_think("  Just an answer  ") == "Just an answer"


def test_unclosed_think_dropped():
    assert S.strip_think("<think>trail off...") == ""


def test_main_rewrites_col(tmp_path):
    src = str(tmp_path / "in.csv")
    out = str(tmp_path / "final.csv")
    with open(src, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "context", "resp"])
        w.writeheader()
        w.writerow({"idx": "0", "context": "c", "resp": "<think>t</think> Final"})
        w.writerow({"idx": "1", "context": "c", "resp": "Plain"})
    n, dropped = S.main(["--in", src, "--col", "resp", "--out", out])
    assert (n, dropped) == (2, 0)
    with open(out, newline="", encoding="utf-8") as f:
        rows = {r["idx"]: r["resp"] for r in csv.DictReader(f)}
    assert rows == {"0": "Final", "1": "Plain"}
