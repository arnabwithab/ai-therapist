"""Tests for eval/mbench_rag/fetch_contexts.py — no network, no chroma, no GPU."""

import json

from backend.core import chat as serving
from eval.mbench_rag import fetch_contexts as F


def _passages():
    return [
        {"text": "Breathe with the belly.", "source": "s1", "chapter": "c1", "score": 0.9},
        {"text": "Notice the feeling.", "source": "s2", "chapter": "c2", "score": 0.8},
    ]


def test_build_rag_block_matches_serving_byte_for_byte():
    served = serving._retrieved_message(_passages())["content"]
    assert F.build_rag_block(_passages()) in served
    # exact body format: [source — chapter]\ntext joined by blank lines
    assert F.build_rag_block(_passages()) == (
        "[s1 — c1]\nBreathe with the belly.\n\n[s2 — c2]\nNotice the feeling."
    )


def test_record_round_trips_through_jsonl(tmp_path):
    out = tmp_path / "ctx.jsonl"
    F.append_record(str(out), 7, _passages())
    F.append_record(str(out), 7, _passages())  # dupes allowed at write time
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [r["idx"] for r in rows] == [7, 7]
    assert rows[0]["passages"][1]["chapter"] == "c2"


def test_load_done_skips_completed(tmp_path):
    out = tmp_path / "ctx.jsonl"
    F.append_record(str(out), 0, _passages())
    assert F.load_done(str(out)) == {0}
    assert F.load_done(str(tmp_path / "missing.jsonl")) == set()


def test_fetch_all_only_calls_retrieve_for_missing(tmp_path):
    out = tmp_path / "ctx.jsonl"
    F.append_record(str(out), 1, _passages())
    calls = []

    def fake_retrieve(q, n=5, source_filter=None):
        calls.append(q)
        return [{"text": "t", "source": "s", "chapter": "c", "score": 1.0}]

    rows = [(0, "ctx0"), (1, "ctx1"), (2, "ctx2")]
    n = F.fetch_all(rows, str(out), retrieve_fn=fake_retrieve, top_k=5)
    assert n == 2
    assert calls == ["ctx0", "ctx2"]
    assert F.load_done(str(out)) == {0, 1, 2}
