"""Tests for eval/mbench_rag/fetch_contexts.py — no network, no chroma, no GPU."""

import json

import pytest

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


def test_build_rag_block_is_servings_formatter():
    from backend.core import chat as serving_mod

    assert F.build_rag_block is serving_mod.format_passages


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


def test_main_forwards_top_k_to_fetch(tmp_path, monkeypatch, caplog):
    seen = {}

    def fake_retrieve(q, n=5, source_filter=None):
        seen.setdefault("n", n)
        return [{"text": "t", "source": "s", "chapter": "c", "score": 1.0}]

    monkeypatch.setattr(F, "retrieve", fake_retrieve)
    monkeypatch.setattr(F, "write_meta", lambda *a: None)
    base = tmp_path / "base.csv"
    base.write_text("idx,context\n0,hello\n", encoding="utf-8")
    out = tmp_path / "ctx.jsonl"
    F.main(
        [
            "--baseline",
            str(base),
            "--out",
            str(out),
            "--csv",
            str(tmp_path / "o.csv"),
            "--top-k",
            "3",
        ]
    )
    assert seen["n"] == 3
    assert F.load_done(str(out)) == {0}


def test_emit_csv_refuses_incomplete_run(tmp_path):

    base = tmp_path / "base.csv"
    base.write_text("idx,context\n0,a\n1,b\n", encoding="utf-8")
    out = tmp_path / "ctx.jsonl"
    F.append_record(str(out), 0, _passages())
    with pytest.raises(SystemExit, match="incomplete"):
        F.emit_csv(str(out), str(tmp_path / "o.csv"), str(base))


def test_emit_csv_rejects_unknown_idx(tmp_path):

    base = tmp_path / "base.csv"
    base.write_text("idx,context\n0,a\n", encoding="utf-8")
    out = tmp_path / "ctx.jsonl"
    F.append_record(str(out), 9, _passages())
    with pytest.raises(SystemExit, match="not in baseline"):
        F.emit_csv(str(out), str(tmp_path / "o.csv"), str(base))


def test_check_resume_meta():

    ok = {"top_k": 5, "embed_model": "m", "collection_count": 10}
    F.check_resume_meta(ok, top_k=5, embed_model="m", collection_count=10)
    with pytest.raises(SystemExit, match="top_k"):
        F.check_resume_meta({**ok, "top_k": 3}, top_k=5, embed_model="m", collection_count=10)
    with pytest.raises(SystemExit, match="embed_model"):
        F.check_resume_meta(
            {**ok, "embed_model": "x"}, top_k=5, embed_model="m", collection_count=10
        )
    with pytest.raises(SystemExit, match="collection_count"):
        F.check_resume_meta(
            {**ok, "collection_count": 11}, top_k=5, embed_model="m", collection_count=10
        )
