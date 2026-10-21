"""Tests for eval/mbench_rag/generate_rag.py — pure logic, no GPU/network."""

import argparse
import csv
import json

import httpx

from eval.mbench_rag import generate_rag as R


def _ns(tmp_path, **kw):
    base = {
        "model": "test-model",
        "vllm_url": "http://localhost:8000",
        "col": "response",
        "out": str(tmp_path / "out.csv"),
        "shard": 0,
        "num_shards": 1,
        "rag_top_k": 5,
        "no_rag": False,
        "max_tokens": 512,
        "workers": 2,
        "limit": 0,
        "push_every": 250,
        "hf_repo": "",
    }
    base.update(kw)
    return argparse.Namespace(**base)


class _Resp:
    def __init__(self, text=None, error=None):
        self._text = text
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        return {"choices": [{"message": {"content": self._text}}]}


class _Client:
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def post(self, path, json=None):
        self.calls += 1
        item = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    def close(self):
        pass


def test_shard_rows_cover_all_without_overlap():
    rows = [(i, f"c{i}") for i in range(10)]
    parts = [R.shard_rows(rows, s, 3) for s in range(3)]
    assert sorted(i for p in parts for i, _ in p) == list(range(10))
    assert [i for i, _ in parts[0]] == [0, 3, 6, 9]
    assert [i for i, _ in parts[1]] == [1, 4, 7]


def test_part_and_remote_paths():
    assert R.part_path("data/out.csv", 2) == "data/out_shard_2.csv"
    assert R.remote_path("response", "out_shard_0.csv") == "response/out_shard_0.csv"


def test_format_sources_roundtrip():
    passages = [{"text": "t", "source": "s", "chapter": "c", "score": 0.9}]
    assert json.loads(R.format_sources(passages)) == [{"source": "s", "chapter": "c", "score": 0.9}]
    assert json.loads(R.format_sources([])) == []


def test_request_body_matches_base_sampling():
    body = R.request_body([{"role": "user", "content": "hi"}], "m", max_tokens=512)
    assert body["model"] == "m"
    assert (body["temperature"], body["top_p"], body["max_tokens"]) == (0.7, 0.8, 512)
    assert (body["top_k"], body["repetition_penalty"], body["stream"]) == (20, 1.1, False)


def test_generate_one_returns_stripped_text():
    assert R.generate_one(_Client([_Resp("  hello  ")]), [], "m") == "hello"


def test_generate_one_retries_then_succeeds():
    err = httpx.HTTPError("boom")
    c = _Client([err, _Resp("ok")])
    assert R.generate_one(c, [], "m") == "ok"
    assert c.calls == 2


def test_generate_one_raises_after_retries():
    c = _Client([httpx.HTTPError("down")])
    try:
        R.generate_one(c, [], "m", retries=2)
    except httpx.HTTPError:
        assert c.calls == 2
    else:  # shortcut: fail loudly if the retry loop ever swallows the error
        raise AssertionError("generate_one swallowed the error")


def test_answer_one_no_rag_has_no_retrieved_block(monkeypatch):
    monkeypatch.setattr(R, "retrieve", lambda *a, **k: (_ for _ in ()).throw(Exception("no")))
    row = R.answer_one(_Client([_Resp("r")]), 0, "c", "m", 512, 5, True)
    assert row["sources"] == "[]"
    assert row["text"] == "r"


def test_answer_one_rag_injects_and_records_sources(monkeypatch):
    seen = {}

    def fake_retrieve(q, n=5):
        seen["q"] = q
        return [{"text": "t", "source": "s", "chapter": "c", "score": 0.5}]

    def fake_generate(client, messages, model, max_tokens):
        seen["retrieved"] = any("<retrieved_techniques>" in m["content"] for m in messages)
        return "r"

    monkeypatch.setattr(R, "retrieve", fake_retrieve)
    monkeypatch.setattr(R, "generate_one", fake_generate)
    row = R.answer_one(object(), 3, "worry", "m", 512, 5, False)
    assert seen == {"q": "worry", "retrieved": True}
    assert json.loads(row["sources"]) == [{"source": "s", "chapter": "c", "score": 0.5}]


def test_merge_parts_sorts_and_keeps_sources(tmp_path):
    out = str(tmp_path / "out.csv")
    for shard, rows in (
        (0, [{"idx": "2", "context": "c2", "response": "r2", "sources": "[]"}]),
        (1, [{"idx": "1", "context": "c1", "response": "r1", "sources": "[s]"}]),
    ):
        with open(R.part_path(out, shard), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["idx", "context", "response", "sources"])
            w.writeheader()
            w.writerows(rows)
    assert R.merge_parts(out, "response", 2) == 2
    with open(out, newline="", encoding="utf-8") as f:
        merged = list(csv.DictReader(f))
    assert [(r["idx"], r["sources"]) for r in merged] == [("1", "[s]"), ("2", "[]")]


def test_run_shard_resumes_and_merges(monkeypatch, tmp_path):
    out = str(tmp_path / "out.csv")
    with open(R.part_path(out, 0), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "context", "response", "sources"])
        w.writeheader()
        w.writerow({"idx": "0", "context": "c0", "response": "old", "sources": "[]"})
    monkeypatch.setattr(R, "load_contexts", lambda: [(0, "c0"), (1, "c1"), (2, "c2")])
    calls = []

    def fake_answer(client, idx, context, *a, **k):
        calls.append(idx)
        return {"idx": idx, "context": context, "text": f"new{idx}", "sources": "[]"}

    monkeypatch.setattr(R, "answer_one", fake_answer)
    R.run_shard(_ns(tmp_path, shard=0, num_shards=2))
    assert calls == [2]  # idx 0 done, idx 1 belongs to shard 1
    with open(out, newline="", encoding="utf-8") as f:
        merged = {r["idx"]: r["response"] for r in csv.DictReader(f)}
    assert merged == {"0": "old", "2": "new2"}
