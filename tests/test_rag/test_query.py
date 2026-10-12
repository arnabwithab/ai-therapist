"""Tests for src/rag/query.py using a temp on-disk Chroma collection."""

import chromadb

from src.rag import query as qmod
from src.rag.query import retrieve


def _seed_collection(path):
    client = chromadb.PersistentClient(path=str(path))
    col = client.get_or_create_collection("test-kb")
    docs = [
        "Mindfulness meditation reduces anxiety through breath awareness.",
        "Cognitive behavioral therapy uses thought records to reframe beliefs.",
    ]
    col.add(
        ids=["a", "b"],
        documents=docs,
        metadatas=[
            {"source": "mind.txt", "chapter": "", "chunk_id": 0},
            {"source": "cbt.txt", "chapter": "", "chunk_id": 0},
        ],
        embeddings=[[float(hash(d) % 100)] * 8 for d in docs],
    )
    return col


def test_retrieve_returns_text_source_and_score(tmp_path, monkeypatch):
    col = _seed_collection(tmp_path / "chroma")
    monkeypatch.setattr(qmod, "_collection", lambda: col)
    monkeypatch.setattr(qmod, "_embed", lambda texts: [[float(hash(t) % 100)] * 8 for t in texts])
    hits = retrieve("breath awareness for anxiety", n=1)
    assert len(hits) == 1
    assert set(hits[0]) == {"text", "source", "chapter", "score"}
    assert hits[0]["source"].endswith(".txt")


def test_retrieve_honours_source_filter(tmp_path, monkeypatch):
    col = _seed_collection(tmp_path / "chroma")
    monkeypatch.setattr(qmod, "_collection", lambda: col)
    monkeypatch.setattr(qmod, "_embed", lambda texts: [[float(hash(t) % 100)] * 8 for t in texts])
    hits = retrieve("therapy", n=5, source_filter="cbt.txt")
    assert hits and all(h["source"] == "cbt.txt" for h in hits)
