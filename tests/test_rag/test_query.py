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


def test_lazy_init_survives_concurrent_first_calls(monkeypatch):
    """Concurrent first requests must build one client/embedder, not race."""
    import threading
    import time

    made_clients, made_embedders = [], []

    class FakeClient:
        def __init__(self, *a, **k):
            time.sleep(0.05)  # widen the race window
            made_clients.append(1)

        def get_collection(self, name):
            class C:
                def query(self, **k):
                    return {
                        "documents": [["d"]],
                        "metadatas": [[{"source": "s", "chapter": "c"}]],
                        "distances": [[0.0]],
                    }

            return C()

    class FakeEmbedder:
        def __init__(self, *a, **k):
            time.sleep(0.05)
            made_embedders.append(1)

        def encode(self, texts, **k):
            return [[0.0]] * len(texts)

    monkeypatch.setattr(qmod.chromadb, "PersistentClient", FakeClient)
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", FakeEmbedder)
    monkeypatch.setattr(qmod, "_client", None)
    monkeypatch.setattr(qmod, "_embedder", None)
    threads = [threading.Thread(target=retrieve, args=(f"q{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(made_clients) == 1
    assert len(made_embedders) == 1
