"""Tests for src/rag/query.py using a temp on-disk Chroma collection."""

import chromadb

from src.rag import query as qmod
from src.rag.query import detect_faith_signal, resolve_religion, retrieve


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


DIM = 4
_QVEC = [1.0] * DIM  # query embedding; general docs sit on it, faith docs farther out


def _seed_mixed(path, n_faith=4):
    """Temp collection: n_faith hindu docs + 4 general docs, controlled embeddings."""
    client = chromadb.PersistentClient(path=str(path))
    col = client.get_or_create_collection("test-kb")
    faith_docs = [f"Hindu counselling passage {i} on dharma and the Gita." for i in range(n_faith)]
    general_docs = [f"General therapy passage {i} on thought records." for i in range(4)]
    col.add(
        ids=[f"h{i}" for i in range(n_faith)] + [f"g{i}" for i in range(4)],
        documents=faith_docs + general_docs,
        metadatas=[
            {"source": f"hindu{i}.txt", "chapter": "", "religion": "hindu"} for i in range(n_faith)
        ]
        + [{"source": f"gen{i}.txt", "chapter": "", "religion": "general"} for i in range(4)],
        embeddings=[[2.0] * DIM] * n_faith + [[1.0] * DIM] * 4,
    )
    return col


def _patch_mixed(tmp_path, monkeypatch, n_faith=4):
    col = _seed_mixed(tmp_path / "chroma", n_faith=n_faith)
    monkeypatch.setattr(qmod, "_collection", lambda: col)
    monkeypatch.setattr(qmod, "_embed", lambda texts: [_QVEC for _ in texts])
    return col


def test_retrieve_religion_filter_hit(tmp_path, monkeypatch):
    _patch_mixed(tmp_path, monkeypatch)
    hits = retrieve("anxiety about exams", n=5, religion="hindu")
    assert len(hits) == 5
    assert all(h["source"].startswith("hindu") for h in hits[:3])


def test_retrieve_religion_split_shape_3_2(tmp_path, monkeypatch):
    _patch_mixed(tmp_path, monkeypatch)
    hits = retrieve("anxiety about exams", n=5, religion="hindu")
    faith = [h for h in hits if h["source"].startswith("hindu")]
    general = [h for h in hits if h["source"].startswith("gen")]
    assert len(hits) == 5
    assert len(faith) == 3
    assert len(general) == 2


def test_retrieve_religion_backfills_general_when_faith_short(tmp_path, monkeypatch):
    _patch_mixed(tmp_path, monkeypatch, n_faith=1)
    hits = retrieve("anxiety about exams", n=5, religion="hindu")
    assert len(hits) == 5
    assert sum(1 for h in hits if h["source"].startswith("hindu")) == 1


def test_retrieve_religion_zero_hit_falls_back_to_general(tmp_path, monkeypatch):
    _patch_mixed(tmp_path, monkeypatch, n_faith=0)
    hits = retrieve("anxiety about exams", n=4, religion="jain")
    assert len(hits) == 4
    assert all(h["source"].startswith("gen") for h in hits)


def test_retrieve_religion_combines_with_source_filter(tmp_path, monkeypatch):
    _patch_mixed(tmp_path, monkeypatch)
    hits = retrieve("anxiety", n=5, source_filter="hindu1.txt", religion="hindu")
    assert hits and all(h["source"] == "hindu1.txt" for h in hits)


def test_retrieve_backward_compat_single_arg(tmp_path, monkeypatch):
    col = _seed_collection(tmp_path / "chroma")
    monkeypatch.setattr(qmod, "_collection", lambda: col)
    monkeypatch.setattr(qmod, "_embed", lambda texts: [[float(hash(t) % 100)] * 8 for t in texts])
    hits = retrieve("breath awareness for anxiety")
    assert hits and set(hits[0]) == {"text", "source", "chapter", "score"}


def test_resolve_religion_gating():
    assert resolve_religion("Hindu", "anchor", "yes, when it fits", "") == "hindu"
    assert resolve_religion("muslim", "a little", "yes", "") == "islam"
    assert resolve_religion("Hindu", "none", "yes", "") is None
    assert resolve_religion("Hindu", "not at all", "yes", "") is None
    assert resolve_religion("Hindu", "anchor", "keep it neutral", "") is None
    assert resolve_religion("Hindu", "anchor", "no", "") is None
    assert resolve_religion("other", "anchor", "yes", "") is None
    assert resolve_religion("", "anchor", "yes", "") is None


def test_resolve_religion_override():
    assert resolve_religion("muslim", "none", "no", "I stopped doing namaaz") == "islam"
    assert resolve_religion("Hindu", "none", "keep it neutral", "God is punishing me") == "hindu"
    assert resolve_religion("Hindu", "none", "no", "work stress lately") is None
    assert resolve_religion("Hindu", "none", "no", "I stopped doing namaaz") is None


def test_detect_faith_signal():
    assert detect_faith_signal("I stopped doing namaaz") == "islam"
    assert detect_faith_signal("my parents' puja") == "hindu"
    assert detect_faith_signal("gurbani kirtan") == "sikh"
    assert detect_faith_signal("reading the bible") == "christian"
    assert detect_faith_signal("metta practice") == "buddhist"
    assert detect_faith_signal("work deadline stress") is None
    assert detect_faith_signal("") is None
