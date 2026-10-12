"""Read-only retrieval API — the only thing the serving layer imports."""

import logging

import chromadb

from src.rag import config

log = logging.getLogger(__name__)

_client = None
_embedder = None


def _collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return _client.get_collection(config.COLLECTION)


def _embed(texts):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    return _embedder.encode(texts, normalize_embeddings=True).tolist()


def retrieve(query, n=5, source_filter=None):
    """Return top-n passages as {text, source, chapter, score} dicts."""
    where = {"source": source_filter} if source_filter else None
    res = _collection().query(
        query_embeddings=_embed([query]),
        n_results=n,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append(
            {
                "text": doc,
                "source": meta.get("source", ""),
                "chapter": meta.get("chapter", ""),
                "score": 1.0 - float(dist) / 2.0,
            }
        )
    log.info("retrieve: %d hits for %r", len(hits), query[:60])
    return hits
