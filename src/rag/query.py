"""Read-only retrieval API — the only thing the serving layer imports."""

import logging
import re
import threading

import chromadb

from src.rag import config

log = logging.getLogger(__name__)

_client = None
_embedder = None
_lock = threading.Lock()  # lazy init races under concurrent first requests

# Profile religion label -> KB manifest tag (manifest vocab:
# hindu|islam|sikh|christian|buddhist|jain|secular|general).
PROFILE_RELIGION_TO_KB = {
    "hindu": "hindu",
    "hinduism": "hindu",
    "muslim": "islam",
    "islam": "islam",
    "sikh": "sikh",
    "sikhi": "sikh",
    "sikhism": "sikh",
    "christian": "christian",
    "christianity": "christian",
    "buddhist": "buddhist",
    "buddhism": "buddhist",
    "jain": "jain",
    "jainism": "jain",
    "secular": "secular",
}

# Curated per-tradition signals for the last-message keyword scan.
FAITH_KEYWORDS = {
    "hindu": (
        "puja",
        "gita",
        "krishna",
        "mandir",
        "bhagwan",
        "karma",
        "dharma",
        "diwali",
        "upanishad",
        "shiva",
        "vishnu",
        "aarti",
        "prasad",
    ),
    "islam": (
        "namaaz",
        "namaz",
        "salah",
        "quran",
        "allah",
        "mosque",
        "masjid",
        "dua",
        "ramadan",
        "roza",
        "muhammad",
        "hajj",
        "tawakkul",
        "sabr",
    ),
    "sikh": (
        "gurbani",
        "gurdwara",
        "waheguru",
        "simran",
        "sangat",
        "seva",
        "khalsa",
        "guru granth",
        "satnam",
        "chardi kala",
        "paath",
        "kirtan",
    ),
    "christian": (
        "jesus",
        "christ",
        "church",
        "bible",
        "gospel",
        "easter",
        "christmas",
        "pastor",
        "psalm",
    ),
    "buddhist": (
        "buddha",
        "sutta",
        "metta",
        "sangha",
        "nirvana",
        "dhamma",
        "vipassana",
    ),
    "jain": ("mahavira", "anekantavada", "aparigraha", "tirthankara", "ahimsa"),
}

# Tradition-neutral faith language: with a known profile these re-include the
# user's own tradition even when the profile gates faith content off.
GENERIC_FAITH_WORDS = (
    "god",
    "faith",
    "pray",
    "worship",
    "sin",
    "bless",
    "divine",
    "sacred",
    "soul",
    "scripture",
    "heaven",
    "hell",
    "punish",
    "forgive",
    "mercy",
    "devotion",
    "ritual",
    "holy",
    "spiritual",
    "religion",
    "religious",
)


def _has_word(text, word):
    return re.search(r"\b" + re.escape(word) + r"\b", text) is not None


def detect_faith_signal(text):
    """Return the KB religion key named in text, or None."""
    text = (text or "").lower()
    if not text.strip():
        return None
    for key, words in FAITH_KEYWORDS.items():
        if any(_has_word(text, w) for w in words):
            return key
    return None


def _norm(value):
    return re.sub(r"[-_]+", " ", (value or "").strip().lower())


def resolve_religion(profile_religion, faith_importance=None, faith_consent=None, user_text=""):
    """Map profile + gating answers to a KB religion key, or None for general-only.

    importance none/not-at-all or consent no/keep-it-neutral gates faith off,
    unless the user's own message names their tradition (or uses generic faith
    words with a known profile) — people talk faith when they talk faith.
    """
    key = PROFILE_RELIGION_TO_KB.get(_norm(profile_religion))
    if key is None:
        return None
    imp = _norm(faith_importance)
    gated = "none" in imp or "not at all" in imp
    gated = gated or _norm(faith_consent) == "no" or "neutral" in _norm(faith_consent)
    if not gated:
        return key
    text = user_text or ""
    if not text.strip():
        return None
    if detect_faith_signal(text) == key:
        return key
    if any(_has_word(text.lower(), w) for w in GENERIC_FAITH_WORDS):
        return key
    return None


def _collection():
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return _client.get_collection(config.COLLECTION)


def _embed(texts):
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                from sentence_transformers import SentenceTransformer

                _embedder = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    return _embedder.encode(texts, normalize_embeddings=True).tolist()


def _query(query, n, where):
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


def retrieve(query, n=5, source_filter=None, religion=None):
    """Return top-n passages as {text, source, chapter, score} dicts.

    religion=None keeps the v1 single unfiltered query. With a religion key,
    ~3/5 of n comes from that tradition and the rest backfills general,
    faith-first and deduped (zero faith hits degrades to n general).
    """
    if religion is None:
        return _query(query, n, {"source": source_filter} if source_filter else None)
    faith_n = (3 * n + 4) // 5
    faith_where = {"religion": religion}
    if source_filter:
        faith_where = {"$and": [{"religion": religion}, {"source": source_filter}]}
    faith_hits = _query(query, faith_n, faith_where)
    general_hits = _query(query, n, {"source": source_filter} if source_filter else None)
    seen = {(h["text"], h["source"], h["chapter"]) for h in faith_hits}
    merged = list(faith_hits)
    for h in general_hits:
        if len(merged) >= n:
            break
        if (h["text"], h["source"], h["chapter"]) not in seen:
            seen.add((h["text"], h["source"], h["chapter"]))
            merged.append(h)
    return merged
