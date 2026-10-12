"""Shared RAG settings."""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DEVICE = os.getenv("RAG_DEVICE", "cpu")
EMBED_BATCH = int(os.getenv("RAG_EMBED_BATCH", "32"))
COLLECTION = os.getenv("RAG_COLLECTION", "therapist-kb")
CHROMA_DIR = os.getenv("RAG_CHROMA_DIR", os.path.join(REPO_ROOT, "chroma"))
KB_TEXT_DIR = os.getenv("RAG_KB_DIR", os.path.join(REPO_ROOT, "data", "rag-kb", "text"))

CHUNK_TOKENS = int(os.getenv("RAG_CHUNK_TOKENS", "512"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
