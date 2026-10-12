"""Paragraph-aware text chunker. No external deps."""

import re

_WORD_RE = re.compile(r"\S+")


def _count(text):
    # ~4 chars/token is close enough for chunk sizing; the embed model
    # truncates anything over its limit anyway.
    return max(1, len(text) // 4)


def chunk_text(text, source, chapter="", max_tokens=512, overlap=50):
    """Split text into ~max_tokens chunks on paragraph boundaries.

    Returns a list of {"text", "source", "chapter", "chunk_id"} dicts.
    """
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = []

    def flush():
        body = "\n\n".join(current).strip()
        if body:
            chunks.append(
                {
                    "text": body,
                    "source": source,
                    "chapter": chapter,
                    "chunk_id": len(chunks),
                }
            )
        current.clear()

    for para in paras:
        # A single oversized paragraph gets hard-split on sentences.
        pieces = [para] if _count(para) <= max_tokens else re.split(r"(?<=[.!?])\s+", para)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            current_size = sum(_count(p) for p in current)
            if current and current_size + _count(piece) > max_tokens:
                flush()
                if overlap > 0 and chunks:
                    tail = chunks[-1]["text"].split()[-overlap:]
                    current.append(" ".join(tail))
            current.append(piece)
    flush()
    return chunks
