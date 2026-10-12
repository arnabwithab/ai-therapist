"""Tests for src/rag/chunk.py."""

from src.rag.chunk import chunk_text


def test_short_text_is_single_chunk():
    chunks = chunk_text("Hello world. This is a test.", source="s.txt")
    assert len(chunks) == 1
    assert chunks[0]["text"] == "Hello world. This is a test."
    assert chunks[0]["source"] == "s.txt"


def test_long_text_splits_on_paragraphs():
    paras = [f"Paragraph {i}. " + "word " * 200 for i in range(6)]
    chunks = chunk_text("\n\n".join(paras), source="s.txt", max_tokens=512)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c["text"].split()) > 0


def test_chunks_carry_sequential_ids_and_metadata():
    text = "First para " * 300 + "\n\n" + "Second para " * 300
    chunks = chunk_text(text, source="book.txt", chapter="Ch 1", max_tokens=200)
    assert len(chunks) >= 2
    assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert all(c["chapter"] == "Ch 1" for c in chunks)
    assert all(c["source"] == "book.txt" for c in chunks)


def test_empty_text_gives_no_chunks():
    assert chunk_text("", source="s.txt") == []
    assert chunk_text("   \n\n  ", source="s.txt") == []
