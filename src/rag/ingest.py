"""Build (or rebuild) the Chroma index from data/rag-kb/text/."""

import argparse
import glob
import logging
import os
import re

import chromadb
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

from src.rag import config
from src.rag.chunk import chunk_text

log = logging.getLogger(__name__)

CHAPTER_RE = re.compile(r"^#{1,3}\s+(.+)$", re.MULTILINE)


def _chapter_for(text, offset):
    """Best-effort chapter: nearest markdown heading before offset."""
    headings = [(m.start(), m.group(1).strip()) for m in CHAPTER_RE.finditer(text)]
    chapter = ""
    for pos, title in headings:
        if pos > offset:
            break
        chapter = title
    return chapter


def load_documents(kb_dir):
    docs = []
    for path in sorted(glob.glob(os.path.join(kb_dir, "*.txt"))):
        name = os.path.basename(path)
        if name.startswith("_"):
            continue
        with open(path, encoding="utf-8") as f:
            docs.append((name, f.read()))
    return docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="wipe collection first")
    parser.add_argument("--kb-dir", default=config.KB_TEXT_DIR)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    docs = load_documents(args.kb_dir)
    if not docs:
        raise SystemExit(f"no .txt files in {args.kb_dir}")
    log.info("%d source files", len(docs))

    client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    if args.reset:
        try:
            client.delete_collection(config.COLLECTION)
        except NotFoundError:
            log.info("no existing collection to wipe")
    collection = client.get_or_create_collection(config.COLLECTION)

    model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)
    total = 0
    for name, text in docs:
        chunks = chunk_text(
            text,
            source=name,
            max_tokens=config.CHUNK_TOKENS,
            overlap=config.CHUNK_OVERLAP,
        )
        # per-chunk chapter from nearest heading
        offset = 0
        for ch in chunks:
            idx = text.find(ch["text"][:60], offset)
            ch["chapter"] = _chapter_for(text, idx if idx >= 0 else offset)
            offset = idx + 1 if idx >= 0 else offset
        ids = [f"{name}#{c['chunk_id']}" for c in chunks]
        for i in range(0, len(chunks), args.batch):
            batch = chunks[i : i + args.batch]
            embs = model.encode(
                [c["text"] for c in batch],
                batch_size=config.EMBED_BATCH,
                normalize_embeddings=True,
            ).tolist()
            collection.upsert(
                ids=ids[i : i + args.batch],
                documents=[c["text"] for c in batch],
                metadatas=[
                    {
                        "source": c["source"],
                        "chapter": c["chapter"],
                        "chunk_id": c["chunk_id"],
                    }
                    for c in batch
                ],
                embeddings=embs,
            )
        total += len(chunks)
        log.info("%s: %d chunks", name, len(chunks))
    log.info("done: %d chunks in %s", total, config.CHROMA_DIR)


if __name__ == "__main__":
    main()
