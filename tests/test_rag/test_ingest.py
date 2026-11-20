"""Tests for F021: manifest<->text/ consistency and religion metadata (ingest)."""

import json
import os

import chromadb

from src.rag import config, ingest
from src.rag.ingest import load_manifest, religion_for

ALLOWED = {"hindu", "islam", "sikh", "christian", "buddhist", "jain", "secular", "general"}


def _txt_names():
    return [
        n for n in os.listdir(config.KB_TEXT_DIR) if n.endswith(".txt") and not n.startswith("_")
    ]


def test_every_txt_resolves_to_allowed_tag():
    manifest = load_manifest(config.KB_TEXT_DIR)
    names = _txt_names()
    assert names, "no .txt sources found"
    for name in names:
        assert religion_for(name, manifest) in ALLOWED, name


def test_no_orphan_manifest_keys():
    manifest = load_manifest(config.KB_TEXT_DIR)
    orphans = set(manifest) - set(_txt_names())
    assert not orphans, orphans


def test_manifest_tags_in_allowed_set():
    path = os.path.join(os.path.dirname(config.KB_TEXT_DIR), "manifest.json")
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw, "manifest is empty"
    for name, entry in raw.items():
        assert entry["religion"] in ALLOWED, (name, entry)


def test_load_manifest_missing_file_returns_empty(tmp_path):
    assert load_manifest(str(tmp_path / "no-such-kb-dir")) == {}


def test_load_manifest_unknown_tag_warns_and_falls_back_to_general(tmp_path, caplog):
    kb = tmp_path / "text"
    kb.mkdir()
    with open(tmp_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"a.txt": {"religion": "jedi"}}, f)
    with caplog.at_level("WARNING", logger=ingest.__name__):
        manifest = load_manifest(str(kb))
    assert manifest == {"a.txt": "general"}
    assert any("jedi" in r.message for r in caplog.records)


def test_religion_metadata_attach_and_where_filter(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection("test-kb-religion")
    docs = [
        "Karma yoga: act without attachment to the fruits of action.",
        "Cognitive behavioral therapy uses thought records to reframe beliefs.",
    ]
    col.add(
        ids=["h1", "g1"],
        documents=docs,
        metadatas=[
            {"source": "gita.txt", "chapter": "", "chunk_id": 0, "religion": "hindu"},
            {"source": "cbt.txt", "chapter": "", "chunk_id": 0, "religion": "general"},
        ],
        embeddings=[[float(hash(d) % 100)] * 8 for d in docs],
    )
    res = col.query(
        query_embeddings=[[float(hash("karma yoga action")) % 100] * 8],
        n_results=5,
        where={"religion": "hindu"},
    )
    assert res["ids"][0] == ["h1"]
    assert res["metadatas"][0][0]["religion"] == "hindu"
