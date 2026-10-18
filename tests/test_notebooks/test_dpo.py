"""Tests for notebooks/dpo/train_dpo.py — pure mapping logic, no GPU/network."""

import sys

sys.path.insert(0, "notebooks/dpo")

from train_dpo import find_parquet, format_dpo_row


class StubTok:
    def apply_chat_template(self, msgs, **kwargs):
        role = msgs[0]["role"]
        return f"<{role}>{msgs[0]['content']}"


def test_format_dpo_row_shapes_prompt_chosen_rejected():
    d = format_dpo_row("why sad?", "hug", "deal with it", StubTok())
    assert set(d) == {"prompt", "chosen", "rejected"}
    assert d["prompt"].startswith("<user>why sad?")
    assert d["chosen"].startswith("<assistant>hug")
    assert d["rejected"].startswith("<assistant>deal with it")


def test_format_dpo_row_strips_blank_rows():
    assert format_dpo_row("  ", "hug", "deal", StubTok()) is None
    assert format_dpo_row("q", "", "deal", StubTok()) is None


def test_find_parquet_prefers_train_split(tmp_path):
    (tmp_path / "test-00000-of-00001.parquet").write_bytes(b"fake")
    want = tmp_path / "train-00000-of-00001.parquet"
    want.write_bytes(b"fake")
    assert find_parquet([str(tmp_path)]) == str(want)


def test_find_parquet_none_when_missing(tmp_path):
    assert find_parquet([str(tmp_path)]) is None
