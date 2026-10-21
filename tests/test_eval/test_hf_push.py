"""Tests for eval/hf_push.py — no network (HfApi stubbed)."""

import pytest

from eval import hf_push

_CREATED = []


class _Api:
    def __init__(self, token=None):
        self.token = token
        self.calls = []
        _CREATED.append(self)

    def create_repo(self, *a, **k):
        self.calls.append(("create_repo", a, k))

    def upload_file(self, *a, **k):
        self.calls.append(("upload_file", a, k))


def test_push_file_noop_without_repo(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_push, "HfApi", _Api)
    _CREATED.clear()
    hf_push.push_file(str(tmp_path / "x.csv"), "", "x.csv")
    assert _CREATED == []


def test_push_file_needs_token(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_push, "HfApi", _Api)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(SystemExit):
        hf_push.push_file(str(tmp_path / "x.csv"), "user/repo", "x.csv")


def test_push_file_creates_repo_and_uploads(tmp_path, monkeypatch):
    p = tmp_path / "part.csv"
    p.write_text("idx\n", encoding="utf-8")
    monkeypatch.setattr(hf_push, "HfApi", _Api)
    monkeypatch.setenv("HF_TOKEN", "tok")
    _CREATED.clear()
    hf_push.push_file(str(p), "user/repo", "col/part.csv")
    (api,) = _CREATED
    assert api.token == "tok"
    kinds = [c[0] for c in api.calls]
    assert kinds == ["create_repo", "upload_file"]
    assert api.calls[1][2]["path_in_repo"] == "col/part.csv"
    assert api.calls[1][2]["repo_type"] == "dataset"
