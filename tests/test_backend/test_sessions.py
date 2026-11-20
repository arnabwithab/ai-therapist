"""F025 server-side sessions — persistence round-trip, isolation, stream persist, list."""

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.core import chat as core
from backend.core import db as db_module


@pytest.fixture()
def tdb(monkeypatch, tmp_path):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "t.db")
    return TestClient(app_module.app)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _register(tdb, username="ara"):
    return tdb.post(
        "/api/auth/register",
        json={"username": username, "password": "s3cret-pw", "age_18_plus": True},
    ).json()["token"]


def _fake_chat(reply="Sleep well.", sources=None, seen=None):
    def go(user_text, **kwargs):
        if seen is not None:
            seen.update(kwargs)
            seen["user_text"] = user_text
        return {"reply": reply, "sources": sources or []}

    return go


def test_chat_mints_session_and_persists_round_trip(tdb, monkeypatch):
    monkeypatch.setattr(core, "chat", _fake_chat())
    token = _register(tdb)
    r = tdb.post("/api/chat", headers=_auth(token), json={"message": "can't sleep"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["session_id"] and body["reply"] == "Sleep well."
    assert db_module.get_history(1, body["session_id"]) == [
        {"role": "user", "content": "can't sleep"},
        {"role": "assistant", "content": "Sleep well."},
    ]


def test_chat_second_turn_sees_db_history(tdb, monkeypatch):
    seen = {}
    monkeypatch.setattr(core, "chat", _fake_chat(reply="second", seen=seen))
    token = _register(tdb)
    sid = tdb.post("/api/chat", headers=_auth(token), json={"message": "first"}).json()[
        "session_id"
    ]
    r = tdb.post("/api/chat", headers=_auth(token), json={"session_id": sid, "message": "next"})
    assert r.status_code == 200, r.text
    assert r.json()["session_id"] == sid
    assert seen["history"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    assert len(db_module.get_history(1, sid)) == 4


def test_chat_passes_profile_through(tdb, monkeypatch):
    seen = {}
    monkeypatch.setattr(core, "chat", _fake_chat(seen=seen))
    token = _register(tdb)
    tdb.put("/api/profile", headers=_auth(token), json={"call_name": "Ara"})
    tdb.post("/api/chat", headers=_auth(token), json={"message": "hi"})
    assert seen["profile"] == {"call_name": "Ara"}
    # DPO thinking traces need room: truncated thinking means null content.
    assert seen["max_tokens"] == 1024


def test_chat_isolated_between_users_and_sessions(tdb, monkeypatch):
    seen = {}
    monkeypatch.setattr(core, "chat", _fake_chat(seen=seen))
    a = _register(tdb, "a")
    b = _register(tdb, "b")
    sid = tdb.post("/api/chat", headers=_auth(a), json={"message": "hi a"}).json()["session_id"]
    r = tdb.post("/api/chat", headers=_auth(b), json={"message": "hi b"})
    assert r.status_code == 200
    assert seen["history"] == []  # b sees none of a's turns
    assert db_module.get_history(2, sid) == []


def test_chat_requires_auth_and_nonempty_message(tdb, monkeypatch):
    monkeypatch.setattr(core, "chat", _fake_chat())
    assert tdb.post("/api/chat", json={"message": "hi"}).status_code == 401
    assert tdb.post("/api/chat", headers=_auth("bogus"), json={"message": "hi"}).status_code == 401
    token = _register(tdb)
    assert tdb.post("/api/chat", headers=_auth(token), json={"message": "  "}).status_code == 422


def test_chat_model_failure_is_502(tdb, monkeypatch):
    def bad(user_text, **kwargs):
        raise core.ModelError("vllm down")

    monkeypatch.setattr(core, "chat", bad)
    token = _register(tdb)
    assert tdb.post("/api/chat", headers=_auth(token), json={"message": "hi"}).status_code == 502


def test_chat_stream_persists_assembled_reply(tdb, monkeypatch):
    def fake_stream(user_text, **kwargs):
        yield b'data: {"choices":[{"delta":{"content":"Sleep"}}]}\n\n'
        yield b'data: {"choices":[{"delta":{"content":" well."}}]}\n\n'
        yield b"data: [DONE]\n\n"

    monkeypatch.setattr(core, "chat_stream", fake_stream)
    token = _register(tdb)
    r = tdb.post("/api/chat", headers=_auth(token), json={"message": "insomnia", "stream": True})
    assert r.status_code == 200, r.text
    assert "Sleep" in r.text
    sid = r.headers["x-session-id"]
    assert db_module.get_history(1, sid) == [
        {"role": "user", "content": "insomnia"},
        {"role": "assistant", "content": "Sleep well."},
    ]


def test_sessions_list(tdb, monkeypatch):
    monkeypatch.setattr(core, "chat", _fake_chat())
    token = _register(tdb)
    s1 = tdb.post("/api/chat", headers=_auth(token), json={"message": "first chat"}).json()[
        "session_id"
    ]
    s2 = tdb.post("/api/chat", headers=_auth(token), json={"message": "second chat"}).json()[
        "session_id"
    ]
    assert s1 != s2
    body = tdb.get("/api/sessions", headers=_auth(token)).json()
    assert {s["id"] for s in body["sessions"]} == {s1, s2}
    assert all({"id", "preview", "updated_at"} <= set(s) for s in body["sessions"])
    assert tdb.get("/api/sessions").status_code == 401


def test_sessions_newest_first_on_same_second_timestamps(tdb, monkeypatch):
    monkeypatch.setattr(db_module, "_now", lambda: "2026-11-20T00:00:00+00:00")
    uid = db_module.create_user("ara", "s3cret-pw")
    db_module.add_message(uid, "s1", "user", "first")
    db_module.add_message(uid, "s2", "user", "second")
    sessions = db_module.list_sessions(uid)
    assert [s["id"] for s in sessions] == ["s2", "s1"]


def test_chat_stream_split_chunks_still_persist(tdb, monkeypatch):
    def fake_stream(user_text, **kwargs):
        yield b'data: {"choices":[{"delta":{"cont'
        yield b'ent":"Sleep well."}}]}\n\n'
        yield b"data: [DONE]\n\n"

    monkeypatch.setattr(core, "chat_stream", fake_stream)
    token = _register(tdb)
    r = tdb.post("/api/chat", headers=_auth(token), json={"message": "insomnia", "stream": True})
    assert r.status_code == 200, r.text
    assert db_module.get_history(1, r.headers["x-session-id"]) == [
        {"role": "user", "content": "insomnia"},
        {"role": "assistant", "content": "Sleep well."},
    ]


def test_chat_blank_session_id_mints_new(tdb, monkeypatch):
    monkeypatch.setattr(core, "chat", _fake_chat())
    token = _register(tdb)
    r = tdb.post("/api/chat", headers=_auth(token), json={"session_id": "  ", "message": "hi"})
    assert r.status_code == 200, r.text
    assert r.json()["session_id"].strip()
