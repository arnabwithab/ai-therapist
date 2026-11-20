"""Route tests — model and RAG are stubbed, only HTTP behaviour is exercised."""

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module
from backend.core import chat as core
from backend.core import db as db_module


def _client(monkeypatch, reply="Sleep well."):
    def fake_chat(user_text, **kwargs):
        return {"reply": reply, "sources": []}

    monkeypatch.setattr(core, "chat", fake_chat)
    return TestClient(app_module.app)


def test_health():
    assert TestClient(app_module.app).get("/health").json() == {"status": "ok"}


def test_models_lists_our_model():
    data = TestClient(app_module.app).get("/v1/models").json()
    assert core.MODEL_ID in [m["id"] for m in data["data"]]


def test_completions_happy_path_ignores_client_model(monkeypatch):
    c = _client(monkeypatch)
    r = c.post(
        "/v1/chat/completions",
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "insomnia"}]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["choices"][0]["message"] == {"role": "assistant", "content": "Sleep well."}
    assert body["model"] == core.MODEL_ID


def test_completions_missing_messages_is_422(monkeypatch):
    c = _client(monkeypatch)
    assert c.post("/v1/chat/completions", json={"model": "m"}).status_code == 422


def test_completions_without_user_message_is_422(monkeypatch):
    c = _client(monkeypatch)
    r = c.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "system", "content": "hi"}]},
    )
    assert r.status_code == 422


def test_completions_stream_true_relays_sse(monkeypatch):
    def fake_stream(user_text, **kwargs):
        yield b'data: {"choices":[{"delta":{"content":"Sleep."}}]}\n\n'
        yield b"data: [DONE]\n\n"

    monkeypatch.setattr(core, "chat_stream", fake_stream)
    c = _client(monkeypatch)
    r = c.post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/event-stream")
    assert "Sleep." in r.text


def test_completions_stream_failure_is_502(monkeypatch):
    def bad_stream(user_text, **kwargs):
        raise core.ModelError("vllm down")
        yield b""

    monkeypatch.setattr(core, "chat_stream", bad_stream)
    r = TestClient(app_module.app).post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert r.status_code == 502


def test_completions_forwards_sampling_params(monkeypatch):
    seen = {}

    def fake_chat(user_text, **kwargs):
        seen.update(kwargs)
        return {"reply": "ok", "sources": []}

    monkeypatch.setattr(core, "chat", fake_chat)
    r = TestClient(app_module.app).post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "top_k": 30,
            "repetition_penalty": 1.2,
        },
    )
    assert r.status_code == 200, r.text
    assert seen["top_k"] == 30 and seen["repetition_penalty"] == 1.2


def test_completions_forwards_history_before_current_turn(monkeypatch):
    seen = {}

    def fake_chat(user_text, **kwargs):
        seen["user_text"] = user_text
        seen["history"] = kwargs.get("history")
        return {"reply": "ok", "sources": []}

    monkeypatch.setattr(core, "chat", fake_chat)
    r = TestClient(app_module.app).post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "second"},
                {"role": "user", "content": "third"},
            ],
        },
    )
    assert r.status_code == 200, r.text
    assert seen["user_text"] == "third"
    assert seen["history"] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]


def test_completions_model_failure_is_502(monkeypatch):
    def bad_chat(user_text, **kwargs):
        raise core.ModelError("vllm down")

    monkeypatch.setattr(core, "chat", bad_chat)
    r = TestClient(app_module.app).post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 502


def test_completions_rag_failure_is_502(monkeypatch):
    def bad_chat(user_text, **kwargs):
        raise core.RagError("chroma down")

    monkeypatch.setattr(core, "chat", bad_chat)
    r = TestClient(app_module.app).post(
        "/v1/chat/completions",
        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 502


def test_completions_answers_cors_preflight():
    c = TestClient(app_module.app)
    r = c.options(
        "/v1/chat/completions",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200, r.text
    assert r.headers["access-control-allow-origin"] == "*"


# --- F024 auth + profile routes (DB swapped to tmp_path, no network) ---


@pytest.fixture()
def tdb(monkeypatch, tmp_path):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "t.db")
    return TestClient(app_module.app)


def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def test_register_login_me_profile_round_trip(tdb):
    r = tdb.post(
        "/api/auth/register", json={"username": "ara", "password": "s3cret-pw", "age_18_plus": True}
    )
    assert r.status_code == 200, r.text
    token = r.json()["token"]
    assert tdb.get("/api/me", headers=_auth(token)).json()["username"] == "ara"

    r = tdb.put("/api/profile", headers=_auth(token), json={"call_name": "Ara", "religion": "Sikh"})
    assert r.status_code == 200, r.text
    assert tdb.get("/api/me", headers=_auth(token)).json()["profile"]["call_name"] == "Ara"

    r = tdb.post("/api/auth/login", json={"username": "ara", "password": "s3cret-pw"})
    assert r.status_code == 200 and r.json()["token"]


def test_users_isolated(tdb):
    a = tdb.post(
        "/api/auth/register", json={"username": "a", "password": "pw-a-long1", "age_18_plus": True}
    ).json()["token"]
    b = tdb.post(
        "/api/auth/register", json={"username": "b", "password": "pw-b-long1", "age_18_plus": True}
    ).json()["token"]
    tdb.put("/api/profile", headers=_auth(a), json={"call_name": "Aya"})
    assert tdb.get("/api/me", headers=_auth(b)).json()["profile"] == {}


def test_register_requires_attestation(tdb):
    r = tdb.post(
        "/api/auth/register",
        json={"username": "ara", "password": "s3cret-pw", "age_18_plus": False},
    )
    assert r.status_code == 400
    r = tdb.post("/api/auth/register", json={"username": "ara", "password": "s3cret-pw"})
    assert r.status_code in (400, 422)


def test_register_duplicate_is_409_and_login_wrong_pw_is_401(tdb):
    body = {"username": "ara", "password": "s3cret-pw", "age_18_plus": True}
    assert tdb.post("/api/auth/register", json=body).status_code == 200
    assert tdb.post("/api/auth/register", json=body).status_code == 409
    assert (
        tdb.post("/api/auth/login", json={"username": "ara", "password": "nope-nope"}).status_code
        == 401
    )


def test_bad_token_is_401_and_logout_revokes(tdb):
    assert tdb.get("/api/me").status_code == 401
    assert tdb.get("/api/me", headers=_auth("bogus")).status_code == 401
    token = tdb.post(
        "/api/auth/register", json={"username": "ara", "password": "s3cret-pw", "age_18_plus": True}
    ).json()["token"]
    assert tdb.post("/api/auth/logout", headers=_auth(token)).status_code == 200
    assert tdb.get("/api/me", headers=_auth(token)).status_code == 401


def test_profile_under_18_warmly_declined(tdb):
    token = tdb.post(
        "/api/auth/register",
        json={"username": "kid", "password": "pw-kid-long1", "age_18_plus": True},
    ).json()["token"]
    r = tdb.put("/api/profile", headers=_auth(token), json={"age_bracket": "<18"})
    assert r.status_code == 403, r.text
    assert "1098" in r.text and "14416" in r.text


def test_completions_answers_cors_preflight_from_non_localhost_origin():
    c = TestClient(app_module.app)
    r = c.options(
        "/v1/chat/completions",
        headers={
            "Origin": "http://192.168.1.5:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200, r.text
