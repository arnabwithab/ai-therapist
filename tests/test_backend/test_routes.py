"""Route tests — model and RAG are stubbed, only HTTP behaviour is exercised."""

from fastapi.testclient import TestClient

from backend import app as app_module
from backend.core import chat as core


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
