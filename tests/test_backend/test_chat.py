"""Tests for backend/core/chat.py — no network, no GPU, no model downloads."""

import httpx
import pytest

from backend.core import chat as core


def _passages():
    return [
        {
            "text": "Mindfulness of breathing calms anxiety.",
            "source": "s1",
            "chapter": "c1",
            "score": 0.9,
        },
        {"text": "Yoga nidra aids sleep.", "source": "s2", "chapter": "c2", "score": 0.8},
    ]


def test_build_messages_injects_system_context_and_user_text():
    msgs = core.build_messages("I can't sleep.", _passages())
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "I can't sleep."}
    ctx = msgs[1]["content"]
    assert "Mindfulness of breathing" in ctx and "Yoga nidra" in ctx


def test_build_messages_without_passages_still_works():
    msgs = core.build_messages("hello", [])
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "hello"}


def _mock_client(status=200, payload=None, exc=None):
    def handler(request):
        if exc is not None:
            raise exc
        return httpx.Response(status, json=payload)

    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))


def _completion(text="Take a breath."):
    return {
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def test_generate_posts_to_vllm_and_returns_content_plus_usage():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = __import__("json").loads(request.content)
        return httpx.Response(200, json=_completion())

    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    content, usage = core.generate(
        client, [{"role": "user", "content": "hi"}], model="m", temperature=0.7
    )
    assert content == "Take a breath."
    assert usage["total_tokens"] == 15
    assert seen["url"].endswith("/v1/chat/completions")
    assert seen["body"]["model"] == "m" and seen["body"]["temperature"] == 0.7


def test_generate_server_error_raises_model_error():
    client = _mock_client(status=500, payload={"error": "boom"})
    with pytest.raises(core.ModelError):
        core.generate(client, [{"role": "user", "content": "hi"}])


def test_generate_timeout_raises_model_error():
    client = _mock_client(exc=httpx.ConnectError("refused"))
    with pytest.raises(core.ModelError):
        core.generate(client, [{"role": "user", "content": "hi"}])


def test_chat_wires_retrieve_and_model():
    client = _mock_client(payload=_completion("Sleep well."))
    out = core.chat("insomnia", client=client, retrieve_fn=lambda q, n=5: _passages())
    assert out["reply"] == "Sleep well."
    assert [s["source"] for s in out["sources"]] == ["s1", "s2"]


def test_chat_rag_failure_raises_rag_error():
    def bad_retrieve(q, n=5):
        raise RuntimeError("chroma down")

    with pytest.raises(core.RagError):
        core.chat("hi", client=_mock_client(payload=_completion()), retrieve_fn=bad_retrieve)


def test_chat_blank_input_raises_value_error():
    with pytest.raises(ValueError):
        core.chat("   ", client=_mock_client(payload=_completion()))
