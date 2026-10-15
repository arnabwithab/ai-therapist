"""Chat logic: RAG context injection + model call. No HTTP framework here.

Stateless by design — history lives in the request (OpenAI messages array),
so there is nothing to store, expire, or lose.
"""

import json
import logging
import os

import httpx

from src.rag.query import retrieve

log = logging.getLogger(__name__)

MODEL_ID = os.getenv("AI_THERAPIST_MODEL", "ai-therapist-qwen3-4b")
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
TIMEOUT_S = float(os.getenv("MODEL_TIMEOUT_S", "120"))

SYSTEM_PROMPT = (
    "You are a compassionate counselling assistant. Respond in natural, supportive "
    "paragraphs. You are not a licensed therapist and this is not a substitute for "
    "professional care; if the user is in crisis, encourage them to contact a local "
    "helpline or emergency services."
)


class RagError(Exception):
    """Retrieval failed; the model must not answer without its context."""


class ModelError(Exception):
    """The model server failed or timed out."""


def build_messages(user_text, passages):
    """OpenAI-style messages: system prompt, optional RAG context, user turn."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if passages:
        context = "\n\n".join(f"[{p['source']} — {p['chapter']}]\n{p['text']}" for p in passages)
        messages.append(
            {
                "role": "system",
                "content": "Use this background only if relevant:\n" + context,
            }
        )
    messages.append({"role": "user", "content": user_text})
    return messages


def generate(client, messages, model=MODEL_ID, temperature=0.7, top_p=0.8, max_tokens=512):
    """POST one non-streaming chat completion; return (text, usage)."""
    try:
        resp = client.post(
            "/v1/chat/completions",
            content=json.dumps(
                {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "stream": False,
                }
            ),
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        body = resp.json()
        msg = body["choices"][0]["message"]
        return msg.get("content", "").strip(), body.get("usage")
    except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
        raise ModelError(f"model call failed: {e}") from e


def generate_stream(client, messages, model=MODEL_ID, temperature=0.7, top_p=0.8, max_tokens=512):
    """Yield raw SSE byte chunks from vLLM (pass-through for OpenAI-style clients)."""
    try:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            for chunk in resp.iter_bytes():
                yield chunk
    except httpx.HTTPError as e:
        raise ModelError(f"model stream failed: {e}") from e


def chat_stream(
    user_text,
    *,
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    top_k=RAG_TOP_K,
    client=None,
    retrieve_fn=None,
):
    """Streaming turn: same retrieval + prompt build, chunks relayed from vLLM."""
    if not user_text or not user_text.strip():
        raise ValueError("empty user turn")
    retrieve_fn = retrieve_fn or retrieve
    try:
        passages = retrieve_fn(user_text, n=top_k)
    except Exception as e:
        raise RagError(f"retrieval failed: {e}") from e
    own_client = client is None
    client = client or httpx.Client(base_url=VLLM_URL, timeout=TIMEOUT_S)
    stream = generate_stream(
        client,
        build_messages(user_text.strip(), passages),
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    try:
        yield from stream
    finally:
        if own_client:
            client.close()


def chat(
    user_text,
    *,
    temperature=0.7,
    top_p=0.8,
    max_tokens=512,
    top_k=RAG_TOP_K,
    client=None,
    retrieve_fn=None,
):
    """Full turn: retrieve passages, inject, generate. Returns reply + sources."""
    if not user_text or not user_text.strip():
        raise ValueError("empty user turn")
    retrieve_fn = retrieve_fn or retrieve
    try:
        passages = retrieve_fn(user_text, n=top_k)
    except Exception as e:
        raise RagError(f"retrieval failed: {e}") from e
    own_client = client is None
    client = client or httpx.Client(base_url=VLLM_URL, timeout=TIMEOUT_S)
    try:
        content, _ = generate(
            client,
            build_messages(user_text.strip(), passages),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    finally:
        if own_client:
            client.close()
    return {
        "reply": content,
        "sources": [
            {"source": p["source"], "chapter": p["chapter"], "score": p["score"]} for p in passages
        ],
    }
