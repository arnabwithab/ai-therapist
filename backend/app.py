"""Thin FastAPI routes over backend.core.chat. No logic lives here."""

import itertools
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.core import chat as core

app = FastAPI(title="ai-therapist")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = core.MODEL_ID
    messages: list[Message]
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    repetition_penalty: float = 1.1
    max_tokens: int = 256
    stream: bool = False


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/v1/models")
def models():
    return {
        "object": "list",
        "data": [{"id": core.MODEL_ID, "object": "model", "owned_by": "ai-therapist"}],
    }


@app.post("/v1/chat/completions")
def completions(req: ChatRequest):
    idx = max(
        (i for i, m in enumerate(req.messages) if m.role == "user" and m.content.strip()),
        default=None,
    )
    if idx is None:
        raise HTTPException(422, "no non-empty user message")
    history = [{"role": m.role, "content": m.content} for m in req.messages[:idx]]
    current = req.messages[idx].content
    # cap generation so prompt + completion fits our 2048-token window
    max_tokens = min(req.max_tokens, 256)
    if req.stream:
        stream = core.chat_stream(
            current,
            history=history,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_tokens=max_tokens,
        )
        try:
            first = next(stream)  # surface RAG/model failures as 502, not a broken stream
        except (core.RagError, core.ModelError) as e:
            raise HTTPException(502, str(e)) from e
        return StreamingResponse(itertools.chain([first], stream), media_type="text/event-stream")
    try:
        out = core.chat(
            current,
            history=history,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
            max_tokens=max_tokens,
        )
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except (core.RagError, core.ModelError) as e:
        raise HTTPException(502, str(e)) from e
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": core.MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": out["reply"]},
                "finish_reason": "stop",
            }
        ],
    }
