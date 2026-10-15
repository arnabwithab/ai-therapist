"""Thin FastAPI routes over backend.core.chat. No logic lives here."""

import time
import uuid

from fastapi import FastAPI, HTTPException
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
    max_tokens: int = 512
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
    if req.stream:
        raise HTTPException(400, "streaming is not supported yet")
    user_msgs = [m.content for m in req.messages if m.role == "user" and m.content.strip()]
    if not user_msgs:
        raise HTTPException(422, "no non-empty user message")
    try:
        out = core.chat(
            user_msgs[-1],
            temperature=req.temperature,
            top_p=req.top_p,
            max_tokens=req.max_tokens,
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
