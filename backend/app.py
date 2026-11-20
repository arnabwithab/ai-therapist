"""Thin FastAPI routes over backend.core.chat. No logic lives here."""

import itertools
import json
import time
import uuid

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.core import chat as core
from backend.core import db

app = FastAPI(title="ai-therapist")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local demo server with no credentials; any origin may use it
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class RegisterRequest(BaseModel):
    username: str
    password: str
    age_18_plus: bool = False


class LoginRequest(BaseModel):
    username: str
    password: str


class ApiChatRequest(BaseModel):
    session_id: str | None = None
    message: str
    stream: bool = False


def _current_user(authorization: str | None = Header(default=None)) -> int:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    user_id = db.user_id_for_token(authorization[7:])
    if user_id is None:
        raise HTTPException(401, "invalid token")
    return user_id


@app.post("/api/auth/register")
def register(req: RegisterRequest):
    if not req.age_18_plus:
        raise HTTPException(400, "18+ attestation required")
    try:
        user_id = db.create_user(req.username, req.password)
    except ValueError as e:
        raise HTTPException(409 if "taken" in str(e) else 400, str(e)) from e
    return {"username": req.username.strip(), "token": db.issue_token(user_id)}


@app.post("/api/auth/login")
def login(req: LoginRequest):
    user_id = db.verify_user(req.username, req.password)
    if user_id is None:
        raise HTTPException(401, "invalid username or password")
    return {"username": req.username.strip(), "token": db.issue_token(user_id)}


@app.post("/api/auth/logout")
def logout(authorization: str | None = Header(default=None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    db.revoke_token(authorization[7:])
    return {"status": "ok"}


@app.get("/api/me")
def me(user_id: int = Depends(_current_user)):
    return {"username": db.username_for(user_id), "profile": db.get_profile(user_id)}


@app.put("/api/profile")
def put_profile(fields: dict, user_id: int = Depends(_current_user)):
    try:
        saved = db.put_profile(user_id, fields)
    except db.MinorError as e:
        raise HTTPException(403, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return saved


@app.get("/api/sessions")
def sessions(user_id: int = Depends(_current_user)):
    return {"sessions": db.list_sessions(user_id)}


def _delta_text(chunk: bytes) -> str:
    """Best-effort extract of one SSE chunk's delta content (stream persist only)."""
    text = ""
    for line in chunk.decode("utf-8", "ignore").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            continue
        try:
            text += json.loads(payload)["choices"][0].get("delta", {}).get("content", "")
        except (ValueError, KeyError, IndexError, TypeError):
            continue
    return text


@app.post("/api/chat")
def api_chat(req: ApiChatRequest, user_id: int = Depends(_current_user)):
    if not req.message or not req.message.strip():
        raise HTTPException(422, "no non-empty user message")
    sid = (req.session_id or "").strip() or uuid.uuid4().hex
    history = db.get_history(user_id, sid)
    profile = db.get_profile(user_id) or None
    message = req.message.strip()
    # user turn is persisted before the model call: a failed turn still leaves a
    # durable transcript of the attempt (a retry may duplicate it — intended).
    if req.stream:
        db.add_message(user_id, sid, "user", message)
        stream = core.chat_stream(message, history=history, profile=profile)
        try:
            first = next(stream)  # surface RAG/model failures as 502, not a broken stream
        except (core.RagError, core.ModelError) as e:
            raise HTTPException(502, str(e)) from e

        def relay():
            raw = bytearray()
            try:
                for chunk in itertools.chain([first], stream):
                    raw += chunk
                    yield chunk
            finally:
                text = _delta_text(bytes(raw))
                if text.strip():
                    db.add_message(user_id, sid, "assistant", text)

        return StreamingResponse(
            relay(), media_type="text/event-stream", headers={"X-Session-Id": sid}
        )
    db.add_message(user_id, sid, "user", message)
    try:
        out = core.chat(message, history=history, profile=profile)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e
    except (core.RagError, core.ModelError) as e:
        raise HTTPException(502, str(e)) from e
    db.add_message(user_id, sid, "assistant", out["reply"])
    return {"session_id": sid, "reply": out["reply"], "sources": out["sources"]}


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
    # cap generation so prompt + completion fits our 8192-token window
    # (1024 fits the DPO adapter's thinking traces plus the reply)
    max_tokens = min(req.max_tokens, 1024)
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
