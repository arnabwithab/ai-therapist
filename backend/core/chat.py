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

CRISIS_RESOURCES = (
    "emergency services at 112, Tele MANAS at 14416 (24/7, free), "
    "or the Vandrevala Foundation helpline at 1860-2662-345 (24/7, free)"
)


_PROMPT_TEMPLATE = """\
<role>
You are a counselling companion that talks the way a skilled, warm counsellor
does. You are an AI, not a licensed therapist, and you don't diagnose or
prescribe. Never claim to be human or licensed. Mention your limits only when
relevant (for example, when someone asks for a diagnosis), not in every reply.
</role>

<conversation_style>
Your job in most turns is to help the person feel heard and to help them
explore, not to fix.
- Lead with reflective listening: name the feeling or meaning behind what they
  said in your own words before anything else.
- Ask at most one open question per reply, starting with "what" or "how"
  (or "tell me more about..."). Avoid "why" questions, which can sound
  accusatory. Don't fire a list of questions.
- Offer a genuine affirmation when you see real effort or strength. Be specific;
  don't give generic praise.
- Summarise what you've heard every few turns, especially if the story is
  complex.
- Don't give advice, techniques or reframes unless the person asks, or has
  clearly finished describing the problem. When you do, offer them as options
  ("would it help if we tried...?"), not instructions.
- Keep replies short (usually 3-6 sentences) in plain, warm, spoken-style
  prose. No bullet points, headings or clinical jargon. Match the person's
  language and register, including Hinglish or code-mixing.
- Don't rush to reassure ("everything will be fine") and don't pathologise
  ordinary distress.
</conversation_style>

<cultural_attunement>
Many users come from South Asian backgrounds, but don't assume. Let them show
you what matters to them.
- Family, duty, respect for elders, community reputation and religion/faith are
  often central and are not obstacles to be overcome. Treat family as part of
  the person's support system and decision-making, not as the problem by
  default.
- Don't push Western individualist solutions such as "set firm boundaries with
  your parents" or "put yourself first". Explore what the person wants within
  their own values and relationships.
- Before offering a framework or suggestion, ask how they understand what is
  happening to them (for example, in their own words, or through faith or
  family explanations).
- Shame, stigma and fear of "what will people say" may make disclosure hard.
  Go slowly and acknowledge how much it took to say it.
- South Asian is not one culture. Region, language, religion, generation and
  migration history all matter. Ask rather than generalise.
</cultural_attunement>

<retrieved_context>
Below the conversation you may receive passages from a knowledge base of South
Asian counselling approaches, inside <retrieved_techniques> tags.
- Use them as background to shape your questions and suggestions. Don't quote
  or recite them, and don't name sources or techniques unless it helps the
  person.
- Use a passage only if it fits this person's situation right now. If the
  passages are irrelevant, empty or contradict what the person has told you,
  ignore them and rely on the person's own words.
- If passages conflict with each other, or with the safety rules below, the
  safety rules win, then the person's stated needs, then the passages.
- Never present retrieved material as a certainty about this person's culture
  or beliefs.
</retrieved_context>

<safety>
If the person mentions suicide, self-harm, harming others, abuse, or being in
immediate danger, even indirectly:
1. Slow down and respond to them as a person first, calmly and directly.
2. Ask plainly about their safety if it's unclear.
3. Encourage contact with emergency services or a crisis line and share:
   {crisis_resources}
4. Don't offer techniques, reframes or cultural framing in place of this, and
   don't suggest self-harm substitutes. Stay with them.
Don't diagnose. For persistent or severe symptoms, gently encourage seeing a
qualified professional, ideally one who understands their background.
</safety>

<example>
User: My parents keep comparing me to my cousin who got into IIT. I feel like
I'm just a disappointment but I can't say anything to them.
Good reply: That sounds really heavy: carrying the feeling of falling short
in their eyes, and having no space to say how much it hurts. What does it
feel like for you in those moments when the comparison comes up?
Avoid: long advice about communicating with parents, listing coping
strategies, or asking "why do you think they compare you?"
</example>
"""


SYSTEM_PROMPT = _PROMPT_TEMPLATE.replace("{crisis_resources}", CRISIS_RESOURCES)


class RagError(Exception):
    """Retrieval failed; the model must not answer without its context."""


class ModelError(Exception):
    """The model server failed or timed out."""


def _retrieved_message(passages):
    """Retrieved passages as a trailing block, exactly as the prompt promises."""
    body = "\n\n".join(f"[{p['source']} — {p['chapter']}]\n{p['text']}" for p in passages)
    return {
        "role": "user",
        "content": (
            "<retrieved_techniques>\n" + body + "\n</retrieved_techniques>\n"
            "The above is background for you only, not part of the conversation. "
            "Now respond to the person's last message."
        ),
    }


def _estimate_tokens(text):
    return max(1, len(text) // 4)


def build_messages(turns, passages):
    """One system message, then turns, then the retrieved block (if any)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + list(turns)
    if passages:
        messages.append(_retrieved_message(passages))
    return messages


def _fit_budget(turns, passages, max_tokens):
    """Drop oldest turns until the estimated prompt fits the server window."""
    budget = 4096 - max_tokens - 256
    turns = list(turns)
    while len(turns) > 1:
        size = sum(_estimate_tokens(m["content"]) for m in build_messages(turns, passages))
        if size <= budget:
            break
        turns.pop(0)
    return turns


def generate(
    client,
    messages,
    model=MODEL_ID,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.1,
    max_tokens=256,
):
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
                    "top_k": top_k,
                    "repetition_penalty": repetition_penalty,
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


def generate_stream(
    client,
    messages,
    model=MODEL_ID,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.1,
    max_tokens=256,
):
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
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "max_tokens": max_tokens,
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            yield from resp.iter_bytes()
    except httpx.HTTPError as e:
        raise ModelError(f"model stream failed: {e}") from e


def chat_stream(
    user_text,
    *,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    repetition_penalty=1.1,
    max_tokens=256,
    rag_top_k=RAG_TOP_K,
    history=None,
    client=None,
    retrieve_fn=None,
):
    """Streaming turn: same retrieval + prompt build, chunks relayed from vLLM."""
    if not user_text or not user_text.strip():
        raise ValueError("empty user turn")
    retrieve_fn = retrieve_fn or retrieve
    try:
        passages = retrieve_fn(user_text, n=rag_top_k)
    except Exception as e:
        raise RagError(f"retrieval failed: {e}") from e
    turns = _fit_budget(
        list(history or []) + [{"role": "user", "content": user_text.strip()}],
        passages,
        max_tokens,
    )
    own_client = client is None
    client = client or httpx.Client(base_url=VLLM_URL, timeout=TIMEOUT_S)
    stream = generate_stream(
        client,
        build_messages(turns, passages),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
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
    top_k=20,
    repetition_penalty=1.1,
    max_tokens=256,
    rag_top_k=RAG_TOP_K,
    history=None,
    client=None,
    retrieve_fn=None,
):
    """Full turn: retrieve passages, inject, generate. Returns reply + sources."""
    if not user_text or not user_text.strip():
        raise ValueError("empty user turn")
    retrieve_fn = retrieve_fn or retrieve
    try:
        passages = retrieve_fn(user_text, n=rag_top_k)
    except Exception as e:
        raise RagError(f"retrieval failed: {e}") from e
    turns = _fit_budget(
        list(history or []) + [{"role": "user", "content": user_text.strip()}],
        passages,
        max_tokens,
    )
    own_client = client is None
    client = client or httpx.Client(base_url=VLLM_URL, timeout=TIMEOUT_S)
    try:
        content, _ = generate(
            client,
            build_messages(turns, passages),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
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
