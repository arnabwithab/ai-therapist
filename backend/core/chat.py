"""Chat logic: RAG context injection + model call. No HTTP framework here.

Stateless by design — history lives in the request (OpenAI messages array),
so there is nothing to store, expire, or lose.
"""

import json
import logging
import os
import re

import httpx

from src.rag.query import GENERIC_FAITH_WORDS, detect_faith_signal, resolve_religion, retrieve

log = logging.getLogger(__name__)

MODEL_ID = os.getenv("AI_THERAPIST_MODEL", "ai-therapist-qwen3-1.7b-dpo")
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
Your users are in India. Let them show you what matters to them.
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
- India is not one culture. Region, language, religion, generation and
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


def format_passages(passages):
    """Passage block in serving format. Single source of truth — evals import this."""
    return "\n\n".join(f"[{p['source']} — {p['chapter']}]\n{p['text']}" for p in passages)


def _retrieved_message(passages):
    """Retrieved passages as a trailing block, exactly as the prompt promises."""
    body = format_passages(passages)
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


def _blank(value):
    """Blank or an opted-out 'prefer not to say' answer (never echoed)."""
    if value is None:
        return True
    s = str(value).strip()
    if not s:
        return True
    return re.sub(r"\s+", " ", re.sub(r"[-_]+", " ", s.lower())).strip() == "prefer not to say"


def render_person(profile):
    """Render the <person> block, or None for an empty/unusable profile."""
    if not profile:
        return None
    lines = []
    parts = []
    if not _blank(profile.get("call_name")):
        parts.append(f"Calls themselves {str(profile['call_name']).strip()}")
    for key in ("age_bracket", "gender"):
        if not _blank(profile.get(key)):
            parts.append(str(profile[key]).strip())
    if parts:
        lines.append("- " + "; ".join(parts) + ".")
    if not _blank(profile.get("language_pref")):
        lines.append(f"- Most comfortable in {str(profile['language_pref']).strip()} — match it.")
    rel_ok, fam_ok = not _blank(profile.get("religion")), not _blank(profile.get("family_religion"))
    if rel_ok and fam_ok:
        lines.append(
            f"- Follows {str(profile['religion']).strip()};"
            f" grew up in a {str(profile['family_religion']).strip()} family."
        )
    elif rel_ok:
        lines.append(f"- Follows {str(profile['religion']).strip()}.")
    elif fam_ok:
        lines.append(f"- Grew up in a {str(profile['family_religion']).strip()} family.")
    if not _blank(profile.get("faith_importance")):
        lines.append(
            f"- Faith matters {str(profile['faith_importance']).strip()} to them right now."
        )
    sup_ok, con_ok = not _blank(profile.get("supports")), not _blank(profile.get("concern"))
    if sup_ok and con_ok:
        lines.append(
            f"- Supports: {str(profile['supports']).strip()}."
            f" First shared concern: {str(profile['concern']).strip()}."
        )
    elif sup_ok:
        lines.append(f"- Supports: {str(profile['supports']).strip()}.")
    elif con_ok:
        lines.append(f"- First shared concern: {str(profile['concern']).strip()}.")
    if not lines:
        return None
    return (
        "<person>\nWhat they told us at intake — use naturally, don't recite:\n"
        + "\n".join(lines)
        + "\nTreat these as starting points to confirm in conversation, not fixed facts.\n</person>"
    )


def _user_raised_faith(text):
    """Light per-turn faith-signal check (specific idiom or generic God-language)."""
    if not text or not str(text).strip():
        return False
    if detect_faith_signal(text) is not None:
        return True
    low = str(text).lower()
    return any(re.search(r"\b" + re.escape(w) + r"\b", low) for w in GENERIC_FAITH_WORDS)


_HARD_RULES = (
    "Never proselytize or push toward/away from any faith."
    " Their faith community is ONE support option among friends, family, professionals."
    " Family-faith gaps: stay neutral."
    " Safety beats faith framing: crisis first, helplines first."
)

_SECULAR_CORE = (
    "Keep support fully secular and human: no religious framing, no God-language, no scripture."
    " If they bring up faith themselves, follow their lead without endorsing or challenging it."
    " Treat any faith passages above as inadmissible unless they directly asked about faith."
)

_ONLY_QUIET_CORE = (
    "Keep support fully secular this turn: no religious framing, no God-language, no scripture."
    " They said to bring faith in only if they raise it, and they have not."
    " Treat any faith passages above as inadmissible."
)

_ONLY_RAISED_CORE = (
    "Bring in their faith only when they raise it themselves."
    " Then mirror their own idiom in plain words, one idea at a time, as an option"
    " — never more than one religious idea per reply, never a prescription."
)

_ANCHOR_CORE = (
    "Their faith is a live resource. Mirror their own idiom in plain words when it fits."
    " Offer tradition ideas only as an option, one idea at a time,"
    " never a prescription, never more than one religious idea per reply."
)

_STRAIN_CORE = (
    "Watch for spiritual struggle: feeling punished or abandoned, guilt about sin or devotion,"
    " or repeated prayer to quiet anxiety. Their tradition also carries"
    " mercy, forgiveness and compassion. Never amplify guilt, never suggest more devotion as a fix."
    " If ritualizing looks compulsive, name it kindly and encourage a professional"
    " who understands scrupulosity."
)


def render_faith_style(profile, user_text=""):
    """Render the <faith_style> block, or None when the profile carries no signal."""
    if not profile:
        return None
    imp = str(profile.get("faith_importance") or "")
    consent = str(profile.get("faith_consent") or "")
    stance = str(profile.get("faith_stance") or "")
    if _blank(imp) and _blank(consent) and _blank(stance):
        return None
    imp_l, con_l, sta_l = imp.strip().lower(), consent.strip().lower(), stance.strip().lower()
    gated = (not _blank(imp) and ("none" in imp_l or "not at all" in imp_l)) or (
        not _blank(consent) and (con_l == "no" or "neutral" in con_l)
    )
    if gated:
        # shortcut: retrieval's explicit-signal override may still surface faith hits, so the
        # secular block marks them inadmissible unless the user directly asked about faith.
        core_text = _SECULAR_CORE
    elif not _blank(stance) and ("weigh" in sta_l or "strain" in sta_l or "both" in sta_l):
        core_text = _STRAIN_CORE
    elif not _blank(consent) and "only" in con_l:
        # shortcut: F022 retrieval leaves only_if_raised ungated, so presentation gates per-turn —
        # faith idiom only when this message raises faith, secular that turn otherwise.
        core_text = _ONLY_RAISED_CORE if _user_raised_faith(user_text) else _ONLY_QUIET_CORE
    elif not _blank(imp) or not _blank(consent):
        core_text = _ANCHOR_CORE
    else:
        return None
    return f"<faith_style>\n{core_text} {_HARD_RULES}\n</faith_style>"


def build_system_prompt(profile=None, user_text=""):
    """V1 prompt byte-identical for an empty profile; person+faith blocks otherwise."""
    if not profile:
        return SYSTEM_PROMPT
    person = render_person(profile)
    faith = render_faith_style(profile, user_text=user_text)
    if person is None and faith is None:
        return SYSTEM_PROMPT
    return "\n\n".join([SYSTEM_PROMPT] + [b for b in (person, faith) if b])


def _last_user_text(turns):
    for m in reversed(list(turns or [])):
        if m.get("role") == "user" and str(m.get("content") or "").strip():
            return m["content"]
    return ""


def build_messages(turns, passages, profile=None):
    """One system message, then turns, then the retrieved block (if any)."""
    system = build_system_prompt(profile, user_text=_last_user_text(turns))
    messages = [{"role": "system", "content": system}] + list(turns)
    if passages:
        messages.append(_retrieved_message(passages))
    return messages


def _fit_budget(turns, passages, max_tokens, profile=None):
    """Drop oldest turns until the estimated prompt fits the server window."""
    # Conservative 6144 budget against the 8192 server window: history is cut
    # early, but prompt + completion always fits with room for estimator error.
    budget = 6144 - max_tokens - 256
    turns = list(turns)
    while len(turns) > 1:
        size = sum(_estimate_tokens(m["content"]) for m in build_messages(turns, passages, profile))
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
        return (msg.get("content") or "").strip(), body.get("usage")
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
    profile=None,
):
    """Streaming turn: same retrieval + prompt build, chunks relayed from vLLM."""
    if not user_text or not user_text.strip():
        raise ValueError("empty user turn")
    if retrieve_fn is None:
        if profile:
            _prof = profile
            retrieve_fn = lambda q, n=5: retrieve(
                q,
                n=n,
                religion=resolve_religion(
                    _prof.get("religion"),
                    _prof.get("faith_importance"),
                    _prof.get("faith_consent"),
                    user_text=q,
                ),
            )
        else:
            retrieve_fn = retrieve
    try:
        passages = retrieve_fn(user_text, n=rag_top_k)
    except Exception as e:
        raise RagError(f"retrieval failed: {e}") from e
    turns = _fit_budget(
        list(history or []) + [{"role": "user", "content": user_text.strip()}],
        passages,
        max_tokens,
        profile,
    )
    own_client = client is None
    client = client or httpx.Client(base_url=VLLM_URL, timeout=TIMEOUT_S)
    stream = generate_stream(
        client,
        build_messages(turns, passages, profile),
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
    profile=None,
):
    """Full turn: retrieve passages, inject, generate. Returns reply + sources."""
    if not user_text or not user_text.strip():
        raise ValueError("empty user turn")
    if retrieve_fn is None:
        if profile:
            _prof = profile
            retrieve_fn = lambda q, n=5: retrieve(
                q,
                n=n,
                religion=resolve_religion(
                    _prof.get("religion"),
                    _prof.get("faith_importance"),
                    _prof.get("faith_consent"),
                    user_text=q,
                ),
            )
        else:
            retrieve_fn = retrieve
    try:
        passages = retrieve_fn(user_text, n=rag_top_k)
    except Exception as e:
        raise RagError(f"retrieval failed: {e}") from e
    turns = _fit_budget(
        list(history or []) + [{"role": "user", "content": user_text.strip()}],
        passages,
        max_tokens,
        profile,
    )
    own_client = client is None
    client = client or httpx.Client(base_url=VLLM_URL, timeout=TIMEOUT_S)
    try:
        content, _ = generate(
            client,
            build_messages(turns, passages, profile),
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
