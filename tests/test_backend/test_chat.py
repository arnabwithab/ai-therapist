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


def test_build_messages_puts_passages_in_trailing_tags_block():
    turns = [{"role": "user", "content": "I can't sleep."}]
    msgs = core.build_messages(turns, _passages())
    assert msgs[0]["role"] == "system"
    assert msgs[1] == {"role": "user", "content": "I can't sleep."}
    tags = msgs[-1]
    assert tags["content"].startswith("<retrieved_techniques>")
    assert "Mindfulness of breathing" in tags["content"]
    assert "[s1 \u2014 c1]" in tags["content"]
    assert tags["content"].rstrip().endswith("respond to the person's last message.")


def test_build_messages_omits_tags_block_without_passages():
    msgs = core.build_messages([{"role": "user", "content": "hello"}], [])
    assert len(msgs) == 2
    assert msgs[-1] == {"role": "user", "content": "hello"}


def test_system_prompt_has_crisis_resources_resolved():
    assert "{crisis_resources}" not in core.SYSTEM_PROMPT
    assert "Tele MANAS" in core.SYSTEM_PROMPT


def test_build_messages_without_passages_still_works():
    msgs = core.build_messages([{"role": "user", "content": "hello"}], [])
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "hello"}


def test_build_messages_single_system_message_with_history():
    turns = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    msgs = core.build_messages(turns, _passages())
    assert sum(1 for m in msgs if m["role"] == "system") == 1
    assert [m["content"] for m in msgs[1:-1]] == ["first", "second", "third"]
    assert msgs[-1]["content"].startswith("<retrieved_techniques>")


def test_fit_budget_drops_oldest_turns_first():
    turns = [{"role": "user", "content": "x" * 4000}] + [{"role": "user", "content": "latest"}]
    kept = core._fit_budget(turns, [], max_tokens=4000)
    assert [m["content"] for m in kept] == ["latest"]


def test_chat_includes_history_in_model_body():
    seen = {}

    def handler(request):
        seen["body"] = __import__("json").loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "ok"}}],
                "usage": None,
            },
        )

    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    core.chat(
        "current",
        history=[{"role": "user", "content": "prior"}],
        client=client,
        retrieve_fn=lambda q, n=5: [],
    )
    contents = [m["content"] for m in seen["body"]["messages"]]
    assert "prior" in contents and "current" in contents
    assert contents.index("prior") < contents.index("current")


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
    assert seen["body"]["top_k"] == 20 and seen["body"]["repetition_penalty"] == 1.1


def test_generate_server_error_raises_model_error():
    client = _mock_client(status=500, payload={"error": "boom"})
    with pytest.raises(core.ModelError):
        core.generate(client, [{"role": "user", "content": "hi"}])


def test_generate_timeout_raises_model_error():
    client = _mock_client(exc=httpx.ConnectError("refused"))
    with pytest.raises(core.ModelError):
        core.generate(client, [{"role": "user", "content": "hi"}])


def test_generate_null_content_returns_empty_string():
    # reasoning models (Qwen3 via vLLM) can return content: null when the
    # completion holds only thinking traces — must not 500 (live traceback).
    client = _mock_client(payload=_completion(text=None))
    content, _ = core.generate(client, [{"role": "user", "content": "hi"}])
    assert content == ""


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


def _sse_client(chunks, status=200):
    def handler(request):
        body = (
            b"".join(b"data: " + __import__("json").dumps(c).encode() + b"\n\n" for c in chunks)
            + b"data: [DONE]\n\n"
        )
        return httpx.Response(status, content=body)

    return httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))


def test_generate_stream_yields_delta_contents():
    chunks = [
        {"choices": [{"delta": {"content": "Take "}}]},
        {"choices": [{"delta": {"content": "a breath."}}]},
    ]
    out = list(core.generate_stream(_sse_client(chunks), [{"role": "user", "content": "hi"}]))
    assert b"Take " in b"".join(out) and b"a breath." in b"".join(out)


def test_generate_stream_server_error_raises_model_error():
    with pytest.raises(core.ModelError):
        list(
            core.generate_stream(
                _mock_client(status=500, payload={}), [{"role": "user", "content": "hi"}]
            )
        )


def test_chat_stream_returns_iterator_starting_with_delta():
    it = core.chat_stream(
        "insomnia",
        client=_sse_client([{"choices": [{"delta": {"content": "Sleep."}}]}]),
        retrieve_fn=lambda q, n=5: _passages(),
    )
    assert b"Sleep." in b"".join(it)


def _full_profile(**over):
    p = {
        "call_name": "Aarav",
        "age_bracket": "25-34",
        "gender": "man",
        "language_pref": "Hinglish",
        "religion": "Hindu",
        "family_religion": "Sikh",
        "faith_importance": "it's my anchor",
        "faith_stance": "comforts",
        "faith_consent": "yes, when it fits",
        "supports": "family, friends",
        "concern": "work stress",
    }
    p.update(over)
    return p


def test_system_prompt_guards_substance_free_openers():
    # regression: "hey there" must not get feelings presumed (line-40 guard).
    assert "no substance yet" in core.SYSTEM_PROMPT
    assert "don't presume feelings or problems" in core.SYSTEM_PROMPT


def test_empty_profiles_render_none_and_v1_exact():
    assert core.render_person(None) is None
    assert core.render_person({}) is None
    assert core.render_faith_style(None) is None
    assert core.render_faith_style({}) is None
    assert core.build_system_prompt(None) is core.SYSTEM_PROMPT
    assert core.build_system_prompt({}) is core.SYSTEM_PROMPT
    msgs = core.build_messages([{"role": "user", "content": "hi"}], [], profile={})
    assert msgs[0]["content"] is core.SYSTEM_PROMPT


def test_person_full_omits_blank_and_pnts():
    p = _full_profile(age_bracket="", gender="prefer not to say", supports="Prefer-Not-To-Say")
    person = core.render_person(p)
    assert "Aarav" in person and "Hinglish" in person
    assert "Hindu" in person and "work stress" in person
    assert "prefer not to say" not in person.lower()
    assert "use naturally, don't recite" in person
    assert "starting points to confirm" in person


def test_faith_anchor_is_option_not_prescription():
    faith = core.render_faith_style(_full_profile(), user_text="work stress")
    low = faith.lower()
    assert "one idea at a time" in low
    assert "as an option" in low
    assert "never a prescription" in low
    assert "never more than one religious idea per reply" in low
    assert "never proselytize" in low
    assert "you should pray" not in low


def test_faith_strain_mercy_not_more_devotion():
    p = _full_profile(faith_importance="quite a bit", faith_stance="weighs down")
    faith = core.render_faith_style(p, user_text="I feel punished")
    low = faith.lower()
    assert "mercy" in low and "forgiveness" in low and "compassion" in low
    assert "never amplify guilt" in low
    assert "never suggest more devotion as a fix" in low
    assert "scrupulosity" in low and "professional" in low
    assert "pray more" not in low


def test_faith_gated_stays_secular_unprompted():
    p = _full_profile(faith_importance="not at all", faith_consent="keep it neutral")
    faith = core.render_faith_style(p, user_text="work stress")
    low = faith.lower()
    assert "fully secular" in low and "no god-language" in low
    assert "never proselytize" in low
    for w in ("dua", "puja", "paath", "simran"):
        assert w not in low
    # even an explicit faith signal in this turn stays secular for no/neutral
    faith2 = core.render_faith_style(p, user_text="I did namaaz today")
    assert "fully secular" in faith2.lower()


def test_faith_only_if_raised_gates_per_turn():
    p = _full_profile(faith_importance="quite a bit", faith_consent="only when I raise it")
    quiet = core.render_faith_style(p, user_text="work stress").lower()
    assert "fully secular" in quiet
    raised = core.render_faith_style(p, user_text="I missed namaaz today").lower()
    assert "only when they raise" in raised
    assert "one idea at a time" in raised
    assert "never more than one religious idea per reply" in raised


def test_system_prompt_does_not_parrot_profile():
    sys = core.build_system_prompt(_full_profile(), user_text="work stress")
    assert "As a " not in sys and "As an " not in sys
    assert "use naturally, don't recite" in sys


def test_person_faith_within_budget():
    p = _full_profile(faith_importance="quite a bit", faith_stance="both")
    person = core.render_person(p) or ""
    faith = core.render_faith_style(p, user_text="work stress") or ""
    assert core._estimate_tokens(person + faith) <= 250


def test_build_messages_single_system_with_profile():
    turns = [{"role": "user", "content": "hello"}]
    msgs = core.build_messages(turns, _passages(), profile=_full_profile())
    assert sum(1 for m in msgs if m["role"] == "system") == 1
    assert core.SYSTEM_PROMPT in msgs[0]["content"]
    assert "<person>" in msgs[0]["content"] and "<faith_style>" in msgs[0]["content"]


def test_chat_stream_and_chat_thread_profile(monkeypatch):
    seen = {}

    def fake_retrieve(q, n=5, religion=None):
        seen["religion"] = religion
        seen["q"] = q
        return []

    monkeypatch.setattr(core, "retrieve", fake_retrieve)
    bodies = {}

    def handler(request):
        bodies["chat"] = __import__("json").loads(request.content)
        return httpx.Response(200, json=_completion("ok"))

    profile = _full_profile(
        religion="Muslim", faith_importance="quite a bit", faith_consent="yes, when it fits"
    )
    out = core.chat(
        "I missed namaaz",
        profile=profile,
        client=httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)),
    )
    assert out["reply"] == "ok" and seen["religion"] == "islam" and seen["q"] == "I missed namaaz"
    assert "<person>" in bodies["chat"]["messages"][0]["content"]

    def sse_handler(request):
        bodies["stream"] = __import__("json").loads(request.content)
        body = b'data: {"choices": [{"delta": {"content": "ok"}}]}\n\n' + b"data: [DONE]\n\n"
        return httpx.Response(200, content=body)

    sse = httpx.Client(base_url="http://test", transport=httpx.MockTransport(sse_handler))
    assert b"ok" in b"".join(core.chat_stream("I missed namaaz", profile=profile, client=sse))
    assert seen["religion"] == "islam"
    assert "<person>" in bodies["stream"]["messages"][0]["content"]


def test_chat_profile_none_is_v1(monkeypatch):
    seen = {}

    def handler(request):
        seen["body"] = __import__("json").loads(request.content)
        return httpx.Response(200, json=_completion("fine"))

    monkeypatch.setattr(core, "retrieve", lambda q, n=5: [])
    out = core.chat(
        "hello",
        profile=None,
        client=httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler)),
    )
    assert out["reply"] == "fine"
    assert seen["body"]["messages"][0]["content"] == core.SYSTEM_PROMPT


def test_fit_budget_keeps_history_with_full_rag_at_1024():
    # F020 regression: five full 512-token passages + multi-turn history must
    # all fit at max_tokens=1024 (budget 6144-1024-256 = 4864 est-tokens).
    passages = [
        {"text": "p" * 2048, "source": f"s{i}", "chapter": f"c{i}", "score": 0.9} for i in range(5)
    ]
    turns = [
        {"role": "user", "content": "u" * 400},
        {"role": "assistant", "content": "a" * 400},
        {"role": "user", "content": "v" * 400},
        {"role": "assistant", "content": "b" * 400},
        {"role": "user", "content": "current worry"},
    ]
    kept = core._fit_budget(turns, passages, max_tokens=1024)
    assert kept == turns
