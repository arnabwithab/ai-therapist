"""Tests for eval/mbench_baseline/judge.py — no network (httpx MockTransport)."""

import csv

from eval.mbench_baseline import judge as J


def test_build_prompt_contains_context_response_and_all_keys():
    p = J.build_prompt("exam anxiety", "breathe deeply")
    assert "exam anxiety" in p and "breathe deeply" in p
    for k in J.FIELDS:
        assert k in p


def test_parse_judgment_accepts_valid_json():
    txt = (
        '{"Guidance": 4, "Informativeness": 4, "Relevance": 5, "Safety": 5,'
        ' "Empathy": 4, "Helpfulness": 4, "Understanding": 4, "Overall": 4,'
        ' "Explanation": "ok"}'
    )
    assert J.parse_judgment(txt)["Overall"] == 4


def test_parse_judgment_strips_code_fences():
    body = (
        '{"Guidance": 1, "Informativeness": 1, "Relevance": 2, "Safety": 2,'
        ' "Empathy": 1, "Helpfulness": 1, "Understanding": 1,'
        ' "Overall": 1, "Explanation": "x"}'
    )
    assert J.parse_judgment("```json\n" + body + "\n```")["Guidance"] == 1


def test_parse_judgment_rejects_out_of_range():
    import pytest

    with pytest.raises(ValueError):
        J.parse_judgment(
            '{"Guidance": 9, "Informativeness": 1, "Relevance": 1, "Safety": 1,'
            ' "Empathy": 1, "Helpfulness": 1, "Understanding": 1, "Overall": 1,'
            ' "Explanation": "x"}'
        )


def test_load_todo_takes_first_n_unjudged(tmp_path):
    base = tmp_path / "base.csv"
    base.write_text(
        "idx,context,qwen3_4b_instruct_2507\n0,c0,r0\n1,c1,r1\n2,c2,r2\n", encoding="utf-8"
    )
    out = tmp_path / "out.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx"])
        w.writeheader()
        w.writerow({"idx": "1"})
    todo = J.load_todo(str(base), str(out), n=500)
    assert [t[0] for t in todo] == [0, 2]


def test_judge_one_posts_responses_api_and_parses(tmp_path):
    import httpx

    body = (
        '{"Guidance": 3, "Informativeness": 3, "Relevance": 4, "Safety": 4,'
        ' "Empathy": 3, "Helpfulness": 3, "Understanding": 3, "Overall": 3,'
        ' "Explanation": "fine"}'
    )
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["session"] = request.headers.get("x-opencode-session")
        import json as _json

        assert _json.loads(request.content)["model"] == J.MODEL
        return httpx.Response(
            200,
            json={
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": body}]}],
            },
        )

    client = httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(handler), timeout=30
    )
    row = J.judge_one(client, "ses-x", 7, "ctx", "resp")
    assert row["idx"] == 7 and row["Overall"] == 3
    assert seen["url"].endswith("/responses") and seen["session"] == "ses-x"


def _judge_body(overall=3):
    return (
        '{"Guidance": 3, "Informativeness": 3, "Relevance": 4, "Safety": 4,'
        f' "Empathy": 3, "Helpfulness": 3, "Understanding": 3, "Overall": {overall},'
        ' "Explanation": "fine"}'
    )


def _ok_response(body):
    return {
        "status": "completed",
        "output": [{"type": "message", "content": [{"type": "output_text", "text": body}]}],
    }


def test_judge_one_retries_transient_failure():
    import httpx

    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(500, json={"error": "busy"})
        return httpx.Response(200, json=_ok_response(_judge_body()))

    client = httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(handler), timeout=30
    )
    row = J.judge_one(client, "ses-x", 7, "ctx", "resp", retries=2)
    assert row["Overall"] == 3 and len(calls) == 2


def test_load_todo_second_batch_continues_where_first_stopped(tmp_path):
    base = tmp_path / "base.csv"
    base.write_text(
        "idx,context,qwen3_4b_instruct_2507\n0,c0,r0\n1,c1,r1\n2,c2,r2\n", encoding="utf-8"
    )
    out = tmp_path / "out.csv"
    first = J.load_todo(str(base), str(out), n=2)
    assert [t[0] for t in first] == [0, 1]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx"])
        w.writeheader()
        for idx, _, _ in first:
            w.writerow({"idx": str(idx)})
    assert [t[0] for t in J.load_todo(str(base), str(out), n=2)] == [2]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx"])
        w.writeheader()
        for i in range(3):
            w.writerow({"idx": str(i)})
    assert J.load_todo(str(base), str(out), n=2) == []


def test_judge_one_honors_retry_after_on_429():
    import httpx

    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": "slow down"}, headers={"retry-after": "0"})
        return httpx.Response(200, json=_ok_response(_judge_body()))

    client = httpx.Client(
        base_url="http://test", transport=httpx.MockTransport(handler), timeout=30
    )
    assert J.judge_one(client, "ses-x", 7, "ctx", "resp", retries=2)["Overall"] == 3
    assert len(calls) == 2


def test_load_todo_uses_resp_col(tmp_path):
    base = tmp_path / "base.csv"
    base.write_text("idx,context,qwen3_4b_sft5\n0,c0,r0\n1,c1,\n2,c2,r2\n", encoding="utf-8")
    out = tmp_path / "out.csv"
    todo = J.load_todo(str(base), str(out), n=500, resp_col="qwen3_4b_sft5")
    assert [t[0] for t in todo] == [0, 2]
    assert todo[0][2] == "r0"
