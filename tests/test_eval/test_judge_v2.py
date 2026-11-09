"""Tests for eval/judge_v2.py — no network (call_model/httpx are stubbed)."""

import csv
import json

import pytest

from eval import judge_v2 as J


def _csv(tmp_path, rows, name="in.csv"):
    p = tmp_path / name
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return p


def test_culture_prompt_injects_reference_norm():
    p = J.build_prompt_culture("ctx", "resp", "name the caste slur, do not minimise")
    assert "name the caste slur" in p
    assert "No reference standard" in J.build_prompt_culture("ctx", "resp")


def test_gdb_severity_two_marks_displaced(monkeypatch):
    d = {
        "ds_severity": 2,
        "substantive_left": False,
        "unsolicited_fixing": False,
        "Explanation": "Support content is about half the reply.",
    }
    monkeypatch.setattr(J, "call_model", lambda *a, **k: json.dumps(d))
    assert J.judge_row(None, "s", "muse", 1, "gdb", "c", "r")["displaced"] == 1


def test_prompts_contain_context_response_and_keys():
    prompts = {
        "miti": J.build_prompt_miti("exam anxiety", "breathe"),
        "safety": J.build_prompt_safety("exam anxiety", "breathe"),
        "culture": J.build_prompt_culture("caste slur at work", "reflect"),
        "gdb": J.build_prompt_gdb("exam anxiety", "breathe"),
    }
    for inst, p in prompts.items():
        assert "exam anxiety" in p or "caste slur" in p
        for k in J.PROMPT_FNS[inst][1]:
            assert k in p, (inst, k)


def test_parse_judgment_accepts_all_instruments():
    miti = {k: 0 for k in J.MITI_FIELDS} | {"Explanation": "ok"}
    safety = {k: False for k in J.SAFETY_FIELDS} | {
        "Explanation": "No medication advice and no disclosed risk is mishandled."
    }
    culture = {k: False for k in J.CULTURE_FIELDS} | {
        "Explanation": "The reply names the norm and stays specific to this client."
    }
    gdb = {
        "ds_severity": 0,
        "substantive_left": False,
        "unsolicited_fixing": False,
        "Explanation": "One concrete next step remains after removing validation.",
    }
    assert J.parse_judgment(json.dumps(miti), "miti")["open_questions"] == 0
    assert J.parse_judgment(json.dumps(safety), "safety")["adv_apathetic"] is False
    assert J.parse_judgment(json.dumps(culture), "culture")["bias_caste"] is False
    assert J.parse_judgment(json.dumps(gdb), "gdb")["substantive_left"] is False


def test_miti_does_not_require_explanation():
    miti = {k: 0 for k in J.MITI_FIELDS}
    assert J.parse_judgment(json.dumps(miti), "miti")["open_questions"] == 0


def test_parse_judgment_strips_code_fences():
    d = {
        "ds_severity": 1,
        "substantive_left": True,
        "unsolicited_fixing": False,
        "Explanation": "A question that moves the work forward remains.",
    }
    assert J.parse_judgment("```json\n" + json.dumps(d) + "\n```", "gdb")["Explanation"].startswith(
        "A question"
    )


def test_parse_judgment_rejects_bool_as_count_and_non_bool_flag():
    bad = {k: 0 for k in J.MITI_FIELDS} | {
        "Explanation": "a real sentence here",
        "open_questions": True,
    }
    with pytest.raises(ValueError):
        J.parse_judgment(json.dumps(bad), "miti")
    bad = {"ds_severity": 0, "substantive_left": 1, "unsolicited_fixing": False, "Explanation": "x"}
    with pytest.raises(ValueError):
        J.parse_judgment(json.dumps(bad), "gdb")
    for sev in (-1, 4, True):
        bad = {
            "ds_severity": sev,
            "substantive_left": True,
            "unsolicited_fixing": False,
            "Explanation": "x",
        }
        with pytest.raises(ValueError):
            J.parse_judgment(json.dumps(bad), "gdb")
    bad = {k: 0 for k in J.MITI_FIELDS} | {
        "Explanation": "a real sentence here",
        "open_questions": -1,
    }
    with pytest.raises(ValueError):
        J.parse_judgment(json.dumps(bad), "miti")


def test_miti_ratios_zero_denominators():
    empty = J.miti_ratios({})
    assert empty["reflection_question_ratio"] is None
    assert empty["pct_complex_reflections"] is None
    r = J.miti_ratios(
        {
            "open_questions": 1,
            "closed_questions": 1,
            "simple_reflections": 1,
            "complex_reflections": 3,
        }
    )
    assert r["reflection_question_ratio"] == 2.0
    assert r["pct_complex_reflections"] == 75.0


def test_crisis_helpline_regex():
    for text in ["call 14416", "Tele MANAS is free", "tele-manas helpline", "dial 112 now"]:
        assert J.crisis_helpline(text), text
    for text in ["call 1123", "1124 people", "911 works", "the 1441 code", "", None]:
        assert not J.crisis_helpline(text), text


def test_load_todo_resumes_written_pairs(tmp_path):
    base = _csv(
        tmp_path,
        [
            {"idx": "0", "context": "c0", "r": "resp0"},
            {"idx": "1", "context": "c1", "r": "resp1"},
            {"idx": "2", "context": "c2", "r": "  "},
        ],
    )
    out = tmp_path / "out.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["idx", "instrument"])
        w.writeheader()
        w.writerow({"idx": 0, "instrument": "miti"})
    todo = J.load_todo(base, out, "r", ["miti", "gdb"], 10)
    assert (0, "gdb", "c0", "resp0", "") in todo
    assert not any(t[0] == 0 and t[1] == "miti" for t in todo)
    assert not any(t[0] == 2 for t in todo)  # blank response skipped


def test_judge_row_adds_ratios_and_helpline(tmp_path, monkeypatch):
    miti = {k: 0 for k in J.MITI_FIELDS} | {"Explanation": "ok", "open_questions": 2}
    monkeypatch.setattr(J, "call_model", lambda *a, **k: json.dumps(miti))
    row = J.judge_row(None, "s", "muse", 7, "miti", "ctx", "call 112")
    assert row["idx"] == 7 and row["crisis_helpline"] == 1
    assert row["reflection_question_ratio"] == 0.0  # 2 questions, 0 reflections
    assert row["pct_complex_reflections"] is None


def test_judge_pair_swap_unswaps(tmp_path, monkeypatch):
    # judge always prefers the reply shown first, so the swap flips the label
    monkeypatch.setattr(J, "call_model", lambda *a, **k: '{"winner": "A", "Explanation": "e"}')
    same = J.judge_pair(None, "s", "muse", 1, "c", "x", "y", 0.0, 0)
    flipped = J.judge_pair(None, "s", "muse", 1, "c", "x", "y", 1.0, 0)
    assert same["winner"] == "A" and same["swapped"] == 0
    assert flipped["winner"] == "B" and flipped["swapped"] == 1


def test_judge_pair_tie_never_flips(monkeypatch):
    monkeypatch.setattr(J, "call_model", lambda *a, **k: '{"winner": "TIE", "Explanation": "e"}')
    assert J.judge_pair(None, "s", "muse", 1, "c", "x", "y", 1.0, 0)["winner"] == "TIE"


def test_pair_swap_is_deterministic(monkeypatch):
    monkeypatch.setattr(J, "call_model", lambda *a, **k: '{"winner": "A", "Explanation": "e"}')
    a = J.judge_pair(None, "s", "muse", 5, "c", "x", "y", 0.5, 0)
    b = J.judge_pair(None, "s", "muse", 5, "c", "x", "y", 0.5, 0)
    assert a["swapped"] == b["swapped"]


def test_judge_pair_forced_order_swaps(monkeypatch):
    monkeypatch.setattr(J, "call_model", lambda *a, **k: '{"winner": "A", "Explanation": "e"}')
    r = J.judge_pair(None, "s", "muse", 1, "c", "x", "y", 0.0, 0, order=1)
    assert r["swapped"] == 1 and r["order"] == 1 and r["winner"] == "B"


def test_write_rows_header_once(tmp_path):
    p = tmp_path / "o.csv"
    fields = ["idx", "instrument", "Explanation"]
    J.write_rows(p, [{"idx": 1, "instrument": "miti", "Explanation": "a"}], fields)
    J.write_rows(p, [{"idx": 2, "instrument": "miti", "Explanation": "b"}], fields)
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and rows[1]["idx"] == "2"


class _FakeClient:
    def __init__(self, texts):
        self.texts = list(texts)
        self.budgets = []

    def post(self, path, json=None, headers=None):
        self.budgets.append(json["max_output_tokens"])
        text = self.texts.pop(0)

        class R:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "status": "completed",
                    "output": [{"type": "message", "content": [{"text": text}]}],
                }

        return R()


def test_call_model_escalates_budget_on_empty_text():
    c = _FakeClient(["", '{"ok": 1}'])
    out = J.call_model(c, "s", "p", "muse")
    assert out == '{"ok": 1}'
    assert c.budgets == [J.MAX_TOKENS, J.MAX_TOKENS * 2]


def test_safe_call_drops_failures():
    def boom(_):
        raise ValueError("x")

    assert J.safe_call(boom, ("idx", "inst")) is None
    assert J.safe_call(lambda t: t[0], ("idx", "inst")) == "idx"


def test_write_incremental_keeps_completed_rows_on_failure(tmp_path):

    p = tmp_path / "o.csv"
    fields = ["idx", "instrument"]

    def fn(t):
        if t == 1:
            raise ValueError("bad row")
        return {"idx": t, "instrument": "miti"}

    n = J.write_incremental(p, fields, [0, 1, 2], lambda t: J.safe_call(fn, t), workers=3)
    assert n == 2
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert sorted(r["idx"] for r in rows) == ["0", "2"]


def test_run_pairs_passes_order_to_judge_pair(tmp_path, monkeypatch):
    import argparse

    calls = []
    monkeypatch.setattr(
        J,
        "judge_pair",
        lambda *a, **k: calls.append(a)
        or {"idx": a[3], "winner": "A", "swapped": 0, "order": a[-1], "Explanation": "e"},
    )
    base = _csv(
        tmp_path,
        [{"idx": "0", "context": "c", "a": "r1", "b": "r2"}],
    )
    out = tmp_path / "p.csv"
    ns = argparse.Namespace(
        baseline=base,
        out=out,
        resp_col="a",
        pair_col="b",
        context_col="context",
        n=10,
        both_orders=True,
        swap_frac=0.0,
        seed=0,
        workers=1,
        session="s",
        model="m",
        hf_repo="",
        hf_path="",
    )
    assert J.run_pairs(ns, None) == 2
    assert sorted(a[-1] for a in calls) == [0, 1]


def test_parse_judgment_rejects_placeholder_explanation():
    d = {k: False for k in J.SAFETY_FIELDS} | {"Explanation": "one sentence"}
    with pytest.raises(ValueError):
        J.parse_judgment(json.dumps(d), "safety")
    d = {k: False for k in J.SAFETY_FIELDS} | {"Explanation": "short"}
    with pytest.raises(ValueError):
        J.parse_judgment(json.dumps(d), "safety")


def test_parse_json_accepts_single_line_fence():
    assert J.parse_json('```json {"a": 1} ```') == {"a": 1}


def test_assert_header_rejects_mismatched_file(tmp_path):
    p = tmp_path / "o.csv"
    J.write_rows(p, [{"idx": 1, "instrument": "miti"}], ["idx", "instrument"])
    with pytest.raises(SystemExit):
        J.assert_header(p, ["idx", "instrument", "extra"])
    J.assert_header(p, ["idx", "instrument"])


def test_write_incremental_round_trip(tmp_path):
    p = tmp_path / "o.csv"
    fields = ["idx", "instrument", "crisis_helpline"]
    J.write_incremental(
        p, fields, [0, 1], lambda t: {"idx": t, "instrument": "safety", "crisis_helpline": 1}, 2
    )
    with open(p, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["crisis_helpline"] for r in rows] == ["1", "1"]
