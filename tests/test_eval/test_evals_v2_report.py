"""Tests for eval/evals_v2_report.py — pure aggregation, no network."""

from eval import evals_v2_report as R


def _miti(idx, **kw):
    base = {k: 0 for k in R.MITI_FIELDS}
    base.update(kw)
    return {"idx": idx, "instrument": "miti", **base, "Explanation": "x"}


def _safety(idx, **kw):
    base = {k: 0 for k in R.SAFETY_FIELDS}
    base.update(kw)
    return {"idx": idx, "instrument": "safety", **base, "Explanation": "x"}


def test_summarize_miti_means_and_zero_question_rows():
    rows = [
        _miti("0", open_questions=2, complex_reflections=1),
        _miti("1", open_questions=0, simple_reflections=1),
    ]
    s = R.summarize(rows)["instruments"]["miti"]
    assert s["n"] == 2 and s["open_questions"] == 1.0
    assert s["zero_question_rows"] == 1
    assert s["pct_complex_reflections"] == 50.0  # mean(100, 0)


def test_summarize_binary_rates():
    rows = [_safety("0"), _safety("1", adv_apathetic=1), _safety("2", adv_apathetic=1)]
    s = R.summarize(rows)["instruments"]["safety"]
    assert s["adv_apathetic"] == 0.6667
    assert s["tg_failed_crisis"] == 0.0


def test_summarize_gdb_derives_displaced_from_severity():
    rows = [
        {"idx": "0", "instrument": "gdb", "ds_severity": "1"},
        {"idx": "1", "instrument": "gdb", "ds_severity": "3"},
    ]
    b = R.summarize(rows)["instruments"]["gdb"]
    assert b["mean_ds_severity"] == 2.0 and b["displaced"] == 0.5


def test_agreement_uses_shared_rows():
    a = [_safety("0", adv_apathetic=1), _safety("1")]
    b = [_safety("0", adv_apathetic=1), _safety("1", adv_apathetic=1)]
    agree = R.agreement(a, b)
    assert agree["adv_apathetic"] == {"n": 2, "agreement": 0.5}
    assert agree["tg_failed_crisis"]["agreement"] == 1.0


def test_pair_stats_win_and_flip():
    rows = [
        {"idx": "0", "winner": "A", "order": "0"},
        {"idx": "0", "winner": "B", "order": "1"},  # flipped
        {"idx": "1", "winner": "TIE", "order": "0"},
        {"idx": "1", "winner": "TIE", "order": "1"},
    ]
    s = R.pair_stats(rows)
    assert s["n_pairs"] == 2
    assert s["a_win_rate"] == 0.5 and s["tie_rate"] == 0.5
    assert s["both_order_pairs"] == 2 and s["flip_rate"] == 0.5


def test_pair_stats_dedupes_repeated_pairs():
    rows = [
        {"idx": "0", "winner": "A", "order": "0"},
        {"idx": "0", "winner": "B", "order": "0"},  # last wins
        {"idx": "0", "winner": "A", "order": "1"},
    ]
    s = R.pair_stats(rows)
    assert s["n_pairs"] == 1 and s["b_win_rate"] == 1.0
    assert s["flip_rate"] == 1.0


def test_gate_passes_on_clean_pilot():
    summary = {
        "instruments": {
            "miti": {"idx": 100},
            "safety": {"idx": 100, "adv_apathetic": 0.2},
            "gdb": {"idx": 100, "substantive_left": 0.6},
        }
    }
    agree = {"adv_apathetic": {"n": 100, "agreement": 0.8}}
    pairs = {"both_order_pairs": 20, "flip_rate": 0.05, "tie_rate": 0.2}
    checks = R.gate(summary, agree, pairs, expected_idx=100)
    assert all(ok for _, ok, _ in checks), checks


def test_gate_fails_on_flat_flag_and_low_completeness():
    summary = {
        "instruments": {
            "miti": {"idx": 50},
            "safety": {"idx": 50, "tg_failed_crisis": 1.0},
            "gdb": {"idx": 50},
        }
    }
    checks = {name: ok for name, ok, _ in R.gate(summary, {}, None, expected_idx=100)}
    assert not checks["safety completeness"]
    assert not checks["no all-true flag"]


def test_gate_crisis_criteria():
    summary = {"instruments": {"miti": {"idx": 100}}}
    good = {"direct": 16, "direct_hit": 14, "control": 8, "control_hit": 0}
    names = {n: ok for n, ok, _ in R.gate(summary, {}, None, crisis=good)}
    assert names["crisis regex hit >=0.75"] and names["crisis regex control FP =0"]
    bad = {"direct": 16, "direct_hit": 4, "control": 8, "control_hit": 2}
    names = {n: ok for n, ok, _ in R.gate(summary, {}, None, crisis=bad)}
    assert not names["crisis regex hit >=0.75"] and not names["crisis regex control FP =0"]


def test_to_markdown_renders_all_blocks():
    report = {
        "judge_a": {
            "instruments": {
                "miti": {
                    "n": 2,
                    "open_questions": 1.0,
                    "closed_questions": 0.0,
                    "simple_reflections": 1.0,
                    "complex_reflections": 0.5,
                    "affirmations": 0.0,
                    "mi_nonadherent": 0.0,
                    "reflection_question_ratio": 1.0,
                    "pct_complex_reflections": 50.0,
                },
                "safety": {"n": 2, "adv_apathetic": 0.5},
                "gdb": {"n": 2, "mean_ds_severity": 1.5, "displaced": 0.5},
            }
        },
        "pairs": {
            "n_pairs": 10,
            "a_win_rate": 0.6,
            "b_win_rate": 0.3,
            "tie_rate": 0.1,
            "flip_rate": 0.0,
        },
        "crisis_regex": {"direct": 17, "direct_hit": 15, "control": 8, "control_hit": 0},
        "agreement": {"adv_apathetic": {"n": 2, "pos": 2, "agreement": 0.5}},
    }
    md = R.to_markdown(report)
    assert "| miti | 2 |" in md and "| safety | 2 |" in md
    assert "pairwise" in md and "crisis regex" in md
