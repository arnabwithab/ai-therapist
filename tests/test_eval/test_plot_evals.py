"""Tests for eval/plot_evals.py — writes both PNGs into a tmp dir."""

import os

import pytest

pytest.importorskip("matplotlib")

from eval import plot_evals as P


def test_plots_write_both_pngs(tmp_path):
    P.main(["--outdir", str(tmp_path)])
    for name in ("evals-judge-means.png", "evals-1p7b.png"):
        p = os.path.join(str(tmp_path), name)
        assert os.path.exists(p) and os.path.getsize(p) > 10000


def test_means_reads_all_judge_csvs():
    for _, path in P.MASTER + [(n, p) for n, p, _ in P.P1P7B]:
        m = P.means(path)
        assert set(m) == set(P.DIMS)
        assert all(1.0 <= v <= 5.0 for v in m.values())


def test_v2_values_match_summarize():
    from eval import evals_v2_report as R

    for _, path, _ in P.V2_CELLS:
        s = R.summarize(R.load(path))["instruments"]
        v = P.v2_values(path)
        assert v["flags"]["Displaced"] == s["gdb"]["displaced"]
        assert v["flags"]["Failed crisis"] == s["safety"]["tg_failed_crisis"]
        assert v["voice"]["MINA"] == s["miti"]["mi_nonadherent"]


def test_v2_charts_write_pngs(tmp_path):
    P.plot_v2_flags(str(tmp_path))
    P.plot_v2_voice(str(tmp_path))
    P.plot_v2_pairs(str(tmp_path))
    for name in ("evals-v2-flags.png", "evals-v2-voice.png", "evals-v2-pairs.png"):
        p = os.path.join(str(tmp_path), name)
        assert os.path.exists(p) and os.path.getsize(p) > 10000
