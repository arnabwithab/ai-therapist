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
