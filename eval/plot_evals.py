"""Regenerate eval charts from judge CSVs (make plots). No GPU/network."""

import argparse
import csv
import os
import statistics

import matplotlib.pyplot as plt
import numpy as np

DIMS = [
    "Guidance",
    "Informativeness",
    "Relevance",
    "Safety",
    "Empathy",
    "Helpfulness",
    "Understanding",
    "Overall",
]

# Series name -> judge CSV (all n=10000, same rubric/judge).
MASTER = [
    ("4B base", "data/eval/mbench-judge-500.csv"),
    ("1.7B DPO", "data/eval/mbench-judge-1p7b-dpo-1024.csv"),
    ("0.6B DPO", "data/eval/mbench-judge-0p6b-dpo.csv"),
]

P1P7B = [
    ("Base 1.7B", "data/eval/mbench-judge-1p7b.csv", "#5B9BD5"),
    ("DPO 1.7B", "data/eval/mbench-judge-1p7b-dpo-1024.csv", "#7BC47F"),
    ("DPO+OARS", "data/eval/mbench-judge-1p7b-dpo-oars.csv", "#ED7D31"),
    ("DPO+OARS+RAG", "data/eval/mbench-judge-1p7b-dpo-oars-rag.csv", "#2E7D32"),
    ("DPO+reg+RAG", "data/eval/mbench-judge-1p7b-dpo-regular-rag.csv", "#C00000"),
]


def means(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 10000, f"{path}: {len(rows)} rows"
    return {d: statistics.mean(int(r[d]) for r in rows) for d in DIMS}


def plot_master(outdir):
    labels, vals = zip(*[(name, means(p)["Overall"]) for name, p in MASTER])
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(labels, vals, color=["C0", "C1", "C2"])
    ax.set_ylim(2.5, 4.5)
    ax.set_ylabel("Mean Overall score (1-5)")
    ax.set_title(
        "MentalBench-Align Overall means: best run per model (muse-spark, temp 0, n=10000 each)"
    )
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "evals-judge-means.png"), dpi=150)
    plt.close(fig)


def plot_1p7b(outdir):
    data = [(name, means(p), color) for name, p, color in P1P7B]
    x = np.arange(len(DIMS))
    w = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (name, m, color) in enumerate(data):
        vals = [m[d] for d in DIMS]
        bars = ax.bar(x + (i - (len(data) - 1) / 2) * w, vals, w, label=name, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_ylim(1, 5.6)
    ax.set_xticks(x)
    ax.set_xticklabels(DIMS)
    ax.set_ylabel("Mean judge score (1-5)")
    ax.set_title(
        "Qwen3-1.7B: DPO gains (+0.12), RAG drags both prompts (muse-spark, temp 0, n=10000 each)"
    )
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "evals-1p7b.png"), dpi=150)
    plt.close(fig)


from eval.evals_v2_report import load as v2_load
from eval.evals_v2_report import pair_stats as v2_pair_stats
from eval.evals_v2_report import summarize as v2_summarize

V2_CELLS = [
    ("Base regular", "data/eval/v2/compare-1p7b-base.csv", "#5B9BD5"),
    ("DPO regular", "data/eval/v2/compare-1p7b-dpo.csv", "#7BC47F"),
    ("DPO OARS", "data/eval/v2/compare-1p7b-dpo-oars.csv", "#ED7D31"),
    ("DPO OARS+RAG", "data/eval/v2/compare-1p7b-dpo-oars-rag.csv", "#2E7D32"),
]

# (chart label, instrument, summarize key). All are fail rates: lower is better.
V2_FLAGS = [
    ("Displaced", "gdb", "displaced"),
    ("Assumptions", "safety", "adv_assumptions"),
    ("Technique Rx", "safety", "adv_technique_prescription"),
    ("Failed crisis", "safety", "tg_failed_crisis"),
    ("Failed abuse", "safety", "tg_failed_abuse"),
]

# (chart label, summarize key). Counts per reply; direction in docs how-to-read.
V2_VOICE = [
    ("Open Q", "open_questions"),
    ("Complex refl.", "complex_reflections"),
    ("Affirm", "affirmations"),
    ("MINA", "mi_nonadherent"),
]

# (x label, pairs csv). A/B/tie from pair_stats; A/B mapping in docs table.
V2_PAIRS = [
    ("Base/DPO", "data/eval/v2/compare-pairs-base-dpo.csv"),
    ("Reg/OARS", "data/eval/v2/compare-pairs-dpo-oars.csv"),
    ("OARS/RAG", "data/eval/v2/compare-pairs-oars-rag.csv"),
]


def v2_values(path):
    """Chart-ready numbers for one compare cell: flag rates + voice counts."""
    s = v2_summarize(v2_load(path))["instruments"]
    return {
        "flags": {label: s[inst][key] for label, inst, key in V2_FLAGS},
        "voice": {label: s["miti"][key] for label, key in V2_VOICE},
    }


def plot_v2_flags(outdir):
    labels = [label for label, _, _ in V2_FLAGS]
    data = [(name, v2_values(p)["flags"], color) for name, p, color in V2_CELLS]
    x = np.arange(len(labels))
    w = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (name, vals, color) in enumerate(data):
        vs = [vals[label] for label in labels]
        bars = ax.bar(x + (i - (len(data) - 1) / 2) * w, vs, w, label=name, color=color)
        for b, v in zip(bars, vs):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_ylim(0, 0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Flag rate (fraction of replies, lower is better)")
    ax.set_title("Evals v2: fail flags by cell (1.7B ladder, muse, n=300 each)")
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "evals-v2-flags.png"), dpi=150)
    plt.close(fig)


def plot_v2_voice(outdir):
    labels = [label for label, _ in V2_VOICE]
    data = [(name, v2_values(p)["voice"], color) for name, p, color in V2_CELLS]
    x = np.arange(len(labels))
    w = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (name, vals, color) in enumerate(data):
        vs = [vals[label] for label in labels]
        bars = ax.bar(x + (i - (len(data) - 1) / 2) * w, vs, w, label=name, color=color)
        for b, v in zip(bars, vs):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_ylim(0, 3.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean count per reply (direction: see how-to-read)")
    ax.set_title("Evals v2: counselling voice by cell (1.7B ladder, muse, n=300 each)")
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "evals-v2-voice.png"), dpi=150)
    plt.close(fig)


def plot_v2_pairs(outdir):
    stats = [(label, v2_pair_stats(v2_load(p))) for label, p in V2_PAIRS]
    x = np.arange(len(stats))
    w = 0.25
    series = [("A wins", "a_win_rate", "#5B9BD5"), ("B wins", "b_win_rate", "#ED7D31"), ("Tie", "tie_rate", "#A6A6A6")]
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (name, key, color) in enumerate(series):
        vs = [st[key] or 0 for _, st in stats]
        bars = ax.bar(x + (i - 1) * w, vs, w, label=name, color=color)
        for b, v in zip(bars, vs):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in stats])
    ax.set_ylabel("Win rate (A/B mapping in table below)")
    ax.set_title("Evals v2: pairwise preferences (same judge, 300 pairs, both orders)")
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "evals-v2-pairs.png"), dpi=150)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    plot_master(args.outdir)
    plot_1p7b(args.outdir)
    plot_v2_flags(args.outdir)
    plot_v2_voice(args.outdir)
    plot_v2_pairs(args.outdir)
    print(f"wrote charts -> {args.outdir}")


if __name__ == "__main__":
    main()
