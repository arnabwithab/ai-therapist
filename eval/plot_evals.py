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


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="docs")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)
    plot_master(args.outdir)
    plot_1p7b(args.outdir)
    print(f"wrote charts -> {args.outdir}")


if __name__ == "__main__":
    main()
