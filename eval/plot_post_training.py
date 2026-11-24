"""Bar chart: MentalBench-Align Overall for Qwen3-1.7B post-training methods."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = 3.72


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/eval/post-training-1p7b.csv")
    parser.add_argument("--out", default="public/post-training-1p7b.png")
    args = parser.parse_args()

    rows = load(args.csv)
    labels = [r["method"] for r in rows]
    values = [float(r["overall"]) for r in rows]
    colors = [
        "#9aa0a6" if r["method"] == "Base" else "#1a73e8" if r["method"] == "DPO" else "#5f6368"
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=160)
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.axhline(BASE, color="#9aa0a6", linewidth=1, linestyle="--")
    ax.set_ylim(2.6, 4.0)
    ax.set_ylabel("MentalBench-Align Overall (1–5)")
    ax.set_title("Qwen3-1.7B after each post-training method")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e8eaed", linewidth=0.8)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    print(args.out)


if __name__ == "__main__":
    main()
