# Evals

## Baseline (Qwen3-4B-Instruct-2507, n=10000)

MentalBench-Align judge (CSS + ARS, 1–5 each) over
`data/eval/mbench-qwen3-4b-baseline/output_mbench.csv`,
scored by `muse-spark-1.3-contributor` via `eval/mbench_baseline/judge.py`
(`make judge`, reasoning effort minimal, temp 0). Full output (gitignored):
`data/eval/mbench-judge-500.csv`. Judged 2026-10-16, idx 0–9999, no gaps.

| Dimension       | Mean | SD   | %5  | %≤2 |
|-----------------|------|------|-----|-----|
| Guidance        | 3.00 | 0.64 | 0   | 16  |
| Informativeness | 3.71 | 0.49 | 1   | 1   |
| Relevance       | 4.85 | 0.43 | 87  | 0   |
| Safety          | 4.67 | 0.80 | 82  | 5   |
| Empathy         | 4.98 | 0.15 | 98  | 0   |
| Helpfulness     | 4.11 | 0.51 | 19  | 0   |
| Understanding   | 4.57 | 0.65 | 64  | 2   |
| Overall         | 4.06 | 0.48 | 14  | 1   |

Notes: Guidance is the weak axis (never a 5). Empathy is saturated —
textbook LLM-judge inflation; treat affective scores as inflated +0.4–0.8
per the MentalBench paper. The ~5% Safety ≤2 rows still need the human
spot-check (F009-3).
