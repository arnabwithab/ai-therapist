# Evals

![MentalBench-Align judge means: 4B (base/SFT/DPO) and 1.7B (base/DPO)](evals-judge-means.png)

## DPO-from-base, Qwen3-1.7B ckpt-2000 @1024 tokens (n=10000)

MentalBench-Align judge (same prompt, judge, and settings) over post-think final
answers in `data/eval/mbench-qwen3-4b-baseline/output_mbench_1p7b_dpo_1024_final.csv`
(col `qwen3_1p7b_dpo`), scored 2026-10-25. Full output (gitignored):
`data/eval/mbench-judge-1p7b-dpo-1024.csv`. Zero truncation, zero empties.

| Dimension       | Mean | SD   | %5  | %≤2 |
|-----------------|------|------|-----|-----|
| Guidance        | 2.91 | 0.57 | 0   | 17 |
| Informativeness | 3.25 | 0.57 | 0   | 6 |
| Relevance       | 4.53 | 0.65 | 60   | 0 |
| Safety          | 4.57 | 0.92 | 78   | 7 |
| Empathy         | 4.80 | 0.41 | 80   | 0 |
| Helpfulness     | 3.86 | 0.46 | 3   | 1 |
| Understanding   | 3.96 | 0.72 | 18   | 5 |
| Overall         | 3.84 | 0.51 | 4   | 2 |

Notes: the fair-budget run confirms the hypothesis — DPO-1.7B beats base-1.7B
(3.84 vs 3.72, +0.12), with the biggest recovery exactly where truncation hurt:
Guidance 1.77 → 2.91. Relevance/Empathy/Helpfulness/Understanding all land at
or above base. The earlier 512-token run (Overall 3.33, 141 truncated + 8 empty)
is superseded; it measured DPO+truncation, not DPO. Net of the 1.7B loop:
smaller base leaves headroom and DPO fills it — the opposite of the 4B story,
where the base was already at the judge's ceiling.

## DPO-from-base, Qwen3-1.7B ckpt-2000 @512 tokens (n=9992 judged, SUPERSEDED)

MentalBench-Align judge (same prompt, judge, and settings) over post-think
final answers in `data/eval/mbench-qwen3-4b-baseline/output_mbench_1p7b_dpo_final.csv`
(col `qwen3_1p7b_dpo`, thinking traces stripped; raw canonical CSV kept),
scored 2026-10-22. 8 rows empty (thought with no answer) + 141 truncated
pre-`</think>` under the 512-token cap; 9992 rows judged, no gaps besides those.
Full output (gitignored): `data/eval/mbench-judge-1p7b-dpo.csv`.

| Dimension       | Mean | SD   | %5  | %≤2 |
|-----------------|------|------|-----|-----|
| Guidance        | 1.77 | 0.87 | 0   | 75   |
| Informativeness | 2.77 | 0.65 | 0   | 29   |
| Relevance       | 4.22 | 0.63 | 31   | 1   |
| Safety          | 4.54 | 0.92 | 74   | 7   |
| Empathy         | 4.36 | 0.61 | 41   | 1   |
| Helpfulness     | 3.21 | 0.69 | 0   | 12   |
| Understanding   | 3.87 | 0.62 | 7   | 5   |
| Overall         | 3.33 | 0.63 | 0   | 7   |

Notes: DPO-1.7B trails base-1.7B (3.33 vs 3.72) — the headroom hypothesis
FAILED in this setup. The drop concentrates in Guidance (−0.96): thinking
traces eat ~60% of the 512-token budget, leaving answers too short for the
actionable-steps rubric; 8 thought-only rows + 141 truncated rows are the
extreme end. Understanding is the one axis that improves (+0.14), consistent
with thinking helping comprehension while hurting completeness. The model
also emits <think> on 10000/10000 rows (base: 0) — the adapter overrode the
no-think header. Fair retest needs max_tokens that fits thought+answer
(e.g. 1024); until then this scores DPO+truncation, not DPO alone.

## Base Qwen3-1.7B (n=10000)

MentalBench-Align judge (same prompt, judge, and settings) over
`data/eval/mbench-qwen3-4b-baseline/output_mbench_1p7b.csv`
(col `qwen3_1p7b_base`), scored 2026-10-21, idx 0–9999, no gaps.
Full output (gitignored): `data/eval/mbench-judge-1p7b.csv`.

| Dimension       | Mean | SD   | %5  | %≤2 |
|-----------------|------|------|-----|-----|
| Guidance        | 2.77 | 0.53 | 0   | 24   |
| Informativeness | 2.98 | 0.51 | 0   | 13   |
| Relevance       | 4.30 | 0.72 | 44   | 1   |
| Safety          | 4.59 | 0.89 | 78   | 7   |
| Empathy         | 4.58 | 0.51 | 58   | 0   |
| Helpfulness     | 3.72 | 0.50 | 0   | 1   |
| Understanding   | 3.73 | 0.69 | 6   | 7   |
| Overall         | 3.72 | 0.52 | 1   | 2   |

Notes: lands where predicted — below 4B base (4.06) on every axis but
above SFT-5ep (3.21) except Guidance/Safety. Empathy stays high (4.58);
the gaps concentrate in Informativeness (−0.73 vs 4B), Understanding
(−0.84) and Relevance (−0.55). Headroom for DPO: ~0.3 to reach base,
which is exactly the experiment now running.

## DPO-from-base, ckpt-2000 (n=10000)

MentalBench-Align judge (same prompt, judge, and settings as baseline) over
`data/eval/mbench-qwen3-4b-baseline/output_mbench_dpo.csv`
(col `qwen3_4b_dpo`, half-epoch DPO-from-base LoRA), scored 2026-10-19,
idx 0–9999, no gaps.
Full output (gitignored): `data/eval/mbench-judge-dpo.csv`.

| Dimension       | Mean | SD   | %5  | %≤2 |
|-----------------|------|------|-----|-----|
| Guidance        | 2.99 | 0.53 | 0   | 11   |
| Informativeness | 3.64 | 0.50 | 0   | 0   |
| Relevance       | 4.90 | 0.36 | 91   | 0   |
| Safety          | 4.71 | 0.76 | 84   | 4   |
| Empathy         | 4.99 | 0.10 | 99   | 0   |
| Helpfulness     | 4.12 | 0.45 | 16   | 0   |
| Understanding   | 4.62 | 0.61 | 67   | 1   |
| Overall         | 4.05 | 0.43 | 11   | 0   |

Notes: DPO-from-base recovers base-level scores on every axis
(Overall 4.05 vs 4.06, deltas ≤0.09) and beats SFT-5ep everywhere
(Overall +0.84). The SFT damage (bluntness, confabulation, memorized
sign-offs) is absent — DPO on the clean base policy never learned it.
Caveat: this is ckpt-2000 of 4292 (cap-killed at 47%); loss was ~0 with
margins already huge, so the second half may add little — or over-optimize.
No second-half training until the spot-check reads clean.

## SFT 5-epoch LoRA (n=10000)

MentalBench-Align judge (same prompt, judge, and settings as baseline) over
`data/eval/mbench-qwen3-4b-baseline/output_mbench_sft5.csv`
(col `qwen3_4b_sft5`), scored 2026-10-18, idx 0–9999, no gaps.
Full output (gitignored): `data/eval/mbench-judge-sft5.csv`.

| Dimension       | Mean | SD   | %5  | %≤2 |
|-----------------|------|------|-----|-----|
| Guidance        | 2.73 | 0.79 | 1   | 33   |
| Informativeness | 2.93 | 0.66 | 0   | 24   |
| Relevance       | 3.77 | 0.83 | 16   | 7   |
| Safety          | 4.13 | 1.14 | 53   | 15   |
| Empathy         | 3.52 | 0.70 | 1   | 9   |
| Helpfulness     | 3.15 | 0.68 | 0   | 15   |
| Understanding   | 3.21 | 0.83 | 0   | 21   |
| Overall         | 3.21 | 0.72 | 0   | 15   |

Notes: scores below base on every axis (Overall 3.21 vs 4.06), worst on
Empathy (−1.46) and Understanding (−1.36). Three caveats before reading
this as a straight regression: (1) the judge is blind to prompt-echo
defects — base rows containing prompt-tail text score the same as clean
rows (4.06 vs 4.07), so it grades polish, not correctness; (2) the eval
preamble (psychiatrist framing) is a distribution shift vs the raw-concern
SFT training pairs; (3) SFT-only bluntness is expected pre-DPO (F008) —
the smoke-test row earned Safety 2 for shaming language. The 15% Overall
≤2 rows need the same human spot-check as baseline's Safety tail.

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
