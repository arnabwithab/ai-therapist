# Evals v2 results (F019)

Serving cell: **Qwen3-1.7B DPO, OARS prompt, no RAG**, 10,000 MentalBench contexts
(`data/eval/mbench-pair-oars.csv`, col `norag`). Pairwise compares the same cell
with and without RAG (`rag` col). Culture/crisis probes: 100 India probes
(75 cultural + 17 crisis + 8 control) in `data/eval/culture-probe-output.csv`,
same serving cell. Primary judge `muse-spark-1.3-contributor`; second judge
`deepseek-v4.1-flash` ran the pilot + 280 rows then stopped by decision.
Machine-readable: `data/eval/v2/report.json` / `.md`, pilot `data/eval/v2/pilot-gate.json` / `.md`.
Method: `docs/evals-v2.md`, `eval/prompts/paper_rubrics.md`.

## Pilot gate (100 main rows, 100 India probes x 2 judges, 100 order-swapped pairs)

| check | result |
|---|---|
| completeness, main cell (miti/safety/gdb) | PASS, 100/100 each |
| completeness, culture probes | PASS, 75/75 cultural probes |
| no flat binary | PASS |
| cross-judge agreement, binary flags with 5+ positives | PASS, min 0.83 |
| pairwise not degenerate | PASS, tie 0.02 |
| crisis probe join | PASS, 17 crisis / 8 control |
| pairwise order-swap flip | WARN, 17% on 100 pairs -> exploratory only |
| GATE | **PASS** (`data/eval/v2/pilot-gate.json`, `.md`) |

## Pilot findings (n=100)

- MITI: 1.11 open questions, 0.15 closed, 1.82 complex reflections, 0.18 simple, 0.55
  affirmations, 0.09 MI-non-adherent; R:Q 1.59, %CR 91.0. The serving voice reflects.
- GDB: mean DS 1.93, 81% of replies displaced (DS >= 2), 82% still hold something
  substantive. Short OARS replies fail the deletion test often.
- Safety: failed crisis 5%, failed abuse 5%, unsupported assumptions 6%, technique
  prescription 1%, symptom speculation 1%, medication opinion 0%, judgmental 0%,
  apathetic 0%.
- Crisis routing: **0/17 crisis probes mention 14416, Tele MANAS, or 112**; no control
  probe mentions them (regex precision ok). One direct-suicide probe that named a planned
  date drew a reply that accepted the date as a ceremony choice and asked about timing.
  This is a model failure, not a judge failure: the regex is deterministic and unit-tested.
- Culture: no all-true flag on the 75 cultural probes; the judges separate coverage and
  specificity from the bias flags.

## Caveats

- Pairwise win-rate is order-unstable (flip 17% with muse, 26% with deepseek at n=100; the
  two replies are the same model with and without RAG, so many pairs are near-ties). Flag
  rates are the headline; pairwise is reported as exploratory with its flip rate.
- Deletion-test severity is ordinal and the judges agree exactly on only 54% of rows; the
  displaced *rate* is therefore reported per judge, never averaged.
- Culture probes are synthetic, written from fetched sources; they test the instrument, not
  Indian clinical ground truth.
- Two of 1,800 pilot calls returned a template-echo answer and were dropped by the
  placeholder guard; both were retried rows that stayed empty.
- Reproduce the gate exactly with `make probe-data` then
  `WORKERS=60 PILOT_ONLY=1 bash eval/run_evals_v2.sh`.

## Full run (n=10000, primary judge)

| instrument | n | result |
|---|---|---|
| MITI | 10000 | open Q 1.09, closed Q 0.17, SR 0.18, CR 1.56, affirm 0.58, MINA 0.025; R:Q 1.38, %CR 89.4 |
| Safety | 10000 | failed crisis 2.42%, failed abuse 2.54%, functional impairment 1.36%, assumptions 4.21%, technique 0.36%, symptom speculation 0.90%, judgmental 0.12%, medication opinion 0.04%, apathetic 0% |
| GDB | 10000 | mean DS 1.71; displaced (DS>=2) 67.3%; substantive content left 87.0% |
| CURE (75 cultural probes) | 75 | coverage 0.0, specificity 0.027, connotation 0.013, coherence 0.293 |
| DECASTE biases (75 cultural probes) | 75 | caste 0.040, religion 0.027, disability 0.027, gender 0.013, SES 0.013 |
| India crisis routing | 17 probes | **0/17 name 14416 / Tele MANAS / 112**; 0/8 controls |
| Pairwise, no-RAG (A) vs RAG (B) | 10000 | A 60.9%, B 38.5%, tie 0.6% -> RAG loses head-to-head; **exploratory**, single-order at 10k, pilot flip was 17% |

Cross-family agreement (`deepseek-v4.1-flash`, n=276 shared rows; the second judge was
stopped after 280 rows by decision): binaries 0.84-1.00 (`adv_assumptions` lowest at 0.84),
`displaced` 0.77, exact `ds_severity` 0.50, crisis-helpline agreement 1.00.

Reading the numbers:

- The serving voice is genuinely reflective (R:Q 1.38, 89% complex reflections) and almost
  never lectures (MINA 2.5%).
- Displacement is the main quality problem: two thirds of replies lose their substance when
  emotional-support sentences are deleted. Short OARS replies trade help for warmth.
- Safety is not catastrophic on average but has a hard tail: ~2.5% crisis and abuse failures
  at 10k means roughly 240 replies that miss a disclosure.
- India crisis routing is the clearest failure: none of 17 crisis probes names a usable Indian
  helpline, and one reply accepted a planned suicide date as a ceremony detail.
- Culture: coverage sits at the floor and coherence is 0.29, so replies rarely engage the
  Indian norm the client raised; bias flags stay low (1-4%).

## Deviations from the frozen protocol

- Second judge: full for the pilot and gate, then stopped at 280 rows by decision. The 10k
  agreement figures come from n=276 shared rows, not all 10k.
- Pairwise at 10k is single-order (10% swapped) because the pilot flip rate was 17%; it is
  reported as exploratory, not as a headline.
- The full pass survived a harness restart and then a machine restart; resume-by-row meant
  only the in-flight rows were rescored.

## Cross-cell comparison (1.7B ladder, 300-row subset, 2026-11-11)

Same 300 idxs (0-299, contexts verified identical across files), muse judge,
miti+safety+gdb. Rerun with `make compare-v2`. Files: `data/eval/v2/compare-*.csv`,
pairs input `data/eval/v2/compare-pairs-300.csv`.

| cell | OQ | CQ | %CR | MINA | mean DS | displaced | failed crisis | failed abuse | assumptions | technique Rx |
|---|---|---|---|---|---|---|---|---|---|---|
| base, regular | 0.02 | 0.04 | 65.4 | 2.66 | 1.77 | 0.707 | 0.013 | 0.033 | 0.110 | 0.070 |
| DPO, regular | 0.23 | 0.16 | 81.4 | 2.74 | 1.32 | 0.523 | 0.007 | 0.017 | 0.137 | 0.207 |
| DPO, OARS | 1.12 | 0.15 | 88.5 | 0.06 | 1.79 | 0.730 | 0.023 | 0.063 | 0.057 | 0.013 |
| DPO, OARS+RAG | 1.11 | 0.24 | 88.2 | 0.05 | 1.81 | 0.700 | 0.017 | 0.063 | 0.067 | 0.017 |

R:Q is not tabulated: the regular cells ask ~0.07 questions/row, so the pooled ratio
explodes (31.1 base, 7.7 DPO) and means nothing; the OQ+CQ counts above are the signal.
OARS cells: R:Q 1.46 both, no-RAG vs RAG.

Pairwise, 300 pairs each, both orders, same judge:

| A vs B | A wins | B wins | tie | flip | read |
|---|---|---|---|---|---|
| base vs DPO (regular) | 0.303 | 0.597 | 0.100 | 0.21 | DPO direction, **exploratory** (flip >0.15) |
| DPO regular vs DPO OARS | 0.197 | 0.803 | 0.000 | 0.12 | OARS wins head-to-head, headline-grade stability |
| DPO OARS vs OARS+RAG | 0.630 | 0.357 | 0.013 | 0.20 | no-RAG direction, **exploratory**; same direction as full-10k single-order (0.609/0.385) |

Reading:

- DPO effect (regular prompt): more complex reflections (65 -> 81%), less displacement
  (0.71 -> 0.52), fewer crisis/abuse misses — but MORE technique-prescription flags
  (0.07 -> 0.21) and assumptions (0.11 -> 0.14). DPO writes longer advice-giving answers
  that trip Adv flags; MINA is unchanged (~2.7) because both voices lecture.
- Prompt effect dominates training effect: OARS moves questions 0.4 -> 1.3/row and MINA
  2.7 -> 0.06, and beats regular-DPO 80/20 head-to-head — even though v1 Overall scored
  OARS lower (3.19 vs 3.84). The v1 Guidance rubric punishes exactly what the v2 judge
  rewards here (questions over advice).
- RAG under OARS: flag deltas all <= 0.013; pairwise leans no-RAG but flip 0.20 keeps it
  exploratory. Consistent with the 10k result, not stronger than it.
- Subset caveat: n=300 cannot resolve ~2% tail rates (crisis/abuse deltas above are noise);
  use the full-run table for tails, this table for voice-level differences.
