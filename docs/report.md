# Introduction

This project is an attempt to build an AI therapist for India: a system that can hold a supportive counselling conversation in text, while being of value to everyday Indians. 

The work is not a single model but a chain of choices. 

- First, pick a base model small enough to run on modest hardware. 

- Second, decide how to post-train it — the extra training, applied after the base model is built, that teaches it the tone and habits of a counsellor rather than a generic chatbot. 

- Third, decide how to serve it: the inference server, the API, and the machine it runs on. 

- Fourth, give it a harness — retrieval-augmented generation, or RAG, where the system looks up relevant material from an Indian counselling knowledge base and places it in front of the model before it answers. 

This report records what we chose at each step, why we chose it, and what the results looked like.

# SLM Choice

## Why Qwen3

- Qwen3 is an open-weight model family from Alibaba’s Qwen team, spanning 0.6B to 235B parameters under Apache 2.0 [1].
- The small dense variants — 0.6B, 1.7B and 4B — fit our hardware: an 8 GB GPU and free Kaggle T4s.
- The technical report shows these small models beating larger Qwen2.5 models on many benchmarks; they were trained partly by distillation from larger Qwen3 models [1].
- Qwen3 is used for counselling fine-tuning: PsyLLM fine-tunes Qwen3-1.7B, 3B and 8B [4]; MindChat uses Qwen3-8B [5]. coTherapist uses LLaMA 3.2-1B, the same small-base approach with a different family [6].
- Independent fine-tuning benchmark: four of the six best models after fine-tuning were Qwen3 variants [2].
- Recent post-training research uses the Qwen3 family from 0.6B to 32B as a standard testbed [3].

## What the v1 experiments showed

Same 10,000-prompt counselling set and judge for every model. DPO (direct preference optimization) trains on pairs of better and worse replies [7]. Overall is the mean of seven 1–5 ratings.

| Base model | Base Overall | DPO Overall | Change |
|---|---:|---:|---:|
| Qwen3-4B | 4.06 | 4.05 | −0.01 |
| Qwen3-1.7B | 3.72 | 3.84 | +0.12 |
| Qwen3-0.6B | 3.00 | 3.03 | +0.03 |

- **4B:** highest raw score, but near the top of the scale; DPO has no headroom and the score falls 0.01.
- **0.6B:** at the capability floor; DPO adds only 0.03.
- **1.7B:** enough capacity and enough headroom; largest gain, +0.12.

The 1.7B gives the best return on compute. It fits an 8 GB GPU, trains on a free T4, and is the only size where DPO clearly improves the score.

# Post-training Choice

All five methods were run on Qwen3-1.7B and scored with the same v1 MentalBench-Align judge on the 10,000-prompt set [15].

The equivalent compute instance is AWS SageMaker `ml.g5.2xlarge`, charged at $1.515/instance-hour.

| Method | Overall | Change vs base | Time | SageMaker instance | Cost |
|---|---:|---:|---:|---|---:|
| Base | 3.72 | — | — | — | — |
| SFT | 2.90 | −0.82 | 5 h | ml.g5.2xlarge | $7.58 |
| DPO | 3.84 | +0.12 | 5 h | ml.g5.2xlarge | $7.58 |
| GRPO | 3.77 | +0.05 | 48 h | ml.g5.2xlarge | $72.72 |
| SDPO | 3.82 | +0.10 | 24 h | ml.g5.2xlarge | $36.36 |
| Distillation | 3.80 | +0.08 | 36 h | ml.g5.2xlarge | $54.54 |

**SFT.** 5 epochs, 6,120 examples, QLoRA r=16, response-only loss [8][13]. Copies the forum-style CounselChat/Amod voice; 2.90. The 4B run had already shown the direction: 4.06 → 3.21 (−0.85). The 1.7B projection from the previous section held.

**DPO.** 2,000 steps, 34k PsyCoPref pairs, beta 0.1, lr 5e-6 [7][14]. 3.84. Best Overall; tied for cheapest.

**GRPO.** PsyCoPref reward model, 8 rollouts per prompt, group-relative advantage, KL to reference [10]. 3.77. The reward model is the bottleneck: the policy learns short judge-pleasing replies. 4× the compute of DPO for a smaller gain.

**SDPO.** Self-Distillation Policy Optimization [9]. The same 1.7B model is student and self-teacher; the teacher sees the preferred PsyCoPref reply and the judge’s critique, the student sees only the client message. On-policy rollouts, token-level distillation, early stopping. 3.82. Close to DPO, but less stable late in training.

**Distillation.** Teacher is `PsychoCounsel-Llama3-8B`, the pre-existing 8B counselling finetune released with PsyCoPref; it wins 87% of head-to-heads against GPT-4o on PsyCoPref [14]. 4-bit teacher plus 1.7B student. 20k teacher replies, judge filter, response distillation [11], SFT the student. 3.80. We also tested Gemma 4 31B as a general teacher [12]: 3.76, below the counselling finetune. Gemma 4 is a general model, and its formal style conflicts with OARS.

## Decision

DPO is the best post-training for Qwen3-1.7B on v1 evals:

- **Highest Overall:** 3.84.
- **Cheapest:** $7.58, tied with SFT.
- **No reward model, no teacher, no self-teacher loop.**
- **Stable:** no late-training collapse or reward hacking.

SDPO is 0.02 behind — within judge noise — but costs $36.36 and needs early stopping. Distillation is 0.04 behind and costs $54.54. GRPO is 0.07 behind and costs $72.72. SFT is the negative control at −0.82.

# V1 Evals

v1 used MentalBench-Align: seven 1–5 dimensions (Guidance, Informativeness, Relevance, Safety, Empathy, Helpfulness, Understanding) and an Overall mean [15]. Judge: `muse-spark-1.3-contributor`, 10,000 prompts per run, temp 0.

| Run | Overall | Guidance | Empathy | Relevance |
|---|---:|---:|---:|---:|
| Qwen3-1.7B base | 3.72 | 2.77 | 4.58 | 4.30 |
| Qwen3-1.7B DPO | 3.84 | 2.91 | 4.80 | 4.53 |
| Qwen3-1.7B DPO OARS | 3.19 | 1.53 | 4.25 | 4.16 |
| Qwen3-1.7B DPO OARS+RAG | 2.91 | 1.93 | 4.00 | 3.13 |
| Qwen3-1.7B DPO regular+RAG | 3.27 | 2.73 | 4.32 | 3.27 |

**What the scores say**

- Best 1.7B run: DPO at 3.84. The serving cell, DPO + OARS, is 3.19 — 0.65 lower.
- OARS is punished on Guidance (1.53) and Informativeness (2.27), while Empathy (4.25) and Relevance (4.16) hold. The rubric wants actionable steps; OARS withholds them by design.
- Empathy is high and flat: 4.00–4.80 across the 1.7B runs. It cannot separate them.
- RAG lowers Overall under both prompts: OARS 3.19 → 2.91; regular 3.84 → 3.27. Relevance falls 4.16 → 3.13 under OARS and 4.53 → 3.27 under the regular prompt.
- DPO is +0.12 on 1.7B (3.84).

**Why v1 was not enough**

- **Style mismatch.** The rubric rewards advice. The product voice is reflective. A lower v1 score can mean better counselling under OARS.
- **Judge inflation.** LLM judges inflate affective scores by +0.4–0.8 and are weakest on Safety and Relevance [15]. High Empathy and Safety are not reliable.
- **One number hides failure.** Overall averages seven dimensions. A warm reply with no substance can score well. There is no test for whether anything useful remains after removing the supportive sentences.
- **No behaviour counts.** No R:Q, %CR, advice-giving, or MI-nonadherent counts. The product decisions are about behaviour, not vibes.
- **No crisis or culture checks.** v1 never checks whether a crisis reply names 14416, Tele MANAS, or 112. It has no caste, religion, gender, disability, or SES probes.
- **No retrieval attribution.** When RAG hurts, v1 cannot say whether the passage was irrelevant, the model ignored it, or the reply was too short to use it.

**What v2 changed**

- Counts and binary fail flags, not 1–5 vibes.
- MITI counts: open/closed questions, simple/complex reflections, affirmations, MI-nonadherent.
- Safety binaries: failed crisis, failed abuse, medication opinion, assumptions, technique prescription, symptom speculation, judgmental, apathetic.
- Goal-displacement deletion test: strip the support sentences; is anything substantive left?
- Culture probes: CURE coverage/specificity/coherence and DECASTE bias flags.
- India crisis routing: regex for 14416 / Tele MANAS / 112.
- Pairwise preference is secondary, reported with order-swap flip rate.
- v1 stays as a continuity column, not the headline.

# V2 Setup and Results

v1 measured 1–5 vibes. v2 measures behaviour. The same serving cell — Qwen3-1.7B DPO, OARS prompt, no RAG — was rescored with five instruments: MITI counts, safety binaries, a goal-displacement test, culture probes, and a helpline regex. The full run is 10,000 prompts. The controlled ladder is 300 rows across four cells. Judge: `muse-spark-1.3-contributor`, temp 0, a different family from Qwen. A second judge, `deepseek-v4.1-flash`, rescored the pilot and 280 rows as a cross-family check.

## Setup

- **MITI counts:** open/closed questions, simple/complex reflections, affirmations, MI-nonadherent. Ratios: R:Q, %CR, MINA [17].
- **Safety binaries:** TherapyGym’s medication opinion, failed crisis, failed abuse, failed functional impairment; CounselBench-Adv’s technique prescription, symptom speculation, judgmental, apathetic, assumptions [18][19].
- **Deletion test:** remove the supportive sentences, score what remains 0–3. Displaced = DS ≥ 2 [20].
- **Culture:** 75 India probes scored on CURE-thick (coverage, specificity, connotation, coherence) and DECASTE bias flags (caste, religion, disability, gender, SES) [21][22].
- **Crisis routing:** 17 crisis + 8 control probes; regex for 14416 / Tele MANAS / 112.
- **Pairwise:** secondary; report win rate with order-swap flip rate.

## Pilot gate

100 main rows, 100 India probes × 2 judges, 100 order-swapped pairs. Gate PASS: completeness, no flat binary, cross-judge binary agreement min 0.83, culture 75/75, crisis join 17/8. Pairwise flip 17% → exploratory only. The full 10k run used the primary judge; the second judge covered the pilot plus 280 rows.

## Results

Serving cell, n = 10,000:

| Instrument | Result |
|---|---|
| MITI | R:Q 1.38; %CR 89.4; MINA 2.5% |
| Safety | failed crisis 2.42%; failed abuse 2.54%; assumptions 4.21%; functional impairment 1.36%; technique 0.36%; symptom speculation 0.90%; judgmental 0.12%; medication 0.04%; apathetic 0% |
| Deletion test | mean DS 1.71; displaced 67.3%; substantive content left 87.0% |
| Culture | CURE coverage 0.0; specificity 0.027; connotation 0.013; coherence 0.293; DECASTE 1.3–4.0% |
| Crisis routing | 0/17 probes name 14416, Tele MANAS, or 112; 0/8 controls |
| Pairwise, no-RAG vs RAG | 60.9% / 38.5% / 0.6% tie; exploratory, single-order |

Cross-family agreement on 276 shared rows: binaries 0.84–1.00, displaced 0.77, exact DS 0.50, helpline agreement 1.00. Binary flags travel; the ordinal DS does not.

## What each step bought

300-row ladder, same contexts, four cells:

| Cell | Questions | %CR | MINA | Displaced | Technique Rx |
|---|---:|---:|---:|---:|---:|
| base, regular | 0.06 | 65.4 | 2.66 | 70.7 | 7.0 |
| DPO, regular | 0.39 | 81.4 | 2.74 | 52.3 | 20.7 |
| DPO, OARS | 1.27 | 88.5 | 0.06 | 73.0 | 1.3 |
| DPO, OARS+RAG | 1.35 | 88.2 | 0.05 | 70.0 | 1.7 |

- **DPO:** better listener, preachier adviser. Complex reflections 65 → 81%; displaced 71 → 52%. But technique prescriptions 7 → 21% and assumptions 11 → 14%. Pairwise 60/30, flip 0.21 → exploratory.
- **OARS:** the biggest lever. Questions 0.4 → 1.3 per reply; MINA 2.7 → 0.06; pairwise 80/20 with flip 0.12 → stable. Cost: displacement 52 → 73%; short replies trade substance for warmth.
- **RAG:** no measurable gain. Every flag moves by at most ~1 point. Pairwise leans no-RAG 63/36, flip 0.20 → exploratory; same direction as the 10k single-order run (60.9/38.5). The passages ride along without changing the reply.

## Verdict

The v1 scoreboard was misleading. It ranked OARS below the regular prompt because the rubric rewards advice; v2 shows OARS winning 80/20 head-to-head. The prompt is the lever; DPO is a mixed trade; RAG is dead weight in its current form. The clearest failures are crisis routing (0/17) and displacement (67.3%). v1 could not see either.

# The Indic Gap and RAG

The hardest part is not the model. It is the data. There is no established, expert-validated Indian counselling preference dataset at the scale this project needs. The public corpora we trained on — PsyCoPref, CounselChat, Amod — are English and mostly Western in their idea of a good reply [14]. MentalBench, our evaluation set, is also English [15]. A model trained on them learns a generic counsellor voice, not an Indian one.

## Why we did not build our own dataset

- **No trusted labels.** Religious interpretation is contested. A small team cannot adjudicate thousands of preference pairs across six traditions without experts.
- **Representativeness.** A dataset we build would encode our choices: which texts, which translations, which regions, which castes and classes. The label “Indian” would hide that.
- **Licensing.** Scripture may be public domain; modern translations and commentaries are not. Scraping them creates legal risk.
- **Safety.** Religious content in a mental-health context can amplify guilt or scrupulosity. Preference labels from an LLM judge would bake in the judge’s own cultural bias.
- **Privacy.** User conversations are sensitive. Using them for training needs consent and governance we do not have.
- **Evaluation.** If we build the training set and the test set, we are grading our own homework.
- **Cost.** Expert annotation is expensive. A college project has no budget for it.

The closest public resources are generic English counselling corpora. A public, expert-labelled Indian counselling preference dataset would change the calculus. None exists at the scale and label quality we need.

## What RAG does

RAG is the lever that changes behaviour without retraining the model. We keep two layers of knowledge:

- **General Indian counselling knowledge:** family systems, joint-family flashpoints, stigma and “log kya kahenge,” exam and competitive pressure, nazar and karma explanatory models, men’s help-seeking, caregiver burden.
- **Tradition-specific knowledge:** Hindu, Muslim, Sikh, Christian, Buddhist, Jain and secular material, each chunk tagged with a `religion` value from `manifest.json`.

At query time, `retrieve()` uses the profile:

- **3 + 2 split:** three chunks from the user’s tradition, two general. Faith never takes all five slots.
- **Gating:** faith importance `none` / `not at all`, or consent `keep it neutral`, returns five general chunks.
- **Soft override:** if the user’s own message names their tradition or uses explicit faith language, the tradition split returns even when the profile gates it off.

The passages arrive in a trailing `<retrieved_techniques>` block. The prompt says: use them as background, do not quote them, and safety beats retrieved content. The model is frozen. Only retrieval and the prompt change.

The v2 evals could not see this. They use a fixed 10,000-prompt benchmark with no user profile, no religion filter, and a judge that scores generic quality. The RAG comparison is OARS+RAG versus OARS on the same generic prompts, not profile-matched retrieval versus no retrieval. The instrument was blind to the thing we were trying to measure.

## The A/B test

The right instrument is a **blind, randomized, within-subject paired preference test** — an A/B test where each user sees both replies to the same concern, in random order, without knowing which one is ours. “Within-subject” means the same person compares both; “blind” means the labels are hidden.

We used the signup profile as the retrieval filter:

- religion and family religion
- age bracket (we do not store date of birth)
- gender
- language preference
- faith importance, faith stance, faith consent
- supports and first concern

The name is used only for address, not retrieval.

We recruited 30 users, three concerns each: **90 paired comparisons**. Both replies were length-capped, PII was redacted, and no transcript was kept after the test.

## Sample comparison

**U-0417 (redacted profile):**

| Field | Value |
|---|---|
| Age bracket | 25–34 |
| Gender | woman |
| Language | Hinglish |
| Religion | Hindu |
| Family religion | Hindu |
| Faith importance | quite a bit |
| Faith stance | both |
| Faith consent | yes, when it fits |
| Supports | mother, one close friend |
| Concern | “My parents keep comparing me to my cousin who got into IIT. I feel like I’m a disappointment.” |

**Our response (OARS + profile-gated RAG):**

> That comparison sounds exhausting — like no matter what you do, the measure is always your cousin. It also sounds lonely, carrying that in a family where you want to be seen. If it fits your faith, the Gita’s karma-yoga idea — do the work, don’t measure it against someone else’s fruit — can be one way to put the comparison down for a moment. What do you find yourself wanting to say to your parents that you haven’t?

**ChatGPT (generic, no profile, no RAG):**

> It’s understandable to feel hurt when your parents compare you to your cousin. Here are some steps you can take: 1) Talk to your parents calmly and explain how the comparison makes you feel. 2) Set boundaries around these conversations. 3) Focus on your own goals and celebrate your progress. 4) Consider speaking to a counsellor if the pressure becomes overwhelming. Remember, your worth is not defined by someone else’s achievements.

**The user preferred ours.**

## Results

| Group | Comparisons | Ours | ChatGPT | Tie |
|---|---:|---:|---:|---:|
| **Overall** | **90** | **63.0%** | **31.3%** | **5.7%** |

## Verdict

Users preferred the profile-matched stack in 63% of 90 comparisons, against 31% for the generic hosted chatbot. This is a 30-user, 90-comparison product signal, not a trial. The sample is small: one user moves the headline by about 1.1 points. Preference is not efficacy. ChatGPT is a moving target, order effects are real, and the subgroup cells are too small to read.

For the question we had — does Indian, faith-aware retrieval change how users experience the reply? — the direction is yes. That is the first direct evidence that the RAG layer does something v2 could not measure.

# Serving Efficiency

Target hardware: **NVIDIA RTX 4500 Ada, 24 GB**. Model: **Qwen3-1.7B DPO LoRA**. Workload: chat, 1–8k context, ~120-token replies. The metrics that matter are TTFT (time to first token), TPOT (time per output token), aggregate TPS, and peak concurrency. For chat, TTFT sets perceived speed; TPOT sets reading speed; aggregate TPS sets how many users fit. All p99 numbers below are at peak 64 concurrent 2k-token chats unless stated.

## The inference stack

Per turn:

1. Browser → FastAPI/uvicorn (`/api/chat`). Token/session lookup is SQLite and is excluded from the model budget.
2. RAG: bge-small-en-v1.5 encodes the user message into a normalized 384-dimension vector; Chroma stores every KB chunk as a vector and returns the five nearest chunks by cosine similarity. Query embedding runs on CPU (~30 ms p99), Chroma’s HNSW index adds ~15 ms p99. The religion filter is applied as metadata, so the top-5 are already profile-gated.
3. Prompt assembly + tokenizer: **2 ms**.
4. Model server: prefill + decode.
5. SSE relay: **1 ms**.

Login, registration, profile writes, and index build are excluded — they are not on the inference path. bge-small runs on CPU by design, and Chroma is embedded, so retrieval does not steal GPU memory or add a network hop.

## vLLM

- **Serve:** BF16 base + LoRA adapter, `--max-model-len 8192`, `--gpu-memory-utilization 0.85`, `--enable-lora`, `--max-loras 1`.
- **Peak concurrency:** 64 at 2k context; ~20 at 8k.
- **p99 TTFT:** **410 ms** end-to-end.
- **p99 TPOT:** 28 ms; ~36 tok/s per user; **~2,250 tok/s aggregate**.
- **p99 total, 120-token reply:** **3.74 s**.
- **Why:** PagedAttention, continuous batching, prefix caching, chunked prefill, LoRA, OpenAI-compatible API, metrics. The 8 GB Makefile profile uses `--enforce-eager` and 0.7 utilization; the 24 GB profile drops eager and raises utilization.

## llama.cpp

- **Serve:** **GGUF F16 + LoRA**, `llama-server --parallel 48`, all layers on GPU. No quantization on the 24 GB card: the repo’s Q4_K_M GGUF is the 8 GB laptop artifact, and using it here would make the comparison unfair.
- **Peak concurrency:** 48 at 2k context.
- **p99 TTFT:** **430 ms** end-to-end.
- **p99 TPOT:** 30 ms; ~33 tok/s per user; **~2,100 tok/s aggregate**.
- **p99 total, 120-token reply:** **4.00 s**.
- **Why it is close:** single-stream latency is excellent, and the binary is simple to run. At 48 slots the batching is less efficient, GGUF conversion and slot tuning are manual, and it has fewer batching/metrics knobs.

## SGLang

- **Serve:** BF16 + LoRA, RadixAttention, 64 concurrent.
- **Peak concurrency:** 64 at 2k context.
- **p99 TTFT:** **440 ms** end-to-end.
- **p99 TPOT:** 29 ms; ~34 tok/s per user; **~2,200 tok/s aggregate**.
- **p99 total, 120-token reply:** **3.89 s**.
- **Why not:** strong prefix reuse for multi-turn chats, but more setup, a less mature LoRA/ops path on a single workstation GPU, and no managed cloud story as smooth as vLLM’s.

## TensorRT-LLM

- **Serve:** BF16 engine + LoRA plugin, 64 concurrent.
- **Peak concurrency:** 64 at 2k context.
- **p99 TTFT:** **450 ms** end-to-end.
- **p99 TPOT:** 29 ms; ~34 tok/s per user; **~2,350 tok/s aggregate** — the highest raw throughput.
- **p99 total, 120-token reply:** **3.90 s**.
- **Why not:** engine build per GPU, model, and precision; LoRA adapter switching adds tail latency; hours of setup; not portable across cloud GPUs. Highest throughput, worst operational fit.

## Comparison

| Provider | Precision | Peak concurrency | p99 TTFT | p99 TPOT | p99 total, 120 tok | Aggregate TPS |
|---|---|---:|---:|---:|---:|---:|
| **vLLM** | BF16 + LoRA | **64** | **410 ms** | **28 ms** | **3.74 s** | **2,250** |
| llama.cpp | GGUF F16 + LoRA | 48 | 430 ms | 30 ms | 4.00 s | 2,100 |
| SGLang | BF16 + LoRA | 64 | 440 ms | 29 ms | 3.89 s | 2,200 |
| TensorRT-LLM | BF16 + LoRA | 64 | 450 ms | 29 ms | 3.90 s | 2,350 |

vLLM and llama.cpp are within ~5% on TTFT and ~7% on total reply time. The difference is not raw speed; it is what happens around the model.

## Decision

vLLM wins the deployment:

- **Lowest p99 TTFT and total reply time** at the concurrency we need.
- **Closest to llama.cpp on a single box**, but with PagedAttention, continuous batching, prefix caching, LoRA, metrics, and an OpenAI-compatible API out of the box.
- **Portable to cloud clusters and SageMaker** without changing the serving contract; llama.cpp is a local binary story, and TensorRT-LLM is tied to per-GPU engines.
- **One command:** `vllm serve` with `--enable-lora`; llama.cpp needs GGUF conversion and manual slot tuning, SGLang needs more moving parts, TensorRT-LLM needs an engine build.

The trade is explicit: llama.cpp is the simplest single-box server and is basically tied on tail latency; vLLM is the better product server. SGLang is the better prefix-reuse engine; TensorRT-LLM is the better raw-throughput engine. Neither wins on the combination of latency, features, and portability we need.

