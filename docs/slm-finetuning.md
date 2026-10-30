# SLM finetuning — conversational only (no tools)

## Model choices
Preferred base for future experimentation: `Qwen/Qwen3-1.7B` (the serving model is its DPO LoRA; DPO-from-base gained +0.12 Overall over base at 1024 tokens, F016).

1. `Qwen/Qwen3-1.7B` (preferred). Train with `enable_thinking=False` (non-thinking, temp 0.7 / top_p 0.8). Unsloth: `unsloth/Qwen3-1.7B`, free-T4 notebook.

2. `Qwen/Qwen3-4B-Instruct` (original primary, superseded for future work). Best post-LoRA open-QA / multi-turn + human-preference alignment (role-play, creative writing). Unsloth: `unsloth/Qwen3-4B-unsloth-bnb-4bit`. History kept for the 4B eval ladder (F010/F015/F008).

3. `microsoft/Phi-4-mini-instruct` (3.8B, MIT, 128K ctx). Detour abandoned after 4 failures (F016-1).

## Finetuning
SFT on Unsloth QLoRA:
- CounselChat: https://huggingface.co/datasets/nbertagnolli/counsel-chat + https://github.com/nbertagnolli/counsel-chat
- Amod mental_health_counseling_conversations: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations

DPO (after SFT):
- PsyCoPref (listed as PsychoCounsel-Preference): https://huggingface.co/datasets/Psychotherapy-LLM/PsyCoPref

## Evaluation — LLM-as-a-judge
1. MentalBench-Align (F009, done): https://github.com/abeerbadawi/MentalBench-Align
2. Better eval harness (F011, deferred — design pending, needs more thinking):
   - Goal: an eval better than the current LLM-as-judge. T-BARS RCC below is one reference among others, alongside the F009 harness.
   - Reference — what T-BARS RCC is: Relational & Communication Competence, 1 of 4 T-BARS pillars (BSA / CRF / RCC / TTE, 20 subskills total).
   - The 5 subskills it scores: empathy expression / rapport building / emotional validation / gentle challenging / context sensitivity.
   - How it scores: 0–4 per subskill (0 = absent/harmful, 4 = excellent therapist-aligned).
   - Its judge setup: LLaMA-3.1-8B-Instruct, CoT rationale + strict JSON (per paper §5.2 + App. A.4).
   - Paper: https://arxiv.org/abs/2601.10246 / https://dl.acm.org/doi/10.1145/3774904.3792988
   - Author pages (no code/data yet): https://proadhikary.github.io/ , https://github.com/proadhikary , https://www.lcs2.in/
   - Note: https://www.cotherapistai.com/ is an unrelated product, ignore it.