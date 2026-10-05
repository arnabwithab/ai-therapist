# SLM finetuning — conversational only (no tools)

## Model choices
1. `Qwen/Qwen3-4B-Instruct` (primary). Best post-LoRA open-QA / multi-turn + human-preference alignment (role-play, creative writing). Train with `enable_thinking=False` (non-thinking, temp 0.7 / top_p 0.8). Unsloth: `unsloth/Qwen3-4B-unsloth-bnb-4bit`, free-T4 notebook.

2. `microsoft/Phi-4-mini-instruct` (3.8B, MIT, 128K ctx). Best reasoning/GB, simplest chat template, no think-mix needed. Good for long sessions.

## Finetuning
SFT on Unsloth QLoRA:
- CounselChat: https://huggingface.co/datasets/nbertagnolli/counsel-chat + https://github.com/nbertagnolli/counsel-chat
- Amod mental_health_counseling_conversations: https://huggingface.co/datasets/Amod/mental_health_counseling_conversations

DPO (after SFT):
- PsyCoPref (listed as PsychoCounsel-Preference): https://huggingface.co/datasets/Psychotherapy-LLM/PsyCoPref

## Evaluation — LLM-as-a-judge
1. MentalBench-Align: https://github.com/abeerbadawi/MentalBench-Align
2. T-BARS RCC pillar (rebuild locally):
   - What it is: Relational & Communication Competence, 1 of 4 T-BARS pillars (BSA / CRF / RCC / TTE, 20 subskills total).
   - The 5 subskills to score: empathy expression / rapport building / emotional validation / gentle challenging / context sensitivity.
   - How to score: 0–4 per subskill (0 = absent/harmful, 4 = excellent therapist-aligned).
   - Judge setup: LLaMA-3.1-8B-Instruct, CoT rationale + strict JSON (per paper §5.2 + App. A.4).
   - Paper: https://arxiv.org/abs/2601.10246 / https://dl.acm.org/doi/10.1145/3774904.3792988
   - Author pages (no code/data yet): https://proadhikary.github.io/ , https://github.com/proadhikary , https://www.lcs2.in/
   - Note: https://www.cotherapistai.com/ is an unrelated product, ignore it.