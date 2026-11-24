#!/usr/bin/env bash
# SGLang serving artifact for the 24 GB RTX 4500 Ada profile.
#
# BF16 base + LoRA, RadixAttention on, 64 running requests. This is the
# throughput-oriented comparison; it is not the deployed server.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
LORA="${LORA:-arnabwithab/ai-therapist-qwen3-1.7b-dpo-lora}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_RUNNING="${MAX_RUNNING:-64}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
CONTEXT="${CONTEXT:-8192}"

exec python -m sglang.launch_server \
  --model-path "$MODEL" \
  --enable-lora \
  --lora-paths "ai-therapist-qwen3-1.7b-dpo=$LORA" \
  --max-running-requests "$MAX_RUNNING" \
  --mem-fraction-static "$MEM_FRACTION" \
  --context-length "$CONTEXT" \
  --host "$HOST" \
  --port "$PORT" \
  --enable-metrics
# Optional, after the baseline is stable:
#   --enable-torch-compile
