#!/usr/bin/env bash
# TensorRT-LLM serving artifact for the 24 GB RTX 4500 Ada profile.
#
# The engine is per-GPU. This script only serves an engine built by
# tensorrt_llm_build.sh. LoRA adapter switching adds tail latency; pre-warm
# the adapter before measuring p99.

set -euo pipefail

ENGINE_DIR="${ENGINE_DIR:-engines/qwen3-1.7b-dpo}"
LORA_DIR="${LORA_DIR:-models/adapters/ai-therapist-qwen3-1.7b-dpo}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

if [[ ! -d "$ENGINE_DIR" ]]; then
  echo "missing engine: $ENGINE_DIR" >&2
  echo "run tensorrt_llm_build.sh first." >&2
  exit 1
fi

if [[ ! -d "$LORA_DIR" ]]; then
  echo "missing LoRA adapter: $LORA_DIR" >&2
  exit 1
fi

exec trtllm-serve "$ENGINE_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --lora_dir "$LORA_DIR"
