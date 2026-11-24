#!/usr/bin/env bash
# llama.cpp serving artifact for the 24 GB RTX 4500 Ada profile.
#
# F16 GGUF, not Q4_K_M: the 24 GB card has room for 16-bit weights, and the
# quantized GGUF is the 8 GB laptop artifact. Serving it here would make the
# comparison against BF16 vLLM unfair.
#
# The PEFT adapter is not a GGUF LoRA. Convert it once:
#   python llama.cpp/convert_lora_to_gguf.py \
#     models/adapters/ai-therapist-qwen3-1.7b-dpo \
#     --outfile models/adapters/ai-therapist-qwen3-1.7b-dpo.gguf
#
# And convert the base model once:
#   python llama.cpp/convert_hf_to_gguf.py Qwen/Qwen3-1.7B \
#     --outfile models/gguf/ai-therapist-qwen3-1.7b-f16.gguf --outtype f16

set -euo pipefail

MODEL="${MODEL:-models/gguf/ai-therapist-qwen3-1.7b-f16.gguf}"
LORA="${LORA:-models/adapters/ai-therapist-qwen3-1.7b-dpo.gguf}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
PARALLEL="${PARALLEL:-48}"
CTX="${CTX:-8192}"

if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  echo "convert the base model to F16 GGUF first (see header)." >&2
  exit 1
fi

if [[ ! -f "$LORA" ]]; then
  echo "missing LoRA: $LORA" >&2
  echo "convert the PEFT adapter to GGUF LoRA first (see header)." >&2
  exit 1
fi

exec llama-server \
  --model "$MODEL" \
  --lora "$LORA" \
  --n-gpu-layers 99 \
  --ctx-size "$CTX" \
  --parallel "$PARALLEL" \
  --flash-attn \
  --host "$HOST" \
  --port "$PORT" \
  --metrics
