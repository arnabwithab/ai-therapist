#!/usr/bin/env bash
# TensorRT-LLM engine build artifact for the 24 GB RTX 4500 Ada profile.
#
# The engine is tied to the GPU, model, precision, and max batch/sequence
# shape. Rebuild after moving to a different card or cloud instance.

set -euo pipefail

CKPT_DIR="${CKPT_DIR:-models/trtllm/qwen3-1.7b-dpo}"
ENGINE_DIR="${ENGINE_DIR:-engines/qwen3-1.7b-dpo}"
MAX_BATCH="${MAX_BATCH:-64}"
MAX_INPUT="${MAX_INPUT:-6144}"
MAX_SEQ="${MAX_SEQ:-8192}"

mkdir -p "$ENGINE_DIR"

trtllm-build \
  --checkpoint_dir "$CKPT_DIR" \
  --output_dir "$ENGINE_DIR" \
  --gemm_plugin auto \
  --lora_plugin auto \
  --max_batch_size "$MAX_BATCH" \
  --max_input_len "$MAX_INPUT" \
  --max_seq_len "$MAX_SEQ" \
  --max_num_tokens "$MAX_SEQ"

echo "engine written to $ENGINE_DIR"
