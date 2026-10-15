"""Export the 2-epoch Qwen3-4B LoRA as a merged GGUF q4_k_m (CPU-only Kaggle kernel).

Unsloth is GPU-only at import, so this uses transformers+peft for the merge and a
minimal llama.cpp build (llama-quantize target only) for the k-quant. Intermediates
are deleted as we go: /kaggle/working holds 20GB and the fp16 stages are ~8GB each.
"""
import glob
import logging
import os
import shutil
import subprocess
import sys

ADAPTER = "arnabwithab/ai-therapist-qwen3-4b-lora"  # 2-epoch LoRA, public
BASE = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
WORK = "/kaggle/working"
FP16_DIR = os.path.join(WORK, "qwen_fp16")
F16_GGUF = os.path.join(WORK, "qwen-f16.gguf")
OUT_DIR = os.path.join(WORK, "qwen_gguf")
OUT_FILE = os.path.join(OUT_DIR, "ai-therapist-qwen3-4b-Q4_K_M.gguf")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("export")


def sh(cmd, **kwargs):
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def main():
    sh([sys.executable, "-m", "pip", "install", "-q",
        "transformers", "peft", "bitsandbytes", "accelerate", "gguf", "torch",
        "torchao>=0.16"])  # PEFT routes bnb-LoRA attach through torchao; image ships 0.10
    sh([sys.executable, "-m", "pip", "install", "-q", "cmake", "ninja"])
    import torch

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    log.info("merging %s into %s ...", ADAPTER, BASE)
    tok = AutoTokenizer.from_pretrained(ADAPTER, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, device_map="cpu",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
    merged.save_pretrained(FP16_DIR)
    tok.save_pretrained(FP16_DIR)
    del merged, base
    log.info("fp16 saved")

    sh(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp",
        os.path.join(WORK, "llama.cpp")])
    sh(["cmake", "-B", "build", "-DCMAKE_BUILD_TYPE=Release"],
       cwd=os.path.join(WORK, "llama.cpp"))
    sh(["cmake", "--build", "build", "--target", "llama-quantize", "-j"],
       cwd=os.path.join(WORK, "llama.cpp"))
    quantize = glob.glob(os.path.join(WORK, "llama.cpp", "build", "bin", "llama-quantize"))[0]
    convert = glob.glob(os.path.join(WORK, "llama.cpp", "**", "convert_hf_to_gguf.py"),
                        recursive=True)[0]
    sh([sys.executable, convert, FP16_DIR, "--outfile", F16_GGUF])
    shutil.rmtree(FP16_DIR)  # ~8GB, no longer needed
    os.makedirs(OUT_DIR, exist_ok=True)
    sh([quantize, F16_GGUF, OUT_FILE, "Q4_K_M"])
    os.remove(F16_GGUF)  # ~8GB, no longer needed
    log.info("DONE %s (%.1f MB)", OUT_FILE, os.path.getsize(OUT_FILE) / 1e6)


if __name__ == "__main__":
    main()
