"""Export the 2-epoch Qwen3-4B LoRA as a merged GGUF q4_k_m (CPU-only Kaggle kernel).

Reads the public adapter + 4-bit base from HF, so no secrets are needed; the .gguf
is fetched with `kaggle kernels output` and uploaded from local (F007-2).
"""
import logging
import os
import subprocess
import sys

ADAPTER = "arnabwithab/ai-therapist-qwen3-4b-lora"  # 2-epoch LoRA, public
BASE = "unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit"
OUT = "/kaggle/working/qwen_gguf"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("export")


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)
    from unsloth import FastLanguageModel

    log.info("loading %s + adapter %s ...", BASE, ADAPTER)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ADAPTER,  # Unsloth resolves the base from adapter_config
        max_seq_length=2048,
        load_in_4bit=True,
    )
    log.info("exporting merged GGUF q4_k_m -> %s ...", OUT)
    model.save_pretrained_gguf(OUT, tokenizer, quantization_method="q4_k_m")
    log.info("DONE %s", os.listdir(OUT))


if __name__ == "__main__":
    main()
