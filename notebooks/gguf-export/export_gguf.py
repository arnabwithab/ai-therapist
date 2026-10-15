"""Export the 2-epoch Qwen3-4B LoRA as a merged GGUF q4_k_m (GPU kernel, Unsloth).

Unsloth's save_pretrained_gguf does merge + quant in one call. Needs ~20 min on T4.
"""
import logging
import os
import subprocess
import sys

ADAPTER = "arnabwithab/ai-therapist-qwen3-4b-lora"  # 2-epoch LoRA, public
OUT = "/kaggle/working/qwen_gguf"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("export")


def main():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)
    from unsloth import FastLanguageModel

    log.info("loading %s ...", ADAPTER)
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
