"""DPO on base Qwen3-4B-Instruct-2507 (Unsloth QLoRA) with PsyCoPref pairs.

Skips SFT deliberately: the 5-epoch SFT adapter memorized forum voice
(sign-offs, confident nonsense). DPO from the clean base policy instead.
Script kernel (not notebook): runs top-to-bottom on a Kaggle T4, ~5.4h for
1 epoch over 34,329 pairs. Saves qwen_dpo/ + tokenizer + log_history.json;
fetch via `kaggle kernels output`. No GGUF export (eval reads the adapter).
"""

import glob
import json as _json
import logging
import os
import subprocess
import sys

OUT_DIR = "/kaggle/working/qwen_dpo"
MODEL_ID = "unsloth/Qwen3-4B-Instruct-2507"
MAX_SEQ = 2048
SEED = 3407
BATCH = 1  # v3: micro-batch 1 (logits on the concatenated pair blew 4GB at batch 2)
GRAD_ACC = 8  # effective batch unchanged at 8

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("dpo")


def _empty(v):
    return str(v).strip() in ("", "nan", "None")


def find_parquet(roots=("/kaggle/input",)):
    """Train parquet: prefer *train*.parquet under the attached datasets."""
    hits = []
    for root in roots:
        hits += glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)
    trains = sorted(p for p in hits if "train" in os.path.basename(p))
    if trains:
        return trains[0]
    return sorted(hits)[0] if hits else None


def format_dpo_row(question, chosen, rejected, tok):
    """Single pair -> TRL string columns. None if any side is blank."""
    if _empty(question) or _empty(chosen) or _empty(rejected):
        return None

    def turn(role, content):
        return tok.apply_chat_template(
            [{"role": role, "content": str(content).strip()}],
            add_generation_prompt=False,
            enable_thinking=False,
            tokenize=False,
        )

    return {
        "prompt": turn("user", question),
        "chosen": turn("assistant", chosen),
        "rejected": turn("assistant", rejected),
    }


def bootstrap():
    """Same install recipe as the proven SFT notebook (Kaggle branch)."""
    import importlib.util

    if importlib.util.find_spec("unsloth") is None:
        log.info("installing unsloth stack...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "unsloth"])
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "transformers==4.57.6"]
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "trl==0.22.2"]
    )
    importlib.import_module("unsloth")  # fail fast before the long downloads
    log.info("unsloth import ok")


def main():
    bootstrap()
    from datasets import load_dataset

    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    parq = find_parquet()
    assert parq, "no *.parquet under /kaggle/input (attach eros483/ai-therapist-dpo-pref)"
    log.info("preference data: %s", parq)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_ID,
        max_seq_length=MAX_SEQ,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )

    raw = load_dataset("parquet", data_files=parq, split="train")
    log.info("raw pairs: %d", len(raw))

    def _map(ex):
        return format_dpo_row(ex["question"], ex["chosen"], ex["rejected"], tokenizer)

    ds = raw.map(_map, remove_columns=raw.column_names)
    ds = ds.filter(lambda ex: ex["prompt"] is not None and ex["chosen"] is not None)
    log.info("kept pairs: %d", len(ds))
    assert len(ds) > 30000, f"suspicious row loss: {len(ds)}"
    assert ds[0]["prompt"].endswith("<|im_end|>\n") or "im_start" in ds[0]["prompt"]
    log.info("example prompt tail: %r", ds[0]["prompt"][-120:])

    from trl import DPOConfig, DPOTrainer

    args = DPOConfig(
        beta=0.1,
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=GRAD_ACC,
        warmup_steps=50,
        num_train_epochs=1,
        learning_rate=5e-6,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="linear",
        seed=SEED,
        max_length=MAX_SEQ,
        max_prompt_length=512,
        fp16=True,  # T4 has no bf16; without this TRL trains in fp32 -> OOM (v1 died at 731s)
        bf16=False,
        report_to="none",
        output_dir="/kaggle/working/dpo_out",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL builds a frozen copy; ~9-11GB total on T4
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    stats = trainer.train()
    log.info("train done: %s", stats)
    _json.dump(trainer.state.log_history, open("/kaggle/working/log_history.json", "w"))

    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    assert os.path.isdir(OUT_DIR), "adapter save failed"
    assert any(f.endswith(".safetensors") for f in os.listdir(OUT_DIR)), "no weights saved"
    log.info("saved %s", OUT_DIR)


if __name__ == "__main__":
    main()
