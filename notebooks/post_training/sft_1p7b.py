"""SFT artifact: Qwen3-1.7B, 5 epochs, QLoRA r=16, response-only loss."""

import json
from pathlib import Path

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

BASE = "unsloth/Qwen3-1.7B"
OUT = "qwen_lora_sft_1p7b"
MAX_SEQ = 2048


def load_mixture():
    counsel = load_dataset("csv", data_files="data/finetune/counsel-chat/*.csv", split="train")
    amod = load_dataset("json", data_files="data/finetune/amod-counseling/*.json", split="train")
    rows = []
    for r in counsel:
        q, a = (r.get("question") or "").strip(), (r.get("answer") or "").strip()
        if q and a:
            rows.append(
                {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
            )
    for r in amod:
        q, a = (r.get("Context") or "").strip(), (r.get("Response") or "").strip()
        if q and a:
            rows.append(
                {"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}
            )
    return rows


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        BASE, max_seq_length=MAX_SEQ, load_in_4bit=True
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=load_mixture(),
        args=SFTConfig(
            output_dir=OUT,
            num_train_epochs=5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_length=MAX_SEQ,
            learning_rate=2e-4,
            logging_steps=25,
            save_strategy="epoch",
            completion_only_loss=True,
        ),
    )
    trainer.train()
    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)
    Path(OUT, "run.json").write_text(
        json.dumps({"examples": len(trainer.train_dataset), "epochs": 5})
    )


if __name__ == "__main__":
    main()
