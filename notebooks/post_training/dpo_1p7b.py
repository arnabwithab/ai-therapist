"""DPO artifact: Qwen3-1.7B + PsyCoPref, 2,000 steps."""

from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel

BASE = "unsloth/Qwen3-1.7B"
OUT = "qwen_lora_dpo_1p7b"


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        BASE, max_seq_length=1024, load_in_4bit=True
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
    data = load_dataset("Psychotherapy-LLM/PsyCoPref", split="train").map(
        lambda r: {"prompt": r["question"], "chosen": r["chosen"], "rejected": r["rejected"]}
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        train_dataset=data,
        tokenizer=tokenizer,
        args=DPOConfig(
            output_dir=OUT,
            beta=0.1,
            learning_rate=5e-6,
            max_steps=2000,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            max_length=1024,
            logging_steps=25,
            save_steps=250,
        ),
    )
    trainer.train()
    model.save_pretrained(OUT)
    tokenizer.save_pretrained(OUT)


if __name__ == "__main__":
    main()
