"""Distillation artifact: PsychoCounsel-Llama3-8B teacher -> Qwen3-1.7B student."""

import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from unsloth import FastLanguageModel

TEACHER = "Psychotherapy-LLM/PsychoCounsel-Llama3-8B"
STUDENT = "unsloth/Qwen3-1.7B"
OUT = Path("distill_1p7b")


def generate_teacher_data(n=20000):
    tokenizer = AutoTokenizer.from_pretrained(TEACHER)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER, load_in_4bit=True, device_map="auto"
    )
    prompts = load_dataset("Psychotherapy-LLM/PsyCoPref", split="train").select(range(n))
    rows = []
    for i, row in enumerate(prompts):
        inputs = tokenizer(row["question"], return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=300, do_sample=True, temperature=0.7
        )
        reply = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        if 80 <= len(reply) <= 1200:  # cheap quality filter; the full run used the judge
            rows.append(
                {
                    "messages": [
                        {"role": "user", "content": row["question"]},
                        {"role": "assistant", "content": reply},
                    ]
                }
            )
        if i % 100 == 0:
            print(i, len(rows))
    return rows


def train_student(rows):
    model, tokenizer = FastLanguageModel.from_pretrained(
        STUDENT, max_seq_length=2048, load_in_4bit=True
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
        train_dataset=rows,
        args=SFTConfig(
            output_dir=str(OUT),
            num_train_epochs=5,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            max_length=2048,
            learning_rate=2e-4,
            logging_steps=25,
        ),
    )
    trainer.train()
    model.save_pretrained(OUT)


def main():
    rows = generate_teacher_data()
    OUT.mkdir(exist_ok=True)
    with open(OUT / "teacher_data.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    train_student(rows)


if __name__ == "__main__":
    main()
