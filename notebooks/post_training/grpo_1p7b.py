"""GRPO artifact: Qwen3-1.7B, 8 rollouts per prompt, PsyCoPref reward model."""

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from unsloth import FastLanguageModel

BASE = "unsloth/Qwen3-1.7B"
REWARD = "Psychotherapy-LLM/PsychoCounsel-Llama3-8B-Reward"
OUT = "qwen_lora_grpo_1p7b"


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

    reward_tokenizer = AutoTokenizer.from_pretrained(REWARD)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        REWARD, num_labels=1, load_in_4bit=True, device_map="auto"
    )

    def reward_fn(prompts, completions, **kwargs):
        scores = []
        for prompt, completion in zip(prompts, completions):
            inputs = reward_tokenizer(
                prompt + completion, return_tensors="pt", truncation=True, max_length=1024
            ).to(reward_model.device)
            with torch.no_grad():
                scores.append(float(reward_model(**inputs).logits.squeeze()))
        return scores

    data = load_dataset("Psychotherapy-LLM/PsyCoPref", split="train").select(range(5000))
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        train_dataset=data,
        tokenizer=tokenizer,
        args=GRPOConfig(
            output_dir=OUT,
            num_generations=8,
            max_steps=2000,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            learning_rate=1e-6,
            max_completion_length=1024,
            logging_steps=25,
            save_steps=250,
        ),
    )
    trainer.train()
    model.save_pretrained(OUT)


if __name__ == "__main__":
    main()
