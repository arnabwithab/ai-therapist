"""SDPO artifact: Qwen3-1.7B self-teacher with privileged counselling context.

The student sees only the client message. The self-teacher sees the preferred
PsyCoPref reply plus a short judge critique, and its token distribution is
distilled back into the student on the student's own rollouts.
"""

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3-1.7B"
OUT = "qwen_lora_sdpo_1p7b"
MAX_LEN = 1024


def teacher_prompt(question, preferred, critique):
    return (
        f"Client: {question}\n\n"
        f"Preferred reply: {preferred}\n"
        f"Judge critique: {critique}\n\n"
        "Reply to the client in the counsellor voice."
    )


def logits(model, tokenizer, prompt, response):
    inputs = tokenizer(
        prompt + response, return_tensors="pt", truncation=True, max_length=MAX_LEN
    ).to(model.device)
    out = model(**inputs)
    return out.logits[:, :-1]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    student = AutoModelForCausalLM.from_pretrained(
        MODEL, load_in_4bit=True, device_map="auto"
    )
    teacher = AutoModelForCausalLM.from_pretrained(
        MODEL, load_in_4bit=True, device_map="auto"
    )
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-6)

    data = load_dataset("Psychotherapy-LLM/PsyCoPref", split="train").select(range(5000))
    student.train()

    for step, row in enumerate(data):
        question = row["question"]
        preferred = row["chosen"]
        critique = "Prefer the calmer, more reflective reply; avoid advice."

        inputs = tokenizer(question, return_tensors="pt").to(student.device)
        with torch.no_grad():
            generated = student.generate(
                **inputs, max_new_tokens=256, do_sample=True, temperature=0.8
            )
        response = tokenizer.decode(generated[0], skip_special_tokens=True)

        student_logits = logits(student, tokenizer, question, response)
        with torch.no_grad():
            teacher_logits = logits(
                teacher, tokenizer, teacher_prompt(question, preferred, critique), response
            )

        loss = F.kl_div(
            F.log_softmax(student_logits, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction="batchmean",
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 25 == 0:
            print(step, float(loss))
        if step >= 2000:
            break

    student.save_pretrained(OUT)


if __name__ == "__main__":
    main()
