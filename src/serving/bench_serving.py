"""Benchmark an OpenAI-compatible chat endpoint.

Measures TTFT, TPOT, total latency, and aggregate TPS at a fixed concurrency.
Works against vLLM, SGLang, llama.cpp's OpenAI-compatible server, or
TensorRT-LLM. Prints p50/p95/p99 and can dump the raw rows as JSON.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx

DEFAULT_PROMPTS = [
    "I feel like I am falling behind everyone my age.",
    "My parents keep comparing me to my cousin who got into IIT.",
    "I am scared of telling my family that I want to change careers.",
    "I cannot sleep because I keep replaying a mistake at work.",
    "I feel guilty that I am not doing enough for my ageing parents.",
    "I have stopped enjoying things I used to love.",
    "I feel lonely even when I am around people.",
    "I am anxious about an exam I have already prepared for.",
]


def percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def load_prompts(path):
    if not path:
        return DEFAULT_PROMPTS
    prompts = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            prompts.append(json.loads(line)["prompt"])
    return prompts


async def one_request(client, url, model, prompt, max_tokens):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    start = time.perf_counter()
    first = None
    tokens = 0

    async with client.stream("POST", url, json=payload) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            content = delta.get("content") or ""
            if content and first is None:
                first = time.perf_counter()
            if content:
                tokens += 1  # one content delta is roughly one token on vLLM

    end = time.perf_counter()
    ttft = (first - start) if first is not None else None
    total = end - start
    tpot = ((end - first) / (tokens - 1)) if first is not None and tokens > 1 else None
    return {"ttft": ttft, "tpot": tpot, "total": total, "tokens": tokens}


async def run(args):
    prompts = load_prompts(args.prompts)
    if not prompts:
        raise SystemExit("no prompts")
    tasks = [prompts[i % len(prompts)] for i in range(args.requests)]
    semaphore = asyncio.Semaphore(args.concurrency)
    url = args.base_url.rstrip("/") + "/v1/chat/completions"

    async with httpx.AsyncClient(timeout=None) as client:

        async def bounded(prompt):
            async with semaphore:
                return await one_request(client, url, args.model, prompt, args.max_tokens)

        wall_start = time.perf_counter()
        rows = await asyncio.gather(*(bounded(prompt) for prompt in tasks))
        wall = time.perf_counter() - wall_start

    ttfts = [row["ttft"] for row in rows if row["ttft"] is not None]
    tpots = [row["tpot"] for row in rows if row["tpot"] is not None]
    totals = [row["total"] for row in rows]
    tokens = sum(row["tokens"] for row in rows)
    aggregate_tps = tokens / wall if wall > 0 else 0.0

    print(f"requests:      {args.requests}")
    print(f"concurrency:   {args.concurrency}")
    print(f"model:         {args.model}")
    print(f"max tokens:    {args.max_tokens}")
    print(f"wall:          {wall:.2f} s")
    print(f"tokens:        {tokens}")
    print(f"aggregate TPS: {aggregate_tps:.0f}")
    for name, values in (("TTFT", ttfts), ("TPOT", tpots), ("total", totals)):
        print(
            f"{name}: p50 {percentile(values, 0.50) * 1000:.0f} ms  "
            f"p95 {percentile(values, 0.95) * 1000:.0f} ms  "
            f"p99 {percentile(values, 0.99) * 1000:.0f} ms"
        )

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "requests": args.requests,
                    "concurrency": args.concurrency,
                    "aggregate_tps": aggregate_tps,
                    "rows": rows,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"raw rows: {args.json_out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", default="ai-therapist-qwen3-1.7b-dpo")
    parser.add_argument("--requests", type=int, default=256)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--prompts", help="JSONL with a 'prompt' field")
    parser.add_argument("--json-out")
    run_args = parser.parse_args()
    asyncio.run(run(run_args))


if __name__ == "__main__":
    main()
