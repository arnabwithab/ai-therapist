<h1 align="center">AI counsellor</h1>

<p align="center">Finetuned SLM for counselling, served with RAG over South Asian techniques, with a voice interface layer.</p>

<p align="center">
  <a href="https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507"><img src="https://img.shields.io/badge/base_model-Qwen3--4B-yellow?logo=huggingface&logoColor=white" alt="base model: Qwen3-4B"></a>
  <a href="https://huggingface.co/arnabwithab/ai-therapist-qwen3-4b-lora"><img src="https://img.shields.io/badge/🤗_LoRA-ai--therapist--qwen3--4b--lora-orange" alt="LoRA adapter on Hugging Face"></a>
  <img src="https://img.shields.io/badge/python-3.11+-blue?logo=python&logoColor=white" alt="python 3.11+">
  <img src="https://img.shields.io/badge/uv-0.12.19-black?logo=astral&logoColor=white" alt="managed with uv 0.12.19">
  <img src="https://img.shields.io/badge/black-26.10.1-black?logo=python&logoColor=white" alt="formatted with black 26.10.1">
  <img src="https://img.shields.io/badge/FastAPI-0.143.1-009688?logo=fastapi&logoColor=white" alt="FastAPI backend 0.143.1">
</p>

> **Disclaimer:** This is an AI powered tool, not a therapist. It can get things wrong and is not for crisis use. If you or someone you know is struggling: call **112**, **Tele MANAS 14416**, or **Vandrevala Foundation +91-9999-666-555**.



## Get started

**Requirements:** Python 3.11+, `uv`, Node 18+, and a GPU with **min 8 GB VRAM** for the model server.

```bash
# 1. clone & enter
git clone https://github.com/Eros483/ai-therapist.git
cd ai-therapist

# 2. update submodules
git submodule update --init --recursive

# 3. install & fetch model
make setup

# 4. build the RAG index
make ingest
```

Then run three processes (three terminals):

```bash
# terminal 1 — model server on :8000 (needs 8GB VRAM)
make model-server

# terminal 2 — backend API on :8001
make serve

# terminal 3 — chat UI (set its OpenAI provider Base URL to http://localhost:8001)
make frontend
```


## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
