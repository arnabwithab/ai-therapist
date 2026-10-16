.PHONY: setup test style check ci ingest serve model-server frontend judge clean

setup: ## install all deps + fetch the GGUF (idempotent, skips if present)
	uv sync --group dev
	test -f $(GGUF) || uv run --no-sync hf download $(GGUF_REPO) $(GGUF_FILE) --local-dir $(dir $(GGUF))

test: ## run all tests
	uv run pytest

style: ## format + lint
	uv run black src/ backend/ tests/
	uv run ruff check src/ backend/ tests/

check: ## verify formatting + lint without changing files
	uv run black --check src/ backend/ tests/
	uv run ruff check src/ backend/ tests/

ci: check test ## clean-room gate for CI

ingest: ## rebuild Chroma index from data/rag-kb/
	uv run python -m src.rag.ingest --reset

serve: ## run the backend API (needs the model server at VLLM_URL)
	uv run uvicorn backend.app:app --host 0.0.0.0 --port 8001

GGUF_REPO ?= arnabwithab/ai-therapist-qwen3-4b-gguf
GGUF_FILE ?= ai-therapist-qwen3-4b-Q4_K_M.gguf
GGUF ?= models/qwen_gguf/$(GGUF_FILE)

model-server: ## serve the GGUF via vLLM on :8000 (syncs the serving group on demand)
	test -f $(GGUF) || (echo "missing $(GGUF) — run make setup to fetch it" && exit 1)
	uv sync --group serving --frozen
	uv run --no-sync vllm serve $(GGUF) --tokenizer Qwen/Qwen3-4B-Instruct-2507 --port 8000 --max-model-len 4096 --enforce-eager --gpu-memory-utilization 0.7 --served-model-name ai-therapist-qwen3-4b

frontend: ## run the anse chat UI (set its Base URL to http://localhost:8001)
	cd frontend/anse && npm ci --no-audit --no-fund && npm run dev

judge: ## score the baseline with muse-spark-1.3-contributor (needs OPENCODE_GO_KEY)
	uv run python eval/mbench_baseline/judge.py --all

clean: ## remove build artifacts, caches, chroma index
	rm -rf __pycache__ .pytest_cache .ruff_cache chroma/
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
