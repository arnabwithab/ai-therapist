.PHONY: init setup test style check ci ingest serve model-server model-server-24g eval-rag frontend judge judge-eval fetch-rag-contexts clean

init: ## check system prereqs (Ubuntu only) and install what's missing: python3.11+, uv
	grep -qi ubuntu /etc/os-release || (echo "make init supports Ubuntu only" && exit 1)
	if ! command -v python3 >/dev/null || ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then \
		sudo apt-get update && sudo apt-get install -y python3; \
	fi
	if ! command -v uv >/dev/null; then \
		if ! command -v curl >/dev/null; then sudo apt-get update && sudo apt-get install -y curl; fi; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
		echo "uv installed — ensure ~/.local/bin is on PATH, then run make setup"; \
	fi
	python3 --version && (uv --version || echo 'uv not on PATH yet: export PATH="$$HOME/.local/bin:$$PATH"')

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

MODEL_ID ?= Qwen/Qwen3-4B-Instruct-2507
MAX_MODEL_LEN ?= 4096
GPU_MEM_UTIL ?= 0.9
LORA_MODULES ?=
EVAL_MODEL ?= $(MODEL_ID)
EVAL_COL ?= response
RAG_TOP_K ?= 5
RAG_CSV ?=
RAG_JSONL ?=
MAX_TOKENS ?= 512
PROMPT ?= oars
N_SHARDS ?= 1
SHARD ?= 0
OUT ?= data/eval/mbench-rag/output_rag.csv
HF_EVAL_REPO ?= arnabwithab/ai-therapist-mbench-rag

model-server-24g: ## serve full-precision MODEL_ID on :8000 (24GB box; LORA_MODULES="name=path" optional)
	uv sync --group serving --frozen
	uv run --no-sync vllm serve $(MODEL_ID) --port 8000 --max-model-len $(MAX_MODEL_LEN) --gpu-memory-utilization $(GPU_MEM_UTIL) --served-model-name $(MODEL_ID) $(if $(LORA_MODULES),--enable-lora --lora-modules $(LORA_MODULES))

eval-rag: ## one OARS+RAG eval shard (EVAL_MODEL/SHARD/N_SHARDS/OUT/EVAL_COL/RAG_TOP_K/MAX_TOKENS/PROMPT/HF_EVAL_REPO; NO_RAG=1 = ablation, RAG_CSV/RAG_JSONL = precomputed, HF_EVAL_REPO= to disable push)
	uv run python -m eval.mbench_rag.generate_rag --model $(EVAL_MODEL) --out $(OUT) --col $(EVAL_COL) --shard $(SHARD) --num-shards $(N_SHARDS) --rag-top-k $(RAG_TOP_K) --max-tokens $(MAX_TOKENS) --prompt $(PROMPT) --hf-repo $(HF_EVAL_REPO) $(if $(NO_RAG),--no-rag) $(if $(RAG_CSV),--rag-csv $(RAG_CSV)) $(if $(RAG_JSONL),--rag-jsonl $(RAG_JSONL))

judge: ## score the baseline with muse-spark-1.3-contributor (needs OPENCODE_GO_KEY)
	uv run python -m eval.mbench_baseline.judge --all

judge-eval: ## judge any eval CSV with the same rubric (BASELINE=.. RESP_COL=.. OUT=..)
	test -n "$(BASELINE)$(RESP_COL)$(OUT)" || (echo "set BASELINE=.. RESP_COL=.. OUT=.." && exit 1)
	uv run python -m eval.mbench_baseline.judge --baseline $(BASELINE) --resp-col $(RESP_COL) --out $(OUT) --all --hf-repo $(HF_EVAL_REPO)

strip-think: ## strip <think> blocks for judging (IN=.. COL=.. OUT=..)
	test -n "$(IN)$(COL)$(OUT)" || (echo "set IN=.. COL=.. OUT=.." && exit 1)
	uv run python -m eval.mbench_rag.strip_think --in $(IN) --col $(COL) --out $(OUT)

fetch-rag-contexts: ## precompute serving-identical RAG blocks for all 10k mbench contexts (RAG_TOP_K=.., default 5)
	uv run python -m eval.mbench_rag.fetch_contexts $(if $(RAG_TOP_K),--top-k $(RAG_TOP_K))

clean: ## remove build artifacts, caches, chroma index
	rm -rf __pycache__ .pytest_cache .ruff_cache chroma/
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
