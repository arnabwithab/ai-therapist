.PHONY: init setup test style check ci ingest serve model-server model-server-24g eval-rag frontend judge judge-v2 report-v2 evals-v2 compare-v2 judge-eval fetch-rag-contexts probe-data clean

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

setup: ## install all deps (model weights stream from HF at serve time)
	uv sync --group dev

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

model-server: ## serve our DPO model (base + LoRA) via vLLM on :8000
	uv sync --group serving --frozen
	# native sampler: flashinfer 0.6.11 sampling JIT doesn't compile on CUDA 13
	VLLM_USE_FLASHINFER_SAMPLER=0 uv run --no-sync vllm serve Qwen/Qwen3-1.7B --port 8000 --max-model-len 8192 --enforce-eager --gpu-memory-utilization 0.7 --enable-lora --lora-modules ai-therapist-qwen3-1.7b-dpo=arnabwithab/ai-therapist-qwen3-1.7b-dpo-lora --served-model-name ai-therapist-qwen3-1.7b-dpo --reasoning-parser qwen3

frontend: ## serve the chat UI at http://localhost:3000 (talks to the backend on :8001)
	cd frontend && python3 -m http.server 3000

EVAL_MODEL ?= ai-therapist-qwen3-1.7b-dpo
EVAL_COL ?= response
RAG_TOP_K ?= 5
RAG_CSV ?=
RAG_JSONL ?=
CONTEXTS ?=
MAX_TOKENS ?= 512
PROMPT ?= oars
WORKERS ?= 16
N_SHARDS ?= 1
SHARD ?= 0
OUT ?= data/eval/mbench-rag/output_rag.csv
HF_EVAL_REPO ?= arnabwithab/ai-therapist-mbench-rag

model-server-24g: ## serve our DPO model on :8000 (24GB box)
	uv sync --group serving --frozen
	VLLM_USE_FLASHINFER_SAMPLER=0 uv run --no-sync vllm serve Qwen/Qwen3-1.7B --port 8000 --max-model-len 8192 --gpu-memory-utilization 0.9 --enable-lora --lora-modules ai-therapist-qwen3-1.7b-dpo=arnabwithab/ai-therapist-qwen3-1.7b-dpo-lora --served-model-name ai-therapist-qwen3-1.7b-dpo --reasoning-parser qwen3

eval-rag: ## one eval shard (EVAL_MODEL/SHARD/N_SHARDS/OUT/EVAL_COL/RAG_TOP_K/MAX_TOKENS/PROMPT/WORKERS/HF_EVAL_REPO; NO_RAG=1 = ablation, RAG_CSV/RAG_JSONL = precomputed, HF_EVAL_REPO= to disable push)
	uv run python -m eval.mbench_rag.generate_rag --model $(EVAL_MODEL) --out $(OUT) --col $(EVAL_COL) --shard $(SHARD) --num-shards $(N_SHARDS) --rag-top-k $(RAG_TOP_K) --max-tokens $(MAX_TOKENS) --prompt $(PROMPT) --workers $(WORKERS) $(if $(HF_EVAL_REPO),--hf-repo $(HF_EVAL_REPO)) $(if $(NO_RAG),--no-rag) $(if $(RAG_CSV),--rag-csv $(RAG_CSV)) $(if $(RAG_JSONL),--rag-jsonl $(RAG_JSONL)) $(if $(CONTEXTS),--contexts $(CONTEXTS))

judge: ## score the baseline with muse-spark-1.3-contributor (needs OPENCODE_GO_KEY)
	uv run python -m eval.mbench_baseline.judge --all

V2_MODEL ?= muse-spark-1.3-contributor
V2_INSTR ?= miti,safety,gdb
V2_BASELINE ?= data/eval/mbench-rag/output_norag.csv
V2_COL ?= norag
V2_OUT ?= data/eval/mbench-judge-v2.csv
V2_WORKERS ?= 50

judge-v2: ## evals-v2 judge (V2_MODEL/V2_INSTR/V2_BASELINE/V2_COL/V2_OUT; PAIR_COL=.. PAIR=1 BOTH_ORDERS=1)
	uv run python -m eval.judge_v2 --baseline $(V2_BASELINE) --resp-col $(V2_COL) --instruments $(V2_INSTR) --model $(V2_MODEL) --out $(V2_OUT) --workers $(V2_WORKERS) --n $(if $(N),$(N),500) $(if $(ALL),--all) $(if $(PAIR_COL),--pair-col $(PAIR_COL)) $(if $(BOTH_ORDERS),--both-orders) $(if $(HF_EVAL_REPO),--hf-repo $(HF_EVAL_REPO))

report-v2: ## evals-v2 report/gate (A=.. B=.. PAIRS=.. PROBES=.. AXES=.. EXPECTED=.. GATE=1 JSON=..)
	test -n "$(A)" || (echo "set A=judge csv" && exit 1)
	uv run python -m eval.evals_v2_report --judge-a $(A) $(if $(B),--judge-b $(B)) $(if $(PAIRS),--pairs $(PAIRS)) $(if $(PROBES),--probes $(PROBES)) $(if $(AXES),--probe-axes $(AXES)) $(if $(EXPECTED),--expected-idx $(EXPECTED)) $(if $(GATE),--gate) $(if $(JSON),--json $(JSON))

evals-v2: ## F019 pilot gate then full 10k (JUDGES=.. PILOT_N=.. PAIRS=.. PILOT_ONLY=1)
	bash eval/run_evals_v2.sh

compare-v2: ## v2 1.7B cross-cell comparison: 4 cells x 300 rows + 3 both-order pairs (muse only, resumable)
	uv run python -c "import csv;N=300;ld=lambda f:{r['idx']:r for r in csv.DictReader(open(f))};a=ld('data/eval/mbench-qwen3-4b-baseline/output_mbench_1p7b.csv');b=ld('data/eval/mbench-qwen3-4b-baseline/output_mbench_1p7b_dpo_1024_final.csv');c=ld('data/eval/mbench-pair-oars.csv');f=open('data/eval/v2/compare-pairs-300.csv','w',newline='');w=csv.DictWriter(f,fieldnames=['idx','context','base','dpo_reg','oars_norag','oars_rag']);w.writeheader();[w.writerow({'idx':str(i),'context':a[str(i)]['context'],'base':a[str(i)]['qwen3_1p7b_base'],'dpo_reg':b[str(i)]['qwen3_1p7b_dpo'],'oars_norag':c[str(i)]['norag'],'oars_rag':c[str(i)]['rag']}) for i in range(N)];f.close()"
	$(MAKE) judge-v2 V2_BASELINE=data/eval/mbench-qwen3-4b-baseline/output_mbench_1p7b.csv V2_COL=qwen3_1p7b_base V2_OUT=data/eval/v2/compare-1p7b-base.csv N=300 HF_EVAL_REPO=
	$(MAKE) judge-v2 V2_BASELINE=data/eval/mbench-qwen3-4b-baseline/output_mbench_1p7b_dpo_1024_final.csv V2_COL=qwen3_1p7b_dpo V2_OUT=data/eval/v2/compare-1p7b-dpo.csv N=300 HF_EVAL_REPO=
	$(MAKE) judge-v2 V2_BASELINE=data/eval/mbench-pair-oars.csv V2_COL=norag V2_OUT=data/eval/v2/compare-1p7b-dpo-oars.csv N=300 HF_EVAL_REPO=
	$(MAKE) judge-v2 V2_BASELINE=data/eval/mbench-pair-oars.csv V2_COL=rag V2_OUT=data/eval/v2/compare-1p7b-dpo-oars-rag.csv N=300 HF_EVAL_REPO=
	$(MAKE) judge-v2 V2_BASELINE=data/eval/v2/compare-pairs-300.csv V2_COL=base PAIR_COL=dpo_reg BOTH_ORDERS=1 V2_OUT=data/eval/v2/compare-pairs-base-dpo.csv N=600 HF_EVAL_REPO=
	$(MAKE) judge-v2 V2_BASELINE=data/eval/v2/compare-pairs-300.csv V2_COL=dpo_reg PAIR_COL=oars_norag BOTH_ORDERS=1 V2_OUT=data/eval/v2/compare-pairs-dpo-oars.csv N=600 HF_EVAL_REPO=
	$(MAKE) judge-v2 V2_BASELINE=data/eval/v2/compare-pairs-300.csv V2_COL=oars_norag PAIR_COL=oars_rag BOTH_ORDERS=1 V2_OUT=data/eval/v2/compare-pairs-oars-rag.csv N=600 HF_EVAL_REPO=

judge-eval: ## judge any eval CSV with the same rubric (BASELINE=.. RESP_COL=.. OUT=..)
	test -n "$(BASELINE)$(RESP_COL)$(OUT)" || (echo "set BASELINE=.. RESP_COL=.. OUT=.." && exit 1)
	uv run python -m eval.mbench_baseline.judge --baseline $(BASELINE) --resp-col $(RESP_COL) --out $(OUT) --all --hf-repo $(HF_EVAL_REPO)

plots: ## regenerate the two live eval charts (means + 1.7B); 4B/0.6B PNGs frozen 2026-10-27
	uv run --with matplotlib python -m eval.plot_evals --outdir docs

strip-think: ## strip <think> blocks for judging (IN=.. COL=.. OUT=..)
	test -n "$(IN)$(COL)$(OUT)" || (echo "set IN=.. COL=.. OUT=.." && exit 1)
	uv run python -m eval.mbench_rag.strip_think --in $(IN) --col $(COL) --out $(OUT)

probe-data: ## derive data/eval/culture_probes.csv (idx,axis,context,expected_standard) from the tracked probes
	uv run python -c "import csv;rows=list(csv.DictReader(open('eval/prompts/culture_probes.csv')));f=open('data/eval/culture_probes.csv','w',newline='');w=csv.DictWriter(f,fieldnames=['idx','axis','context','expected_standard']);w.writeheader();[w.writerow({'idx':i,'axis':r['axis'],'context':r['scenario'],'expected_standard':r['expected_standard']}) for i,r in enumerate(rows)];f.close()"

fetch-rag-contexts: ## precompute serving-identical RAG blocks for all 10k mbench contexts (RAG_TOP_K=.., default 5)
	uv run python -m eval.mbench_rag.fetch_contexts $(if $(RAG_TOP_K),--top-k $(RAG_TOP_K))

clean: ## remove build artifacts, caches, chroma index
	rm -rf __pycache__ .pytest_cache .ruff_cache chroma/
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
