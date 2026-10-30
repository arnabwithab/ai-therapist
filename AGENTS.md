# AGENTS.md

> Remote access: see `SSH.md` (gitignored, local-only) for cselab11-41 connection details.

## Project Overview
AI counsellor: a finetuned small language model for counselling (Qwen3-1.7B DPO LoRA — Qwen3-1.7B is the preferred base for future experimentation — served via vLLM with --enable-lora and --reasoning-parser qwen3 so thinking stays out of replies), served with RAG over South Asian therapeutic techniques (Chroma + bge-small embeddings) and, eventually, a voice interface layer. No agentic work. College project demo — counselling-style conversational quality, evaluated with MentalBench-Align (F009, done); a better eval harness is still to be designed (F011, deferred — T-BARS RCC is one reference among others).

## Development Philosophy
- TDD first: write the test, then the implementation. Never skip.
- Tests mirror the structure of the module they test
- No function ships without a test
- API routes are thin — logic lives in core/
- Explicit over clever — readable code beats smart code
- If it isn't runnable via `make`, it isn't done

## Tech Stack
- ML training: Unsloth QLoRA notebooks (`notebooks/`), run on Kaggle T4 (Eros438 account)
- RAG: Chroma (embedded persistent, `./chroma/`, gitignored) + `BAAI/bge-small-en-v1.5` via sentence-transformers (CPU-only)
- Backend (planned): FastAPI (Python)
- Frontend: vendored Local-LLM-Chat v3 single HTML (MIT) at `frontend/index.html`, default endpoint `http://localhost:8001`, served via `make frontend`
- Package Manager: uv (Python) — never pip directly
- Build/Task Runner: **Make** — root `Makefile` is the single entry point

## Key Commands
```bash
make setup    # uv sync, installs all deps
make test     # runs all tests
make style    # black + ruff
make ingest   # rebuild Chroma index from data/rag-kb/
make clean    # removes build artifacts, caches, __pycache__ etc.
```

## Directory Structure
```
ai-therapist/
├── src/rag/            # chunk.py, ingest.py, query.py, config.py
├── notebooks/          # Unsloth finetune notebooks (Qwen3_4B-Instruct.ipynb)
├── data/
│   ├── finetune/       # SFT/DPO datasets (gitignored contents)
│   └── rag-kb/         # RAG sources: raw/ + text/
├── docs/
│   ├── research.md
│   └── slm-finetuning.md   # model choice, datasets, VRAM, eval plan
├── chroma/              # Chroma persist dir (gitignored, rebuilt via make ingest)
├── Makefile
├── .env.example
└── pyproject.toml      # uv project
```

## Conventions
### Makefile
- Root `Makefile` is mandatory and the canonical control surface. New workflow steps become targets, not prose.
- Targets are thin wrappers over `uv run` / `uvx`.
- `make setup` is idempotent and safe to re-run.

### Python
- **Package manager: `uv`** — `uv add`, `uv run`, `uv sync`. Never pip.
- Formatter: `black`, Linter: `ruff`
- snake_case for files, variables, functions
- RAG: `query.py:retrieve()` is the only API the serving layer imports; chunking/embedding stay store-agnostic (Chroma→Qdrant migration = re-ingest only)
- Env vars via `.env`; never commit secrets
- No `print` logging in library code — stdlib `logging`

### General
- Conventional commits (feat:, fix:, chore:, docs:, test:, refactor:)
- `data/*/`, `chroma/`, `.env` are never committed
- Notebooks: strip outputs before committing (keeps diffs small)

## Deployment Philosophy
Hosting undecided; persistent disk confirmed. Keep everything runnable with zero infra assumptions: embedded Chroma, local embeddings, no Docker requirement. If serving outgrows embedded mode, migrate to Qdrant (re-ingest, no RAG code rewrite).

## Agent Guidelines
- Always run `make style` before considering code done; `make test` after changes
- Always use `uv`, never pip
- Keep `docs/features.json` current: flip a feature/subtask status, refresh `last_updated` and the `summary` counts whenever work lands
- Never modify `docs/` unless explicitly asked (the exception: keeping `docs/features.json` in sync is mandatory)
- Check `docs/slm-finetuning.md` before touching training/eval — it takes precedence
- New workflow step? Add a Makefile target
- If something feels out of scope, flag it rather than silently doing it

## Project-Specific Notes
- Kaggle account is Eros438; Hugging Face + GitHub is arnabwithab. Kaggle CLI cannot attach secrets — train on Kaggle, push to HF from local.
- Amod dataset is gated (RAIL-D): accept terms on HF before downloading.
- `cotherapistai.com` is an unrelated commercial product — ignore it. coTherapist paper authors (IIT Delhi) have not released code/data; T-BARS RCC stays a reference for the better-eval design (F011), not a rebuild task.
- Previews in rag-kb (`hku-preview`, `pageplace-preview`) are front-matter only — keep, don't treat as full books.
