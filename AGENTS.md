# AGENTS.md

> Remote access: see `SSH.md` (gitignored, local-only) for cselab11-41 connection details.

## Project Overview
AI counsellor: a finetuned small language model for counselling (Qwen3-1.7B DPO LoRA — the model is frozen for phase 2, no retraining — served via vLLM with --enable-lora and --reasoning-parser qwen3 so thinking stays out of replies), served with RAG over South Asian therapeutic techniques (Chroma + bge-small embeddings) and, eventually, a voice interface layer. No agentic work.

Phase 2 (from 2026-11-20) is product mode: a local, login-backed chat tailored to Indian users around religion and culture — first-login profile form (FICA-informed, 11 fields), religion-tagged RAG, a person/faith-aware OARS prompt, server-side memory. Plan of record: `docs/phase2-plan.md`. v1 research is closed and its eval harnesses (F009/F017/F019) are frozen — do not extend them or design new evals unless explicitly asked.

## Development Philosophy
- TDD first: write the test, then the implementation. Never skip.
- Tests mirror the structure of the module they test
- No function ships without a test
- API routes are thin — logic lives in core/
- Explicit over clever — readable code beats smart code
- If it isn't runnable via `make`, it isn't done

## Tech Stack
- ML training: Unsloth QLoRA notebooks (`notebooks/`), run on Kaggle T4 (Eros438 account)
- RAG: Chroma (embedded persistent, `./chroma/`, gitignored) + `BAAI/bge-small-en-v1.5` via sentence-transformers (CPU-only); phase 2: sources carry religion tags from `data/rag-kb/manifest.json`, retrieval is profile-gated
- Backend: FastAPI (Python) — thin routes over `backend/core/`
- App DB (phase 2, planned): SQLite via stdlib `sqlite3` at `data/therapist.db` (gitignored) — users/tokens/profiles/messages; scrypt password hashing; no ORM, no JWT
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
├── src/
│   ├── rag/            # chunk.py, ingest.py, query.py, config.py
│   └── serving/        # vLLM/llama.cpp/SGLang/TensorRT-LLM artifacts
├── backend/            # FastAPI routes (app.py) + core/ (chat.py; db.py in phase 2)
├── frontend/           # vendored single-file chat UI (index.html)
├── public/             # charts and static images
├── notebooks/          # Unsloth finetune notebooks (Qwen3_4B-Instruct.ipynb)
├── data/
│   ├── finetune/       # SFT/DPO datasets (gitignored contents)
│   └── rag-kb/         # RAG sources: raw/ + text/ (+ manifest.json, phase 2)
├── docs/
│   ├── phase1-plan.md       # v1 research + finetuning record (frozen)
│   ├── phase2-plan.md       # phase 2 design source
│   ├── evals.md / evals-v2.md / evals-v2-results.md
│   ├── prompt.md
│   ├── citations.md
│   └── report.md
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
- Phase 2 personalisation: tradition-specific guidance comes only from the user profile + religion-tagged retrieved passages — never hardcode it in code; the prompt is the single place faith rules live (`backend/core/chat.py`)
- Profile data (religion, age, concerns) is sensitive: local-only gitignored DB, no telemetry, never logged beyond debug level
- Env vars via `.env`; never commit secrets
- No `print` logging in library code — stdlib `logging`

### General
- Conventional commits (feat:, fix:, chore:, docs:, test:, refactor:)
- `data/*/`, `chroma/`, `.env` are never committed
- Notebooks: strip outputs before committing (keeps diffs small)

## Deployment Philosophy
Hosting undecided; persistent disk confirmed. Keep everything runnable with zero infra assumptions: embedded Chroma, local embeddings, embedded SQLite, no Docker requirement. If serving outgrows embedded mode, migrate to Qdrant (re-ingest, no RAG code rewrite).

## Agent Guidelines
- Always run `make style` before considering code done; `make test` after changes
- Always use `uv`, never pip
- Keep `docs/features.json` current: flip a feature/subtask status, refresh `last_updated` and the `summary` counts whenever work lands
- Never modify `docs/` unless explicitly asked (the exception: keeping `docs/features.json` in sync is mandatory)
- Training + evals are frozen (v1 closed 2026-11-20): don't extend `eval/`, don't retrain. Read `docs/phase2-plan.md` before backend/RAG/frontend work; `docs/phase1-plan.md` is the frozen v1 record
- New workflow step? Add a Makefile target
- If something feels out of scope, flag it rather than silently doing it

## Project-Specific Notes
- Kaggle account is Eros438; Hugging Face + GitHub is arnabwithab. Kaggle CLI cannot attach secrets — train on Kaggle, push to HF from local.
- Amod dataset is gated (RAIL-D): accept terms on HF before downloading.
- `cotherapistai.com` is an unrelated commercial product — ignore it. The T-BARS RCC/coTherapist eval reference is obsolete — evals are dropped in phase 2 (product mode).
- KB coverage decision (2026-11-20): six dominant Indian traditions + secular; Ambedkarite/Navayana, Adivasi (Sarna/Donyi-Polo), Parsi/Zoroastrian, Baha'i, Jewish users are deliberately skipped (plan §7) — the free-text "other" answer handles them.
- 18+ only (user decision 2026-11-20): registration requires an age attestation; an honest <18 profile answer is declined warmly with Childline 1098 / Tele MANAS 14416 signposting. Never build a path that serves minors.
- Previews in rag-kb (`hku-preview`, `pageplace-preview`) are front-matter only — keep, don't treat as full books.
