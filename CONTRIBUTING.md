# Contributing

Small, reviewable PRs. Read [AGENTS.md](AGENTS.md) first — it takes precedence over this file.

## Workflow

1. Check [`docs/features.json`](docs/features.json) for the feature ID your work belongs to.
2. **TDD first:** write the test, then the implementation. No function ships without a test. Tests mirror the module they test (`tests/test_rag/…`, `tests/test_backend/…`).
3. Keep API routes thin — logic lives in `core/` (`backend/core/`, `src/rag/`).
4. Run before pushing:
   ```bash
   make style   # black + ruff
   make test
   ```
   `make ci` is the clean-room gate (format check + lint + tests).

## Conventions

- **Package manager: `uv`** — `uv add`, `uv run`, `uv sync`. Never pip.
- `snake_case` for files, variables, functions. Explicit over clever.
- `src/rag/query.py:retrieve()` is the only API the serving layer imports — keep chunking/embedding store-agnostic.
- Env vars via `.env`; never commit secrets.
- No `print` logging in library code — stdlib `logging`.
- New workflow step? Add a `Makefile` target — if it isn't runnable via `make`, it isn't done.
- Conventional commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`).
- Notebooks: strip outputs before committing.

## Never commit

`data/*/` contents, `chroma/`, `models/`, `.env`. (See `.gitignore`.)

## Docs

- Don't modify `docs/` unprompted — exception: keep `docs/features.json` in sync (flip subtask status, refresh `last_updated` and `summary` counts when work lands).
- Check `docs/slm-finetuning.md` before touching training/eval — it takes precedence.
- If something feels out of scope, flag it rather than silently doing it.
