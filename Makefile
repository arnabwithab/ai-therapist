.PHONY: setup test style ingest clean

setup: ## install all deps (idempotent)
	uv sync --group dev

test: ## run all tests
	uv run pytest

style: ## format + lint
	uv run black src/ tests/
	uv run ruff check src/ tests/

ingest: ## rebuild Chroma index from data/rag-kb/
	uv run python -m src.rag.ingest --reset

clean: ## remove build artifacts, caches, chroma index
	rm -rf __pycache__ .pytest_cache .ruff_cache chroma/
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
