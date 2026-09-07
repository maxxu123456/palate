.PHONY: sync fmt lint type test check

sync:
	uv sync --extra api --extra eval --group dev

fmt:
	uv run ruff format src tests
	uv run ruff check --fix src tests

lint:
	uv run ruff check src tests

type:
	uv run mypy src

test:
	uv run pytest

check: lint type test
