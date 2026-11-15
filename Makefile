# Cross-platform dev tasks. Requires uv: https://docs.astral.sh/uv/
.PHONY: develop lint format typecheck test check build clean

develop:  ## create .venv with the library, training deps, and dev tools
	uv sync --all-packages
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy

test:
	uv run pytest --cov=aimlabs --cov-report=term-missing

check: lint typecheck test

build:
	uv build --package aimlabs

clean:
	rm -rf dist .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
