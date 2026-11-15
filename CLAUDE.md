# aimlabs

uv workspace: `model/` (the `aimlabs` library, src layout) and `training/`
(scripts; `training/scripts` is on the pytest pythonpath so `common` imports).

- Setup: `uv sync --all-packages`
- Check everything: `make check` (ruff, mypy on `model/src`, pytest)
- Tests use a tiny on-disk BERT from the root `conftest.py`; never require
  network access in tests.
- Python >= 3.10; use modern typing (`list[str]`, `X | None`).
