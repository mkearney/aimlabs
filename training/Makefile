BIN = .venv/Scripts
PIP = $(BIN)/pip.exe
PYTHON = $(BIN)/python.exe
PYTEST = $(BIN)/pytest

.PHONY: develop
develop:
	python -m venv .venv
	$(PIP) install \
		-e ../model/. \
		-r requirements/dev.txt
		# -r requirements/training.txt

.PHONY: test
test:
	$(PYTEST) -s --cov=src .
