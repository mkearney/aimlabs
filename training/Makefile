# Convenience wrapper; see the Makefile at the repository root.
.PHONY: develop test train

develop:
	$(MAKE) -C .. develop

test:
	uv run pytest tests

# e.g. make train ARGS="--input-dir ~/data/imdb --save"
train:
	uv run python scripts/train.py $(ARGS)
