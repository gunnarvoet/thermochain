.PHONY: lint test docs ghdocs servedocs schematics help
.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := uv run python -c "$$BROWSER_PYSCRIPT"

help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

lint: ## check style with flake8
	ruff check src/thermochain

test: ## run tests
	pytest

docs: ## generate documentation using pdoc
	rm -rf docs
	uv run pdoc --math -t .pdoc-theme-gv -d numpy -o docs thermochain
	$(BROWSER) docs/index.html

ghdocs: ## generate documentation for GitHub Pages (CI-safe, no browser)
	rm -rf docs
	PDOC_ALLOW_EXEC=1 pdoc --math -t .pdoc-theme-gv -d numpy -o docs thermochain

servedocs: ## compile the docs & watch for changes
	uv run pdoc --math -t .pdoc-theme-gv -d numpy thermochain
	# $(BROWSER) http://localhost:8080

schematics: ## regenerate schematic figures (docs SVGs + paper PDFs)
	uv run python schematics/pipeline_schematic.py
	uv run python schematics/drift_procedure_schematic.py
