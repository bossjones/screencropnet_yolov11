# Makefile for easy development workflows.
# See development.md for docs.
# Note GitHub Actions call uv directly, not this Makefile.

.DEFAULT_GOAL := help

.PHONY: default install lint test upgrade build clean agent-rules help

default: agent-rules install lint test ## Run agent-rules, install, lint, and test

.PHONY: install
install: ## Install dependencies with all extras
	@echo "🚀 Installing dependencies with all extras"
	@uv sync --all-extras

.PHONY: lint
lint: ## Run linting tools
	@echo "🚀 Running linting tools"
	@uv run python devtools/lint.py

.PHONY: test
test: ## Run tests with pytest
	@echo "🚀 Running tests with pytest"
	@uv run pytest

.PHONY: upgrade
upgrade: ## Upgrade all dependencies to latest versions
	@echo "🚀 Upgrading all dependencies to latest versions"
	@uv sync --upgrade --all-extras --dev

.PHONY: build
build: ## Build the package distribution
	@echo "🚀 Building package distribution"
	@uv build

.PHONY: agent-rules
agent-rules: CLAUDE.md AGENTS.md ## Generate CLAUDE.md and AGENTS.md from .cursor/rules

# Use .cursor/rules for sources of rules.
# Create Claude and Codex rules from these.
CLAUDE.md: .cursor/rules/general.mdc .cursor/rules/python.mdc
	@echo "🚀 Generating CLAUDE.md from .cursor/rules"
	@cat .cursor/rules/general.mdc .cursor/rules/python.mdc > CLAUDE.md

AGENTS.md: .cursor/rules/general.mdc .cursor/rules/python.mdc
	@echo "🚀 Generating AGENTS.md from .cursor/rules"
	@cat .cursor/rules/general.mdc .cursor/rules/python.mdc > AGENTS.md

.PHONY: clean
clean: ## Remove build artifacts and cache directories
	@echo "🚀 Removing build artifacts and cache directories"
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .venv/
	@rm -rf CLAUDE.md AGENTS.md
	@find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: help
help: ## Show this help message
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"
