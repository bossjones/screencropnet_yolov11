# Makefile for easy development workflows.
# See development.md for docs.
# Note GitHub Actions call uv directly, not this Makefile.

.DEFAULT_GOAL := help

.PHONY: default install lint test check open-coverage upgrade build clean agent-rules help monkeytype-create monkeytype-apply autotype

ML_BACKEND_DIR := tools/labeling/ml_backend

default: install lint test ## Run install, lint, and test
# default: agent-rules install lint test ## Run agent-rules, install, lint, and test

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

.PHONY: check
check: ## Run type checking with ty
	@echo "🚀 Running type checking with ty"
	@uv run ty check

.PHONY: open-coverage
open-coverage: ## Open coverage HTML report in browser
	@open htmlcov/index.html

.PHONY: upgrade
upgrade: ## Upgrade all dependencies to latest versions
	@echo "🚀 Upgrading all dependencies to latest versions"
	@uv sync --upgrade --all-extras --dev

.PHONY: build
build: ## Build the package distribution
	@echo "🚀 Building package distribution"
	@uv build

.PHONY: ml-backend-build ml-backend-up ml-backend-up-d ml-backend-down

ml-backend-build: ## Build the Label Studio ML-backend Docker image
	@$(MAKE) -C $(ML_BACKEND_DIR) build

ml-backend-up: ## Start the ML backend in the foreground
	@$(MAKE) -C $(ML_BACKEND_DIR) up

ml-backend-up-d: ## Start the ML backend detached (daemonized)
	@$(MAKE) -C $(ML_BACKEND_DIR) up-d

ml-backend-down: ## Stop and remove the ML-backend container
	@$(MAKE) -C $(ML_BACKEND_DIR) down

.PHONY: label-studio label-studio-local ml-backend

# Label Studio is installed as an isolated uv tool (`uv tool install
# label-studio`): its pinned requests/pillow versions conflict with this
# project's deps, so it cannot live in the project venv. `uvx` == `uv tool run`.
label-studio: ## launch Label Studio annotation UI on http://localhost:8080
	uvx label-studio

label-studio-local: ## launch Label Studio serving the screenshots dir as local files
	# LOCAL_FILES_* lets you import the on-disk screenshots without uploading them
	LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
	LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(CURDIR)/scratch/datasets/twitter_screenshots_raw \
	uvx label-studio

# Local (non-Docker) ML backend launch, mirroring README step 2. Runs as an
# isolated uv tool; reads the checkpoint from scratch/checkpoints/ (override with
# CHECKPOINT_PATH). The backslash-joined recipe is one shell, so cd persists.
ml-backend: ## launch the ML backend locally via uvx on http://localhost:9090
	cd $(ML_BACKEND_DIR) && \
	uvx --from label-studio-ml --with torch --with timm --with albumentations \
	    --with opencv-python-headless label-studio-ml start . --port 9090

# Disabled: agent-rules auto-generates CLAUDE.md/AGENTS.md from .cursor/rules,
# clobbering hand-edited content. Re-enable (and restore `agent-rules` in
# `default` above) once we decide how to merge generated + hand-written rules.
# .PHONY: agent-rules
# agent-rules: CLAUDE.md AGENTS.md ## Generate CLAUDE.md and AGENTS.md from .cursor/rules
#
# # Use .cursor/rules for sources of rules.
# # Create Claude and Codex rules from these.
# CLAUDE.md: .cursor/rules/general.mdc .cursor/rules/python.mdc
# 	@echo "🚀 Generating CLAUDE.md from .cursor/rules"
# 	@cat .cursor/rules/general.mdc .cursor/rules/python.mdc > CLAUDE.md
#
# AGENTS.md: .cursor/rules/general.mdc .cursor/rules/python.mdc
# 	@echo "🚀 Generating AGENTS.md from .cursor/rules"
# 	@cat .cursor/rules/general.mdc .cursor/rules/python.mdc > AGENTS.md

.PHONY: monkeytype-create
monkeytype-create: ## Run tests with monkeytype tracing
	@echo "🚀 Running tests with monkeytype tracing"
	@uv run monkeytype run `uv run which pytest`

.PHONY: monkeytype-apply
monkeytype-apply: ## Apply monkeytype stubs to all modules
	@echo "🚀 Applying monkeytype stubs to all modules"
	@uv run monkeytype list-modules | xargs -n1 -I{} sh -c 'uv run monkeytype apply {}'

.PHONY: autotype
autotype: monkeytype-create monkeytype-apply ## Run monkeytype tracing and apply stubs

.PHONY: clean
clean: ## Remove build artifacts and cache directories
	@echo "🚀 Removing build artifacts and cache directories"
	@rm -rf dist/
	@rm -rf *.egg-info/
	@rm -rf .pytest_cache/
	@rm -rf .mypy_cache/
	@rm -rf .venv/
	# @rm -rf CLAUDE.md AGENTS.md  # disabled while agent-rules is commented out
	@find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: help
help: ## Show this help message
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"
