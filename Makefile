# Makefile for easy development workflows.
# See development.md for docs.
# Note GitHub Actions call uv directly, not this Makefile.

.DEFAULT_GOAL := help

.PHONY: default install lint test check open-coverage upgrade build clean agent-rules help monkeytype-create monkeytype-apply autotype

ML_BACKEND_DIR := tools/labeling/ml_backend

# Labeling-flow paths (override on the command line, e.g. `make labeling-export LS_EXPORT=foo.zip`).
PYTORCH_LAB ?= /Users/bossjones/dev/bossjones/pytorch-lab
DATASET_DIR ?= datasets/twitter_screenshots_localization_dataset
LS_EXPORT ?= ./ls_export.zip
RAW_DIR := scratch/datasets/twitter_screenshots_raw

# Append-only log for label-studio-local (gitignored via *.log). Query it later
# when something looks wrong; `make label-studio-log-truncate` empties it.
LABEL_STUDIO_LOG := $(CURDIR)/label-studio.log

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

# ---- Ingest/classify pipeline: services, migrations, run targets ----

.PHONY: services-up services-down services-logs migrate api worker test-integration

services-up: ## Start Postgres, RabbitMQ, Prometheus (:9091), Grafana (:3001) via docker compose
	@echo "🚀 Starting supporting services"
	@docker compose up -d
	@docker compose ps

services-down: ## Stop and remove the supporting services
	@echo "🚀 Stopping supporting services"
	@docker compose down

services-logs: ## Follow logs from the supporting services
	@docker compose logs -f

migrate: ## Apply Alembic migrations to Postgres
	@echo "🚀 Applying database migrations"
	@uv run alembic upgrade head

api: ## Run the FastAPI ingest/classify service on 127.0.0.1:8000
	@uv run uvicorn screencropnet_yolo.server.api:create_app --factory --host 127.0.0.1 --port 8000

worker: ## Run the RabbitMQ classification worker (needs the `worker` dep group + weights)
	@uv run screencrop-worker

test-integration: ## Run the integration suite against real Postgres + RabbitMQ
	@echo "🚀 Running integration tests"
	@uv run pytest -m integration

.PHONY: ml-backend-build ml-backend-up ml-backend-up-d ml-backend-down

ml-backend-build: ## Build the Label Studio ML-backend Docker image
	@$(MAKE) -C $(ML_BACKEND_DIR) build

ml-backend-up: ## Start the ML backend in the foreground
	@$(MAKE) -C $(ML_BACKEND_DIR) up

ml-backend-up-d: ## Start the ML backend detached (daemonized)
	@$(MAKE) -C $(ML_BACKEND_DIR) up-d

ml-backend-down: ## Stop and remove the ML-backend container
	@$(MAKE) -C $(ML_BACKEND_DIR) down

.PHONY: label-studio label-studio-local label-studio-log-truncate ml-backend

# Label Studio is installed as an isolated uv tool (`uv tool install
# label-studio`): its pinned requests/pillow versions conflict with this
# project's deps, so it cannot live in the project venv. `uvx` == `uv tool run`.
# Pin --python 3.12: Label Studio's metadata allows up to 3.14, but its
# django-environ dep imports the removed pkgutil.find_loader and crashes on 3.14.
label-studio: ## launch Label Studio annotation UI on http://localhost:8080
	uvx --python 3.12 label-studio

label-studio-local: ## launch Label Studio serving the screenshots dir as local files (tees to label-studio.log)
	# LOCAL_FILES_* lets you import the on-disk screenshots without uploading them.
	# `2>&1 | tee -a` shows output live while appending stdout+stderr to the log;
	# `set -o pipefail` preserves label-studio's exit code past the tee pipe.
	set -o pipefail; \
	LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
	LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=$(CURDIR)/scratch/datasets/twitter_screenshots_raw \
	uvx --python 3.12 label-studio 2>&1 | tee -a $(LABEL_STUDIO_LOG)

label-studio-log-truncate: ## empty label-studio.log
	@: > $(LABEL_STUDIO_LOG)
	@echo "✔︎ truncated $(LABEL_STUDIO_LOG)"

# Local (non-Docker) ML backend launch, mirroring README step 2. Runs as an
# isolated uv tool; reads the checkpoint from scratch/checkpoints/ (override with
# CHECKPOINT_PATH). The backslash-joined recipe is one shell, so cd persists.
ml-backend: ## launch the ML backend locally via uvx on http://localhost:9090
	cd $(ML_BACKEND_DIR) && \
	uvx --python 3.11 \
	    --from "git+https://github.com/HumanSignal/label-studio-ml-backend.git" \
	    --with torch --with timm --with albumentations \
	    --with opencv-python-headless --with redis --with rq \
	    label-studio-ml start . --port 9090

.PHONY: labeling-stage labeling-tasks labeling-setup-project labeling-export dataset-validate train

# Labeling pipeline targets, mirroring docs/label-studio-annotation-guide.md. The
# raw commands still live in that guide; these are the one-shot equivalents.
labeling-stage: ## stage raw images + checkpoint from PYTORCH_LAB into scratch/ (guide step 1)
	mkdir -p $(RAW_DIR) scratch/checkpoints scratch/labeling
	cp "$(PYTORCH_LAB)/scratch/datasets/twitter_screenshots_localization_dataset/labels_pascal_temp.csv" \
	   $(RAW_DIR)/labels_pascal_temp.csv
	cp -R "$(PYTORCH_LAB)/scratch/datasets/twitter_screenshots_localization_dataset/train_images" \
	   $(RAW_DIR)/train_images
	cp "$(PYTORCH_LAB)/screencropnet/models/ScreenCropNetV1_378_epochs.pth" \
	   scratch/checkpoints/screencropnet_efficientnet_b0_378.pth

labeling-tasks: ## build Label Studio tasks.json with boxes pre-drawn (guide step 2)
	uv run scripts/pascal_csv_to_ls_tasks.py \
	    --csv $(RAW_DIR)/labels_pascal_temp.csv \
	    --images-root $(RAW_DIR)/train_images \
	    --images-url-prefix "/data/local-files/?d=train_images" \
	    --out scratch/labeling/tasks.json

labeling-setup-project: ## create+configure the screencropnet LS project via SDK (needs LABEL_STUDIO_API_KEY)
	# --local-files-document-root must match label-studio-local's LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT
	uv run scripts/setup_ls_project.py \
	    --title screencropnet \
	    --tasks scratch/labeling/tasks.json \
	    --ml-backend-url http://localhost:9090 \
	    --local-files-document-root $(RAW_DIR)

labeling-export: ## convert a Label Studio export (LS_EXPORT) into DATASET_DIR (guide step 8)
	uv run scripts/ls_yolo_export_to_dataset.py \
	    --export $(LS_EXPORT) \
	    --out $(DATASET_DIR)/ \
	    --val-ratio 0.2 --test-ratio 0.1 --seed 42

dataset-validate: ## validate the canonical dataset without training (guide step 8)
	uv run python -m screencropnet_yolo.train --validate-only

train: ## train YOLO26 on the canonical dataset — runs a real training session (guide step 8)
	uv run python -m screencropnet_yolo.train

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
