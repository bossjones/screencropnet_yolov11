.PHONY: build up up-d down

build: ## Build the ML-backend Docker image
	docker compose build

up: ## Start the ML backend in the foreground
	docker compose up

up-d: ## Start the ML backend detached (daemonized)
	docker compose up -d

down: ## Stop and remove the ML-backend container
	docker compose down
