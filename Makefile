.PHONY: help test lint format check typecheck coverage security run run-worker run-admin docker-up docker-down docker-build docker-logs migrate clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Development ---

test: ## Run all tests
	pytest tests/ -v --tb=short

coverage: ## Run tests with coverage report
	pytest tests/ -v --tb=short --cov=services --cov=config --cov-report=term-missing --cov-fail-under=35

lint: ## Run ruff linter
	ruff check .

format: ## Format code with ruff
	ruff format .

typecheck: ## Run mypy type checker
	mypy config.py services/ --no-error-summary

security: ## Run bandit security scan
	bandit -r services/ config.py main.py worker.py admin_api.py database.py --severity-level medium

check: lint test ## Run lint + tests

# --- Run Services ---

run: ## Start webhook API (port 8000)
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

run-worker: ## Start background worker
	python worker.py

run-admin: ## Start admin API (port 8080)
	uvicorn admin_api:app --host 127.0.0.1 --port 8080 --reload

# --- Docker ---

docker-up: ## Start all services with Docker Compose
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-build: ## Build Docker images
	docker compose build

docker-logs: ## Tail logs from all services
	docker compose logs -f

# --- Database ---

migrate: ## Run database migrations
	alembic upgrade head

migrate-new: ## Create a new migration (usage: make migrate-new msg="description")
	alembic revision --autogenerate -m "$(msg)"

# --- Cleanup ---

clean: ## Remove cache and compiled files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage coverage.xml
