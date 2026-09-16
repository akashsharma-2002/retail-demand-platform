.PHONY: install data train pipeline test lint typecheck security evals up down load-test web

install:
	uv sync --extra embeddings --extra deep
data:
	uv run python -m retail_platform.data
pipeline:
	uv run python -m retail_platform.pipeline.flow
test:
	uv run pytest --cov=retail_platform --cov-report=term-missing
lint:
	uv run ruff check src tests evals
typecheck:
	uv run mypy src
security:
	uv run bandit -q -r src -ll
	uv run pip-audit --strict --progress-spinner off
evals:
	uv run python -m evals.run_all
up:
	docker compose up -d --build
down:
	docker compose down
load-test:
	uv run locust -f evals/locustfile.py --headless -u 10 -r 2 -t 2m --host http://localhost:8000 --csv artifacts/reports/locust
web:
	cd web && npm ci && npm run dev
