.PHONY: dev test lint security-test e2e live-llm-test format migrate seed run-gateway run-execution run-agent run-channels docker-up docker-down

VENV := .venv
PY := $(VENV)/Scripts/python.exe

dev:
	python -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"
	cp -n .env.example .env || true

test:
	$(PY) -m pytest tests/unit tests/integration -q

security-test:
	$(PY) -m pytest tests/security -q

e2e:
	$(PY) -m pytest tests/e2e -q

# Opt-in only: hits the real Anthropic API, needs ANTHROPIC_API_KEY, costs
# money per run. Never invoked by `test`/`test-all`/CI.
live-llm-test:
	RUN_LIVE_LLM_TESTS=1 $(PY) -m pytest tests/e2e -q

test-all:
	$(PY) -m pytest tests -q

lint:
	$(PY) -m ruff check src tests
	$(PY) -m mypy src

format:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

migrate:
	$(PY) -m alembic upgrade head

run-gateway:
	$(PY) -m uvicorn numi.gateway.api.app:app --reload --port 8001

run-execution:
	$(PY) -m uvicorn numi.execution.api.app:app --reload --port 8002

run-agent:
	$(PY) -m uvicorn numi.agent.api.app:app --reload --port 8000

run-channels:
	$(PY) -m uvicorn numi.channels.api.app:app --reload --port 8003

docker-up:
	docker compose up --build

docker-down:
	docker compose down -v
