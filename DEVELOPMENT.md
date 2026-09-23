# Development

## Setup

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
cp .env.example .env
./.venv/Scripts/python.exe -m pytest
```

No Slack/Teams app or LLM API key is required to run the test suite —
`IDENTITY_PROVIDER=mock` and, with no LLM key set, the deterministic
offline planner. The tests never open a real database connection (they
inject `tests/canned_adapter.py` — see [DATABASE_ADAPTERS.md](DATABASE_ADAPTERS.md)).

**Running the actual system** does connect to real databases. Point
`config/dev_credentials.yaml` (copied from
`config/dev_credentials.example.yaml`, git-ignored) at your dev databases,
or bring up the bundled sample ones:

```bash
docker compose up -d postgres-sample            # ready-to-use Postgres target
docker compose --profile mssql up -d mssql-sample   # opt-in, heavy
```

### LLM providers

Set a key for any provider you want available (`ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`). With none set, the
agent uses the deterministic offline planner. Each configured provider is
selectable in chat via `/models` and `/model <provider> <model>`. See
[POLICY_MODEL.md](POLICY_MODEL.md#llm-selection) for the resolution rules.

## Project layout

See "Package layout" in [ARCHITECTURE.md](ARCHITECTURE.md). In short:
`common/` has no service-specific logic, `gateway/domain/` is the security
boundary's business logic, `gateway/api/` and its equivalents in
`execution/`/`agent/`/`channels/` are thin FastAPI wiring around it.

## Coding standards

- Python 3.12+, full type hints, Pydantic v2 models for every boundary
  (never a bare `dict` crossing a trust boundary).
- Business logic lives in `domain/` modules, not in FastAPI route handlers
  — routers in `api/routers/*.py` should stay a thin translation layer
  between HTTP and a domain call.
- Every `NumiError` carries a `FailureCode` (`common/models/failures.py`)
  and a safe, user-facing `detail` — never let a raw exception message or
  stack trace reach a channel adapter.
- New tools: see the "Adding a new tool" section of
  [TOOL_CATALOG.md](TOOL_CATALOG.md).
- Run `make lint` (ruff + mypy) before committing.

## Running a single service locally

```bash
make run-execution   # port 8002
make run-gateway     # port 8001
make run-agent       # port 8000
make run-channels    # port 8003
```

Each command runs against `sqlite+aiosqlite:///./numi_dev.db` by default
(see `.env.example`); for a closer-to-production setup, run
`docker compose up postgres redis` first and point `CONTROL_DB_URL` at it.

**On Windows**, start the Execution Service with `python -m numi.execution`
instead of `make run-execution` / raw `uvicorn`. psycopg's async mode needs
a selector-based event loop, and uvicorn forces `ProactorEventLoop` on
Windows unless told not to *before* it starts — `uvicorn numi.execution.api.app:app`
can't set that in time (the app module is imported after uvicorn's loop
already exists). `python -m numi.execution` sets the policy first. The
other three services don't open real database connections, so
`make run-gateway`/`run-agent`/`run-channels` are unaffected.

## Local end-to-end smoke test without any UI

```bash
curl -X POST http://localhost:8003/dev/chat \
  -H "Content-Type: application/json" \
  -d '{"user": "dba_l2@example.com", "message": "CoreBanking production is slow. Investigate."}'
```

Approve the resulting action:

```bash
curl -X POST http://localhost:8003/dev/chat/events \
  -H "Content-Type: application/json" \
  -d '{"user": "dba_l2@example.com", "conversation_id": "dev-conversation", "approval_id": "<from previous response>", "decision": "approve"}'
```

## Mock users (development only — see `config/identity.yaml`)

| dev user | Slack account id | Teams AAD id | Role |
|---|---|---|---|
| `dba_l1@example.com` | `U_MOCK_L1` | `aad-mock-l1` | DBA_L1 |
| `dba_l2@example.com` | `U_MOCK_L2` | `aad-mock-l2` | DBA_L2 |
| `dba_l3@example.com` | `U_MOCK_L3` | `aad-mock-l3` | DBA_L3 |
| `dba_l3b@example.com` | `U_MOCK_L3B` | `aad-mock-l3b` | DBA_L3 (second, for dual-approval testing) |
| `dba_manager@example.com` | `U_MOCK_MGR` | `aad-mock-mgr` | DBA_L3 + DBA_MANAGER |
| `notadba@example.com` | `U_MOCK_NONDBA` | `aad-mock-nondba` | not a DBA |

These are fictitious accounts for local development/tests only.
