# Numi — Enterprise AI DBA Agent & Secure DBA Control Gateway

[![CI](https://github.com/Psalmuel69/numi-dba-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Psalmuel69/numi-dba-agent/actions/workflows/ci.yml)

**Numi** (`@Numi`) is an AI database administration assistant that DBA teams talk
to over Slack and Microsoft Teams. It investigates incidents, gathers the
evidence itself, analyzes performance, and — only through a fully
independent, non-bypassable **DBA Control Gateway** — executes approved
remediation.

The single most important property of this system:

```
USER → VERIFIED IDENTITY → AI DBA → UNTRUSTED TOOL REQUEST → DBA CONTROL GATEWAY
     → AUTHORIZATION → POLICY → RISK → APPROVAL (if required) → SCOPED EXECUTION
     → DATABASE → VERIFICATION → AUDIT → AI DBA → USER
```

**The AI agent never holds a database credential, never has direct network
access to a database, and never determines its own authorization.** It can
only *ask* to run something — a separate service, the Gateway, is the only
thing with real access, and it independently re-checks every request
before anything runs. See
[ARCHITECTURE.md](ARCHITECTURE.md), [SECURITY.md](SECURITY.md), and
[THREAT_MODEL.md](THREAT_MODEL.md) for how that's enforced structurally, not
by prompting.

## Services

| Service       | Responsibility                                                                 |
|---------------|----------------------------------------------------------------------------------|
| `channels`    | Slack + Microsoft Teams webhook adapters. Verifies signatures/tokens, resolves identity, forwards to `agent`. Never touches a database. |
| `agent`       | The LLM-backed investigation/planning loop (Anthropic / OpenAI / Gemini / DeepSeek, DBA-selectable per conversation, or a deterministic offline planner). Proposes tool calls; has no DB credential and no authorization authority. For a recognized scenario (slow queries, high CPU, blocking, ...) it follows a fixed, named [playbook](ARCHITECTURE.md#investigation-loop-freeform-vs-playbook-driven) instead of investigating fully freeform. |
| `gateway`     | **The security boundary.** Tool registry, target validation, authorization, policy, risk, approval, data minimization, rate limiting, audit. Every request passes through here for an identity/permission/risk check, and for a human approval when one's needed. |
| `execution`   | The only service with database credentials/network access. Dispatches to `SQLServerAdapter`/`PostgreSQLAdapter`/`MySQLAdapter` (MySQL + MariaDB). |

Supporting infrastructure: PostgreSQL (control-plane database), Redis
(rate limiting in production).

## What it can do beyond answering questions

- **Remembers past investigations.** Before starting a new one, it checks whether it's seen the same server misbehave before, and reuses what it already found.
- **Double-checks its own conclusions.** A second pass reviews the answer before you see it, and asks the model to try again if it doesn't actually follow from the evidence.
- **Notices patterns across servers.** If three servers hit the same kind of problem this week, it says so instead of treating each one as a fresh mystery.
- **Logs its own bad calls.** When it degrades — hesitates, gets rejected by its own review, or falls back to a backup AI provider — that's recorded somewhere you can actually check later, not just buried in logs.

None of this is required reading to run the project — see
[ARCHITECTURE.md](ARCHITECTURE.md) if you want the mechanics.

## Quick start (local development)

The test suite needs no external services. Running the actual system needs
a database to point at (a sample one is included) and, optionally, an API
key for an LLM.

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -e ".[dev]"
cp .env.example .env
cp config/dev_credentials.example.yaml config/dev_credentials.yaml
./.venv/Scripts/python.exe -m pytest        # 908 pass, 24 opt-in skipped
```

Run the whole thing (a sample PostgreSQL database is included, so there's
something to actually talk to):

```bash
docker compose up --build
```

Talk to it via the dev channel without needing real Slack/Teams — this hits a local dev-only
endpoint that talks to the same agent:

```bash
curl -X POST http://localhost:8003/dev/chat \
  -H "Content-Type: application/json" \
  -d '{"user": "dba_l2@example.com", "message": "Check blocking on the production PostgreSQL cluster."}'
```

**Registering databases.** You don't register individual databases. Register each *server* in
`config/servers.yaml` (host, environment, criticality, allowed roles,
maintenance window, per-database overrides) and store its diagnostic
credential in the secrets manager under the same id. Numi discovers the
databases, tables, indexes and extensions on that server itself — reading
catalog and statistics views only, never table or view contents. Ask it
`/servers`, `/catalog <server>`, or `/discover` in chat.

**Choosing an LLM.** Set a key for any of `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY` (e.g.
`ANTHROPIC_API_KEY=sk-ant-... docker compose up`). Each configured provider
becomes selectable in chat with `/models` and `/model <provider> <model>`.
With no key set, the agent uses a deterministic offline planner. If your
provider has a bad moment, Numi retries, then tries a different model,
then — if you've configured more than one provider — a different vendor
entirely, and always says so plainly rather than going quiet. A single
investigative decision is never worse than ~20s late no matter how many
retries or model fallbacks happen underneath — see
[POLICY_MODEL.md](POLICY_MODEL.md#llm-selection) for the exact mechanics.

**Playbooks.** For a recognized scenario (slow queries, high CPU, blocking,
deadlocks, replication lag, connection saturation, ...) the agent follows a
fixed, named diagnostic sequence instead of deciding each step freeform —
faster and more consistent for the handful of situations that come up over
and over. One playbook, `comprehensive_summary`, isn't tied to a specific
symptom at all — it's a broad, single-server sweep ("comprehensive health
check", "daily summary", "full report", ...) a DBA can ask for on demand,
and the building block the **scheduled daily digest** calls once per
registered server. Ask `/playbooks` in chat to see the current list, or read
[ARCHITECTURE.md](ARCHITECTURE.md#investigation-loop-freeform-vs-playbook-driven)
for how and why.

**Daily digest (proactive, and structurally read-only).** Set
`DAILY_REPORT_SLACK_CHANNEL` and, once a day at `DAILY_REPORT_HOUR_UTC`,
Numi sweeps every registered server with `comprehensive_summary` and posts
one combined digest — organized by server, calling out only deviations, and
reporting any server it *couldn't* check as exactly that rather than
silently dropping it. Unset (the default) means no scheduler and no
background job at all. Proactive output is text and a recommendation,
**never** an action: a scheduled run is marked read-only, is offered no
write tool, refuses to submit anything the Gateway's catalog doesn't
confirm as a read, and never creates an approval request — so it can report
that a session should probably be killed, but can never kill one. See
[ARCHITECTURE.md](ARCHITECTURE.md#scheduled-daily-digest-proactive-and-structurally-read-only)
for how that's enforced and [OPERATIONS.md](OPERATIONS.md) for how to turn
it on.

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full local setup and
[TOOL_CATALOG.md](TOOL_CATALOG.md) for what `@Numi` can currently do.

## Commands

```bash
make dev             # create venv, install, copy .env.example
make test            # unit + integration tests
make security-test   # dedicated security test suite (spec §43-47)
make e2e             # end-to-end scenario tests
make lint            # ruff + mypy
make docker-up       # full stack via docker compose
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — service boundaries, data flow, why the Gateway is the only path to a database
- [SECURITY.md](SECURITY.md) — the security model and where each control actually lives
- [THREAT_MODEL.md](THREAT_MODEL.md) — threats, attack paths, mitigations, residual risk, tests
- [API.md](API.md) — REST API reference for all four services
- [TOOL_CATALOG.md](TOOL_CATALOG.md) — every tool, its risk classification, and whether it's enabled by default
- [POLICY_MODEL.md](POLICY_MODEL.md) — how policy/risk/approval decisions are made and configured
- [DATABASE_ADAPTERS.md](DATABASE_ADAPTERS.md) — SQL Server/PostgreSQL adapter design, adding a new engine
- [DEPLOYMENT.md](DEPLOYMENT.md) — production deployment guidance
- [OPERATIONS.md](OPERATIONS.md) — running it day to day
- [DEVELOPMENT.md](DEVELOPMENT.md) — local dev setup, project layout, coding standards
- [TESTING.md](TESTING.md) — test pyramid and how to run each layer
- [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) — what to do if Numi itself is the incident

## Status

**Working today:**

- Real connections to SQL Server, PostgreSQL, MySQL, and MariaDB
- Anthropic, OpenAI, Gemini, and DeepSeek as interchangeable AI providers, switchable per conversation, with automatic fallback if one has an outage
- Playbook-driven investigation for common scenarios, freeform reasoning for everything else
- Investigation memory, self-critique before reporting a conclusion, and cross-server pattern spotting (see [ARCHITECTURE.md](ARCHITECTURE.md) for how these work)
- Real identity (OIDC/SCIM) and real secrets backends (Vault/AWS/Azure/GCP) — not just the mock versions used for local dev
- The opt-in daily digest, read-only by construction — it can never take an action, only report

**Not yet implemented:** an Oracle adapter (the structure for one exists — see [DATABASE_ADAPTERS.md](DATABASE_ADAPTERS.md) if you want to add it).

**Test suite:** ~908 tests passing (unit, integration, and a dedicated security suite), plus 24 opt-in tests that need a real database/LLM key to run.
