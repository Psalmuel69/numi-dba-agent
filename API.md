# API Reference

All four services are FastAPI apps; each exposes interactive OpenAPI docs at
`/docs` when running. This is a hand-written summary of the stable surface
(spec §57).

Every endpoint below except health/ready checks and the Slack/Teams webhook
signature/token verification requires a signed service token in the
`X-Service-Token` header (`common/service_auth.py`), scoped to the
receiving service's audience (`numi-gateway`, `numi-execution`,
`numi-agent`).

## Gateway (`gateway/api/app.py`) — default port 8001

### `POST /v1/tool-calls`

Body: `ToolCallRequest` (`common/models/tool.py`) — `tool_id`, `arguments`,
`target`, `reason`, `conversation_id`, `investigation_id?`, `request_id`,
`channel`, `channel_account_id`, `approval_id?`, `change_id?`.

Response: `ToolCallResponse` — `status` (`EXECUTED`/`APPROVAL_REQUIRED`/
`DENIED`/`FAILED`), `execution_id?`, `approval_id?`, `result?`,
`failure_code?`, `message`, `risk?`, `policy_decision?`.

### `GET /v1/tools`, `GET /v1/tools/{tool_id}`

Lists the tool catalog. `GET /v1/tools?channel=...&channel_account_id=...`
filters to what that resolved role could attempt (convenience only — see
[SECURITY.md](SECURITY.md)).

### `POST /v1/approvals/{approval_id}/approve`, `.../reject`

Body: `{channel, channel_account_id}` — the *approver's* channel account,
independently re-resolved exactly like a tool call. Response includes the
resulting `status` (`APPROVED`, `AWAITING_SECOND_APPROVAL`, `REJECTED`).

### `GET /v1/approvals/{approval_id}`

Returns the approval's current state (never the raw action hash).

### `POST /v1/investigations`, `PATCH /v1/investigations/{investigation_id}`

The Agent's write path for investigation state (`gateway/domain/
investigation_store.py`) — the Gateway never fabricates evidence/findings,
it only persists and serves back what the Agent recorded. `POST` body:
`investigation_id`, `conversation_id`, `user_subject_id`, `server_id?`,
`playbook_id?`, `environment?`, `target?`, `problem?`, `status?`; idempotent
on a duplicate `investigation_id` (returns the existing row rather than
erroring). `PATCH` takes the same fields plus `evidence`/`hypotheses`/
`findings`/`recommendations`/`actions`, all optional — omitted fields are
left unchanged, `404` if the id doesn't exist.

### `POST /v1/investigations/{investigation_id}/events`

Appends one append-only event (`event_type`, `payload`) to an
investigation's timeline.

### `GET /v1/investigations/memory/{server_id}`

Recalls recent, concluded investigations for a server — background context
an investigation's problem statement can cite, never grounding evidence for
it. Query params: `exclude_investigation_id?`, `limit` (default 3). Set
`INVESTIGATION_MEMORY_LOOKBACK=0` to disable recall entirely.

### `GET /v1/investigations/correlate`

Cross-server pattern correlation — "this same symptom happened on N other
servers recently." Matches structurally, by shared `playbook_id` and
optionally `environment`, never by fuzzy text similarity. Query params:
`playbook_id` (required), `environment?`, `exclude_server_id?`, `limit`
(default 5). Returns `[]` immediately if `CROSS_SERVER_CORRELATION_ENABLED=false`
or if `playbook_id` is empty (a fully freeform investigation has no
scenario to correlate on).

### `GET /v1/investigations/{investigation_id}`, `GET /v1/audit/{audit_event_id}`

Read-only projections of control-plane state.

### `POST /v1/decision-events`, `GET /v1/decision-events/summary`

Durable storage for a handful of decision-quality signals — a conclusion
rejected on review, a cross-provider fallback substitution — so they're a
queryable habit, not just log lines. `POST` body: `event_type`,
`conversation_id?`, `investigation_id?`, `provider?`, `model?`, `payload?`.
`GET .../summary?since_hours=24` returns counts grouped by `event_type`
since that cutoff. Gated by `DECISION_EVENT_LOGGING_ENABLED`.

### `GET /v1/catalog/servers`, `GET /v1/catalog/servers/{server_id}`

The registered servers (`config/servers.yaml`) plus, per server, the latest
discovered catalog — engine version/edition, database list, and per-database
object counts (tables, views, indexes, procedures) and available extensions.
Catalog discovery reads catalog and statistics views only; it never reads
table or view contents. `404` if the server id is not registered.

### `POST /v1/catalog/refresh`, `POST /v1/catalog/refresh/{server_id}`

Re-runs discovery now (otherwise it refreshes lazily on first use and every
`DISCOVERY_REFRESH_MINUTES`). Body: `{channel, channel_account_id}` —
independently re-resolved and required to hold `DBA_MANAGER` (`403`
otherwise). `404` if the server id is not registered.

### `GET /health`, `GET /ready`

## Execution Service (`execution/api/app.py`) — default port 8002

### `POST /v1/execute`

Only callable by the Gateway (audience `numi-execution`). Body:
`ExecutionRequest`; response: `ExecutionResult`. Never exposed to any other
service or to the public internet in a real deployment (spec §30).

### `POST /v1/discover`

Only callable by the Gateway. Body: `DiscoveryRequest` (`server_id`,
`platform`, `max_objects_per_database`); response: `ServerCatalog`. Opens a
real connection via the same `CredentialProvider` as `/v1/execute` and reads
the engine's catalog/DMV/stats views — server properties, `sys.databases` /
`pg_database`, object lists with row estimates and sizes, available
extensions. It issues no query that returns user table/view data.

### `GET /health`, `GET /ready`

## Agent (`agent/api/app.py`) — default port 8000

### `POST /v1/chat`

Body: `{channel, channel_account_id, conversation_id, channel_thread_id?, message}`.
Response: `AgentReply` — `text`, `status` (`ok`/`approval_required`/`denied`/
`error`/`clarification`), `approval_card?`, `investigation_id?`.

### `POST /v1/chat/events`

Body: `{channel, channel_account_id, conversation_id, approval_id, decision}`
where `decision` is `"approve"` or `"reject"` — routes to the Gateway's
approval endpoints and resubmits the original tool call on success.

### `POST /v1/alerts/trigger`

Internal — called by Channels' `/webhooks/alerts` after it verifies the
external monitoring system's signature, with a service token audience
`numi-agent` (same as `/v1/chat`). Body:
`{server, metric?, current_value?, threshold?, severity?, source?, message?}`.
Runs one freeform, read-only investigation
(`orchestrator.run_triggered_investigation` — see ARCHITECTURE.md's
"Alert-triggered investigation") against the named server and delivers the
result via Channels' `POST /v1/notify`. Response: `{ok, error?}` — `ok:
false` covers an unresolvable `server`, the feature being unconfigured
(`ALERT_WEBHOOK_SLACK_CHANNEL` unset), or a delivery failure; none of these
are HTTP errors, since the request Channels forwarded was itself valid.

### `GET /health`

## Channels (`channels/api/app.py`) — default port 8003

### `POST /webhooks/slack`

Verifies `X-Slack-Signature`/`X-Slack-Request-Timestamp`
(`channels/slack/signature.py`), handles Slack's `url_verification`
challenge, and forwards `message` events from verified DBA accounts to the
Agent.

### `POST /webhooks/slack/interactive`

Handles Block Kit button clicks (Approve/Reject), routing to the Agent's
`/v1/chat/events`.

### `POST /webhooks/teams`

Verifies the Bot Framework bearer token (`channels/teams/auth.py`) and
routes both plain messages and Adaptive Card `Action.Submit` payloads
(approve/reject).

### `POST /webhooks/alerts`

An external monitoring system (Prometheus Alertmanager, Datadog, a cloud
provider's own alarms, ...) reporting a threshold breach. Verifies
`X-Numi-Alert-Signature`/`X-Numi-Alert-Timestamp`
(`channels/alerts/signature.py` — same HMAC-over-timestamp-bound-body shape
as the Slack signature above); an unset `ALERT_WEBHOOK_SECRET` refuses
every request. Body: `{server, metric?, current_value?, threshold?,
severity?, source?, message?, alert_id?}` — `server` is the only required
field (a registered server id or alias) and `alert_id`, if the sender
includes one, deduplicates a retried delivery the same way Slack's own
event retries are. Forwards to the Agent's `POST /v1/alerts/trigger` and
returns `{ok, error?}` — always HTTP 200/400/401, since a monitoring
system's retry/backoff logic should react to "the request was malformed,"
not to "the server name in a well-formed alert didn't match anything."

### `POST /v1/notify`

The one Agent-initiated (outbound) path on this service: delivers an
unsolicited message to a channel. Today that is only the scheduled daily
health digest (`agent/scheduled_report.py`). Requires a service token with
audience `numi-channels`; body `{channel_id, text}`.

Deliberately minimal — no identity, no `approval_id`, no conversation. It
posts text and nothing else, and the reply it renders never carries an
approval card, so it cannot be used to put a clickable action in front of a
DBA. Delivery lives here rather than in the Agent because this is the only
service holding a channel credential.

### `POST /dev/chat`, `POST /dev/chat/events` (spec §66)

Mock channel for local development — no real Slack/Teams credentials
required. Body: `{user, message, conversation_id?}` where `user` is a
`channel_accounts.dev` value from `config/identity.yaml` (e.g.
`"dba_l2@example.com"`). Still goes through real identity resolution and
the real Agent/Gateway/Execution pipeline — only the transport is mocked.

### `GET /health`
