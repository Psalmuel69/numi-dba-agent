# Operations

## Day-to-day

- **Adding a DBA:** add them to the real enterprise IdP group(s) referenced
  in `config/identity.yaml`'s `identity.groups`/`identity.roles` — nothing
  in this codebase needs to change or redeploy.
- **Onboarding a server:** add an entry to `config/servers.yaml` with its
  host/port, environment, criticality, `allowed_roles`, maintenance window,
  and any per-database overrides, then store its credential in the secrets
  manager under the same server id. The Gateway discovers the databases,
  tables, indexes and extensions on that server automatically (on first use,
  on `/discover`, or eagerly at startup with
  `NUMI_DISCOVERY_ON_STARTUP=true`); individual databases are never
  registered by hand. The Agent selects targets from the registry + the
  discovered catalog, never from a freehand connection string (spec §11).
  Discovery reads catalog and statistics views only — never table contents.
- **Changing what requires approval:** edit `config/policy.yaml`. Changes
  take effect on the Gateway's next restart (config is loaded once at
  process start — restart the Gateway deployment to pick up policy edits).
- **Enabling a restricted tool:** flip the corresponding `ENABLE_*_TOOL`
  environment variable for the Gateway *and* Execution Service, and confirm
  an adapter implementation actually exists for it (see
  [TOOL_CATALOG.md](TOOL_CATALOG.md) — several restricted tools have an
  argument schema but deliberately no execution path yet).
- **Adding/changing a playbook:** edit `agent/playbooks/library.py` —
  each entry is a `playbook_id`, trigger phrases (matched word-boundary,
  case-insensitive), a fixed list of read-only diagnostic steps, and
  conclusion guidance for the LLM's final call. Restart the Agent to pick
  up changes (loaded once at import time, like the tool catalog). See
  [ARCHITECTURE.md](ARCHITECTURE.md#investigation-loop-freeform-vs-playbook-driven).
  Ask `/playbooks` in chat to see what's currently registered.

## Enabling the daily health digest

Once a day, Numi can sweep every registered server with the
`comprehensive_summary` playbook and post one combined digest to a channel.
It is **off by default**, and the destination channel is the only switch:
with `DAILY_REPORT_SLACK_CHANNEL` unset, no scheduler is created and no job
is registered at all.

Set on the **agent** service (it owns the orchestrator; the digest is never
scheduled from `channels` or `execution`):

| Variable | Default | What it does |
|---|---|---|
| `DAILY_REPORT_SLACK_CHANNEL` | `""` (off) | Destination channel id. Empty = feature entirely disabled. |
| `DAILY_REPORT_HOUR_UTC` | `6` | Hour (UTC, minute 0) the digest runs. Must be 0–23 — an out-of-range value fails the process at startup rather than silently running at another hour. |
| `DAILY_REPORT_SERVERS` | `""` (all) | Comma-separated server ids/aliases to narrow the sweep. Empty means every *active* registered server, so a newly onboarded server is covered the next morning with no config change. |
| `DAILY_REPORT_IDENTITY_ACCOUNT` | `""` | The DBA account every call in a scheduled run is authorized and audited as. **Required in practice** — see below. |
| `DAILY_REPORT_IDENTITY_CHANNEL` | `dev` | Which channel namespace that account id belongs to (`slack` / `teams` / `dev`). |

The agent service also needs `CHANNELS_BASE_URL` (default
`http://localhost:8003`) to reach the Channels service, which is what
actually posts the message — the Agent holds no Slack token.

**About the identity.** A scheduled run is not privileged and has no bypass:
the Gateway independently re-resolves `DAILY_REPORT_IDENTITY_ACCOUNT`
through the normal `IdentityProvider` and runs the same
authorization/policy/risk pipeline as a live DBA's message, so the digest
can only see what that account's own role is allowed to see, and every call
lands in the audit trail under it. Point it at a real, least-privilege DBA
account. If it names an account the directory doesn't know, nothing breaks
dangerously — every call is simply denied and each server shows up in the
digest as "could not be checked", which is exactly how you'll notice.

**It never takes an action.** A scheduled run is marked read-only: it is
offered no write tool, refuses to submit anything the Gateway's catalog
doesn't confirm is a read (before the request is even built), and never
creates an approval request or posts an approval card. It can report that a
session should probably be killed; it cannot kill one, and there is no
configuration flag that changes that. See
[ARCHITECTURE.md](ARCHITECTURE.md#scheduled-daily-digest-proactive-and-structurally-read-only).

**Reading the digest.** Servers are grouped: `NEEDS ATTENTION` (only
deviations, one block per server), `COULD NOT BE CHECKED` (with the reason),
then a single closing line naming everything that came back clean. The
header always states both numbers ("6 servers swept: 4 checked, 2 could not
be checked") — if servers are consistently landing in the second group,
that's an availability/credential/discovery problem worth chasing, not a
digest problem. A line reading "Numi would have proposed
`database.kill_session` here" means the agent identified a remediation and
was structurally prevented from taking it; it is a recommendation for you,
never something that happened.

**Turning it off / changing the time.** Unset `DAILY_REPORT_SLACK_CHANNEL`
(or change `DAILY_REPORT_HOUR_UTC`) and restart the Agent — the schedule is
read once at startup, like policy and playbooks. Nothing is persisted: there
is no job store, so there is no stale schedule to clean up.

## Enabling alert-triggered investigation

The event-driven sibling of the digest above: point your monitoring system
(Prometheus Alertmanager, Datadog, a cloud provider's own alarms, ...) at
`POST https://<channels-host>/webhooks/alerts`, and Numi investigates the
specific breach it reports instead of waiting for the next scheduled sweep.
**Off by default**, with two independent switches — one per service:

| Variable | Service | Default | What it does |
|---|---|---|---|
| `ALERT_WEBHOOK_SECRET` | channels | `""` (off) | Signs/verifies inbound alert requests. Empty = the webhook route refuses every request outright. |
| `ALERT_WEBHOOK_SLACK_CHANNEL` | agent | `""` (off) | Destination channel id. Empty = the Agent-side handler is a no-op even if a request somehow reaches it. |
| `ALERT_WEBHOOK_IDENTITY_ACCOUNT` | agent | `""` | The DBA account every call in a triggered run is authorized and audited as. **Required in practice** — same reasoning as `DAILY_REPORT_IDENTITY_ACCOUNT` below, and deliberately a *separate* account so the two features can be enabled, disabled, and audited independently. |
| `ALERT_WEBHOOK_IDENTITY_CHANNEL` | agent | `dev` | Which channel namespace that account id belongs to. |
| `ALERT_WEBHOOK_COOLDOWN_SECONDS` | agent | `900` | Suppresses a re-investigation of the same `(server, metric)` pair within this many seconds, so a flapping alert doesn't re-run the investigation (and re-post to the channel) on every firing. `0` disables the cooldown. |

**Sending an alert.** POST JSON with at minimum `{"server": "<id-or-alias>"}`
(matched against `config/servers.yaml` by exact id/alias, case- and
punctuation-insensitive — an unmatched or ambiguous name is reported back as
`"unknown server"`, never guessed). Add whatever of `metric`,
`current_value`, `threshold`, `severity`, `source`, `message` your
monitoring system has — they become the investigation's opening problem
statement, so a specific symptom drives what actually gets checked, unlike
the digest's fixed checklist. An optional `alert_id` deduplicates a retried
delivery. Sign the request: `X-Numi-Alert-Timestamp` (Unix seconds) and
`X-Numi-Alert-Signature: sha256=<hmac>` computed over
`f"{timestamp}.{raw_body}"` with `ALERT_WEBHOOK_SECRET` — see
`channels/alerts/signature.py`.

**Same read-only guarantee as the digest, same mechanism.** A triggered
investigation is marked read-only exactly like a scheduled one — same three
enforcement layers, same inability to create an approval request. See
[ARCHITECTURE.md](ARCHITECTURE.md#alert-triggered-investigation-the-digests-event-driven-sibling).

**Turning it off.** Unset either `ALERT_WEBHOOK_SECRET` (Channels refuses
every request) or `ALERT_WEBHOOK_SLACK_CHANNEL` (the Agent no-ops even on a
request that got through) and restart that service. No job store, no
scheduler, nothing to clean up.

## Investigation memory, self-critique, and model routing

Five related, independently-switchable features, all on by default with a
kill switch each — none require a redeploy beyond restarting the Agent (and
Gateway, for the two Gateway-side switches):

| Variable | Service | Default | What it does |
|---|---|---|---|
| `INVESTIGATION_MEMORY_LOOKBACK` | agent | `3` | How many recent, concluded past investigations on the same server the Agent recalls as background before starting a new one. `0` disables recall. |
| `SELF_CRITIQUE_ENABLED` | agent | `true` | A second LLM opinion reviews a draft conclusion before it's accepted — catches a conclusion that names nothing fabricated and has nothing pending, but still doesn't follow from the evidence. A failed critique call always fails open (the conclusion is still accepted); only a working, negative verdict rejects. |
| `DECISION_EVENT_LOGGING_ENABLED` | gateway | `true` | Persists a handful of decision-quality signals (a rejected conclusion, a cross-provider fallback substitution) as durable, queryable records instead of only structured log lines — see `GET /v1/decision-events/summary` in [API.md](API.md). |
| `LLM_FAST_MODEL` / `LLM_STRONG_MODEL` | agent | `""` (off) | Task-complexity model routing: simple calls (classifying a message) use the fast model, investigation reasoning and self-critique use the strong one. Both empty means no behavior change at all. Never overrides a DBA's explicit `/model` choice, and never disabled by a locked `LLM_PROVIDER` — it only constrains the vendor, tiering still applies within it. |
| `CROSS_SERVER_CORRELATION_ENABLED` | gateway | `true` | Gateway-side kill switch for `GET /v1/investigations/correlate` — "this same symptom happened on N other servers recently," matched by shared playbook + environment, never fuzzy text. |
| `CROSS_SERVER_CORRELATION_LOOKBACK_DAYS` | gateway | `30` | How far back correlation looks. |

All five degrade gracefully on their own: a Gateway hiccup while recalling
memory or logging a decision event never blocks or fails the DBA's turn —
worst case, the enhancement silently didn't happen that time.

## Monitoring what matters

Per spec §42, track (via the OpenTelemetry wiring in
`common/observability.py` and the audit/security-event tables):

- Policy denials and security events (`security_events` table) — alert on
  spikes, they're the leading indicator of probing/misuse.
- Approval latency (time between `APPROVAL_REQUESTED` and
  `APPROVED`/`REJECTED`/`EXPIRED` audit events) — a consistently-expiring
  approval queue means DBAs aren't seeing the approval cards in time.
- Execution failures and timeouts, by tool and by database — a rising
  `EXECUTION_TIMEOUT` rate on one instance is itself an incident signal.
- Rate-limit rejections — sustained hits usually mean either abuse or a
  legitimately-busy incident response that needs a temporary limit bump
  (edit `config/rate_limits.yaml` and redeploy the Gateway).
- `readonly_run_dropped_write_proposal` / `readonly_run_declined_approval_card`
  (structured log, Agent) — the scheduled digest's read-only guard actually
  firing. Not an error, and expected occasionally: it means the model
  proposed an action during an unattended run and was structurally stopped
  (the finding still reaches the digest as a recommendation). Worth watching
  as a *rate*: a sudden rise means either a genuinely deteriorating estate or
  a model increasingly inclined to act on its own.
- `daily_digest_built` (structured log, Agent) — carries `server_count` and
  `failed`. A `failed` count that is persistently non-zero is an
  availability/credential/discovery problem, not a reporting one.
  `daily_digest_publish_failed` means the digest was built but never reached
  the channel — i.e. a silent morning that isn't a quiet one.
- `llm_call_deadline_exceeded` (structured log, Agent) — a single decision
  hit the ~20s hard ceiling; sustained occurrences mean the configured
  provider is degraded/exhausted across its whole fallback chain, not a
  one-off blip. `gemini_model_unavailable_switching` shows which model and
  why (quota vs. capacity) leading up to it.

## Approval queue hygiene

Approvals expire (default 10 minutes; see `ApprovalEngine`'s
`default_ttl_seconds`) and are never resurrected — an expired request must
be re-proposed by the Agent from scratch, which re-runs the full
policy/risk pipeline against current state. This is intentional: stale
context (a blocking chain that resolved itself, a database that's since
gone into maintenance) should not authorize a now-irrelevant action.

## Investigation/audit retention

Per spec §59, retention for audit logs, chat messages, investigations, and
execution results should be configured according to your organization's
compliance requirements; this repo does not hard-code a retention job, but
every table involved (`audit_events`, `security_events`,
`tool_executions`, `investigations`) is a normal table an operational
retention job can prune by `created_at`. Prefer storing hashes/references
over raw sensitive result data where retention windows are long — the
`arguments_hash` field on `audit_events` already follows this pattern.

## Runbook: a tool call is stuck in `APPROVAL_REQUIRED`

1. `GET /v1/approvals/{approval_id}` to check status/expiry.
2. If it's expired, there's nothing to approve anymore — ask the DBA to
   re-issue the request through the Agent.
3. If it's `AWAITING_SECOND_APPROVAL`, a second, *different*, independently
   authorized approver needs to approve — check `config/policy.yaml`'s
   `dual_approval_required` list and the tool's `allowed_roles` to confirm
   who's eligible.

## Runbook: a DBA reports "I approved it but nothing happened"

Check the audit trail for that `approval_id`/`request_id`:
- `APPROVAL_MISMATCH` means the resubmitted action didn't match what was
  approved — the underlying request must have changed between proposal and
  approval (see [THREAT_MODEL.md](THREAT_MODEL.md) threat #5).
- `APPROVAL_INVALID` with "already been used" means it already executed
  once — check `tool_executions` for the prior run rather than re-approving.
