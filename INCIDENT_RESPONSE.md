# Incident Response

This document is about incidents *in Numi itself* — a suspected security
issue, a misbehaving approval, or the Gateway making a wrong call. For
using Numi to respond to a *database* incident, see the worked example in
[README.md](README.md) and [ARCHITECTURE.md](ARCHITECTURE.md).

## Suspected unauthorized action

1. Pull the `audit_events` and `security_events` rows for the
   `request_id`/`approval_id`/`conversation_id` in question
   (`GET /v1/audit/{id}` or query the control-plane database directly).
2. Check `actor_subject_id` against your enterprise IdP — this is the
   *verified* subject id the Gateway resolved, independent of anything the
   Agent or a chat message claimed.
3. Check `policy_decision` and `risk` on the audit record — was this
   `ALLOW` (should it have been `REQUIRES_APPROVAL`?) or was an approval
   present? If an approval was present, pull its `action_hash` and compare
   against what a fresh `ApprovalContext.action_hash()` would produce for
   the claimed original request — a mismatch here means the binding worked
   as designed and the *execution* was correctly denied; if execution
   somehow succeeded despite a hash mismatch, that is a critical bug in
   `gateway/domain/approval.py` and should be treated as a sev-1.
4. If the action executed and shouldn't have: rotate `SERVICE_JWT_SECRET`
   immediately (invalidates all in-flight service tokens across
   Channels/Agent/Gateway/Execution) and audit `config/policy.yaml` /
   `config/servers.yaml` for unauthorized edits (these are the only two
   files that can widen what's possible without a code change).

## Suspected credential compromise

Database credentials never leave the Execution Service and are fetched
just-in-time via `CredentialProvider` — there is no cache to purge in the
Agent or Gateway. Rotate the credential at the secrets manager
(Vault/AWS/Azure/GCP); the next execution automatically fetches the new
one. If using `local_dev`/`SECRETS_PROVIDER=local_dev` in anything other
than local development, that is itself the incident — it should never be
configured that way outside a developer's machine.

## Approval queue looks wrong / approvals not expiring

Check `ApprovalRecord.expires_at` directly in the control-plane database.
Expiry is computed at creation time (`created_at + default_ttl_seconds`,
default 600s) and re-checked on every `decide`/`verify_for_execution` call
— there is no background job that "sweeps" expired approvals, so a record
can sit in `PENDING`/`APPROVED` status in the table after its `expires_at`
has passed until the next time it's touched. This is expected and benign
(any attempt to use it will still be correctly denied) but can look
confusing when inspecting the table directly.

## Agent producing nonsensical or unsafe-sounding proposals

Remember: **a proposal is not an action.** Every `ProposeToolCall` the
Agent emits still goes through the full Gateway pipeline. If the Agent
proposes something that shouldn't even be *possible* for the requesting
role/environment, and the Gateway correctly denies it, that is the system
working as designed, not an incident — the fix (if any) is a better system
prompt or planner logic, not a security response. If the Gateway
*executes* something that policy should have denied, treat that as a
sev-1 in the Gateway's `policy_engine.py`/`tool_call_handler.py`, not the
Agent.

**Checking whether this is a one-off or a pattern.** `GET /v1/decision-
events/summary?since_hours=24` on the Gateway returns durable, queryable
counts of things like `conclusion_rejected_self_critique`,
`conclusion_rejected_ungrounded_identifiers`,
`conclusion_rejected_pending_verification`,
`llm_cross_provider_fallback_used`, and `self_critique_call_failed`
grouped by type — these used to exist only as log lines. A spike in
`conclusion_rejected_*` counts means the Agent's own conclusions are
regularly failing internal review (the system catching itself, not
necessarily an incident); a spike in `llm_cross_provider_fallback_used`
means a configured LLM vendor is unreliable, not that the Agent is
misbehaving. Both `SELF_CRITIQUE_ENABLED` and `DECISION_EVENT_LOGGING_
ENABLED` (env vars, default `true`) can be turned off without a redeploy
if either pass itself is suspected of misbehaving.

## Rollback

Every controlled write tool declares `reversible: true/false`
(`gateway/domain/tool_catalog.py`). For reversible operations
(`cancel_query`, `update_statistics`, `create_index`, `rebuild_index`),
the corresponding inverse/no-op-safe action is itself just another typed
tool call, subject to the same pipeline. For irreversible operations
(`kill_session`, `modify_configuration`, `restart_instance`, `failover`),
there is no automated rollback — this is exactly why they carry higher risk
scores and (for the last two) mandatory dual approval; recovery is a
database-operational procedure outside this codebase's scope.

## Escalation

This repository does not encode your organization's paging/escalation
policy. Wire `security_events` (severity `WARNING`/`CRITICAL`) into your
alerting pipeline (spec §42) so a burst of `UNAUTHORIZED`,
`APPROVAL_MISMATCH`, or `SEPARATION_OF_DUTIES_VIOLATION` events pages
someone rather than sitting unnoticed in a database table.
