# Policy Model

## Decisions

The Policy Engine (`gateway/domain/policy_engine.py`) returns exactly one
of three decisions for a given `(environment, tool, role)`:

- **ALLOW** — proceed to risk assessment and execution directly.
- **DENY** — refuse. Nothing about the request reaches the Execution Service.
- **REQUIRES_APPROVAL** — an `Approval` must exist and be granted (spec §15)
  before execution proceeds.

**Unlisted combinations always evaluate to `DENY`** (`default_decision: DENY`
in `config/policy.yaml`) — adding a new tool or role without a policy entry
fails closed, not open.

## Configuration

`config/policy.yaml`:

```yaml
default_decision: DENY

environments:
  production:
    database.kill_session:
      DBA_L1: DENY
      DBA_L2: REQUIRES_APPROVAL
      DBA_L3: REQUIRES_APPROVAL
      DBA_MANAGER: REQUIRES_APPROVAL

change_ticket_required:
  production:
    - database.restart_instance

dual_approval_required:
  - database.restart_instance
  - database.failover
```

Business rules live **only** here (and in `ToolDefinition.requires_dual_approval`
as a tool-intrinsic floor) — never in an agent prompt.

## Escalation rules layered on top of the static table

`PolicyEngine.evaluate` can escalate (never de-escalate) a base `ALLOW` to
`REQUIRES_APPROVAL`:

1. **Outside maintenance window** — if the tool has `availability_impact=true`
   and the target database's `maintenance_window` (from
   `config/servers.yaml`, timezone-aware) doesn't currently cover "now."
2. **Current load is critical** — if `current_load_critical=True` is passed
   in (a hook for future integration with a live health check) and the tool
   is a write operation.

`change_ticket_required` is evaluated independently: if a tool requires a
change ticket in this environment and no `change_id` was supplied, the
Gateway raises `CHANGE_TICKET_REQUIRED` regardless of what the base table
said (see `tool_call_handler.py` step 6).

## Risk

`gateway/domain/risk_engine.py::RiskEngine.assess` computes a `RiskAssessment`
independent of policy — policy decides *whether* an action needs approval;
risk decides *how it should be described* (risk level, score, blast radius,
reversibility, reason codes) to the approver and in the audit trail.

Scoring starts from the tool's declared `risk_level` floor and adds points
for: production environment, database criticality, write/availability
impact, irreversibility, and current load — but only for operations that
can actually change state (a pure read never gets penalized for running
against a critical production database, since its blast radius is
identical everywhere).

Blast radius (`common/models/risk.py::BlastRadius`) is looked up per tool
id (`risk_engine.py::_TOOL_BLAST_RADIUS`) and escalated to
`MULTIPLE_OBJECTS` if the caller indicates more than one affected object.

## Approval

See [SECURITY.md](SECURITY.md) rule #10/#11 and
[THREAT_MODEL.md](THREAT_MODEL.md) threat #5 for the approval binding and
expiry model in depth. In short: `ApprovalContext.action_hash()` binds
actor + tool + tool version + target + normalized arguments + environment +
database + risk level; any change to the underlying request invalidates the
approval when re-verified at execution time.

## Maintenance windows

```yaml
maintenance_window:
  timezone: Africa/Lagos
  start: "01:00"
  end: "04:00"
```

`gateway/domain/policy_engine.py::is_within_maintenance_window` handles
windows that wrap midnight and falls back to "no restriction" if a database
has no window configured. **The LLM never declares a maintenance window
active** — this is computed purely from configuration and wall-clock time.

## Rate limiting

`config/rate_limits.yaml` defines three tiers (`read`, `write`, `critical`),
each with five independent budgets (per user, per conversation, per tool,
per database, per environment), enforced in-memory for local dev/tests
(`InMemoryRateLimitBackend`) or via Redis in production
(`RedisRateLimitBackend`). A `WRITE`/`PRIVILEGED` operation, or any
operation requiring approval, is billed against the stricter `critical`
tier.

## Mandatory Safety Gates mapping

A separate external playbook/policy specification this project has been
checked against enumerates 13 numbered "Mandatory Safety Gates" a DBA agent
must satisfy. This project's Gateway pipeline
(`gateway/domain/tool_call_handler.py::ToolCallHandler._handle_inner`,
the sequence documented at the top of that file: Tool Registry → Argument
Validation → Target Validation → Authorization → Rate Limiting → Policy
Engine → Risk Engine → Approval Engine → Execution → Data Minimization →
Audit) was checked gate-by-gate against the real code below, not assumed
from this document's or ARCHITECTURE.md's prose. Two gates turned out to be
only partially enforced — called out honestly rather than papered over.

1. **Validate environment/engine/host/instance/database.** —
   `gateway/domain/target_validation.py::TargetValidator.validate`, called
   unconditionally at step 3 of `_handle_inner`. Resolves the target to
   exactly one registered server (`ServerRegistry.resolve`), validates the
   database against the discovered catalog when one exists, and checks
   required scope fields (`_check_required_fields`) before anything else
   meaningful runs. Engine/platform and environment come from the resolved
   `ServerEntry`, never from the request's own free-text claims. Fully
   enforced, on every call.

2. **Confirm authenticated user/role/target scope.** —
   `gateway/domain/authorization.py::authorize`. Identity is never taken
   from the request; it's the `VerifiedIdentity` already produced by
   `IdentityProvider.resolve_by_external_account` (see gate 13 below for
   what happens when that fails). `authorize` checks DBA-team membership
   (`identity.is_dba()`), MFA (`identity.mfa_satisfied`), the role against
   `tool.allowed_roles`, and the role against the target's own
   `ctx.allowed_roles` (a per-server/per-database override) — all four
   independently, all raising `UNAUTHORIZED`/`AUTHENTICATION_FAILED` on
   failure. Fully enforced, on every call, and re-derived fresh each time
   (nothing here is cached across a conversation turn, per the module's own
   docstring).

3. **Classify production vs. non-production.** — `Environment` (`common/
   models/target.py`) is resolved onto every target via the server registry
   and threaded through every downstream check: `authorize` checks
   `ctx.environment` against `tool.allowed_environments`;
   `PolicyEngine.evaluate` takes `environment` as a required argument and
   keys its decision table by it; `RiskEngine.assess` adds
   `ReasonCode.PRODUCTION`/`NON_PRODUCTION` and score deltas based on it.
   Fully enforced — see also SECURITY.md rule 15.

4. **Collect read-only evidence first.** — **Only partially enforced, and
   only for playbook-driven investigations.** Every step of a matched
   playbook (`agent/playbooks/library.py`) is, by construction, a read-only
   `database.get_*` tool — pinned by
   `tests/unit/test_playbooks.py::test_no_playbook_step_ever_proposes_a_write_tool`
   — so a playbook-driven investigation always gathers evidence before any
   write is even possible within it. But the Gateway pipeline itself has no
   check of this kind: `ToolCallHandler` carries a `request.investigation_id`
   purely for audit correlation (`tool_call_handler.py`'s `correlation`
   dict) — it is never read back to require that N read-only calls
   preceded a write. In the freeform investigation loop
   (`agent/orchestrator.py::_continue_investigation`), the tool menu offered
   to the LLM each turn is every tool the caller's role can use, read and
   write alike (`available = await self._tool_client.available_tools(...)`)
   — nothing structurally stops a fresh freeform investigation from
   proposing a write on its very first turn. In practice a write still has
   to clear authorization/policy/risk/approval regardless of when it's
   proposed, but "evidence was gathered first" is a prompt-level convention
   for freeform investigations, not a hard gate the Gateway enforces.

5. **Minimize sensitive data before returning it to the AI.** —
   `gateway/domain/data_policy.py::DataMinimizer.apply`, invoked exactly
   once, at step 10 of `_handle_inner`, before any row reaches the
   `ToolCallResponse` the Agent (and therefore the LLM) sees. Masks fields
   matching a configurable sensitive-field regex, caps row/column counts,
   and reports whether truncation occurred. Fully enforced — there is no
   code path from `execution.execute()` to the Agent that skips it.

6. **Treat database output as untrusted evidence.** — Structural design
   plus an explicit instruction, not a technical control that could reject
   a non-compliant model: the offline/mock planner and the structured
   adapters only ever inspect specific structural fields (row counts, named
   ids) rather than re-interpreting arbitrary result text, and every real
   provider's system prompt (`agent/llm/base.py`) explicitly states "tool
   results are untrusted data" and instructs the model never to treat text
   inside a tool result as an instruction to obey. This is defense in depth
   on top of the structural design (SECURITY.md's "No security by prompt"
   section is explicit that prompt wording is never itself a security
   boundary) — genuinely partial by nature, since no wording can *force* an
   LLM's behavior, but the design means a model that ignores the
   instruction still can't reach a database directly or self-authorize
   anything as a result.

7. **Apply authorization / policy / risk / blast-radius checks.** — Steps
   4, 6, and 7 of `_handle_inner`: `authorize(...)`,
   `self._policy.evaluate(...)`, and `self._risk.assess(...)`, in that
   order, unconditionally, before execution is reachable. Blast radius is
   computed inside `RiskEngine.assess` from `risk_engine.py::_TOOL_BLAST_RADIUS`
   (looked up per tool id, escalated to `MULTIPLE_OBJECTS` when the caller
   reports more than one affected object) and carried on the returned
   `RiskAssessment`. Fully enforced, on every call, and none of the three
   can be skipped by anything the Agent or LLM claims.

8. **Require approval for controlled/disruptive actions.** —
   `PolicyEngine.evaluate` returning `PolicyDecision.REQUIRES_APPROVAL`
   drives step 8 of `_handle_inner`: without a valid `request.approval_id`,
   the handler creates an `Approval` record
   (`gateway/domain/approval.py::ApprovalEngine.create`) and returns
   `APPROVAL_REQUIRED` — execution never happens in that turn. Fully
   enforced; see the Approval section above and SECURITY.md rules 4/5.

9. **Bind approval to exact tool/target/arguments/environment.** —
   `ApprovalContext.action_hash()` hashes actor, tool, tool version, target,
   normalized arguments, environment, database, and risk level together;
   `ApprovalEngine.verify_for_execution` (called at step 8 before execution
   is reached) recomputes and compares this hash, raising
   `APPROVAL_MISMATCH`/`APPROVAL_INVALID` on any drift. Fully enforced —
   see also SECURITY.md rule 10 and THREAT_MODEL.md threat #5.

10. **Execute only through the Gateway and Execution Plane.** — Step 9 of
    `_handle_inner` is the only place `ExecutionClient.execute()` is called,
    and it is only reachable once ALLOW or a verified `REQUIRES_APPROVAL`
    has been reached. Structurally backstopped, not just procedurally:
    `agent/` has no database driver dependency at all, and
    `CredentialProvider` (`execution/credentials/provider.py`) is only
    importable/reachable from inside `execution/` — see SECURITY.md rules
    1/2/6. Fully enforced.

11. **Verify the result independently.** — **Enforced, but deliberately
    narrow in scope today.** `agent/orchestrator.py`'s
    `_VERIFICATION_TOOLS_BY_WRITE_TOOL` maps a write tool to the read-only
    tool(s) that can confirm its real-world effect; `InvestigationState.
    pending_verification` is set the moment a mapped write executes, and
    `_finalize_conclude` refuses to let the model conclude while it's still
    set — the same self-correction shape as the existing conclusion-
    grounding check. As of this writing that table covers exactly
    `kill_session`/`cancel_query` (verified against `get_blocking_sessions`/
    `get_sessions`/`get_running_queries`) — the case with an unambiguous,
    single-call re-check. A write like `update_statistics`/`create_index`
    has no entry and therefore no independent re-check enforced today
    (confirming it *helped* needs a follow-up performance observation, not
    one more cheap tool call) — extending coverage is one more table entry,
    not a new mechanism, but until that entry exists for a given write tool,
    this gate does not apply to it.

12. **Record investigation/approval/execution/verification events.** —
    **Enforced for investigation/approval/execution; not as its own
    distinct event type for verification.** `AuditLog.record`
    (`gateway/domain/audit.py`) is called from every branch of
    `_handle_inner`/`handle` (`TOOL_CALL_EXECUTED`, `TOOL_CALL_DENIED`,
    `TOOL_CALL_FAILED`, `APPROVAL_REQUESTED`), and `ApprovalEngine`
    separately records `CREATED`/`APPROVER_1_APPROVED`/
    `APPROVER_2_APPROVED`/`APPROVED`/`REJECTED`/`EXPIRED`/`EXECUTED`
    events. `handle`'s single `except NumiError` block (around
    `tool_call_handler.py` line 109) picks `TOOL_CALL_DENIED` vs.
    `TOOL_CALL_FAILED` from the `FailureCode`: `EXECUTION_FAILED`/
    `EXECUTION_TIMEOUT`/`DATABASE_UNAVAILABLE` — the only codes that can be
    raised *after* `self._execution.execute(...)` actually ran (step 9) —
    audit as `TOOL_CALL_FAILED`; every other code is a pre-execution
    refusal and stays `TOOL_CALL_DENIED`. `request.investigation_id` is
    threaded into every `TOOL_CALL_EXECUTED`/`TOOL_CALL_DENIED`/
    `TOOL_CALL_FAILED` record's correlation ids, so an investigation's tool
    calls are traceable in the audit log. But the gate 11 verification re-check is just another tool
    call from the Gateway's point of view — audited only as an ordinary
    `TOOL_CALL_EXECUTED` event, indistinguishable from any other read. The
    actual verification verdict (`InvestigationState.last_verification`,
    `"RESOLVED"`/`"UNRESOLVED"`) lives in `agent/context_manager.py`'s
    `InvestigationState`, which that module's own docstring says is "kept
    entirely in the Agent process" — not a durable, distinctly-typed audit
    record in the Gateway's append-only log. If that in-process state is
    lost (process restart, no persistence layer), the verification verdict
    itself is not independently reconstructable from the audit trail —
    only the fact that some read tool ran, same as any other read.

13. **Fail closed when identity/target/policy/approval/verification is
    unavailable.** — Identity: `gateway/api/deps.py::resolve_identity`
    raises `HTTPException(401)` the moment
    `IdentityProvider.resolve_by_external_account` returns `None`; nothing
    downstream treats a missing identity as anonymous-allow. Target:
    `TargetValidator.validate` raises `INVALID_TARGET` on an unresolvable or
    ambiguous server. Policy: `PolicyEngine`'s `default_decision` is `DENY`
    (`config/policy.yaml`) — an unlisted `(environment, tool, role)`
    combination denies, not allows. Approval: a `REQUIRES_APPROVAL` action
    with no `approval_id`, or one that fails `verify_for_execution`'s hash
    check, never reaches execution. Verification: per gate 11's actual
    scope, a mapped write with `pending_verification` still set blocks
    `Conclude` from succeeding — the loop is forced to keep checking or run
    out of turn budget, never to silently report success. All five fail
    closed within their documented scope; gate 11/12's narrower coverage
    (above) means "verification unavailable" fails closed only for the
    specific write tools currently in `_VERIFICATION_TOOLS_BY_WRITE_TOOL`.

## LLM selection

Which LLM the agent uses is a *product* choice, not a security control —
`agent/llm/registry.py::LLMRegistry` resolves it, and it never touches
authorization, policy, risk, or approval (all of which live in the Gateway
and are identical regardless of provider).

**Configuration** (`Settings`, see `.env.example`):

| Setting | Effect |
|---|---|
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GEMINI_API_KEY` / `DEEPSEEK_API_KEY` | A provider becomes *selectable* the moment its key is present. |
| `LLM_PROVIDER` = `""` / `auto` | Auto-pick the first configured provider (order: anthropic, openai, gemini, deepseek). DBAs may switch per-conversation. |
| `LLM_PROVIDER` = `<name>` | Force that provider **and disable** per-conversation switching. |
| `LLM_PROVIDER` = `mock` | Force the deterministic offline planner. |
| `LLM_MODEL` | Default model for the resolved provider (`""` = the provider's own default). |
| `ALLOW_USER_MODEL_SELECTION` = `false` | Keep the auto default but disable the `/model` command. |

With no key set at all, `effective_default_llm()` returns `("mock", …)` —
and `validate_for_production()` refuses to start the process in that state
when `NUMI_ENV=production`.

**Resilience.** A provider outage or malformed completion never crashes the
chat or hangs it indefinitely — it degrades to a clear message. Each
`decide_next_action`/`extract_intent` call gets one same-model retry
(`StructuredLLMProvider._CALL_RETRIES`), and the Gemini provider
additionally switches to the next model in a ranked fallback chain on a
quota or capacity error specific to the current model (cooling that model
down rather than blacklisting it permanently — see `_cooldown_seconds`,
which prefers the API's own `retryDelay` hint). Whatever combination of
retries and model switches happens underneath, one call is never worse than
`StructuredLLMProvider._OVERALL_DEADLINE_SECONDS` (20s) late — see
[ARCHITECTURE.md](ARCHITECTURE.md#latency-ceiling-on-a-single-llm-decision).

**Cross-provider fallback.** When a provider is unreachable *as a whole*
(every retry and every one of its own model fallbacks exhausted — verified
live as a free-tier daily quota exhausted across all of Gemini's models),
the same call is re-issued against the next configured provider from
`configured_llm_providers()` before the DBA is told anything failed. This
happens even when `LLM_PROVIDER` is locked or a DBA picked a provider with
`/model` — an answer beats "try again later" while another configured key
sits unused — so the lock is best read as "the provider to use", not "the
only provider that may ever run". It is never silent: a reply produced by a
substituted vendor says so plainly. The escalation shares the same 20s
ceiling rather than adding one per provider — see
[ARCHITECTURE.md](ARCHITECTURE.md#cross-provider-llm-fallback-when-a-whole-vendor-is-down-not-just-a-model)
for the budget arithmetic.

**Task-complexity routing** (`LLM_FAST_MODEL`/`LLM_STRONG_MODEL`, both `""`
by default) is an orthogonal axis on top of the above, not a replacement for
it: a simple call (classifying a message) can route to a cheaper/faster
model while investigation reasoning and self-critique route to a stronger
one, within whatever provider/model the selection above already resolved.
It only ever fills in when the DBA has made no explicit `/model` choice —
same as everything else here, purely a routing default, never a policy or
authorization decision, and never disabled by a locked `LLM_PROVIDER` (that
only constrains the vendor; tiering still applies within it). See
[OPERATIONS.md](OPERATIONS.md) for both settings.

**Per-conversation selection** (chat commands, `agent/orchestrator.py`):

- `/models` — lists each configured provider and the models its key can
  actually use (a live `models.list()` call).
- `/model` — shows the current selection and its source (conversation vs
  deployment default).
- `/model <provider> <model>` — switches for this conversation. Rejected if
  the provider isn't configured, the model isn't in that key's list, or
  selection is locked/disabled.

The selection is stored on `ConversationState` (in Agent process memory)
and is never an input to any Gateway decision.

## "Who can approve this?" — answered honestly, without exposing policy data

A DBA can ask this in plain language (`meta_command="approvers"` on
`IntentExtraction`, handled by `AgentOrchestrator._handle_approvers_command`,
also reachable as the literal `/approvers` command). The reply is a static,
general description of the model above (role tiers, environment-based
escalation, no self-approval, dual approval for critical actions) — **never**
a read of `config/policy.yaml` itself. Two reasons: that file is loaded only
by the Gateway process (see "Configuration" above), and even if the Agent
could read it, the Agent has no way to know a specific role's actual
per-tool/per-environment decision without duplicating the Policy Engine's
own logic outside the Gateway, which this architecture deliberately never
does ("Business rules live only here... never in an agent prompt"). When a
DBA's own request actually needs approval, the real, request-specific
requirement is still always shown at that moment via the normal
`APPROVAL_REQUIRED` flow — this command is a general explainer, not a
lookup.
