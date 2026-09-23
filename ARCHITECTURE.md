# Architecture

## Topology

```
DBA
 │  Slack / Teams
 ▼
Channel Adapter (channels/)
 │  verify signature/token, resolve identity (UX check), forward
 ▼
AI DBA Agent (agent/)
 │  LLM-backed planning; no DB credential; every action is a "proposal"
 ▼  Tool Request (untrusted)
DBA CONTROL GATEWAY (gateway/)
 │  Registry → Args → Target → AuthZ → RateLimit → Policy → Risk → Approval
 ▼  Authorized, scoped execution instruction
Execution Service (execution/)
 │  the ONLY component with database credentials/network access
 ├─→ SQLServerAdapter
 ├─→ PostgreSQLAdapter
 └─→ MySQLAdapter (MySQL + MariaDB)
        │
        ▼
     DATABASE
```

Every arrow above is a real network hop (HTTP + a signed, audience-scoped
service token — `numi.common.service_auth`), not a function call inside one
process. That's deliberate: the trust boundary is the *service* boundary,
not a class boundary a bug could accidentally erase.

## Why four services and not one

| If merged into... | What breaks |
|---|---|
| Agent + Gateway | The LLM process could reach the authorization/policy/approval code paths directly — no network hop to intercept, audit, or rate-limit independently of the model. |
| Gateway + Execution | The Gateway (which parses arbitrary-ish inputs, evaluates policy, and is closer to attacker-influenced data) would hold database credentials. |
| Everything | No independent audit trail, no way to scale/patch/credential each concern separately, no way to run the Agent in a lower-trust network zone than the Execution Service. |

## Request lifecycle (a tool call)

1. **Channel Adapter** (`channels/`) verifies the inbound webhook
   (`channels/slack/signature.py`, `channels/teams/auth.py`), independently
   resolves the sender's enterprise identity via `IdentityProvider`
   (UX-level convenience check — see below), and forwards
   `{channel, channel_account_id, message}` to the Agent over HTTP with a
   signed service token.
2. **Agent** (`agent/orchestrator.py`) resolves an `LLMProvider` for the
   conversation via `agent/llm/registry.py` (the DBA's `/model` choice, or
   the configured default — Anthropic / OpenAI / Gemini / DeepSeek, or the
   deterministic offline planner when no key is set) and, in a bounded
   loop, asks it to classify intent and decide the next investigative step
   — always as a *typed* `AgentAction` (`agent/planner/actions.py`), never
   free text. A `ProposeToolCall` becomes a `ToolCallRequest` sent to the
   Gateway via `agent/tool_client.py`. The choice of model has no bearing
   on the security boundary: every proposal, from any provider, goes
   through the identical Gateway pipeline.
3. **Gateway** (`gateway/domain/tool_call_handler.py`) runs the full
   pipeline: tool registry lookup → argument schema validation → target
   parsing/enrichment → server resolution + catalog validation → **independent identity
   re-resolution** (see below) → rate limiting → policy evaluation → risk
   assessment → approval creation-or-verification → dispatch to the
   Execution Service → data minimization → audit.
4. **Execution Service** (`execution/service.py`) is hit only by the
   Gateway, with a service token scoped to its own audience. It resolves a
   credential via `CredentialProvider`, opens a connection, and dispatches
   to one typed `DatabaseAdapter` method. It returns raw (unmasked) rows —
   masking is the Gateway's job, so there's exactly one place that decision
   is made.
5. Gateway applies `DataMinimizer`, records an `AuditEventRecord`, and
   returns a structured `ToolCallResponse` (EXECUTED / APPROVAL_REQUIRED /
   DENIED / FAILED) — never a stack trace, never raw DB error text.

   **DENIED vs. FAILED** (`tool_call_handler.py::handle`'s single
   `except NumiError`, ~line 109): DENIED means the Gateway refused
   *before* dispatching to the Execution Service — target/auth/rate-limit/
   policy/risk/approval all raise `NumiError` codes that stay DENIED
   (`INVALID_TARGET`, `UNAUTHORIZED`, `POLICY_DENIED`, `APPROVAL_*`,
   `RATE_LIMITED`, etc.). FAILED means step 9 actually dispatched to the
   Execution Service and the attempt itself didn't succeed —
   `EXECUTION_FAILED`, `EXECUTION_TIMEOUT`, or `DATABASE_UNAVAILABLE`,
   the only codes `_handle_inner` can raise *after* `self._execution.execute(...)`
   returns `success=False`. This distinction is what makes a playbook's "one
   failed diagnostic shouldn't abort the rest of the investigation"
   guarantee actually hold: `agent/orchestrator.py::_submit_and_relay`
   treats DENIED as a policy fact that ends the turn, but records FAILED as
   evidence and lets the investigation continue (verified live: a genuine
   `database.get_error_logs` adapter failure was previously misreported as
   DENIED and aborted a `comprehensive_summary` playbook run outright).
6. Agent relays the result (or an approval card) back through the Channel
   Adapter to the human.

## The no-raw-error invariant also covers discovery, not just tool calls

The "never a stack trace, never raw DB error text" rule above is about the
tool-call pipeline specifically; `/discover` (`gateway/domain/discovery.py`'s
`DiscoveryOrchestrator.refresh_all`/`refresh_server`, and the single-server
`POST /v1/catalog/refresh/{id}` route in `gateway/api/routers/catalog.py`)
is a separate path that talks to the Execution Service and can fail the same
way (an unreachable Execution Service, a 500, a timeout) — it needs the same
invariant applied on purpose, not inherited for free. A raw `httpx` exception
bakes in the request URL and, for an `HTTPStatusError`, an MDN documentation
link; `discovery.py`'s `clean_discovery_error()` maps the exception types
that matter to short DBA-facing text (status/unreachable/timeout/generic
fallback) and is the one place both call sites go through, mirroring how
`ExecutionService.execute`'s `except Exception` branch already logs the real
exception and returns a clean, generic message. The raw exception is always
still logged server-side (structlog `error_type` + `error`) — only the
DBA-facing text is generic.

**The same invariant, one layer deeper.** `clean_discovery_error` covers
the Gateway-to-Execution-Service boundary; `execution/discovery/engine.py`'s
`run_discovery` is the dispatcher one layer further in, where each
platform's discoverer actually opens a connection to the real target.
Verified live: asking about a genuinely unreachable dev server crashed
this with an unhandled `psycopg.OperationalError`, surfacing at the API
layer as a raw 500 with a driver traceback — inconsistent with
`ExecutionService.execute`'s own established posture for the identical
class of failure (log the real exception server-side, return a clean
result). `run_discovery` now catches any exception each platform's
discoverer raises and returns an empty `ServerCatalog` carrying a plain
warning instead, so a target being offline degrades exactly like every
other "couldn't reach it" case in this system already does — the
individual discoverer methods still deliberately reraise on a fatal
connection failure (so their own cleanup can run; see
`MySQLDiscoverer`'s own test for why), the catch belongs at the one shared
dispatch point every platform funnels through, not duplicated per engine.

## Investigation loop: freeform vs. playbook-driven

`orchestrator.py::_run_investigation_loop` bounds every investigation to
`_MAX_INVESTIGATION_TURNS` (6) steps so a confused model can't loop forever.
It also bounds a narrower failure mode independently of the turn cap:
verified live, a model can get stuck restating the same finding as one
`record_observation` after another instead of ever calling `conclude`, even
once a plain, complete answer ("no replica configured") was already clear.
`_MAX_CONSECUTIVE_RECORD_OBSERVATIONS` (2, reset by any other action) stops
asking once that pattern is clearly stuck, using whatever evidence already
exists rather than waiting out the rest of the turn budget on further calls
that were never going to conclude either.

Within the turn/observation bounds, a step is decided one of two ways:

- **Freeform** (the default, and the only mode before playbooks existed):
  every turn calls `LLMProvider.decide_next_action` — the model picks the
  next tool from the ones it was offered, or concludes.
- **Playbook-driven**: on a new investigation, `agent.playbooks.library
  .match_playbook` runs a deterministic, zero-LLM-call keyword match
  against the problem text. For 14 known scenarios (slow queries, high CPU,
  high memory, blocking, deadlocks, connection saturation, replication lag,
  backup health, storage capacity, transaction log reuse, error logs,
  a comprehensive single-server summary, general health, configuration
  tuning review), this picks a fixed, named sequence of
  read-only diagnostic calls. Each step of a matched playbook is submitted
  directly — **no LLM call in between** — and once the sequence completes,
  the LLM is called again to interpret everything gathered (nudged by the
  playbook's own conclusion guidance). Unmatched problem text runs fully
  freeform, unchanged.

  Reviewed against Xata's (a competing, now-archived, Postgres-only DBA
  agent) shipped playbook prompts: two gaps were worth adopting. First, its
  slow-query/high-CPU prompts explicitly exclude the engine's own
  introspection/system-catalog queries from being blamed as "the" hot
  query — `slow_queries` and `high_cpu`'s `conclusion_guidance` now carry
  the equivalent, engine-general framing. Second, its `tuneSettings`
  playbook has no analog here — `configuration_review` (get_configuration +
  get_health only) fills that gap, scoped honestly to rule-of-thumb
  misconfiguration flags rather than true capacity-based sizing, since
  Numi's server registry has no instance-class/hardware-sizing data to
  size against.

  Checked against a separate external playbook specification: the original
  combined `storage` playbook (disk/capacity *and* transaction log growth,
  3 steps, one shared trigger list) never checked replication lag or backup
  status — two of the most common real reasons a transaction log can't
  reuse space. Split into `storage` (disk/capacity only — `get_storage`,
  `get_health`) and a new `transaction_log` playbook (`get_transaction_log`,
  `get_replication_status`, `get_backup_status`, `get_health`), each with
  its own trigger list and a `conclusion_guidance` that matches its actual
  hypothesis set — `transaction_log`'s specifically asks the model to name
  which concrete reuse blocker (active/long transaction, replication lag,
  failed/stalled log backup, log shipping/mirroring/snapshots, or
  maintenance) the evidence supports, not just report current usage.

  A further pass against that same external playbook specification deepened
  the remaining 7 playbooks that hadn't had this treatment yet: `deadlocks`,
  `blocking`, `high_memory`, `replication`, `backups`, `errors`, and
  `general_health`. Each now asks the model to test explicit hypotheses
  (root-cause categories for deadlocks and blocking; pressure-source
  categories for high memory; risk/exposure dimensions beyond the raw
  number for replication lag; a calculated recovery-point gap and explicit
  severity for backups; a finding taxonomy plus cross-checking against
  other gathered signals for error logs; an explicit status word for
  general health) rather than reporting raw findings. Two steps were added
  where a genuinely missing, already-registered diagnostic closed a real
  gap: `deadlocks` now also calls `get_sessions` (checking for long-running
  transactions contributing to the lock cycle), and `errors` now also
  calls `get_blocking_sessions` and `get_deadlocks` (so a lock-timeout or
  deadlock-flavored log entry can actually be correlated against current
  evidence instead of only the raw log text). A step was deliberately
  *not* added to `replication` for the analogous "check storage for
  retained WAL" case — `replication` already sat at 3 steps, and a 4th
  would have pushed it to the exact turn-count boundary where
  `_MAX_CONSECUTIVE_RECORD_OBSERVATIONS`'s check-before-call ordering (see
  the loop above) stops catching a stuck model early and lets it burn the
  full `_MAX_INVESTIGATION_TURNS` cap instead — the live-verified regression
  `test_stuck_observation_loop.py::test_the_replication_playbook_no_longer_
  burns_the_full_turn_cap` pins exactly this, and every playbook's step
  count is now asserted to leave at least one turn free
  (`test_deepened_playbooks_stay_within_the_shared_turn_budget`). No new
  diagnostic tool was invented for any of the 7 — every addition is an
  existing, already-registered read tool from `gateway/domain/tool_catalog
  .py`. As with `configuration_review`'s own scoping note, this round is
  honest about what the system still can't do: `high_memory`'s guidance
  now explicitly says host/container-level memory isn't something this
  system can directly measure (no host/OS-level diagnostic tool exists
  here) rather than claiming to assess it, and none of the 7 claim any
  capacity forecasting or trend-over-time analysis — Numi has no
  historical/time-series data store, so every playbook's conclusion is
  still built only from what its own diagnostic calls returned in this one
  investigation.

  **`comprehensive_summary`: a broad, single-server sweep.** The other 13
  playbooks are each triggered by one
  specific symptom (deadlock, high CPU, ...); `comprehensive_summary` is
  the odd one out — a DBA asks for it directly ("comprehensive health
  check", "daily summary", "full report", ...) to sweep this one server's
  state across every dimension the other playbooks check individually, in
  one shot. It is also the building block the scheduled, multi-server
  morning digest calls once per server — that scheduling and multi-server
  orchestration is no longer deferred, but it lives entirely outside this
  playbook (see "Scheduled daily digest" below); the playbook itself is
  still strictly one server per invocation. Two honesty notes, both stated in the playbook's own
  `description` and `conclusion_guidance` (`agent.playbooks.library`, not
  just here): (1) it reports only the server's *current* state — there is
  no historical data store anywhere in this system, so it cannot show a
  trend or delta against yesterday; and (2) it has exactly 5 steps, not one
  per instance-wide read tool. Every investigation — playbook or freeform —
  shares one turn budget, `orchestrator._MAX_INVESTIGATION_TURNS` (6), and
  (per the same invariant the paragraph above pins for the other 7 deepened
  playbooks, `test_deepened_playbooks_stay_within_the_shared_turn_budget`)
  every playbook in this library must leave at least one turn free for the
  model's own unrestricted concluding call: verified live, a 12-step draft
  of this playbook (one call per instance-wide read tool) silently stopped
  submitting steps once the shared budget was spent, with most never
  running and no warning to the DBA, and even a 6-step draft — exactly
  consuming the budget — would have violated that shared invariant. Rather
  than ship either, its step list was cut to the 5 that span the widest
  practical breadth within the existing budget — availability, resource
  pressure, workload/blocking, protection/backups, storage capacity, and
  logs. Configuration was the category dropped (not blocking, backups,
  storage, or logs) because it already has its own dedicated
  `configuration_review` playbook; its `conclusion_guidance` explicitly
  forbids speaking to a category (replication/HA or configuration) it has
  no evidence for, rather than guessing, and instead points the DBA at the
  dedicated `replication`/`configuration_review` playbooks. Raising the
  shared cap, or giving this one playbook a larger budget of its own, is an
  `orchestrator.py` change and out of scope for this addition.
  Distinct from `general_health` (an existing, lighter 4-step pulse check
  for "how's it doing" phrasing) by design: different, non-overlapping
  trigger phrases, and ordered *before* `general_health` in the `PLAYBOOKS`
  tuple specifically because some of its own triggers (e.g. "comprehensive
  health check") contain `general_health`'s "health check" trigger as a
  substring — placed after it, `general_health`'s broader trigger would
  have shadowed every `comprehensive_summary` phrasing that happens to
  contain "health check" (`match_playbook` is first-match-wins); placed
  before it, `general_health`'s own narrower triggers ("how is", "how's",
  "overall status", ...) still route correctly to `general_health`, since
  none of them appear inside any `comprehensive_summary` trigger.

**A playbook's fixed steps are a floor, not a ceiling.** That first
post-playbook call is *not* restricted to `conclude` — `_next_playbook_
action` returning `None` (steps exhausted) simply makes the loop fall
through to the exact same `decide_next_action` call the freeform path
uses, with the same full, unrestricted `available_tool_ids` and the same
`tool_requirements`/`tool_allowed_arguments` plumbing. `_problem_statement_
for_llm` tells the model both directions explicitly: conclude now if the
evidence already suffices, but if it doesn't, propose one or more further
read-only diagnostic tool calls — not limited to this playbook's own
steps — before concluding. Each such extra call goes through
`_submit_and_relay` exactly like any other freeform proposal (including
argument-stripping and Gateway-DENIED self-correction) and is folded into
the same transcript/evidence the eventual conclusion is built from. So "the
LLM is asked only once" is only true for a playbook whose evidence was
already sufficient — verified by a scripted-LLM test
(`tests/unit/test_playbook_freeform_extension.py`) that drives the
blocking playbook's 4 fixed steps to completion, then has the model
propose one further diagnostic outside those 4 steps before concluding,
confirming the extra call is actually submitted and its result reflected
in the final report. Crucially, this never grants extra turns: every
playbook step and every freeform extension increments the exact same
`investigation.turn_count` against the exact same `_MAX_INVESTIGATION_
TURNS`, so a long playbook simply leaves fewer freeform turns available
afterward, and a model that never converges still terminates via the
existing turn-cap/final-chance-to-conclude fallback above — pinned by
`tests/unit/test_turn_budget_playbook_plus_freeform.py`.

Why this exists: freeform investigation was already *capable* of running
any read-only tool in any order and reaching a correct answer — a playbook
adds no new capability. What it fixes is that, for a *known* scenario type,
the freeform loop had no fixed shape or stopping point: verified live, one
real investigation ran several unrelated diagnostics after it already had
its answer and exhausted the turn budget without concluding. A playbook is
a named, reviewable, deterministic answer to "what do we check, in what
order, for this kind of problem" — and because the sequence is fixed in
advance, it also means fewer LLM round-trips per investigation (each one
a chance for a malformed completion or added latency), which matters under
the per-decision latency ceiling below. A playbook only ever pre-selects
*which* read-only diagnostics to run — a recommended remediation, or any
write, still goes through the normal LLM-proposes / Gateway-approves flow
exactly like a freeform investigation's.

**Grounding the conclusion.** The Pydantic/discriminated-union validation
that gates every `AgentAction` (`agent.planner.actions.agent_action_adapter`)
only ever checks an action's *shape* — a `Conclude`'s `summary`/
`likely_root_cause`/`recommendation` are free text with nothing stopping the
model from stating something that never happened. Verified live: a real
conclusion named three CamelCase-looking table names that don't exist in
the database at all, instead of the real ones its own tool call had
actually returned. `orchestrator._ungrounded_identifiers` checks any such
name against everything the investigation actually gathered (transcript,
evidence, and the DBA's own problem statement — so the DBA's own
terminology is never mistaken for a hallucination) and, on a miss, rejects
the conclusion and gives the model one more bounded try — the same
self-correction pattern used for a fixable Gateway denial above, capped by
the same turn count as everything else.

## Scheduled daily digest: proactive, and structurally read-only

`comprehensive_summary` was always written as the building block of a
scheduled morning report — one server per invocation, scheduling deferred.
`agent/scheduled_report.py` is that deferred half: once a day at
`DAILY_REPORT_HOUR_UTC`, run that playbook against every registered server
and post ONE combined digest to `DAILY_REPORT_SLACK_CHANNEL`.

**The constraint the whole feature is built inside: proactive output is text
and a recommendation, never an action.** An investigation a human started
turn-by-turn can legitimately propose a remediation and route it through the
normal LLM-proposes / Gateway-approves flow (spec §7, §37) — a DBA is right
there, reading the approval card, with all the context that produced it. An
investigation nobody started has none of that. So a scheduled run must be
able to say "session 13400 should probably be killed" and must not be able
to kill it, and — just as important — must not leave an approval card in a
channel for someone to rubber-stamp at 6am with no context.

**Where that's enforced, and why not in the scheduler.** The flag is
`InvestigationState.read_only`, set only by
`orchestrator.run_comprehensive_summary`, and everything that acts on it
lives next to the code that submits tool calls rather than in
`scheduled_report.py`. A guarantee that only holds if the caller remembers
to ask for it is not a guarantee. Three layers, each independently tested
(`tests/unit/test_scheduled_digest_never_writes.py`) with every layer above
it assumed broken — because a defense-in-depth layer tested only in
combination is one whose silent failure is invisible:

1. **The menu.** `_continue_investigation` filters `available_tools` to
   `OperationType.READ` on a read-only run, so the model is never offered a
   write. `StructuredLLMProvider.decide_next_action` already refuses a
   `ProposeToolCall` naming a tool outside `available_tool_ids`, so in the
   ordinary case the write never even becomes a proposal the loop sees.
2. **The gate.** `_submit_and_relay` refuses anything not *confirmed* READ
   by the Gateway's own live catalog (`_is_confirmed_read_tool`), **before**
   a `ToolCallRequest` is constructed. Placement is the point: a write that
   reaches the Gateway has already had a policy decision made about it, and
   that decision can be APPROVAL_REQUIRED — which creates a real approval
   record with a real TTL. "Never executes a write" has to mean the write
   never left the Agent process, not that something further down stopped it.
   The predicate fails closed: an unknown tool_id, or no catalog, is refused
   rather than waved through on a `database.get_*` naming convention — the
   cost of being wrong is asymmetric (a missing line in a report versus the
   entire guarantee, in the one context where nobody is watching).
   Layer 1 depends on a provider implementation behaving and on prompt
   wording; this layer depends on nothing but local control flow, which is
   why both exist.
3. **The approval refusal.** Even for a READ tool a deployment's
   `policy.yaml` puts behind approval, a read-only run never stores a
   `PendingApproval` or returns an `ApprovalCard`. The Gateway's own
   approval record is deliberately left alone to expire on its TTL — the
   Agent has no authority to cancel a Gateway decision; what it refuses is
   its own half, so nothing actionable reaches a human.

A blocked proposal is **not** an error and does not end the turn. It's
recorded as an `internal.readonly_guard` transcript entry (the same pattern
`internal.grounding_check` and `internal.verification_check` already use) and
as evidence, and the loop continues — so the model's next turn sees its
proposal went nowhere and writes it up as a recommendation instead, which is
exactly the output this feature wants. It also surfaces in the digest
(`ScheduledSummary.dropped_proposals`), worded so it can't be misread as
something Numi did: "Numi *would have* proposed ...".

**Nothing here is a new route to a database.** Every call goes through the
same ToolClient → Gateway → Execution pipeline as a DBA's message,
authorized as a real configured DBA account
(`DAILY_REPORT_IDENTITY_ACCOUNT`) that the Gateway independently
re-resolves per call (spec §62). There is no scheduled-job bypass and no
elevated service role: a digest sees exactly what that account would have
seen by typing "daily summary" into Slack.

**Reuse, not a second investigation engine.** `run_comprehensive_summary`
builds the same `ConversationState` + `InvestigationState` pair
`handle_message` would have, sets the `playbook_id` `match_playbook` would
have matched, and calls the same `_continue_investigation` — so the turn
budget, argument stripping, DENIED self-correction, "a FAILED step doesn't
abort the run" behavior, conclusion grounding and the playbook's own
`conclusion_guidance` all apply identically and for free.
`tests/unit/test_scheduled_summary_entry_point.py` asserts the submitted
tool sequence against `playbooks.library`'s own step list rather than a
copy, so the two cannot drift. Two deliberate differences: the state is
**never registered with the `ContextManager`** (a scheduled sweep must be
invisible to the conversation layer — registering it could clobber a real
DBA's live `database_context`, in-progress investigation or pending approval
on whatever `conversation_id` it reused), and the environment is **supplied,
not asked for** — `handle_message` refuses to guess an environment, but
there is nobody here to ask, so the caller passes the environment the server
registry itself declares.

**Reporting a failure is the point, not an afterthought.** A DBA reading "6
servers checked, all healthy" when it was really "6 attempted, 2 never
responded" is worse off than with no digest at all — it actively tells them
to stop looking. So `build_digest` always states both numbers, gives
unchecked servers their own labelled section, and never folds them into the
closing "all other checks came back clean" line. Whether a server *was*
checked is decided structurally — `investigation.actions` is appended to
only on an EXECUTED call, so an empty list means not one diagnostic
succeeded — never by reading the model's prose, which will happily narrate
"everything looks healthy" having gathered nothing. The same discipline
decides whether a server gets its own block at all: `ScheduledSummary
.is_clean` reads the typed `Conclude` action's own root-cause/recommendation
fields, because there is no honest way to parse reassurance out of free
text. This applies `comprehensive_summary`'s own "report ONLY deviations,
then say plainly everything else came back clean" guidance a second time,
one level up — without it a ten-server estate produces ten paragraphs of "X
is fine" every morning, which is the same wall of text the playbook's
guidance already rejects, just bigger. Per-server failures never abort the
sweep, and `run_once` never raises into the scheduler: the one outcome
deliberately not available anywhere in this module is silence, because a
digest that simply doesn't arrive is indistinguishable from a quiet morning.

**Delivery goes through Channels, not straight to Slack.** The Agent holds
no channel credential and must not start holding one — it is the service
running attacker-influenceable model output. `ChannelsDigestPublisher` posts
to a new `POST /v1/notify` on the Channels service with the same signed,
audience-scoped service token (`numi-channels`) every other internal hop
uses, and Channels does the rendering and the Slack call, exactly as it
already does for every reply. That endpoint delivers text and nothing else:
it takes no identity, no approval_id and no conversation, and the
`AgentReply` it builds never carries an `approval_card`, so it cannot become
a way to put a clickable action in front of a DBA. A Teams destination later
is a change to that function, not to the Agent (see "Adding a channel").

**Opt-in, with exactly one switch.** An unset `DAILY_REPORT_SLACK_CHANNEL`
means `schedule_daily_digest` constructs nothing, starts nothing and
registers no job — not an idle scheduler waking daily to find it has nowhere
to post. Returning `None` rather than an inert scheduler is what makes that
directly assertable (`tests/unit/test_daily_digest_scheduling.py`, which
checks both directions without ever advancing a clock). There is deliberately
no separate `enable_...` boolean: there is no coherent "enabled but with
nowhere to send it" state, and two switches would only ever be a way to get
them out of sync. APScheduler is used rather than a hand-rolled
`asyncio.sleep` loop because the correctness of "every day at 06:00" lives
almost entirely in edge cases a loop would have to reimplement by hand —
missed occurrences after a restart, overlapping runs, drift, an explicit
timezone — which here are `misfire_grace_time` / `max_instances` /
`coalesce` / a UTC-pinned trigger. The job is registered in the **agent**
service's FastAPI lifespan: it owns the orchestrator, holds no database
credential (unlike `execution`), and is not a webhook front door whose
lifecycle is driven by inbound traffic (unlike `channels`). Starting it in
the lifespan rather than at import time also means importing the module, or
building the app to inspect its routes, never spins up a background job.

## Alert-triggered investigation: the digest's event-driven sibling

The digest above is triggered by a clock; `agent/alert_trigger.py` is the
same shape triggered by an event instead — an external monitoring system
(Prometheus Alertmanager, Datadog, a cloud provider's own alarms, ...)
reports a threshold breach, and Numi investigates it unattended, the same
way it would sweep a server at 6am.

**Every constraint the digest section above documents applies here
identically, for the same reason and via the same mechanism.** This is
deliberate, not incidental: `run_triggered_investigation` and
`run_comprehensive_summary` are two thin, distinctly-shaped callers
(`orchestrator.py`) into the *same* `_continue_investigation` loop, sharing
the same `read_only=True` flag and all three of its enforcement layers, the
same never-registered-with-`ContextManager` ephemeral state, the same
environment-supplied-not-guessed rule, and the same
`_unattended_summary`/`ScheduledSummary` result shape —
`tests/unit/test_scheduled_digest_never_writes.py`'s
`test_the_triggered_entry_point_blocks_a_write_end_to_end` asserts the
identical guarantee through this entry point specifically, not just through
the loop internals the two share. The one deliberate difference: where
`run_comprehensive_summary` always runs the fixed `comprehensive_summary`
playbook, `run_triggered_investigation` is freeform (`playbook_id=None`) —
an alert already names a specific symptom (a metric, a threshold, a current
value), so what should get checked next is exactly the kind of judgment the
LLM planner makes for a live DBA typing that same symptom into chat, not a
fixed checklist. That problem statement (`alert_trigger.build_problem_statement`)
is layer zero of the same read-only guarantee `_SCHEDULED_SUMMARY_PROBLEM`
is for the digest: it states outright that nothing proposed is ever
executed and that nobody is present to answer a clarifying question.

**The inbound path is new; nothing about the outbound or authorization path
is.** An external system is not one of our own services, so it needs its own
trust boundary rather than `common.service_auth`'s internal-only token
scheme: `POST /webhooks/alerts` (Channels) verifies an HMAC signature
(`channels/alerts/signature.py`, the same shape as
`channels/slack/signature.py` — a timestamp-bound signature with a
replay-defeating freshness check) before the body is even parsed, and
`ALERT_WEBHOOK_SECRET` unset is a hard failure at the signature-check level,
never a silent "unauthenticated is fine for now." From there the shape
collapses back onto everything already established: Channels forwards to
the Agent's `POST /v1/alerts/trigger` with the same signed, audience-scoped
internal service token `/v1/chat` uses; the Agent resolves the alert's
`server` field against the registry by exact id/alias match (never a
guessed substring — an ambiguous or absent match is reported as "unknown
server," because unlike a live DBA conversation, nobody is present to
notice or correct a wrong match); every diagnostic call is authorized as a
real configured DBA account (`ALERT_WEBHOOK_IDENTITY_ACCOUNT`, deliberately
separate from the digest's own account so the two features can be enabled,
disabled and audited independently) that the Gateway independently
re-resolves per call, exactly like every other path into this system; and
delivery goes back through Channels' `POST /v1/notify`
(`scheduled_report.ChannelsDigestPublisher`, reused as-is — it already does
precisely "ask Channels to deliver this text to this channel_id"), never
straight from the Agent.

**Opt-in, with exactly one switch, on each side of the boundary
independently.** An unset `ALERT_WEBHOOK_SLACK_CHANNEL` makes
`AlertTriggerRunner.handle_alert` a no-op on the Agent side, mirroring the
digest's own one-switch design; an unset `ALERT_WEBHOOK_SECRET` makes every
request to `/webhooks/alerts` fail signature verification on the Channels
side. Retries are deduplicated the same way Slack's own event retries are
(`channels/api/app.py`'s TTL'd `_seen_slack_event_ids` cache), on an
optional `alert_id` field in the alert payload — kept as an independent
cache rather than sharing Slack's, since the two features' lifecycles
(enabled/disabled, key spaces) have no reason to be coupled.

**A cooldown, distinct from `alert_id` dedup.** Dedup catches a retried
delivery of the *same* firing; it does nothing for a metric that genuinely
re-breaches its threshold every few minutes, which would otherwise run a
full investigation — and post a fresh message — on every occurrence.
`AlertTriggerRunner` enforces a per-`(server, metric)` cooldown
(`ALERT_WEBHOOK_COOLDOWN_SECONDS`, default 900s, 0 disables it) on top of
the dedup, using the exact same `RateLimitBackend` abstraction the
Gateway's own rate limiter runs on (`limit=1` over the cooldown window is
what a cooldown *is* — moved to `common.rate_limit_backend` specifically so
both features share it) — Redis-backed and correctly shared across
replicas the instant `RATE_LIMIT_BACKEND=redis` is set, unlike a
per-process cache that would silently reset on every load-balanced request.

## Memory across investigations: the write-orphaned table finally gets a writer

`InvestigationRecord`/`InvestigationEventRecord` (`gateway/infrastructure/db/models.py`)
have existed in the schema since migration 0001, but until now nothing ever
wrote to them — the only route was a read-only `GET /v1/investigations/{id}`,
whose own docstring already anticipated a
`gateway.domain.investigation_store` module that didn't exist yet. Every
investigation started cold: `ContextManager` is explicitly process-local,
in-memory, per-conversation state (see its own module docstring), so a
prior investigation on the same server left nothing behind for the next
one to build on.

`gateway/domain/investigation_store.py` is that module: `InvestigationStore`
(interface) + `DbInvestigationStore`. Deliberately **not** a
`DbCatalogStore`-style cache-in-front store — a catalog is one row per
registered server, small and bounded, worth loading whole into memory;
investigations are unbounded and append-heavy, so this is a thin
write-through store instead, every operation hitting the DB directly.
`gateway/domain/investigation_memory.py`'s `InvestigationMemory.recall`
is the read side, mirroring `DiscoveryOrchestrator.ensure_fresh`'s own
"look up, degrade to nothing rather than fail hard" shape: a lookup
failure logs a warning and returns `[]` rather than blocking a new
investigation from starting — recall is an enhancement, never a
dependency. New routes, all under the existing service-token dependency:
`POST /v1/investigations`, `PATCH /v1/investigations/{id}`,
`POST /v1/investigations/{id}/events`, `GET /v1/investigations/memory/{server_id}`.

`target` is a JSON blob with no indexable server key, so a new, separately
indexed `server_id` column was added to `investigations` (migration 0003,
following 0002's explicit-`op.add_column` convention — 0001's
implicit-metadata `create_all()` is a one-time exception, never repeated).

On the Agent side, `AgentOrchestrator._bootstrap_investigation_memory` is a
one-time, best-effort hook at the top of `_continue_investigation` —
covering every entry path into it (interactive, resumed, the scheduled
digest, and an alert-triggered run, since they all funnel through the same
loop) — that tells the Gateway an investigation exists and recalls recent,
concluded findings for the same server into a new
`InvestigationState.memory_context` field. That field is kept deliberately
separate from `evidence`: folding a prior investigation's claim into
`evidence` would let `_ungrounded_identifiers` (see "A write executing is
not license to conclude it worked," above) treat something the *previous*
investigation found as if *this* investigation had itself confirmed it —
quietly defeating the exact guarantee that check exists for. Recalled
memory is prompt background only, rendered as `[server: X] (STATUS)
'problem' — findings: [...]` lines, never grounding evidence. Governed by
`INVESTIGATION_MEMORY_LOOKBACK` (default 3; 0 disables recall without
touching any call site). The four new `ToolClient` methods this needs
(`create_investigation`, `update_investigation`,
`append_investigation_event`, `get_investigation_memory`) are all
best-effort by construction — they catch, log, and swallow internally,
the same posture `alert_trigger.py` already takes toward its cooldown
backend, so a Gateway hiccup never blocks or fails an investigation.

**A live bug this surfaced, and its fix.** `state.database_context["instance"]`
is set directly from whatever raw text the model extracts as
`instance_hint` — verified live, a real DBA's "Postgres dev 02" for the
registered server `postgres-dev-02` resolved correctly for the actual tool
calls (the Gateway independently re-resolves the target with its own fuzzy
matching regardless), but was being sent to the Gateway as `server_id`
completely unnormalized. Two conversations about the same physical server,
phrased even slightly differently, would have silently failed to recognize
each other for memory purposes. `AgentOrchestrator._find_matching_servers`
(refactored out of the pre-existing `_environment_for_instance`, so both
now share one matching implementation instead of risking two that drift)
and the new `_canonical_server_id` resolve a raw hint to the registered id
using the exact same fuzzy rules (exact/substring/host/normalized) real
tool-call resolution already relies on — falling back to the raw hint only
when it matches no registered server or more than one, never guessing.

## A second opinion before a conclusion ships

`_finalize_conclude` (see "A write executing is not license to conclude it
worked," above) already ran two free, structural checks before this:
`_ungrounded_identifiers` catches a conclusion *naming* something never
seen, `pending_verification` catches a write reported done without an
independent re-check. Neither catches a conclusion that fabricates nothing
and has nothing pending, yet still doesn't actually follow from the
evidence gathered — a plausible-sounding leap rather than a grounded
finding. Nothing reviewed a conclusion's *reasoning* before it reached the
DBA.

`CritiqueVerdict` (`agent/planner/actions.py`: `sound: bool`,
`issue: str | None`) is a second, independent LLM opinion on exactly that
question. `LLMProvider.critique_conclusion` has a concrete, non-abstract
default returning `sound=True` unconditionally — deliberately **not**
`@abstractmethod`: making it one would force every existing `LLMProvider`
test double, and the deterministic offline mock planner this whole default
test suite runs on, to implement a method they have no reason to care
about. `StructuredLLMProvider`'s one real override — inherited by all four
vendor providers with zero per-vendor code — uses the same
`_call_tool`/`_call_with_retry` machinery every other structured call
already uses. `CrossProviderFallbackLLM.critique_conclusion` delegates to
the primary provider only, deliberately **not** part of the cross-provider
fallback walk — the same reasoning as `summarize_for_human`'s own
exclusion: a critique failing means "skip it," never "the DBA gets stuck,"
so it isn't worth another vendor's slice of the deadline that exists to
bound *decisions*.

`_self_critique_conclude` slots in as a third check inside
`_finalize_conclude`, using the identical reject-and-retry scaffolding the
other two already established (a transcript entry, an evidence note, a
structured warning log, `return None` so the caller's loop retries mid-
investigation) — gated `if not final_chance`, the same reasoning as the
existing `pending_verification` gate: the one bounded last-chance call
after the turn budget runs out offers no tool calls at all, so there is no
way left for the model to act on new guidance, and rejecting there would
only throw away everything the investigation found. Critically, it **fails
open** on any exception — a timeout, a provider outage, a malformed
response after retries — logging `self_critique_call_failed` and treating
it exactly like a `sound=True` verdict: a critique call failing must never
be worse than not having critiqued at all. Verified live against a real,
genuine Gemini outage (every fallback model exhausted, a real 20s
timeout): the wrapper still returned cleanly with the fail-open result,
never hanging or crashing. Governed by `SELF_CRITIQUE_ENABLED` (default
true) as a cheaper-than-a-redeploy kill switch.

## Decision-quality events: a durable, queryable review loop

A rejected conclusion or a forced cross-provider fallback substitution
previously only ever became a structured log line — nothing durable,
nothing queryable, nowhere this becomes a standing habit to review rather
than something noticed only mid-incident while grepping logs. A new
`llm_decision_events` table (migration 0004 — a genuinely new table,
unlike 0003's retrofitted column, so it's automatically covered by
`test_migrations_match_models.py`'s drift check without needing an entry
in that test's `TABLES_COVERED_BY_0001` set) backs
`gateway/domain/decision_events.py`'s `DbDecisionEventStore` — write-mostly,
no in-process cache, since the volume here (a handful of events per
investigation at most) never justifies one. Two new routes:
`POST /v1/decision-events` and `GET /v1/decision-events/summary?since_hours=24`
— the latter is the actual review-loop query, a `GROUP BY event_type`
rollup, not a dashboard.

**Deliberately not exhaustive.** In scope: the three `_finalize_conclude`
rejection paths plus `self_critique_call_failed`, and cross-provider
fallback substitutions — already collected per-message as `FallbackEvent`s
on `ConversationState.llm_fallback_notices` for DBA-facing disclosure, this
just also persists them. Out of scope, staying log-only for now:
`llm_call_retrying`, `llm_call_deadline_exceeded`,
`decide_next_action_validation_failed`, `gemini_model_unavailable_switching`
— these fire from the registry's cached, cross-conversation provider
instances, which have no per-conversation sink and no `ToolClient`
reference to persist through; capturing them durably means threading a new
`events` kwarg through every provider's `decide_next_action`/
`extract_intent` signature, cascading through the ABC, all four vendors,
`CrossProviderFallbackLLM`, and every test double — a real future
extension, not a v1 corner cut being hidden. `ToolClient.log_decision_event`
is best-effort like the four investigation-memory methods above, but logs
locally on total failure too: unlike those, there is no other record of
the event at all if both the Gateway call and this log line were to
vanish. Governed by `DECISION_EVENT_LOGGING_ENABLED` (default true).

## Task-complexity model routing: proactive, not just reactive

Every LLM call previously used one configured model regardless of how
simple or hard the decision was — `extract_intent` (classifying one
message) got exactly the same model as `decide_next_action` (multi-step
investigation reasoning), with fallback purely reactive on total provider
outage (see "Cross-provider LLM fallback," below).
`LLMRegistry.tier_model(call_type: "fast" | "strong")` reads two new
settings, `LLM_FAST_MODEL`/`LLM_STRONG_MODEL`, both empty by default — so a
zero-config deployment resolves the exact same model as before this
existed, byte-for-byte.

`AgentOrchestrator._llm_for` gains a `call_type` parameter (default
`"strong"`), applying a tier override **only** when the DBA has made no
explicit `/model` choice (`state.llm_provider is None and state.llm_model
is None` — the exact existing lock condition) — an explicit choice is
never silently overridden by a tier default. A deployment-level provider
lock (`Settings.llm_provider` forcing one vendor) does not disable tiering
either, since that only constrains which vendor is used; `state.llm_provider`/
`llm_model` stay `None` regardless, so tiering still applies within the
locked vendor. Both `extract_intent` call sites (`handle_message`,
`_classify_potential_topic_shift`) request `call_type="fast"`;
`_continue_investigation` keeps the default `"strong"` — unchanged, since
that's what `decide_next_action` already needed. A tier override doesn't
survive a cross-provider fallback walk (a model id is provider-specific,
and `resilient_for_conversation` already never carries one across vendors
for that reason) — it reverts to that fallback vendor's own default, same
as today. Verified live with both settings actually configured and
deployed: a real conversation's `extract_intent` call correctly used the
fast model and its `decide_next_action` call correctly started on the
strong model, before the pre-existing, unrelated quota-driven fallback
took over.

## Cross-server pattern correlation: reusing Phase 1's store, not new infrastructure

Each investigation was scoped to one server — no way to notice "three
servers hit the same symptom this week," a pattern a human DBA would catch
immediately. This needed no new infrastructure beyond the investigation
store above: the `GET /v1/investigations/{id}` router's own docstring had
already anticipated this exact module. Two more indexed columns on
`investigations` — `playbook_id`, `environment` (migration 0005, same
`op.add_column` pattern as 0003) — since correlation needs to filter on
both alongside excluding the asking server.

`InvestigationStore.find_similar`/`InvestigationMemory.correlate` (the same
"look up, degrade to nothing on failure" shape as `recall`) back
`GET /v1/investigations/correlate?playbook_id=...&environment=...&exclude_server_id=...&limit=...`,
gated by `CROSS_SERVER_CORRELATION_ENABLED` checked in the route itself —
the authoritative enforcement point — and bounded by
`CROSS_SERVER_CORRELATION_LOOKBACK_DAYS` (default 30) so "recently" means
something, not an unbounded historical scan. `ToolClient.get_cross_server_patterns`
is best-effort like every other Phase 1 method.

Correlates **structurally only** — by shared `playbook_id` (the identical
deterministic scenario match `agent.playbooks.library.match_playbook`
already makes for the investigation itself, reused rather than inventing a
second vocabulary) and optionally `environment`. Deliberately not fuzzy
text or embeddings similarity: no vector-search infrastructure exists
anywhere in this codebase, and building one would be a separate, much
larger investment than "cross-server correlation" implies — a candidate
future phase, not something half-built here. A fully freeform investigation
(no playbook matched) has no meaningful scenario to correlate on, so
correlation is skipped entirely rather than querying "everything with no
playbook." Matches fold into the same `memory_context` list Phase 1 built,
tagged with their own `server_id` (`InvestigationMemoryEntry` gained that
field, populated for same-server recall too) so the rendered
`[server: X]` line lets the model/DBA tell "this happened here before"
apart from "this happened elsewhere too."

## Latency ceiling on a single LLM decision

Each layer of `StructuredLLMProvider`'s resilience (per-call timeout →
model-fallback with cooldown, Gemini-specific → one same-model retry) is
individually reasonable but has no bearing on the others' worst case —
verified live, their product left one real decision hanging for minutes
with a provider under sustained load. `_OVERALL_DEADLINE_SECONDS` (20s)
wraps the whole thing: no matter how many retries or model switches happen
underneath, a single `decide_next_action`/`extract_intent` call degrades to
a clear "try again" message within ~20s, never longer. This is the actual
production guarantee — not any individual timeout's own value.

## Cross-provider LLM fallback (when a whole vendor is down, not just a model)

The layer above has one blind spot, confirmed live: it bounds how long a
*failing* provider may take, not what happens when that provider has
nothing left to give. This deployment's default (Gemini) exhausted its
free-tier daily quota — `RESOURCE_EXHAUSTED`, 20 requests/day/model — and
did so across every one of its *own* internal fallback models in turn
(`gemini_provider._MODEL_FALLBACKS`: 3.5-flash → 3.6-flash → 3.7-flash →
3.8-flash → 3.1-pro-preview → 3-flash-preview → 3.1-flash-lite). Every
resilience layer worked exactly as designed and the DBA still got "The
gemini service is temporarily unavailable" — while real
`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `DEEPSEEK_API_KEY` sat configured
and unused in the same `.env`. Retrying harder inside one vendor cannot fix
a vendor that is out of quota; the only useful move is sideways.

`agent/llm/fallback.py` adds that move. When the conversation's provider
proves *unreachable* for one specific `extract_intent`/`decide_next_action`
call, `CrossProviderFallbackLLM` re-issues that same call against the next
provider family from `Settings.configured_llm_providers()` — the same
preference-ordered, key-filtered list `/models` and default resolution
already use, so an unconfigured provider is never even considered. The
split that makes this possible is `LLMProvider.*_or_raise`: "unreachable"
(every retry hit a transport/API error, or the deadline expired) raises
`LLMProviderUnavailableError`, while "answered, but the answer failed
validation" keeps its own long-standing degraded reply and is deliberately
*not* escalated — a malformed completion is a prompt/schema problem another
vendor is no more likely to get right, so spending its latency budget on
one would only add delay.

**The time-budget arithmetic.** The guarantee above is a property of a
*decision*, not of a provider, so cross-provider fallback divides that
ceiling rather than repeating it — four providers at 20s each would mean an
80s decision and break it outright. `per_attempt_budget_seconds` splits the
same `OVERALL_DEADLINE_SECONDS` across the primary plus every configured
fallback (the primary is the likeliest to answer, but is not entitled to
spend the whole allowance and leave nothing for the escape hatch):

| configured providers | per attempt | worst-case total |
| --- | --- | --- |
| 1 | 20.0s | 20s — not wrapped at all; byte-identical to before |
| 2 | 10.0s | 20s |
| 3 | 6.7s | 20s |
| 4 | 5.0s | 20s |

`MIN_PER_ATTEMPT_BUDGET_SECONDS` (5s) floors the slice so that adding a
fifth provider family can never shrink every attempt below a usable
round-trip — it would hand out four viable attempts and let the walk be cut
short instead of five guaranteed-to-fail ones. It does not bind today
(20/4 == 5 exactly), and is itself clamped to the total. Each attempt is
additionally clamped to the time actually remaining, so the sum can never
exceed the ceiling: the ~20s promise holds unchanged.

**It is never silent.** A locked `LLM_PROVIDER`, or a DBA's explicit
`/model` choice, does *not* disable the fallback — an answer beats "try
again later" while usable keys sit idle — but a substituted vendor is
disclosed, not hidden: every substitution is recorded on the per-message
`ConversationState.llm_fallback_notices` sink and appended to the reply by
`AgentOrchestrator.handle_message` ("(The gemini service was unavailable,
so I used anthropic instead.)"). Done at that single outermost return
point, not at each of the dozen-odd places a reply is built, so no future
reply path can forget it. When *every* provider fails, nothing was
substituted and the DBA gets the pre-existing message verbatim, naming the
provider they actually chose — a strict no-regression, pinned by test.

Two paths deliberately never see any of this: the deterministic offline
planner (`llm_provider="mock"`, what the whole default test suite runs on)
and single-provider deployments — today's common case — both get the bare
provider back from `LLMRegistry.resilient_for_conversation`, unwrapped.
Covered by `tests/unit/test_cross_provider_fallback.py` (31 tests, entirely
stub-driven: no keys, no network, no real waiting).

## Instance-wide diagnostics don't demand a database

Every read tool used to default to `required_target_scope = ["environment",
"instance", "database"]` (`gateway/domain/tool_catalog.py`), so a DBA saying
"what's running on postgres-local" with no database named got rejected —
`INVALID_TARGET`: "Which database on postgres-local?" — even though the
underlying diagnostic never needed one. Checked against all three adapters
(`execution/adapters/{postgresql,sqlserver,mysql}.py`): SQL Server's DMVs and
MySQL's `information_schema`/`performance_schema` views were already
genuinely instance-wide for sessions, blocking, deadlocks, running queries,
wait stats, replication, backups, configuration, and error logs — no
per-database filter in the SQL. PostgreSQL's adapter was the one with a real
bug: it added an artificial `where datname = current_database()` to most of
these, even though `pg_stat_activity`/`pg_locks`/`pg_stat_database` are
natively cluster-wide in Postgres. That filter is now removed, and a
`datname`/`database_name` column is surfaced on the affected rows so the
result itself says which database(s) are involved.

`database.get_health`, `get_version`, `get_sessions`, `get_blocking_sessions`,
`get_deadlocks`, `get_running_queries`, `get_wait_statistics`,
`get_replication_status`, `get_backup_status`, `get_configuration`, and
`get_error_logs` now declare `required_target_scope = ["environment",
"instance"]` — no database required. Tools that are genuinely
database-scoped on at least one engine (`get_storage`,
`get_transaction_log`, `get_tables`, `get_indexes`/`get_statistics`,
`get_query_plan`/`get_top_queries`) are unchanged. `execution/service.py`
already fell back to the credential's own default database when none is
given (`if request.database: ...`) — no change needed there.

This is what lets the agent *investigate* which database is affected (e.g.
"check what's running/blocking on X") instead of only ever being told, or
only remembering one once a DBA happens to name it — a DBA asking about an
incident often doesn't know the database yet; that's the point of asking.

### Follow-up: get_storage joined the instance-wide set too

`get_storage` was deliberately left database-scoped in the fix above — a
re-examination found that was the wrong call. All three engines'
`storage()` methods had the same shape of artificial restriction as
Postgres's original bug:

- **PostgreSQL**: `pg_database_size(current_database())` only ever reported
  the one database the connection happened to be on, even though
  `pg_database` is a global catalog (no per-connection restriction) listing
  every database on the cluster and its size. Now
  `select datname as database_name, pg_database_size(datname) as
  database_size_bytes from pg_database where datistemplate = false order by
  database_size_bytes desc` — genuinely cluster-wide, with `database_name`
  on every row.
- **SQL Server**: `sys.master_files` is ALREADY a cluster-wide catalog view
  — every data/log file for every database on the instance. The
  `WHERE database_id = DB_ID()` clause was the artificial restriction;
  removed, with `sys.databases` joined in for the database name
  (`sys.master_files` only carries the numeric `database_id`).
- **MySQL/MariaDB**: `information_schema.TABLES` spans every schema on the
  instance. `WHERE TABLE_SCHEMA = DATABASE()` was the artificial
  restriction; replaced with `GROUP BY TABLE_SCHEMA`, so one call reports
  every schema's size with the schema name as a column.

Honest caveat: Postgres's per-*table* breakdown (`pg_stat_user_tables`,
used by `get_tables`) is itself connection-scoped in Postgres — you can
only see the currently-connected database's own tables through it, unlike
`pg_stat_activity`/`pg_locks`. So only the *database-size* figure in
`get_storage` became instance-wide; the per-table level of detail still
requires `get_tables` (already database-scoped, unchanged) for whichever
database is of interest.

`database.get_storage` now declares `required_target_scope = ["environment",
"instance"]`, same as the rest of the set above. Verified live-effect via a
new integration test
(`test_comprehensive_summary_runs_exactly_five_tool_calls_through_the_real_gateway`
in `tests/integration/test_agent_orchestrator.py`): the `comprehensive_
summary` playbook's `get_storage` step (`target={}`, no database) used to
be denied `INVALID_TARGET` by the real Gateway and burn a 6th tool call on
a self-correction retry — one over its 5-step, 6-turn shared budget. It now
runs cleanly on the first attempt, leaving the playbook's intended one turn
free for the model's own concluding call.

## execute() must surface the query's own result, not just rowcount

Found while live-testing the blocking playbook against a real, currently
blocking session: `database.kill_session` reported `terminated=False` for
*every* session it ever killed across this whole project's live testing —
even ones independently confirmed dead a moment later. The bug was one
layer down from the adapter: `kill_session`'s SQL is `select
pg_terminate_backend(%(pid)s) as terminated` — a SELECT, not a plain
DML/DDL statement — but every engine's `QueryExecutor.execute()`
(`execution/adapters/connections.py`) discarded the cursor's own result row
and returned only `{"rowcount": ...}`. The adapter's `result.get
("terminated", False)` then always fell back to the default. Nothing ever
raised — it was a silently wrong answer, not a visible error, so it survived
this many rounds of live testing before a live kill against a session that
was independently checked before and after finally caught it.

`execute()` on all three engines now also fetches the first returned row
(when the cursor's `description` says the statement produced one) and
merges its columns into the result dict, alongside `rowcount`. A plain
DML/DDL statement with no result columns is unaffected — `rowcount` alone.

## A write executing is not license to conclude it worked

The README's own lifecycle diagram states the property this section
enforces: `... → DATABASE → VERIFICATION → AUDIT → AI DBA → USER` — the
system verifies, then the AI DBA reports the actual outcome. A shared
external playbook spec this project follows says the same thing more
bluntly: "Never mark an incident as resolved merely because an action was
submitted. Resolution requires independent verification." Before this,
that property held only as far as the model chose to make it hold.

`ToolCallStatus.EXECUTED` on a write (e.g. `database.kill_session`) means
the Gateway/Execution pipeline ran the statement — it says nothing about
whether the condition the DBA actually cared about (a session still
blocking something) is now gone. Whether the investigation ever re-checked
that afterward (calling `database.get_blocking_sessions` again, say) was
previously left entirely to the model's own discretion within its turn
budget. Verified live, a real model has voluntarily done exactly the right
thing — "Subsequent session and blocking checks confirmed that session X
has been successfully terminated" — but nothing ever forced it to. Nothing
stopped the same model, on a different run, from proposing `kill_session`
and immediately concluding "Completed" from the bare EXECUTED status alone,
with no independent check that the session was actually gone. A DBA
reading that report has no way to tell the two cases apart.

The fix mirrors an already-established pattern in this same file:
`_ungrounded_identifiers`/`_finalize_conclude`'s existing grounding check
(see `test_conclusion_grounding.py`) already rejects a Conclude that names
something never actually seen in the investigation, and gives the model
one more bounded try. Post-remediation
verification is the same shape, applied to a different failure: a Conclude
that would report a write as done without an independent re-check.

- `orchestrator._VERIFICATION_TOOLS_BY_WRITE_TOOL` is a lookup table from a
  write tool_id to the read-only tool(s) that can cheaply, obviously
  confirm its real-world effect — currently `kill_session`/`cancel_query`
  against `get_blocking_sessions`/`get_sessions`/`get_running_queries`.
  Deliberately scoped to session-termination-shaped writes first: that's
  the case this project has repeatedly hit live and gotten wrong, and it's
  the one with an unambiguous, single-call check (does this session_id
  still show up?). A write like `update_statistics`/`create_index` has no
  equally cheap re-check (confirming it actually *helped* needs a
  follow-up performance observation, not one more tool call), so those
  intentionally stay out of the table for now — extending this to a future
  write tool with its own obvious check is one more table entry, not a new
  mechanism.
- `InvestigationState.pending_verification` is set the moment a mapped
  write executes, and cleared the moment one of its correlated read-only
  tools is itself proposed and executes — regardless of what that check
  finds. `_verification_still_shows_condition` inspects that read tool's
  own result rows for the exact session_id the write targeted (the same
  `session_id`/`blocked_session_id`/`blocking_session_id` fields the real
  adapters already return — see `execution/adapters/*.py`) and records the
  verdict on `InvestigationState.last_verification`: `"RESOLVED"` or
  `"UNRESOLVED"`.
- `_finalize_conclude` rejects a Conclude while `pending_verification` is
  still set — same self-correction shape as the grounding check: feed back
  exactly what's missing (which tool to call), append an
  `internal.verification_check` transcript entry, and let the loop's own
  turn budget bound how many times this can happen. The one exception is
  the single bounded last-chance Conclude call after the turn budget runs
  out (`final_chance=True`): it offers no tool calls at all
  (`available_tool_ids=[]`), so there is no way left for the model to
  actually go check — rejecting there would only discard everything the
  investigation found in favor of the generic no-root-cause fallback, so
  it's accepted instead, with the report saying plainly that it was never
  independently verified.
- `_format_report`/`_verification_note` state the real, structurally-
  derived outcome directly in the DBA-facing reply — never inferred from
  the model's own free-text Conclude wording, and never collapsed into one
  generic "Completed": *independently re-checked and confirmed resolved*,
  *independently re-checked and did NOT resolve*, or *executed but never
  independently checked* are three distinct, separately-worded outcomes.

Whether a tool_id is even eligible for this treatment is double-checked
against the tool catalog's own `operation_type` (already returned in full
by `/v1/tools` — nothing new had to be threaded through the Gateway for
this), the same belt-and-suspenders reasoning
`_strip_unschematized_arguments` already uses: `_VERIFICATION_TOOLS_BY_
WRITE_TOOL`'s own tool_id membership is the primary signal, but if the
catalog no longer classifies that tool_id as `OperationType.WRITE`, this
mechanism stands down rather than trusting a static mapping that might now
be stale.

Verified via `tests/unit/test_post_remediation_verification.py`, including
a scripted reproduction of the exact live pattern: a mock LLM proposes
`kill_session`, gets `EXECUTED` (`terminated=True`), then immediately
proposes `Conclude` claiming success with no re-check at all — rejected,
not accepted at face value — alongside the same scenario with a proper
re-check afterward (accepted, reply states it was independently verified),
one where the re-check shows the session is still there (accepted, reply
states it was NOT resolved), and one where the model never re-checks at
all before the turn budget runs out (accepted only via the bounded
last-chance path, reply states it was never independently verified).

## A NoArgs tool's own required-arguments entry must say "[]", not nothing

Observed repeatedly across live Slack testing: `database.get_blocking_sessions`
and `database.get_sessions` — both `NoArgs` in `gateway/domain/tool_catalog.py`
(`ARGUMENT_MODELS`, `extra="forbid"`) — kept getting proposed with an extra
`reason` (sometimes `session_id` or `database_name`) folded into `arguments`.
The Gateway correctly rejected each one as `INVALID_ARGUMENTS`, and the
orchestrator's self-correction path (`_SELF_CORRECTABLE_DENIAL_CODES`) always
recovered within the same turn — never broke an investigation — but it
wasted an LLM call and a Gateway round-trip every single time it happened.

The root cause was in `orchestrator._continue_investigation`, not the model:
`tool_requirements` (the per-tool required-argument map injected into
`decide_next_action`'s prompt, see `StructuredLLMProvider._ACTION_SYSTEM`)
was built as `{tool_id: reqs for t in available if (reqs := ...required)}` —
a tool with an empty required list (every `NoArgs` read) was walrus-filtered
out of the dict entirely, not included with an empty list. The model was
never actually told "this tool takes nothing"; it only ever saw entries for
tools that DO need something, and reasonably (if wrongly) generalized from
the `arguments` schema's superset of real properties (`session_id`, `reason`,
... — genuine requirements for *other* tools) that the prompt has to declare
up front for the reasons in `llm/base.py`'s own comment on `_FLAT_ACTION_SCHEMA`
(a property-less object schema gives a weaker model nothing to fill in).
`ProposeToolCall` already has its own top-level `reason` for the
human-readable justification — the model was folding that same idea into
`arguments` a second time, unprompted, for tools whose real schema has no
room for it at all.

The fix has two layers, deliberately not just one:

1. **Prompt fix** (root cause): `tool_requirements` now includes every
   available tool, NoArgs ones mapped to `[]` — an explicit "this tool needs
   nothing" signal instead of silence the model has to interpret on its own.
   `_ACTION_SYSTEM` and `_FLAT_ACTION_SCHEMA`'s `arguments` description both
   spell out what an empty list means and explicitly forbid borrowing a
   property that belongs to some *other* tool.
2. **Defense-in-depth** (`orchestrator._strip_unschematized_arguments`,
   called from `_submit_and_relay`): even if a prompt fix doesn't hold for
   every provider/model forever, any `arguments` key outside a tool's own
   real property set is silently dropped before a `ToolCallRequest` is ever
   built — the Gateway never sees it, so there is nothing left to deny.
   Dropping such a key is always safe: it could never have been one that
   tool's own schema would have accepted.

Verified live via the real orchestrator + real Gateway pipeline (in-process
ASGI, `tests/integration/test_no_args_extra_arguments.py`): a scripted
planner reproducing the exact bad completion above now completes on its
first attempt, with zero `INVALID_ARGUMENTS` denials in the transcript.

## A known environment must survive across investigations, not just within one

Live-reproduced in a real Slack thread: the DBA established development/
postgres-local early in a conversation, ran two more successful
investigations that each named postgres-local again, and then sent a plain
follow-up ("So what database is the copy activity happening on?") that
named neither an environment nor a server. `handle_message` (`agent/
orchestrator.py`) wrongly re-asked "which environment should I
investigate?" even though the environment had already been established
earlier in this exact conversation.

`state.database_context` (`agent/context_manager.py::ConversationState`) is
the actual source of truth here — it's a `ConversationState` field, so it
outlives any one `InvestigationState` and is never reset when an
investigation concludes and a new one starts. A brand-new investigation's
environment gate (`if "environment" not in state.database_context: ask`)
already reads this persisted value, not just the current message's
`intent.environment_hint` — so once established, it was never actually at
risk of being forgotten by that check alone. The real gap was the missing
other half of the *instance* symmetry already in place for `database`: a
switch to a different, named instance correctly pops a stale `database`
(it might not exist on the new server) via `state.database_context.pop
("database", None)`, but nothing equivalent protected `environment` — if
that new instance was unregistered or ambiguous (`_environment_for_instance`
returns `None`), the OLD instance's environment silently kept asserting
itself for the new one instead of being dropped, which is exactly the kind
of guess the spec forbids ("for production targets I won't guess"). Fixed
by popping `state.database_context["environment"]` too whenever a genuine
instance switch's environment can't be auto-resolved — mirroring the
existing database-goes-stale-on-switch logic exactly — and by making the
final "not in state.database_context" check's rationale explicit in code
comments, so it can't be quietly narrowed to `intent.environment_hint`
alone by a future change. See the two new cases in
`tests/unit/test_environment_clarification.py` for the exact scenario
pinned: a later fresh investigation never re-asking for an
already-known environment, and switching to an unresolvable instance
correctly forgetting the stale one instead of guessing.

## A stable conversation_id is a channels-layer responsibility, not the Agent's

The previous section's fix (and its tests) assume `handle_message` receives
the SAME `conversation_id` for every message in one ongoing exchange — and
traced in isolation, every piece of the Agent's own persistence logic
(`state.database_context` surviving across investigations,
`InvestigationState.is_concluded` correctly gating fresh-vs-resume,
`_ENVIRONMENT_ANSWER_RE`'s deterministic resume path) held up fine under
that assumption. Live testing then reproduced the identical-looking symptom
again anyway — a DBA's follow-up with zero named entities of its own asked
for an already-known environment, and even a bare "development" answer to
that clarification fell through to the generic chitchat fallback instead of
resuming — and this time it turned out that assumption itself was false.

The real bug was one line in `channels/api/app.py`'s Slack webhook:

```python
conversation_id = f"slack:{event.get('channel')}:{event.get('thread_ts', event.get('ts'))}"
```

A threaded reply's `thread_ts` is shared by every message in that thread, so
that half is fine. But an ordinary, non-threaded message — exactly how a
DBA naturally follows up in a channel that (by this webhook's own design)
"responds to plain messages too", no @-mention or thread reply required —
has no `thread_ts` at all, so this fell back to `event["ts"]`: that
message's own timestamp, unique to it and matched by nothing that comes
after. Every plain follow-up therefore silently started a brand-new, empty
`ConversationState` — `state.database_context` empty, `state.investigation`
`None` — no matter how thoroughly the Agent's own state persisted *within*
one `conversation_id`, because no two consecutive plain messages actually
shared one. Fixed in `_slack_conversation_id` by scoping a non-threaded
message to `(channel, slack_user_id)` instead of `(channel, message)`, so a
DBA's own consecutive plain messages share one conversation while two
different DBAs typing in the same channel still don't cross.

The general lesson: a fix verified end-to-end at the orchestrator level
(stable `conversation_id` held constant across calls, as every unit test in
`tests/unit/` necessarily does) only proves the orchestrator's *own* logic
is correct — it can't by itself prove the id it's keyed on is actually
stable in production, since that id is computed one layer up, by a
different service, from data the Agent never sees. See
`tests/integration/test_channels_api.py`'s
`test_three_plain_non_threaded_slack_messages_from_the_same_dba_share_one_conversation`
for the reproduction, and `tests/unit/test_environment_clarification.py`'s
`test_the_live_three_turn_sequence_is_handled_correctly_given_one_stable_conversation`
for the confirmation that, given that one stable id, the orchestrator side
already had no remaining gap.

### A sibling bug: the interactive-button path had its own, separate copy of the same mistake

Fixing the regular-message path did not fix approvals — a second, structurally
identical bug lived in `/webhooks/slack/interactive` (the Slack
Approve/Reject button handler), found live immediately after: clicking
either button never resolved the pending approval, no matter who clicked or
what card they clicked. It computed its own `conversation_id` inline,
independently of `_slack_conversation_id`, as:

```python
conversation_id = f"slack:{channel_id}:{payload.get('container', {}).get('message_ts', '')}"
```

— the approval CARD MESSAGE's own `message_ts`, which is unique to that one
card and never used as a `conversation_id` anywhere else in the system. An
approval card lives in the same conversation as the investigation that
produced it, so this could never match the `conversation_id` `state.
pending_approval` actually lives under — `AgentOrchestrator.
handle_approval_decision`'s `pending = state.pending_approval` lookup found
a brand-new, empty `ConversationState` every time and returned "There is no
pending approval on this conversation." regardless of who clicked or which
card.

This was the exact same class of mistake as the regular-message bug above
(deriving a conversation-scoped id from a single message's own unique
timestamp instead of from the stable `(channel, user, thread)` triple), just
in a second, independent code path that had never been routed through
`_slack_conversation_id` in the first place — extracting that helper fixed
only the one call site it replaced. Fixed by routing the interactive handler
through the same `_slack_conversation_id(channel, slack_user_id, thread_ts)`
helper, reading `thread_ts` from the interactive payload's `container` or
`message` object (whichever Slack populates — both are checked, `container`
first) exactly as a real block_actions payload can carry it. In this
codebase that resolves to the empty-string fallback in practice today,
since `SlackMessageSender.post_message` doesn't yet thread its own replies
(so an approval card is always posted as a plain, non-threaded message) —
but it now goes through the exact same `(channel, user)`-scoped computation
the regular, non-threaded message path uses, which is what actually matters
here: whatever the originating plain message's `conversation_id` was, the
card's button click now reproduces it exactly, instead of a third, unrelated
derivation. See `tests/integration/test_channels_api.py`'s
`test_slack_interactive_button_click_resolves_to_the_same_conversation_as_the_originating_message`
(isolates the channels-layer contract with a stub Agent) and
`test_slack_approval_card_button_click_actually_resolves_the_pending_approval_end_to_end`
(the full, real Agent/Gateway/Execution round trip: a real investigation
reaches `APPROVAL_REQUIRED`, the real card is captured, and clicking Approve
on it actually resolves and executes the real `state.pending_approval`,
not just a matching id in isolation).

The Microsoft Teams equivalent (`teams_webhook`) does **not** have this bug:
it computes `conversation_id` exactly once, from the Bot Framework's own
stable `activity.conversation.id`, before branching into the approve/reject
(`Action.Submit`) case versus the regular-message case — both branches read
the same already-computed value, so there was never a second, independent
derivation for the two paths to disagree about. Teams was left unchanged.

## An approval card must collapse on a decision — but only when it actually resolved

An approval card's Approve/Reject buttons stayed fully clickable forever
after a real decision was made — nothing ever rewrote the original Slack
message, so a second click (by the same person, or someone else entirely)
was always technically possible, even though the server-side decision was
already final. `slack_interactive` now calls
`SlackMessageSender.update_message` (Slack's `chat.update`) against the
card's own message (`payload["message"]["ts"]`/`blocks` — Slack's own echo
of the message the click happened on) to replace its `actions` block (found
by `block_id == f"numi_approval_{approval_id}"`, set when the card was
first rendered) with a static line: "✅ Approved by \<name\>" or "❌
Rejected by \<name\>". Slack buttons have no disabled-but-visible state to
toggle — swapping the interactive block for a plain one, the same pattern
real Slack apps (GitHub, PagerDuty, ...) use for this, is what actually
removes them.

The first version of this collapsed on *which button was clicked*, decision
alone. Live testing immediately surfaced why that's wrong: a DBA clicked
Approve on their own CRITICAL request and got correctly blocked by
separation of duties ("The requester cannot approve their own critical
action") — but the card still collapsed to "✅ Approved by \<them\>", both
lying about the outcome and hiding a still-open approval from the one
different, eligible DBA who actually could act on it. The same problem
applies to the first leg of a dual-approval requirement
(`AWAITING_SECOND_APPROVAL`): that decision succeeded, but the card must
stay live for a second, different approver.

The fix is a structured signal, not a guess from `status` or the reply
text: `AgentReply.approval_still_pending` (default `False`), set `True` by
`AgentOrchestrator.handle_approval_decision` in exactly those two cases —
both of which deliberately leave `state.pending_approval` un-cleared for
the same reason. `slack_interactive` only calls `update_message` when this
is `False`. See `tests/integration/test_channels_api.py`'s
`test_slack_approval_card_buttons_collapse_after_a_decision_is_clicked`
(the genuine-resolution case) and
`test_slack_approval_card_buttons_stay_live_after_a_failed_decision` (the
separation-of-duties case that exposed the first version's bug).

## A `message` subtype without a top-level "user" must not crash the webhook

`/webhooks/slack` unconditionally read `event["user"]` after checking only
`event.get("type") != "message"` and `event.get("bot_id")` — found live as
a genuine, unhandled `KeyError` 500ing the whole webhook. Slack sends
several `message`-typed events that are not a DBA sending Numi a fresh
instruction and carry no top-level `"user"` at all: `message_changed`
(edits — the author lives nested under `event["message"]["user"]`
instead), `message_deleted`, and others. Since Slack retries any delivery
it doesn't get a fast 200 for, one crash like this risks compounding into
repeated retries rather than one cleanly-ignored event. Fixed by reading
`event.get("user")` and skipping (same as the existing bot-message skip)
whenever it's absent, rather than assuming the key exists. See
`tests/integration/test_channels_api.py`'s
`test_slack_message_edit_event_with_no_top_level_user_does_not_crash_the_webhook`.

## The identity re-resolution point (why "UX check" ≠ "security boundary")

Channel Adapters and the Agent both *can* check whether a channel account is
a recognized DBA — this is a UX nicety (fail fast with a friendly message
instead of round-tripping to the LLM for someone who was never going to be
authorized). **But `ToolCallRequest` carries only `channel` +
`channel_account_id` — never a `VerifiedIdentity` object, never a role.**
The Gateway (`gateway/api/deps.py::resolve_identity`) calls its own
`IdentityProvider` instance, fresh, on every single tool call. If the
Agent were fully compromised and claimed "this user is DBA_MANAGER", it
would have no channel to make that claim in the first place — there is no
such field.

## Real identity and credential providers: configuration, and the injectable-client pattern

Two abstractions in this system front a real external dependency that
cannot exist in dev or CI: `IdentityProvider` (who is this DBA?) and
`CredentialProvider` (what credential opens this database?). Both now have
real implementations alongside their dev-safe defaults, and both are built
the same way — worth reading once before touching either.

### Selecting one

| Env var | Values | Factory |
|---|---|---|
| `IDENTITY_PROVIDER` | `mock` (default), `oidc` | `common/identity/factory.py::build_identity_provider` |
| `SECRETS_PROVIDER` | `local_dev` (default), `vault`, `aws_secrets_manager`, `azure_key_vault`, `gcp_secret_manager` | `execution/credentials/provider.py::build_credential_provider` |

Each factory is the *single* place that maps a config string to a class;
no service has its own `if settings.identity_provider == ...` ladder. An
unrecognized value raises rather than falling back to the dev-safe option —
a typo in `IDENTITY_PROVIDER` must never quietly hand production a
fictitious directory. `Settings.validate_for_production` separately refuses
to start a production process on `mock`/`local_dev` at all.

### What each real backend needs

**OIDC** (`common/identity/oidc_provider.py`) — `OIDC_ISSUER`,
`OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`, plus an `oidc:` section in
`config/identity.yaml` giving the SCIM `directory_endpoint` and, per
channel, which directory attribute holds that channel's account id
(`channel_attributes`). Endpoint metadata comes from OIDC Discovery; the
directory query is SCIM 2.0. The group → role mapping is the *same*
`identity:` section the mock reads, through the same `GroupRoleMapping`
class, so roles cannot drift between dev and production.

The design decision worth knowing: a Slack/Teams webhook hands you a
channel-native account id and no token, so there is nothing to validate —
resolution is necessarily a directory *lookup*
(`?filter=<attr> eq "<id>"`). `oidc.channel_attributes` is the real-directory
analogue of each mock user's `channel_accounts` block: instead of listing
ids per user, it says which attribute holds them and asks the IdP.
Populating that attribute is a directory-sync concern. There is no default
mapping — an unconfigured channel fails closed instead of guessing an
attribute name and risking a wrong match.

**Secrets managers** (`execution/credentials/provider.py`) — one env var
each (`VAULT_ADDR`+`VAULT_TOKEN`, `AWS_REGION`, `AZURE_KEY_VAULT_URL`,
`GCP_PROJECT_ID`); cloud auth is the platform's own ambient chain
(instance/task role, managed identity, ADC), never a key in Numi's
config. All four read the *same* JSON object — the keys
`LocalDevCredentialProvider` already reads from
`config/dev_credentials.yaml` — under a documented per-backend naming
convention (`secret/numi/db/<id>`, `numi/db/<id>`, `numi-db-<id>`).
Moving from dev to a real manager is a transport change, not a re-modelling.

Each SDK is an optional extra (`secrets-vault`, `secrets-aws`,
`secrets-azure`, `secrets-gcp`), imported lazily inside the provider that
needs it — the same treatment the LLM SDKs and database drivers get. A
Vault deployment never needs boto3 installed, and a missing package
surfaces as a `DEPENDENCY_UNAVAILABLE` naming the package and the extra,
not an import crash at startup. The OIDC provider needs no new dependency
at all: `httpx` is already a base dependency.

### The injectable-client pattern (how any of this is testable)

There is no Vault server, OIDC tenant, or cloud account in dev or CI, and
none is faked into existence. Instead every real provider takes an optional
client on the constructor — `client=` on the four credential providers,
`http_client=` on the OIDC provider:

- **`None` (production):** the provider builds the real SDK client lazily,
  on first use, from its own configuration.
- **Injected (tests):** that object is used verbatim. No SDK is imported,
  no network call is made, and the test asserts on *this codebase's*
  behavior — request shape, response parsing, error mapping, fail-closed
  defaults.

This is the established seam for "real thing unavailable in dev/test" here:
see `tests/canned_adapter.py`, `FakeQueryExecutor` in
`tests/unit/test_adapters.py`, and the LLM provider tests. Fakes for these
two live in `tests/unit/test_secrets_providers.py` and
`tests/unit/test_oidc_identity_provider.py`; each mimics only the one or
two calls its provider actually makes.

Type hints for injected clients are `Any`, not the real SDK type — the same
choice `execution/adapters/connections.py` makes for `pyodbc`/`psycopg`
handles — so `mypy src` type-checks cleanly on a machine where none of the
four optional SDKs is installed.

### Fail closed, in each one's own idiom

The two abstractions fail closed *differently*, because their call sites
do:

- A `CredentialProvider` **raises**
  `NumiError(DEPENDENCY_UNAVAILABLE)` — the execution must stop, and a
  raw SDK exception must never escape (it would carry secret paths, vendor
  stack traces, sometimes the secret itself). A secret that exists but is
  incomplete is treated identically to an unreachable manager: never a
  partial credential.
- An `IdentityProvider` **returns `None`** — matching
  `MockIdentityProvider`'s contract, which `gateway/api/deps.py` and
  `channels/api/app.py` are already written against, and which denies the
  request. An identity that cannot be proven is an identity that does not
  exist. Nothing is raised past those two methods.

## Package layout

```
src/numi/
  common/            # shared vocabulary — no service-specific logic
    models/           # failures, target, tool, risk, identity, execution contracts
    identity/         # IdentityProvider, MockIdentityProvider,
                      #   OIDCIdentityProvider (real OIDC/SCIM), factory
    config.py          # Settings (pydantic-settings)
    service_auth.py    # signed service-to-service tokens
    observability.py   # structured logging + tracing, secret redaction

  gateway/
    domain/            # tool_registry, target_validation, authorization,
                        # policy_engine, risk_engine, approval, data_policy,
                        # rate_limiter, audit, sql_validator, tool_call_handler
    infrastructure/     # control-plane DB (SQLAlchemy models + session),
                        # execution_client (HTTP to Execution Service)
    api/                # FastAPI app + routers

  execution/
    adapters/           # DatabaseAdapter interface, SQLServerAdapter,
                        # PostgreSQLAdapter, MySQLAdapter (MySQL + MariaDB),
                        # connections (real drivers)
    discovery/           # ServerDiscoverer per platform (catalog/DMV/stats
                        # views only, never table data) + run_discovery
    credentials/        # CredentialProvider (local_dev/Vault/AWS/Azure/GCP)
    service.py           # dispatcher: ExecutionRequest -> adapter method
    api/                 # FastAPI app

  agent/
    llm/                 # base (LLMProvider), mock, anthropic/openai/gemini/
                        # deepseek providers, registry (provider+model selection)
    planner/actions.py    # structured AgentAction union
    playbooks/library.py   # fixed diagnostic sequences for known scenarios
    orchestrator.py        # investigation loop
    scheduled_report.py     # opt-in daily multi-server digest (read-only)
    context_manager.py     # conversation/investigation state (in-process)
    tool_client.py          # HTTP client to the Gateway
    api/                    # FastAPI app

  channels/
    slack/               # signature verification, Block Kit rendering, sender
    teams/                # Bot Framework auth, Adaptive Cards, sender
    api/                   # FastAPI app: webhooks + /dev/chat mock channel
```

## Adding a database engine (Oracle, ...)

1. Add the platform to `common.models.target.Platform`.
2. Implement `execution.adapters.base.DatabaseAdapter` for it, using the
   engine's native diagnostics (equivalent of DMVs / `pg_stat_*`).
3. Implement `execution.discovery.base.ServerDiscoverer` for it (reads
   catalog/stats views only — never table data).
4. Register both in `execution.service._adapter_class_for` and
   `execution.discovery.engine._DISCOVERERS`.
5. Register servers of that platform in `config/servers.yaml`. Databases,
   tables, indexes and extensions are discovered automatically — they are
   never listed by hand.

Nothing in the Agent or Gateway contract changes — they only ever see the
canonical `DatabaseTarget`/`ToolDefinition`/`ExecutionRequest` models.

## Adding a channel (Discord, email, ...)

Implement a new adapter under `channels/<name>/` that verifies its own
transport's authenticity, resolves identity via the shared
`IdentityProvider`, and calls the Agent's `/v1/chat` — the same contract
Slack and Teams use. No Gateway or Agent code changes.

The one outbound, Agent-initiated path (`POST /v1/notify`, used by the
scheduled daily digest) lives here for the same reason: this is the only
service holding a channel credential, and the only one that knows how a
message should be rendered per channel. Sending the digest to a new channel
type is a change to that handler, never to the Agent.

## `InvestigationState.status`: a bounded set of stages, not the full 17-state spec

`InvestigationState.status` was, until now, only ever the string
`"INVESTIGATING"` or `"CONCLUDED"` — a plain binary that couldn't say
*why* an investigation was concluded (a confirmed fix? an unconfirmed
one? nothing ever attempted?) or that it was currently blocked on the DBA
versus a pending re-check. An external playbook spec this project has
been compared against defines a much richer 17-state investigation
lifecycle: `NEW → TRIAGED → INVESTIGATING → EVIDENCE_COLLECTED →
ROOT_CAUSE_IDENTIFIED → RECOMMENDATION_READY → AWAITING_APPROVAL →
APPROVED → EXECUTING → EXECUTED → VERIFICATION_PENDING →
VERIFIED/UNVERIFIED/FAILED/REJECTED/ESCALATED → RESOLVED/CLOSED`.

Building that state machine verbatim would be overengineering for this
codebase specifically: most of those states aren't independently
observable in its actual control flow (there is no code path where this
Agent process has ever known, distinctly, that an investigation just
became "TRIAGED" or "EVIDENCE_COLLECTED" as opposed to plain
"still investigating"), and the richest part of that lifecycle — approval
and execution — is already tracked correctly elsewhere: `PendingApproval`
and the Gateway's own audit trail, never `InvestigationState`. Duplicating
that into a second, parallel state machine here would only ever be
something that could drift out of sync with the real decision-maker.

Instead, `InvestigationState.status` (see `context_manager.py`'s
`InvestigationStage` type) is a small, deliberately bounded `Literal` of
exactly the stages this codebase's control flow can actually and
accurately observe itself transitioning through:

- `INVESTIGATING` — the existing default/starting stage, evidence still
  being gathered.
- `AWAITING_CLARIFICATION` — set the moment `_run_investigation_loop`
  returns an `AskClarification` question within budget, reusing the
  existing `clarification_count` tracking rather than a second mechanism.
  Reset back to `INVESTIGATING` at the very top of the loop's own `while`
  body — the one point every subsequent pass through the loop (a resumed
  call after the DBA answers, or simply the next playbook step/tool call)
  necessarily starts from, so this can never linger stale once the block
  clears.
- `AWAITING_VERIFICATION` — deliberately **never** a value `status` is
  itself assigned. `InvestigationState.effective_status` computes it on
  the fly from `pending_verification` (see "A write executing is not
  license to conclude it worked" above) whenever `status` is plain
  `INVESTIGATING`: that field is already the one authoritative record of
  whether a post-write re-check is outstanding, so mirroring it into a
  second, independently-written value on `status` would just be two
  things that could drift apart. `AWAITING_CLARIFICATION` and any
  `CONCLUDED_*` stage still take precedence over it — a clarification
  actively blocking the loop right now, or an investigation that's
  already final, both outrank a re-check that can simply wait.
- `CONCLUDED_VERIFIED` / `CONCLUDED_UNRESOLVED` / `CONCLUDED_UNVERIFIED` /
  `CONCLUDED_NO_ACTION` — four distinct conclusion outcomes in place of
  the old flat `CONCLUDED`, computed by `orchestrator._conclusion_stage`
  from exactly the same two signals (`last_verification`/
  `pending_verification`) that `_verification_note` already reads to
  build the DBA-facing verification text — `_verification_note` now maps
  *this* stage to its wording rather than re-deriving its own, separate
  judgment, so the stage recorded in `investigation.status` can never
  disagree with what the reply itself says happened. `_conclusion_stage`
  is called at every point an investigation ends, not only the primary
  `action=conclude` path in `_finalize_conclude` — the turn-cap,
  stuck-observation, and clarification-exhausted fallbacks in
  `_run_investigation_loop` also end an investigation without ever
  building a `Conclude` action of their own, and reuse this exact same
  derivation instead of a third judgment.

`InvestigationState.is_concluded` (`status` in the four `CONCLUDED_*`
values) is the one place that answers "is this investigation concluded or
not" — `handle_message`'s own resume-vs-fresh-investigation gate uses it
instead of the old `status == "CONCLUDED"` equality check, so a call site
that only cares about that distinction never has to enumerate all four
values itself.

`/status` (`_status_reply`) surfaces all of this in plain language via
`_stage_phrase` — "Investigating.", "Awaiting your answer to a clarifying
question.", "Awaiting independent verification of database.kill_session."
(the specific tool_id, read off `pending_verification`), or the
appropriate concluded-with-outcome phrasing — alongside the playbook/step
note and evidence count it already showed.

States deliberately left out, and why: `NEW`/`TRIAGED` (this Agent has no
pre-investigation queue — an investigation is created already
investigating); `EVIDENCE_COLLECTED`/`ROOT_CAUSE_IDENTIFIED`/
`RECOMMENDATION_READY` (there is no distinct moment evidence-gathering
"finishes" before root-cause analysis starts — one LLM call interleaves
all three, and a playbook's own step count is already visible via the
existing playbook/step note, not a status value); `AWAITING_APPROVAL`/
`APPROVED`/`EXECUTING`/`EXECUTED` (this is exactly the approval lifecycle
`PendingApproval` and the Gateway's audit trail already track
authoritatively — see above); `VERIFICATION_PENDING` (this is
`AWAITING_VERIFICATION`, computed rather than stored, as covered above);
`FAILED`/`REJECTED`/`ESCALATED` (a denied or failed tool call is reported
immediately as its own `AgentReply` — `status="denied"`/`"error"` — the
investigation itself simply continues or the DBA sees the failure inline,
never a terminal investigation stage of its own); `RESOLVED`/`CLOSED`
(no separate closing step exists after a Conclude — the four `CONCLUDED_*`
stages already are the terminal state, and reopening happens by simply
starting a fresh investigation in the same conversation, not by
transitioning a closed one).

## A fresh, self-contained instruction must not be swallowed by a stale, unanswered clarification

`handle_message`'s resume rule ("any message while `state.investigation` is
not concluded is a reply to it") is itself the fix for two earlier live
bugs (see `_ENVIRONMENT_ANSWER_RE`'s own comment and the
`_Turn2AsksThenConcludesLLM` test above) — a bare "development" or a bare
server name answering a clarification must never be reclassified by
`extract_intent` and silently dropped. But that same unconditional rule
has a failure mode of its own once a clarification is never answered:
`AWAITING_CLARIFICATION` has no timeout, so an old, abandoned investigation
sits there indefinitely, and *every* later message — no matter how
obviously it's a brand-new, unrelated, fully-specified instruction — keeps
getting framed to `decide_next_action` as "the DBA just replied" to
whatever that stale investigation last asked (`_problem_statement_for_llm`
still prepends the original, never-updated `investigation.problem`
verbatim).

Reproduced live: a deliberate gibberish test ("check blah on the thing pls
fix asap!!!") asked what "blah"/"the thing" meant and was never answered.
32 minutes and several unrelated exchanges later, "Drop the test database
on postgres-local, it's no longer needed" — a real request — got a reply
that rambled about "blah" and "the thing" instead of addressing it.
Reproduced again independently the same session with a different pair of
messages ("rebuild all indexes ... in sql server dev 01" got contaminated
into a proposal to restart an unrelated `postgres-local` instance).

The fix is a narrow, additional check — `_classify_potential_topic_shift`
— inserted only in the one specific situation where staleness risk is
real: `investigation.status == "AWAITING_CLARIFICATION"` (not the other
non-concluded stages — `AWAITING_VERIFICATION` in particular is a live,
recently-created state, not a plausibly-stale one) and no approval is
outstanding (an outstanding approval is real and actionable; never
abandon it for something that merely looks like a new request). Even
there, it deliberately does NOT give `extract_intent`'s classification the
same trust the pre-fix code already learned not to (see above) —
`is_dba_task`/`is_greeting_or_chitchat` alone were exactly what silently
dropped a bare, legitimate answer before. It requires the message to (a)
run to 4+ words — checked first, at zero LLM-call cost, so this added
latency/API spend never lands on the common case of a short, legitimate
answer — and (b) name its own concrete target (`instance_hint`,
`database_hint`, or `environment_hint`) once classified. A bare
"development" or bare server name fails (a); a longer reply that still
names no target of its own ("it's the one from this morning") fails (b)
and correctly still resumes. Only when both hold does the stale
investigation get marked `CONCLUDED_UNRESOLVED` (see the stage vocabulary
above — it was genuinely never resolved) and the message routed through
`_start_fresh_investigation` — extracted out of the tail of
`handle_message` specifically so this pivot case and the ordinary
"no active investigation" case share one meta-command/chitchat/target-
resolution implementation rather than a second copy that could drift.
See `tests/unit/test_environment_clarification.py`'s
`test_a_fresh_fully_specified_instruction_abandons_a_stale_unanswered_clarification`
(the live-reproduced positive case) and
`test_a_longer_reply_naming_no_target_of_its_own_still_resumes_the_stale_clarification`
(the negative case guarding the target-naming check specifically,
independent of the word-count floor the three-turn test above already
covers).

## Masking a sensitive FIELD says nothing about a sensitive VALUE

`DataMinimizer` (`gateway/domain/data_policy.py`) has always masked a field
whose NAME matches a sensitive pattern — `password`, `ssn`,
`account_number`. That is the right treatment for a column that is nothing
but a secret, and it is blind to the case that actually leaks: a field whose
name is entirely innocuous and whose CONTENTS are a verbatim statement a
user ran. `database.get_running_queries` returns a `query_text` column
straight out of `pg_stat_activity` (`execution/adapters/postgresql.py::
running_queries`; SQL Server's `sys.dm_exec_sql_text` and MySQL's
`information_schema.PROCESSLIST` return the same shape), and

```
SELECT * FROM accounts WHERE account_number = '1234567890'
                         AND customer_name = 'Jane Doe'
```

passed through untouched — the field name "query_text" is not sensitive, so
nothing masked it — into the LLM and from there into a Slack channel. That
is real customer data leaving the database through the one component whose
entire premise (SECURITY.md control 13) is that Numi reads diagnostics and
never table contents.

The fix has to keep the statement *useful*. A DBA diagnosing a slow or
deadlocked query needs the shape — which tables, which columns, which
joins, whether there's a leading-wildcard LIKE, whether the ORDER BY is
unindexed. None of that is the literal values, and the literal values are
exactly what must never leave. So `gateway/domain/query_scrubber.py`
parses the text with `sqlglot` — the same real parser `sql_validator.py`
already uses for the read-only SQL tool, and the house pattern this follows
— and replaces every literal node with a placeholder, leaving table names,
column names, keywords and structure intact.

**Where it lives: inside `DataMinimizer`, keyed on field name.** The
alternative considered was a second, separate minimization pass applied only
to the four tool results that obviously return raw text
(`running_queries`, `top_queries`, `deadlocks`, `error_logs`). Rejected,
for two reasons. First, it has a hole on day one:
`database.get_blocking_sessions` is not in that list, and all three
adapters return `blocked_query`/`blocking_query`/`blocked_query_text` from
it (`execution/adapters/*.py::blocking`) — full statement text carrying
exactly the literals this exists to catch; `get_sessions` returns
`query_text` too. A tool-id-keyed list is a list someone must remember to
extend every time an adapter grows a column; a field-name-keyed rule covers
them the moment they appear. Second, `DataMinimizer.apply()` is already the
one mandatory seam — every tool result passes through it, unconditionally,
at step 10 of `tool_call_handler.py::_handle_inner`, with no code path
around it. A second pass would be a second thing to remember to call.
`DataPolicyConfig` therefore grows a `free_text_sql_field_patterns` set
*separate from* `sensitive_field_patterns`, because the treatment differs:
sensitive fields are fully masked, these are literal-scrubbed and keep
their shape. Sensitive-by-name is checked first and wins, which is what
makes the change strictly additive — every field masked before is still
masked, byte for byte (`test_existing_field_name_masking_is_completely_
unchanged`).

**The placeholder is `<redacted>`, not `?`.** `?` is the obvious first
instinct and is wrong: `?` is itself a real bind-parameter marker in the
ODBC/MySQL dialects this system talks to, so `... WHERE account_number = ?`
is indistinguishable from a statement the application genuinely sent
parameterized. A DBA (and the LLM) could not tell "Numi removed a value
here" from "the app used a bind parameter here" — and those call for
different diagnoses. `<redacted>` can never be mistaken for something the
application wrote, and it matches the existing `***MASKED***` convention:
say plainly that something was deliberately removed. It is emitted as a
string literal in every position, numbers included, so the scrubbed
statement still parses as valid SQL. Every literal goes, including ones
harmless in isolation (a `LIMIT 100`, a `WHERE status = 1`): the
alternative is a per-literal judgment about whether a value is sensitive,
which is the kind of guess this codebase refuses to make elsewhere and the
one that fails silently and unrecoverably when it guesses wrong.

**The fallback path is deliberately less precise, and deliberately not a
crash.** A Postgres error-log line, SQL Server's `deadlock_graph` XML,
MySQL's `SHOW ENGINE INNODB STATUS` blob, a plan's text representation, or
a valid statement chopped mid-literal (the adapters themselves truncate at
`left(query, 200)` / `LEFT(INFO, 500)`) are all normal inputs here.
Dropping the row or raising would blind the DBA to exactly the diagnostic
they asked for and turn a successful tool call into a FAILED one, so
`scrub_sql_literals` falls back to a regex scrub of obvious literal shapes
— single-quoted runs and bare numeric runs — instead of the AST. With no
structure there is no way to tell a value from an identifier, so that path
over-redacts on purpose (timestamps and PIDs in a log line go too). Two
deliberate carve-outs: double-quoted text is preserved, because in standard
SQL it is an *identifier* and in Postgres log text it is almost always the
object name the DBA needs (`violates unique constraint "accounts_pkey"`);
and sqlglot is always tried first, so anything that really is SQL gets the
precise treatment.

**Why a successful parse is not enough on its own.** sqlglot is permissive
by design and `ErrorLevel.RAISE` does not cover this. Verified against
sqlglot 30.17: `"Jane Doe"` parses cleanly as an `exp.Alias` and
round-trips as `Jane AS Doe` — the value survives unredacted *and* the text
is mangled; `"1234567890"` parses as a bare `exp.Literal`. Accepting those
would be strictly worse than the fallback. So a parse counts only when the
root node is an actual statement (`_SQL_STATEMENT_TYPES`); a bare
expression or fragment is treated as unparseable and handed to the regex
path, which redacts both of those examples correctly.

Cost, measured on the worst realistic case (a full 100-row result — the
`max_rows` cap — every row carrying a complete statement): ~142ms, about
1.4ms per statement. A result of plain metric columns costs ~0.9ms total,
since no field name matches and no parse is attempted. Both are immaterial
against the database round trip and the 20s per-decision LLM deadline.

Only `str` values are scrubbed. That is not incidental: PostgreSQL's
`deadlocks()` returns a numeric `deadlocks` COUNTER, and several engines
return NULL query text for an idle session — running a text scrubber over
an int would destroy a metric while protecting nothing. The Gateway reports
`literal_scrubbed_fields` alongside the existing `masked_fields` in the
tool result, kept distinct because the two mean different things: a masked
field is gone, a scrubbed field is still there and still readable and only
its values were replaced. See `tests/unit/test_query_scrubber.py` and the
no-regression guards in `tests/unit/test_data_policy.py`.

## The least-privilege premise was documented, never verified

`execution/discovery/base.py`'s module docstring has always stated the
premise this whole architecture rests on: "The diagnostic login only needs
read access to catalog / DMV / stats views ... It should NOT have SELECT on
user tables — Numi never reads table contents." Nothing ever checked it.
That made it a statement of intent rather than a control: a login
provisioned with `db_datareader`, a Postgres superuser, or a MySQL account
carrying a stray `GRANT SELECT ON *.*` all work perfectly and silently hold
far more authority than the design calls for. Every other control in
SECURITY.md is enforced in code; this one was enforced by hoping whoever
provisioned the account read the docs.

Each `ServerDiscoverer` now implements `_check_least_privilege`, run once
per discovery run on the connection the crawl already holds — not per
database, and never per tool call. Discovery is already lazy-on-first-use
and re-run by `/discover`, so this rides along on an existing trip rather
than adding one. The result is a structured `LeastPrivilegeFinding` on
`ServerCatalog`.

**Effective permissions, not grant tables, wherever the engine offers them.**
This is the difference between a check that works and one that misses its
own main cases:

- **PostgreSQL**: `pg_catalog.has_table_privilege(current_user, oid,
  'SELECT')` over `pg_class`, excluding `pg_catalog`/`information_schema`
  and the TOAST/temp schemas. Reading
  `information_schema.table_privileges` instead — the obvious first
  instinct — would miss a superuser entirely: a superuser holds SELECT on
  everything while being listed as grantee of nothing, and that is the
  worst login this check exists to find.
- **SQL Server**: `HAS_PERMS_BY_NAME(..., 'OBJECT', 'SELECT')` over
  `sys.objects` (`type IN ('U','V')`, `is_ms_shipped = 0`, system schemas
  excluded). `sys.database_permissions` records only *explicit* grants, so
  it returns nothing for a login whose SELECT arrives via `db_datareader`
  membership — by far the most common way a diagnostic account ends up able
  to read user data, and likewise nothing for a `sysadmin`.
- **MySQL/MariaDB**: no per-object effective-permission function exists, so
  the grant tables are the only source and all three levels are unioned —
  `USER_PRIVILEGES`, `SCHEMA_PRIVILEGES`, `TABLE_PRIVILEGES`. Reading only
  `TABLE_PRIVILEGES` would miss both broader cases: `GRANT SELECT ON *.*`
  and `GRANT SELECT ON appdb.*` leave no row there at all despite granting
  strictly more access than any per-table grant.

**Read-only, always.** This is introspection of the engine's own privilege
views and nothing else — Numi reports, a human DBA revokes. No code path
here attempts a REVOKE or any other change, pinned per engine by
`test_no_statement_ever_attempts_to_change_a_privilege`.

**A finding is never fatal, and a failed check is never a clean bill of
health.** A privilege view the login can't read is a normal outcome on a
locked-down server, and the catalog is perfectly usable without this field,
so `run_least_privilege_check` converts any failure into
`checked=False` — the same best-effort posture discovery already takes for
a database it can't enumerate. That is deliberately distinct from
`checked=True, has_user_table_select=False`: a check that never ran must
not be reported as clean. Neither renders a warning, but only the latter
means anything.

**Honest about scope.** A positive finding is conclusive. A negative one is
scoped, and says so: PostgreSQL and SQL Server both scope relation
visibility per database, so the check covers the database the discovery
connection is bound to, and `scope_note` carries that caveat to the DBA
rather than implying instance-wide coverage. MySQL's privilege views are
genuinely instance-wide, so there a clean result really is clean
everywhere. The scan is row-capped at 200; hitting the cap sets
`count_is_lower_bound` so the reported number reads "at least N" instead of
being silently wrong.

**Surfacing.** The finding gets its own WARNING-level structlog record in
`gateway/domain/discovery.py::refresh_server` (`least_privilege_violation`)
so it reaches log-based alerting without anyone reading a catalog, and the
DBA-facing copy renders in `/catalog <server>`
(`orchestrator.py::_handle_catalog_command`): "⚠️ This server's diagnostic
login (numi_diag) has SELECT on 12 user table/views (e.g. dbo.Accounts,
dbo.Customers) — should be revoked for least-privilege". Both surfaces
render `LeastPrivilegeFinding.warning_text()` rather than deriving their
own wording, the same single-source-of-truth reasoning `_verification_note`
follows for verification outcomes, so the log and the chat reply can never
disagree.

`least_privilege` is a dedicated optional field rather than another entry
in `ServerCatalog.warnings`: that list is free text about what one crawl
couldn't read, whereas this is a structured, durable security finding about
the login itself that the renderer needs the count, sample and login name
from. Optional so a catalog discovered before this check existed round-trips
unchanged through the catalog store and still renders — pinned by
`test_a_catalog_predating_this_feature_still_renders`. The sample carries
object *names* only, never row data, which would rather defeat the point.

See `tests/unit/test_least_privilege_check.py` (per engine, both the
over-privileged and clean cases, via `FakeQueryExecutor` — no real
connection) and `tests/integration/test_least_privilege_surfacing.py`,
which drives the real `/catalog` command through the real orchestrator and
Gateway to prove the warning actually reaches the DBA, since a finding
nobody ever sees is the same as no finding.

## kill_session requires approval even in development, and a rejected Conclude's evidence note doesn't repeat itself

Two related, live-found issues while testing the deterministic mock
planner end to end against a real blocking scenario.

**`kill_session` was `ALLOW` for every role in every environment,
including development.** Every other write tool in `config/policy.yaml`'s
development tier that actually changes something requires at least an
approval click (`create_index`/`rebuild_index` for `DBA_L1`,
`modify_configuration`, `restart_instance`, `failover`) — `kill_session`
was the one exception, executing with zero human confirmation step at
all. Confirmed live: a genuine blocking chain (a real `FOR UPDATE` lock
held by one session, blocking a second) got investigated and the head
blocker killed automatically, no approval card, no pausedone reviewing. now c. Fixed by making
`database.kill_session` `REQUIRES_APPROVAL` for every role in development
too. This is deliberately a *single*-approval gate, not dual —
`kill_session`'s own `risk_level` is `MEDIUM` (`tool_catalog.py`), so
`ApprovalEngine.approve` never sets `requires_dual_approval` for it (see
its own `requires_dual_approval` branch), which means the requester can
still approve their own request in one extra click — this only adds a
pause-and-confirm step, not a second-person requirement. (`restart_instance`/
`failover`, tested separately this same session, *are* dual-approval — the
self-approval block that test triggered is specific to
`requires_dual_approval=True` tools, not a blanket rule; the code already
drew this distinction correctly, this session's own summary just stated
it too broadly the first time.) See
`tests/unit/test_policy_engine.py::test_kill_session_requires_approval_even_in_development_for_every_role`.

**A rejected `Conclude`'s evidence note could repeat itself verbatim.**
`_finalize_conclude` rejects a `Conclude` that would report a write as
done before an independent re-check has happened
(`investigation.pending_verification`) — a real model, shown the
`internal.verification_check` transcript entry this appends, usually
course-corrects and actually calls the suggested verification tool. The
*deterministic* mock planner can't: its `decide_next_action` only ever
reads the `executed` tool-id set from the transcript, never the freeform
guidance text appended to a rejected turn, so once it has executed
`kill_session` it keeps proposing the exact same `Conclude` every
remaining turn — and every rejection appended the identical sentence to
`investigation.evidence` again. A DBA testing the blocking flow with the
mock provider saw the same "(a draft conclusion after
database.kill_session was rejected — not yet independently verified)"
sentence three times in a row in the final summary. Same problem, same
fix, for the sibling ungrounded-identifier rejection path. Fixed with
`AgentOrchestrator._append_evidence_once`: skip appending to
`investigation.evidence` when the text is a byte-identical repeat of the
entry already at the end — `investigation.transcript` (what a real model
actually reads to try to course-correct) still gets the reminder fresh
every turn, unchanged; only the DBA-facing summary de-duplicates.
