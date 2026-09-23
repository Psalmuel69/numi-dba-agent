"""A small library of fixed, named diagnostic sequences for the handful of
DBA scenarios that come up over and over (slow queries, high CPU, blocking,
...), plus a keyword matcher that picks one for a new investigation.

Why this exists (not a capability the Agent lacked — see the design note
below): freeform investigation already *can* call any read-only tool in any
order, and does so successfully. What it doesn't have on its own is a fixed,
predictable shape for a *known* scenario type — verified live, this is what
let one real investigation ("list the tables in AdventureWorks2019...") run
several unrelated diagnostics (blocking, wait stats, top queries) after
already having the answer, burn the whole turn budget, and return a vague
"no root cause found" instead of just answering. A playbook is a named,
reviewable, deterministic answer to "what do we check, in what order, for
this kind of problem" — and because each playbook's tool sequence is fixed
in advance, `_run_investigation_loop` can *execute* it without an LLM call
per step (only one call at the end, to interpret the gathered evidence and
conclude). That is a deliberate trade: a playbook trades "the model decides
every step" for "the model decides once, from a complete picture" — fewer
LLM round-trips per investigation (faster, cheaper, matters under the
_OVERALL_DEADLINE_SECONDS ceiling), and the same DBA question always
investigates the same way (auditable, reviewable ahead of time — every step
here still goes through the Gateway's own independent authorization/policy/
risk pipeline exactly like any other tool call; a playbook only decides
*which* read-only tool to propose next, never that it's allowed to run).

Every step here is a read-only diagnostic tool (`database.get_*`) with
either no arguments or arguments that are entirely fixed/self-contained
(e.g. `top_queries` ordered by the metric that scenario cares about) — a
playbook never proposes a write. A playbook only pre-selects *which*
diagnostics to run; if a Conclude action recommends a remediation, that
still goes through the normal LLM-proposes / Gateway-approves flow like any
other action (spec §7, §37) — nothing here bypasses approval.

When no playbook's triggers match the problem text, the investigation falls
back to the existing freeform loop exactly as before this feature existed —
this is additive, not a replacement."""

from __future__ import annotations

import dataclasses
import re


@dataclasses.dataclass(frozen=True)
class PlaybookStep:
    tool_id: str
    # Shown to the DBA as this step's `reason` — what this step is checking
    # and why, in the context of the scenario the playbook is for.
    purpose: str
    # Fixed, self-contained arguments (e.g. {"order_by": "cpu"}) — never
    # anything that needs to be inferred from the conversation (a session
    # id, a schema/table name); those stay in the freeform LLM-driven path.
    arguments: dict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class Playbook:
    playbook_id: str
    name: str
    description: str
    # Trigger phrases matched case-insensitively, whole-word/whole-phrase
    # (never a bare substring — see _matches, this is what keeps "log" from
    # matching "login" and "lock" from matching "blocking" unintentionally
    # where that's not wanted, while still matching intended overlaps like
    # "lock" inside "deadlock" being a *separate*, deliberately-word-bounded
    # non-match).
    triggers: tuple[str, ...]
    steps: tuple[PlaybookStep, ...]
    # Steers the final conclude call once all steps have run — what "done"
    # looks like for this specific scenario.
    conclusion_guidance: str


def _step(tool_id: str, purpose: str, **arguments: object) -> PlaybookStep:
    return PlaybookStep(tool_id=tool_id, purpose=purpose, arguments=arguments)


# Ordered most-specific-first: a message can plausibly match more than one
# playbook's triggers (e.g. "deadlock" also concerns blocking), and the
# first match wins — see match_playbook.
PLAYBOOKS: tuple[Playbook, ...] = (
    Playbook(
        playbook_id="deadlocks",
        name="Deadlock Investigation",
        description="A reported deadlock or repeated deadlocking.",
        triggers=("deadlock", "deadlocks", "deadlocked", "deadlocking", "dead lock"),
        steps=(
            _step("database.get_deadlocks", "Pulling recent deadlock graphs."),
            _step("database.get_blocking_sessions", "Checking for blocking still in progress."),
            _step("database.get_running_queries", "Checking what's running now around the same objects."),
            _step(
                "database.get_sessions",
                "Checking session durations for long-running transactions that could "
                "be contributing to the lock-acquisition cycle.",
            ),
        ),
        conclusion_guidance=(
            "Reconstruct the lock-acquisition cycle from the deadlock graph(s): "
            "which sessions/queries were involved, what each held and what each "
            "was waiting on, and in what order. State which session was chosen as "
            "the deadlock victim and whether the same access pattern is still "
            "occurring now. Then identify the root-cause category the evidence "
            "actually supports, rather than a generic 'a deadlock happened': "
            "inconsistent table/object access order across the competing "
            "transactions, a missing index forcing a broader scan-and-lock "
            "footprint than necessary, a long-running transaction (check session "
            "durations) holding locks longer than needed, a large batch operation "
            "or lock escalation turning many row locks into a table lock, an "
            "isolation level stricter than the workload needs, a trigger or "
            "foreign-key check acquiring an additional unexpected lock, or a "
            "maintenance operation contending with normal traffic. If more than "
            "one deadlock graph was returned, note how often this pattern is "
            "recurring and whether it clusters around a particular time. "
            "Recommend a fix that matches the actual root cause — consistent "
            "access order, an index, smaller transactions/batches, "
            "application-level retry handling, an isolation-level review, or "
            "separating conflicting workloads — rather than generic advice."
        ),
    ),
    Playbook(
        playbook_id="blocking",
        name="Blocking / Locking Investigation",
        description="Sessions blocked, stuck, or waiting on locks.",
        triggers=(
            "blocking", "blocked", "block chain", "locking", "lock wait",
            "stuck", "hanging", "won't complete", "not completing",
        ),
        steps=(
            _step("database.get_blocking_sessions", "Mapping the current blocking chain."),
            _step("database.get_running_queries", "Checking what the head blocker and waiters are running."),
            _step("database.get_wait_statistics", "Checking whether lock waits dominate overall wait time."),
            _step("database.get_sessions", "Checking how long the blocking session has been open."),
        ),
        conclusion_guidance=(
            "Identify the head blocker (session, query, how long it's held the "
            "lock) and who's waiting on it. Classify what the head blocker "
            "actually is, not just that it's blocking: an active/long-running "
            "query, an idle-in-transaction session (its statement finished but "
            "the transaction was never committed/rolled back), a large batch "
            "operation, a reporting query, a DDL statement, a maintenance "
            "operation, a replication/log-shipping process, or unknown if the "
            "evidence doesn't clearly say. Quantify the impact: how many "
            "sessions are blocked, the longest single wait, and whether the "
            "chain is static or spreading (more waiters appearing behind the "
            "same head blocker). Then test — not just narrate — the plausible "
            "hypotheses against the evidence actually gathered: a long-running "
            "transaction holding locks past when they were needed, a missing "
            "index forcing broader locking, lock escalation to a table-level "
            "lock, DDL contention (a schema change blocking readers/writers), "
            "an isolation level stricter than necessary, an application "
            "connection/transaction leak (a session left open after its work "
            "finished), a large batch operation, or a broader infrastructure "
            "slowdown making every query — including the blocker itself — run "
            "long. State which hypothesis the evidence supports and which it "
            "rules out. Recommend cancelling the blocking query or killing the "
            "session only if it's clearly safe to do so — never propose it as "
            "the only option without saying what it would affect."
        ),
    ),
    Playbook(
        playbook_id="slow_queries",
        name="Slow Query Investigation",
        description="Queries or the database generally running slower than expected.",
        triggers=(
            "slow", "slowness", "slow query", "slow queries", "performance issue",
            "taking too long", "timing out", "timeout", "high latency", "latency",
            "queries are slow", "query is slow", "running slow",
        ),
        steps=(
            _step("database.get_health", "Establishing a baseline health snapshot."),
            _step("database.get_running_queries", "Checking what's executing right now."),
            _step("database.get_top_queries", "Ranking queries by duration.", order_by="duration"),
            _step("database.get_wait_statistics", "Checking what the engine is spending time waiting on."),
            _step("database.get_blocking_sessions", "Ruling out blocking as the cause."),
        ),
        conclusion_guidance=(
            "Name the specific query/queries responsible if the evidence points "
            "to one, and whether the cause looks like a missing index, stale "
            "statistics, blocking, or resource pressure (CPU/waits) rather than "
            "the query itself. Avoid introspection/system-catalog queries when "
            "picking 'the' slow query — THIS IS VERY IMPORTANT: a top-ranked row "
            "against the engine's own catalog/metadata views (Postgres: "
            "pg_catalog, information_schema; SQL Server: sys.*, "
            "INFORMATION_SCHEMA; MySQL/MariaDB: information_schema, "
            "performance_schema, mysql) is noise from monitoring/introspection, "
            "almost never the reported slowness, and must not be named as the "
            "root cause."
        ),
    ),
    Playbook(
        playbook_id="high_cpu",
        name="High CPU Investigation",
        description="Elevated or spiking CPU usage on the instance.",
        triggers=("cpu", "high cpu", "cpu spike", "cpu usage", "cpu utilization", "processor usage"),
        steps=(
            _step("database.get_health", "Establishing a CPU/resource baseline."),
            _step("database.get_top_queries", "Ranking queries by CPU consumption.", order_by="cpu"),
            _step("database.get_running_queries", "Checking what's currently executing."),
            _step("database.get_wait_statistics", "Checking for CPU-related (signal) waits."),
        ),
        conclusion_guidance=(
            "Name the query/queries driving CPU if the evidence points to one "
            "or a small number, and whether this looks like a single runaway "
            "query, a plan regression, or broad concurrent load. Avoid "
            "introspection/system-catalog queries when picking 'the' hot query "
            "— THIS IS VERY IMPORTANT: a top-ranked row against the engine's "
            "own catalog/metadata views (Postgres: pg_catalog, "
            "information_schema; SQL Server: sys.*, INFORMATION_SCHEMA; "
            "MySQL/MariaDB: information_schema, performance_schema, mysql) is "
            "noise from monitoring/introspection, almost never the reported "
            "CPU driver, and must not be named as the root cause."
        ),
    ),
    Playbook(
        playbook_id="high_memory",
        name="High Memory Investigation",
        description="Memory pressure, high memory usage, or out-of-memory conditions.",
        triggers=(
            "memory", "high memory", "memory pressure", "out of memory", "oom",
            "memory usage", "memory leak", "running out of memory",
        ),
        steps=(
            _step("database.get_health", "Establishing a memory baseline."),
            _step("database.get_configuration", "Checking configured memory limits."),
            _step(
                "database.get_top_queries",
                "Ranking queries by reads (buffer/memory pressure proxy).",
                order_by="reads",
            ),
            _step("database.get_running_queries", "Checking what's currently executing."),
        ),
        conclusion_guidance=(
            "State whether the configured memory limit looks undersized for the "
            "observed load, or whether a small number of queries are driving "
            "the pressure. Before recommending anything, work out where the "
            "pressure is actually coming from: the database engine's own "
            "allocations (buffer pool/shared buffers/plan cache), a specific "
            "query or small set of queries, or pressure outside the engine "
            "entirely (the host, a container/VM memory limit, or another "
            "service sharing the box) — note plainly that host/container-level "
            "memory isn't something this system can directly measure, so treat "
            "that as a hypothesis to raise when the in-engine evidence doesn't "
            "explain the pressure, not a measured finding. Validate — don't "
            "just assume — each plausible hypothesis against the evidence "
            "gathered: an oversized memory grant/setting relative to the actual "
            "workload, a poor query plan requesting more memory than it needs "
            "(e.g. a bad cardinality estimate driving a large sort/hash), "
            "excessive concurrent sessions each holding their own allocation, a "
            "genuine leak (usage that only climbs, with no corresponding "
            "drop), a configuration value left at an unreasonable default, "
            "host-level overcommitment, or the instance being undersized for "
            "its actual load. Treat a restart as a last resort that requires "
            "explicit DBA approval — never the first recommendation, and never "
            "proposed unprompted before the above has actually been checked; "
            "when it is worth mentioning, say plainly that it only clears "
            "symptoms like a leak or plan-cache bloat and does not fix a "
            "genuine undersizing or misconfiguration, which will recur."
        ),
    ),
    Playbook(
        playbook_id="connections",
        name="Connection Saturation Investigation",
        description="Connection limits, refused/failed connections, or too many open sessions.",
        triggers=(
            "connection", "connections", "too many connections", "connection pool",
            "max connections", "can't connect", "cannot connect", "connection refused",
            "connection limit", "out of connections",
        ),
        steps=(
            _step("database.get_sessions", "Listing current sessions."),
            _step("database.get_health", "Establishing a baseline."),
            _step("database.get_configuration", "Checking the configured max-connections limit."),
        ),
        conclusion_guidance=(
            "State the current session count against the configured limit, and "
            "whether one application/account is holding a disproportionate "
            "number of connections. Lead with that comparison: if current usage "
            "is well under the configured max (comfortable headroom), say so "
            "plainly and stop there — do not manufacture a false alarm or "
            "invent tuning advice from otherwise-healthy numbers just because "
            "an investigation was run. Reserve genuine concern for when usage "
            "is meaningfully close to the limit, or one application/account "
            "holds a disproportionate share regardless of the overall total."
        ),
    ),
    Playbook(
        playbook_id="replication",
        name="Replication Investigation",
        description="Replication lag, Always On, or streaming replication issues.",
        triggers=(
            "replication", "replica", "replicas", "replication lag", "lagging",
            "always on", "availability group", "streaming replication", "standby",
        ),
        steps=(
            _step("database.get_replication_status", "Checking replication/Always On status and lag."),
            _step("database.get_health", "Establishing a primary-side health baseline."),
            _step("database.get_wait_statistics", "Checking for waits consistent with replication pressure."),
        ),
        conclusion_guidance=(
            "State the current lag (or sync state) per replica and whether it's "
            "within the expected range for this environment — but go beyond the "
            "raw number: assess stale-read risk for anything reading from that "
            "replica, RPO/RTO exposure if the primary failed over right now, "
            "whether the lag is itself creating primary-side risk (a lagging or "
            "disconnected replica can hold WAL/log open, which shows up as "
            "unexpected storage growth on the primary — worth a separate storage "
            "check if that's suspected), whether the lag/queue is growing or "
            "holding steady, and which applications or read paths are actually "
            "affected. Validate the likely cause "
            "against the evidence rather than only reporting the lag figure: "
            "network latency between primary and replica, storage or CPU "
            "pressure on the replica itself, an unusually large write volume on "
            "the primary outpacing apply capacity, a long-running transaction "
            "on the replica blocking apply, a disconnected or dropped replica, "
            "replication-slot retention holding WAL/log no longer being "
            "consumed, an apply/redo worker that has stalled or errored, or the "
            "replica being undersized for the write rate it needs to keep up "
            "with."
        ),
    ),
    Playbook(
        playbook_id="backups",
        name="Backup Health Investigation",
        description="Missed, failed, or overdue backups.",
        triggers=("backup", "backups", "backup failed", "backup job", "last backup", "backup status"),
        steps=(
            _step("database.get_backup_status", "Checking recent backup history and status."),
            _step("database.get_storage", "Checking available storage for the next backup."),
        ),
        conclusion_guidance=(
            "State the time and status of the most recent full/log backup and "
            "whether it's within the expected recovery-point objective for this "
            "environment — and do the actual arithmetic: explicitly calculate "
            "and state the recovery-point gap (time elapsed since the last "
            "successful full backup, and separately since the last successful "
            "log/differential backup) against the environment's expected RPO, "
            "rather than only reporting whether the most recent job "
            "'succeeded' or 'failed'. If the backup history data returned "
            "includes any signal about restore-test history (e.g. a last "
            "restore-test timestamp or status), report it; if no such signal "
            "is present, say plainly that restore-test history isn't available "
            "from this data rather than guessing or omitting it silently. "
            "Classify the overall finding with an explicit severity instead of "
            "a flat pass/fail: informational (healthy, comfortably within "
            "RPO), warning (approaching the RPO boundary, or a single "
            "non-critical job failed), high (the RPO is already exceeded, or "
            "the full-backup chain is broken), or critical (no successful "
            "backup within a large multiple of the RPO, or backups have been "
            "failing repeatedly with no current recovery point achievable). "
            "State which severity applies and why."
        ),
    ),
    Playbook(
        playbook_id="storage",
        name="Storage Capacity Investigation",
        description="Disk space or overall storage capacity risk (not specifically the transaction log).",
        triggers=(
            "disk space", "disk full", "running out of space", "out of disk",
            "out of space", "storage",
        ),
        steps=(
            _step("database.get_storage", "Checking overall storage/space utilization."),
            _step("database.get_health", "Establishing a baseline."),
        ),
        conclusion_guidance=(
            "State current usage vs. capacity, and — if the evidence points to "
            "one — what's actually driving growth: unexpected data growth, "
            "transaction log retention, a failed/paused backup chain, table/"
            "index bloat, an archive/purge job that stopped running, "
            "auto-growth events, replication backlog holding old WAL/log "
            "segments, or storage throttling. If the transaction log "
            "specifically looks like the driver, say so explicitly and note "
            "that the transaction_log playbook (triggered by phrasing like "
            "'transaction log full' or 'log won't truncate') investigates log "
            "reuse blockers in depth."
        ),
    ),
    Playbook(
        playbook_id="transaction_log",
        name="Transaction Log Investigation",
        description="The transaction log (or WAL) growing, filling up, or failing to reuse space.",
        triggers=(
            "transaction log full", "transaction log growth", "log growing",
            "log full", "log is full", "wal growing", "log can't reuse",
            "log cannot reuse", "log won't truncate", "log will not truncate",
            "log reuse", "transaction log",
        ),
        steps=(
            _step("database.get_transaction_log", "Checking transaction log / WAL usage."),
            _step(
                "database.get_replication_status",
                "Checking for a lagging replica holding the log/WAL open.",
            ),
            _step(
                "database.get_backup_status",
                "Checking for a stalled or failed log backup blocking log reuse.",
            ),
            _step("database.get_health", "Establishing a baseline."),
        ),
        conclusion_guidance=(
            "Identify which concrete reuse blocker is actually responsible, "
            "not just current log usage: an active or long-running "
            "transaction, replication lag (a lagging replica holding the log "
            "open), a failed or stalled log backup (the single most common "
            "reuse blocker on SQL Server, and often tied to WAL retention/"
            "archiving on Postgres), log shipping/mirroring/snapshots holding "
            "old log records, or a maintenance operation blocking log reuse. "
            "Name the specific blocker the evidence supports rather than "
            "describing usage alone."
        ),
    ),
    Playbook(
        playbook_id="errors",
        name="Error Log Investigation",
        description="Errors, exceptions, or failures reported in the error log.",
        triggers=(
            "error log", "error logs", "errors in the log", "exceptions",
            "failing queries", "crashing", "keeps crashing", "seeing errors",
        ),
        steps=(
            _step("database.get_error_logs", "Pulling recent error log entries."),
            _step("database.get_health", "Establishing a baseline."),
            _step(
                "database.get_blocking_sessions",
                "Checking for current blocking to correlate against any lock-timeout "
                "or blocking-related error entries.",
            ),
            _step(
                "database.get_deadlocks",
                "Checking for recent deadlock graphs to correlate against any "
                "deadlock-related error entries.",
            ),
            _step("database.get_running_queries", "Checking what's currently executing."),
        ),
        conclusion_guidance=(
            "Summarize the distinct error(s) found (not just a raw dump), how "
            "recent/frequent each is, and which looks most likely to be the "
            "reported problem. Classify each distinct finding into a category "
            "rather than a raw list of log lines — pick whichever of these "
            "actually fits: availability, authentication, authorization, "
            "storage, memory, CPU, network, corruption, backup, replication, "
            "locking, configuration, security, or application integration. "
            "Correlate against the other signals gathered in this same "
            "investigation when relevant, rather than treating log entries in "
            "isolation: a lock-timeout or deadlock-related error should be "
            "cross-checked against the blocking-chain and deadlock-graph "
            "evidence gathered above to say whether it's still happening or "
            "was transient, and note when a logged error's underlying "
            "condition no longer appears in the other evidence gathered (it "
            "looks resolved) versus still being present now (still ongoing)."
        ),
    ),
    Playbook(
        playbook_id="comprehensive_summary",
        name="Comprehensive Health Summary",
        description=(
            "A DBA-requested comprehensive sweep across this project's "
            "read-only diagnostics for ONE server in one shot — not tied to a "
            "specific symptom, and deliberately broader/heavier than "
            "general_health's quick 4-step pulse check. Scoped honestly, two "
            "ways: (1) this reports the server's CURRENT state only — there "
            "is no historical data store anywhere in this system, so it "
            "cannot do trend/delta comparison against yesterday or any prior "
            "run — and it covers exactly one server per invocation, never a "
            "fleet. Sweeping multiple servers on a schedule is separate "
            "orchestration that calls this playbook once per server; it now "
            "exists (`agent.scheduled_report`, via "
            "`orchestrator.run_comprehensive_summary`), but it is still not "
            "something this playbook does itself, and nothing about this "
            "playbook changes when it is invoked that way except that the "
            "investigation is marked read-only. (2) Every investigation in "
            "this project — playbook or "
            "freeform — shares one turn budget, `orchestrator._MAX_"
            "INVESTIGATION_TURNS` (6 as of this writing), and every playbook "
            "in this library is required to leave at least one turn free for "
            "the model's own unrestricted concluding call (see "
            "`test_deepened_playbooks_stay_within_the_shared_turn_budget`), "
            "so a single invocation can run at most 5 deterministic "
            "diagnostic calls before that final call. Rather than declare 12 "
            "steps — one per instance-wide read tool — and have most of them "
            "silently never run (verified live — the turn budget simply "
            "stops submitting further steps once spent, with no warning), "
            "this playbook's 5 steps were chosen to span the widest "
            "practical breadth — availability, resource pressure, workload/"
            "blocking, protection/backups, storage capacity, and logs — "
            "within that shared budget. Configuration is deliberately left "
            "out of this particular sweep: it already has its own dedicated "
            "`configuration_review` playbook, and cutting it here (rather "
            "than blocking, backups, storage, or logs) keeps the daily-pulse "
            "categories that change moment to moment. Raising the shared cap "
            "(or giving this one playbook a larger budget of its own) is an "
            "orchestrator.py change and explicitly out of scope here."
        ),
        triggers=(
            "daily summary", "comprehensive health check", "full health report",
            "complete status", "everything about this server", "full report",
            "daily report", "comprehensive check", "complete health check",
            "full diagnostic",
        ),
        # Exactly 5 steps, one short of the shared 6-turn budget — see the
        # description above for why this playbook does not simply list every
        # instance-wide read tool, and
        # test_deepened_playbooks_stay_within_the_shared_turn_budget for the
        # invariant every playbook in this library must satisfy: a 6th fixed
        # step here would consume the model's own final unrestricted
        # conclude/extend turn. Ordered availability/workload before
        # protection/config, most-urgent-first — matching how a DBA would
        # actually triage.
        steps=(
            _step("database.get_health", "Establishing the overall instance/database health baseline."),
            _step("database.get_blocking_sessions", "Checking for any active blocking chains."),
            _step("database.get_backup_status", "Checking recent backup history and status."),
            _step("database.get_storage", "Checking storage/space utilization."),
            _step("database.get_error_logs", "Pulling recent error log entries."),
        ),
        conclusion_guidance=(
            "This is the broadest playbook in the library — the concluding "
            "report must synthesize, not restate. Organize findings into the "
            "categories a DBA actually triages by, mirroring the General "
            "Health Check structure used elsewhere in this project: "
            "availability (is it up, overall health snapshot), resource "
            "pressure (from get_health's own numbers), workload/blocking "
            "(active blocking chains), protection/backups (backup status, "
            "storage headroom), and logs (recent error log entries). Only "
            "speak to a category you actually have evidence for — this "
            "playbook's fixed steps do not include a replication/HA or "
            "configuration check, so never state or imply replication is "
            "'fine' or that configuration looks reasonable; simply omit "
            "those categories rather than guessing (point the DBA at the "
            "dedicated `replication` or `configuration_review` playbook if "
            "either looks worth a closer look). THIS IS THE IMPORTANT PART: "
            "for each category you do have evidence for, report ONLY what "
            "deviates from normal or needs attention — do not restate a "
            "clean result at all, for any "
            "category. Close with an explicit sentence to the effect of 'all "
            "other checks came back clean' (naming which categories, if it "
            "reads better) covering everything unremarkable, instead of "
            "walking through every check one by one. A DBA asking for a "
            "comprehensive summary wants a short, scannable list of what "
            "matters, not a wall of 'X is fine, Y is fine, Z is fine' for a "
            "half-dozen checks. Also state plainly, once, that this reflects "
            "only the current moment on this one server — no trend/delta "
            "comparison against a prior day is available (no historical data "
            "store exists in this system), and this is not a multi-server "
            "sweep."
        ),
    ),
    Playbook(
        playbook_id="general_health",
        name="General Health Check",
        description="A general request to check overall health/status, not a specific symptom.",
        triggers=(
            "health check", "how is", "how's", "overall status", "general status",
            "status check", "how is it doing", "how's it doing", "everything okay",
            "everything ok",
        ),
        steps=(
            _step("database.get_health", "Checking overall instance/database health."),
            _step("database.get_running_queries", "Checking current activity."),
            _step("database.get_wait_statistics", "Checking dominant wait types."),
            _step("database.get_storage", "Checking storage headroom."),
        ),
        conclusion_guidance=(
            "Give a concise overall status and call out anything that stood "
            "out, even if nothing looks urgent. Classify the overall finding "
            "with one explicit status word — healthy, informational, warning, "
            "high, critical, or unknown (use unknown only when the "
            "diagnostics themselves failed or returned unusable data, never as "
            "a hedge when they succeeded) — rather than a loose narrative. "
            "Crucially: an all-clear answer must look just as complete and "
            "confident as a problem answer, never vaguer or shorter just "
            "because nothing was wrong — a DBA reading the reply should not be "
            "able to tell, from how thin it is, whether things were actually "
            "checked and found fine versus not checked carefully. State "
            "plainly what was checked (current activity, dominant wait types, "
            "storage headroom) and that nothing concerning was found there, "
            "rather than a one-line 'looks fine'."
        ),
    ),
    Playbook(
        playbook_id="configuration_review",
        name="Configuration Tuning Review",
        description=(
            "A proactive review of server/database configuration for common, "
            "rule-of-thumb misconfigurations — not a request tied to a specific "
            "symptom. Scoped honestly: this flags settings that look clearly "
            "unreasonable by common rule-of-thumb ranges and obvious "
            "misconfigurations, not true capacity-based sizing (Numi's server "
            "registry has no instance-class/hardware-sizing data to size "
            "against)."
        ),
        triggers=(
            "tune", "tuning", "configuration review", "review settings",
            "optimize configuration", "recommend settings", "config recommendations",
            "are our settings okay",
        ),
        steps=(
            _step("database.get_configuration", "Pulling current server/database configuration parameters."),
            _step(
                "database.get_health",
                "Establishing a baseline (connection count, size, uptime) to contextualize the settings.",
            ),
        ),
        conclusion_guidance=(
            "This is a rule-of-thumb review, not a capacity-sized recommendation "
            "— be explicit that no instance-class/hardware (CPU/RAM) data was "
            "available to size against, so never claim a setting is 'correctly "
            "sized for this hardware'; only flag values that look clearly "
            "unreasonable or left at an obvious default. Weigh the parameters "
            "relevant to whichever engine actually responded: Postgres — "
            "shared_buffers, effective_cache_size, work_mem, "
            "maintenance_work_mem, wal_buffers, checkpoint_completion_target, "
            "max_connections, default_statistics_target, random_page_cost, "
            "effective_io_concurrency, min_wal_size/max_wal_size, "
            "max_worker_processes, max_parallel_workers(_per_gather). SQL "
            "Server — 'max server memory (MB)'/'min server memory (MB)', 'max "
            "degree of parallelism', 'cost threshold for parallelism', and the "
            "max-connections-equivalent ('user connections'). MySQL/MariaDB — "
            "innodb_buffer_pool_size, innodb_log_file_size, max_connections, "
            "tmp_table_size/max_heap_table_size, innodb_flush_log_at_trx_commit, "
            "thread_cache_size. Cross-check against get_health's own numbers "
            "(e.g. active_connections vs. max_connections) where relevant, and "
            "say plainly when nothing looks obviously misconfigured rather than "
            "manufacturing tuning advice from reasonable-looking defaults."
        ),
    ),
)

_BY_ID: dict[str, Playbook] = {p.playbook_id: p for p in PLAYBOOKS}


def get_playbook(playbook_id: str | None) -> Playbook | None:
    if playbook_id is None:
        return None
    return _BY_ID.get(playbook_id)


def _matches(text: str, trigger: str) -> bool:
    return re.search(r"\b" + re.escape(trigger) + r"\b", text) is not None


def match_playbook(problem_text: str) -> Playbook | None:
    """Deterministic, zero-latency, zero-LLM-call keyword match — this
    selection has to be cheap and instant since it runs on every new
    investigation, before any LLM call. Returns None (freeform investigation,
    unchanged) when nothing matches."""
    text = (problem_text or "").lower()
    if not text.strip():
        return None
    for playbook in PLAYBOOKS:
        if any(_matches(text, trigger) for trigger in playbook.triggers):
            return playbook
    return None
