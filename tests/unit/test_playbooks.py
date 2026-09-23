"""Playbook library correctness: every step must name a real, currently
registered read-only tool (never a write, never an invented id — a typo
here would silently degrade to "no tool ever runs" or, worse, "the wrong
tool runs"), and the keyword matcher must pick the intended playbook (and
only that one) for representative real-world phrasings, while leaving
ordinary non-matching problem text to the existing freeform path (None)."""

from __future__ import annotations

from numi.agent.playbooks.library import PLAYBOOKS, get_playbook, match_playbook
from numi.common.config import Settings
from numi.gateway.domain.tool_catalog import build_tool_catalog

# The real, currently registered read-only tool ids (spec §8) — a playbook
# step naming anything outside this set (or a write tool) could never
# actually run, or worse, could silently propose a write with no DBA in the
# loop to review the exact call first.
_REAL_READ_TOOL_IDS = {
    t.tool_id
    for t in build_tool_catalog(Settings(_env_file=None))
    if not t.data_modification
}


def test_every_playbook_step_names_a_real_read_only_tool():
    assert _REAL_READ_TOOL_IDS, "sanity check: the real catalog must be non-empty"
    for playbook in PLAYBOOKS:
        assert playbook.steps, f"{playbook.playbook_id} has no steps"
        for step in playbook.steps:
            assert step.tool_id in _REAL_READ_TOOL_IDS, (
                f"{playbook.playbook_id} step names {step.tool_id!r}, which is not a "
                "real, currently-registered read-only tool"
            )


def test_every_playbook_id_is_unique():
    ids = [p.playbook_id for p in PLAYBOOKS]
    assert len(ids) == len(set(ids))


def test_get_playbook_round_trips_every_registered_id():
    for playbook in PLAYBOOKS:
        assert get_playbook(playbook.playbook_id) is playbook


def test_get_playbook_returns_none_for_an_unknown_or_missing_id():
    assert get_playbook(None) is None
    assert get_playbook("not-a-real-playbook") is None


def test_matcher_picks_the_intended_playbook_for_real_world_phrasings():
    cases = {
        "Why is CoreBanking so slow today?": "slow_queries",
        "queries are timing out on production": "slow_queries",
        "CPU usage on sqlserver-prod-01 is spiking": "high_cpu",
        "we're seeing high memory usage on the replica": "high_memory",
        "sessions are blocked and nothing is completing": "blocking",
        "we just hit a deadlock in CoreBanking": "deadlocks",
        "getting connection refused, too many connections": "connections",
        "replication lag on the standby is growing": "replication",
        "did last night's backup job fail?": "backups",
        "the transaction log is full on CoreBanking": "transaction_log",
        "we're running out of disk space on postgres-local": "storage",
        "seeing a lot of errors in the error log": "errors",
        "can you do a health check on CoreBanking": "general_health",
        "can you tune this instance for us?": "configuration_review",
        "we'd like a configuration review of postgres-local": "configuration_review",
        "can you recommend settings for this server": "configuration_review",
        "are our settings okay on the prod instance": "configuration_review",
    }
    for text, expected_id in cases.items():
        playbook = match_playbook(text)
        assert playbook is not None, f"expected a match for {text!r}"
        assert playbook.playbook_id == expected_id, (
            f"{text!r} matched {playbook.playbook_id!r}, expected {expected_id!r}"
        )


def test_deadlock_matches_before_the_broader_blocking_playbook():
    """'deadlock' contains the substring 'lock' — a naive substring matcher
    would hit the blocking playbook's 'lock'-ish triggers first if it came
    first in the list, or word-boundary matching could still misfire if
    implemented carelessly. Pins the intended, more-specific match."""
    playbook = match_playbook("we had a deadlock on the orders table")
    assert playbook is not None
    assert playbook.playbook_id == "deadlocks"


def test_word_boundary_matching_does_not_false_positive_on_substrings():
    # "login" must not trigger the connections playbook via a naive "log"
    # substring match, and "backlog" must not trigger the backups playbook
    # via a naive "back" substring match.
    assert match_playbook("users are failing to login to the app") is None
    assert match_playbook("there's a backlog of jobs to process") is None


def test_unmatched_problem_text_falls_back_to_freeform():
    assert match_playbook("please list the tables in CoreBanking") is None
    assert match_playbook("") is None
    assert match_playbook("   ") is None


def test_no_playbook_step_ever_proposes_a_write_tool():
    write_tool_ids = {
        t.tool_id for t in build_tool_catalog(Settings(_env_file=None)) if t.data_modification
    }
    for playbook in PLAYBOOKS:
        for step in playbook.steps:
            assert step.tool_id not in write_tool_ids


# --- configuration_review (new playbook, added after comparing against ---
# --- Xata's shipped playbook prompts — see library.py's PLAYBOOKS tuple) ---


def test_configuration_review_playbook_uses_only_configuration_and_health():
    playbook = get_playbook("configuration_review")
    assert playbook is not None
    assert [step.tool_id for step in playbook.steps] == [
        "database.get_configuration",
        "database.get_health",
    ]


def test_configuration_review_matches_tune_and_tuning_but_not_unrelated_words():
    # "tune" and "tuning" are both registered triggers, but neither is a
    # substring false-positive off something like "attune" or "fortune".
    assert match_playbook("can you tune this instance for us?") is not None
    assert match_playbook("can you tune this instance for us?").playbook_id == "configuration_review"
    assert match_playbook("we need some tuning done on postgres-local") is not None
    assert match_playbook("we need some tuning done on postgres-local").playbook_id == (
        "configuration_review"
    )
    assert match_playbook("the orchestra needs to attune its instruments") is None
    assert match_playbook("we lost a fortune on that deal") is None


def test_configuration_review_does_not_shadow_or_get_shadowed_by_connections_or_general_health():
    # "configuration review"/"tune"/"tuning" must not accidentally match the
    # connections playbook's "connection"/"configuration"-adjacent wording,
    # and existing connections/general-health phrasing must not accidentally
    # route into configuration_review.
    assert match_playbook("too many connections to the database").playbook_id == "connections"
    assert match_playbook("can you do a health check on CoreBanking").playbook_id == "general_health"
    assert match_playbook("can you recommend settings for this server").playbook_id == (
        "configuration_review"
    )
    assert match_playbook("we'd like a configuration review of postgres-local").playbook_id == (
        "configuration_review"
    )


def test_slow_queries_and_high_cpu_guidance_warns_against_introspection_queries():
    slow_queries = get_playbook("slow_queries")
    high_cpu = get_playbook("high_cpu")
    for playbook in (slow_queries, high_cpu):
        assert playbook is not None
        guidance = playbook.conclusion_guidance
        assert "pg_catalog" in guidance
        assert "information_schema" in guidance
        assert "performance_schema" in guidance
        assert "INFORMATION_SCHEMA" in guidance or "sys.*" in guidance


def test_connections_guidance_frames_a_healthy_count_as_no_alarm():
    connections = get_playbook("connections")
    assert connections is not None
    guidance = connections.conclusion_guidance
    assert "false alarm" in guidance
    assert "comfortable headroom" in guidance or "well under" in guidance


def test_configuration_review_guidance_scopes_itself_honestly_and_names_engine_params():
    playbook = get_playbook("configuration_review")
    assert playbook is not None
    guidance = playbook.conclusion_guidance
    description = playbook.description
    # Must not overclaim true capacity-based sizing (no instance-class /
    # hardware data exists anywhere in Numi's server registry).
    assert "capacity" in guidance
    assert "hardware" in guidance or "instance-class" in guidance
    # Must name real, engine-specific parameters, adapted per engine.
    assert "shared_buffers" in guidance
    assert "max_connections" in guidance
    assert "max degree of parallelism" in guidance
    assert "innodb_buffer_pool_size" in guidance
    # The scoping limitation must also be stated in the description, not just
    # buried in the conclusion guidance.
    assert "instance-class" in description or "hardware" in description


# --- storage / transaction_log split (previously one combined "Storage / ---
# --- Transaction Log Investigation" playbook with 3 steps and no ---
# --- replication/backup checks at all — see library.py's PLAYBOOKS tuple) ---


def test_storage_playbook_is_now_capacity_only():
    playbook = get_playbook("storage")
    assert playbook is not None
    assert [step.tool_id for step in playbook.steps] == [
        "database.get_storage",
        "database.get_health",
    ]
    # The old combined playbook's transaction-log-specific triggers must no
    # longer live on `storage` — they belong to `transaction_log` now.
    assert "transaction log full" not in playbook.triggers
    assert "log growing" not in playbook.triggers
    assert "wal growing" not in playbook.triggers


def test_transaction_log_playbook_exists_with_the_expected_steps():
    playbook = get_playbook("transaction_log")
    assert playbook is not None
    assert playbook.name == "Transaction Log Investigation"
    assert [step.tool_id for step in playbook.steps] == [
        "database.get_transaction_log",
        "database.get_replication_status",
        "database.get_backup_status",
        "database.get_health",
    ]


def test_transaction_log_guidance_names_the_concrete_reuse_blockers():
    playbook = get_playbook("transaction_log")
    assert playbook is not None
    guidance = playbook.conclusion_guidance
    for phrase in (
        "long-running", "replication lag", "failed or stalled log backup",
        "log shipping", "maintenance",
    ):
        assert phrase in guidance, f"expected {phrase!r} in transaction_log guidance"


def test_a_bare_storage_question_still_routes_to_the_capacity_playbook():
    playbook = match_playbook("we're running out of storage on prod-db-01")
    assert playbook is not None
    assert playbook.playbook_id == "storage"


def test_transaction_log_phrasings_route_to_transaction_log_not_storage():
    cases = [
        "the transaction log is full on CoreBanking",
        "our log is full and won't truncate",
        "the wal growing fast on postgres-local",
        "the transaction log can't reuse space",
        "log won't truncate even after a checkpoint",
    ]
    for text in cases:
        playbook = match_playbook(text)
        assert playbook is not None, f"expected a match for {text!r}"
        assert playbook.playbook_id == "transaction_log", (
            f"{text!r} matched {playbook.playbook_id!r}, expected 'transaction_log'"
        )


def test_disk_space_phrasings_still_route_to_storage_not_transaction_log():
    cases = [
        "disk space is running low on prod-db-01",
        "we're out of disk space",
        "running out of space on the data volume",
    ]
    for text in cases:
        playbook = match_playbook(text)
        assert playbook is not None, f"expected a match for {text!r}"
        assert playbook.playbook_id == "storage", (
            f"{text!r} matched {playbook.playbook_id!r}, expected 'storage'"
        )


def test_transaction_log_does_not_overlap_backups_triggers():
    # "backup" and "transaction log" are two different questions ("is my
    # last backup okay" vs. "why can't my transaction log reuse space") —
    # their trigger sets must stay clearly distinct even though
    # transaction_log's steps include a backup-status check.
    backups = get_playbook("backups")
    transaction_log = get_playbook("transaction_log")
    assert backups is not None and transaction_log is not None
    assert not (set(backups.triggers) & set(transaction_log.triggers))
    assert match_playbook("did last night's backup job fail?").playbook_id == "backups"
    assert match_playbook("the transaction log is full").playbook_id == "transaction_log"
    # Neither playbook's triggers should fire on the other's canonical phrasing.
    for trigger in backups.triggers:
        assert match_playbook(f"checking {trigger} now").playbook_id != "transaction_log"


def test_transaction_log_does_not_overlap_replication_or_general_health_triggers():
    replication = get_playbook("replication")
    general_health = get_playbook("general_health")
    transaction_log = get_playbook("transaction_log")
    assert replication is not None and general_health is not None and transaction_log is not None
    assert not (set(replication.triggers) & set(transaction_log.triggers))
    assert not (set(general_health.triggers) & set(transaction_log.triggers))


def test_storage_does_not_overlap_transaction_log_or_backups_triggers():
    storage = get_playbook("storage")
    transaction_log = get_playbook("transaction_log")
    backups = get_playbook("backups")
    assert storage is not None and transaction_log is not None and backups is not None
    assert not (set(storage.triggers) & set(transaction_log.triggers))
    assert not (set(storage.triggers) & set(backups.triggers))


def test_total_playbook_count_after_the_storage_transaction_log_split():
    # 12 playbooks existed going into this change (11 original scenarios
    # plus a concurrently-landed 12th, configuration_review). Splitting the
    # combined storage playbook into storage + transaction_log adds one
    # more, for 13. A later addition (comprehensive_summary, see below)
    # brings the total to 14 — see
    # test_total_playbook_count_after_comprehensive_summary_is_added.
    assert len(PLAYBOOKS) == 14


# --- deepening the remaining 7 playbooks (deadlocks, blocking, high_memory, ---
# --- replication, backups, errors, general_health) against the same ---
# --- external playbook specification already used for the slow_queries/ ---
# --- high_cpu/connections/storage/transaction_log rounds above — see ---
# --- library.py's PLAYBOOKS tuple and ARCHITECTURE.md's playbook section ---


def test_deadlocks_playbook_steps_include_a_session_duration_check():
    playbook = get_playbook("deadlocks")
    assert playbook is not None
    assert [step.tool_id for step in playbook.steps] == [
        "database.get_deadlocks",
        "database.get_blocking_sessions",
        "database.get_running_queries",
        "database.get_sessions",
    ]


def test_deadlocks_guidance_names_root_cause_categories_and_asks_for_the_cycle():
    playbook = get_playbook("deadlocks")
    assert playbook is not None
    guidance = playbook.conclusion_guidance
    for phrase in (
        "lock-acquisition cycle", "deadlock victim", "access order",
        "missing index", "long-running transaction", "lock escalation",
        "isolation level", "foreign-key", "maintenance operation",
    ):
        assert phrase in guidance, f"expected {phrase!r} in deadlocks guidance"


def test_blocking_guidance_classifies_the_blocker_and_tests_hypotheses():
    playbook = get_playbook("blocking")
    assert playbook is not None
    guidance = playbook.conclusion_guidance
    for phrase in (
        "idle-in-transaction", "reporting query", "DDL statement",
        "maintenance operation", "replication/log-shipping process",
        "longest single wait", "spreading", "missing index",
        "lock escalation", "isolation level", "connection/transaction leak",
        "infrastructure slowdown",
    ):
        assert phrase in guidance, f"expected {phrase!r} in blocking guidance"


def test_high_memory_guidance_separates_pressure_sources_and_gates_restart():
    playbook = get_playbook("high_memory")
    assert playbook is not None
    guidance = playbook.conclusion_guidance
    for phrase in (
        "the database engine's own allocations", "the host", "container/VM",
        "another service", "oversized memory grant", "poor query plan",
        "excessive concurrent sessions", "genuine leak", "host-level overcommitment",
        "last resort", "explicit DBA approval", "never the first recommendation",
    ):
        assert phrase in guidance, f"expected {phrase!r} in high_memory guidance"
    # Never claim host/container memory is actually measured — this system
    # has no host-level diagnostic tool, matching the honesty standard the
    # configuration_review playbook already sets for its own scope limits.
    assert "isn't something this system can directly measure" in guidance


def test_replication_playbook_steps_are_unchanged_by_the_guidance_enrichment():
    # No new step was added here — a 4th step (e.g. get_storage) would push
    # this playbook's step count to the same boundary that made
    # test_stuck_observation_loop's replication regression test start
    # burning the full 6-turn cap instead of exiting early on a stuck model
    # (interaction between _MAX_CONSECUTIVE_RECORD_OBSERVATIONS and
    # _MAX_INVESTIGATION_TURNS in orchestrator.py, which this task does not
    # touch) — so this round only deepens conclusion_guidance for replication.
    playbook = get_playbook("replication")
    assert playbook is not None
    assert [step.tool_id for step in playbook.steps] == [
        "database.get_replication_status",
        "database.get_health",
        "database.get_wait_statistics",
    ]


def test_replication_guidance_covers_risk_exposure_not_just_the_lag_number():
    playbook = get_playbook("replication")
    assert playbook is not None
    guidance = playbook.conclusion_guidance
    for phrase in (
        "stale-read risk", "RPO/RTO exposure", "primary-side risk",
        "growing or holding steady", "applications or read paths",
        "network latency", "long-running transaction", "disconnected or dropped replica",
        "replication-slot retention", "apply/redo worker", "undersized",
    ):
        assert phrase in guidance, f"expected {phrase!r} in replication guidance"


def test_backups_guidance_calculates_the_recovery_point_gap_and_classifies_severity():
    playbook = get_playbook("backups")
    assert playbook is not None
    guidance = playbook.conclusion_guidance
    for phrase in (
        "recovery-point gap", "restore-test history",
        "informational", "warning", "high", "critical",
    ):
        assert phrase in guidance, f"expected {phrase!r} in backups guidance"


def test_errors_playbook_steps_include_blocking_and_deadlock_correlation():
    playbook = get_playbook("errors")
    assert playbook is not None
    assert [step.tool_id for step in playbook.steps] == [
        "database.get_error_logs",
        "database.get_health",
        "database.get_blocking_sessions",
        "database.get_deadlocks",
        "database.get_running_queries",
    ]


def test_errors_guidance_classifies_findings_into_categories_and_correlates_signals():
    playbook = get_playbook("errors")
    assert playbook is not None
    guidance = playbook.conclusion_guidance
    for category in (
        "availability", "authentication", "authorization", "storage", "memory",
        "CPU", "network", "corruption", "backup", "replication", "locking",
        "configuration", "security", "application integration",
    ):
        assert category in guidance, f"expected category {category!r} in errors guidance"
    assert "Correlate against the other signals gathered" in guidance


def test_general_health_guidance_requires_an_explicit_status_classification():
    playbook = get_playbook("general_health")
    assert playbook is not None
    guidance = playbook.conclusion_guidance
    for status in ("healthy", "informational", "warning", "high", "critical", "unknown"):
        assert status in guidance, f"expected status {status!r} in general_health guidance"
    # The DBA must get an equally complete/confident answer whether or not
    # anything is actually wrong — not a vaguer reply on the all-clear path.
    assert "just as complete and confident" in guidance


def test_deepened_playbooks_stay_within_the_shared_turn_budget():
    # _MAX_INVESTIGATION_TURNS is 6 (orchestrator.py) and each playbook step
    # spends one turn before the model ever gets its own (unrestricted) turn
    # to conclude or extend — a playbook with 6+ steps would exhaust the
    # entire budget before that final call ever happens, silently changing
    # the "LLM is asked exactly once" property the whole feature relies on.
    # Every playbook, including the 7 deepened here, must leave at least one
    # turn free.
    _MAX_INVESTIGATION_TURNS = 6
    for playbook in PLAYBOOKS:
        assert len(playbook.steps) < _MAX_INVESTIGATION_TURNS, (
            f"{playbook.playbook_id} has {len(playbook.steps)} steps, leaving no "
            "turn free for the model's own unrestricted conclude/extend call"
        )


# --- comprehensive_summary (new playbook — a broad, single-server sweep a ---
# --- DBA can explicitly invoke, or a future scheduled job could call once ---
# --- per server; see library.py's PLAYBOOKS tuple for the full scoping ---
# --- note. Distinct from general_health, which stays the existing, ---
# --- lighter 4-step "quick pulse check" — untouched by this addition) ---


def test_total_playbook_count_after_comprehensive_summary_is_added():
    # 13 playbooks existed going into this change; comprehensive_summary
    # adds a 14th.
    assert len(PLAYBOOKS) == 14


def test_comprehensive_summary_runs_its_five_steps_availability_first():
    # Capped at exactly 5 steps — not all instance-wide read tools, and one
    # short of the shared 6-turn budget — because every investigation
    # (playbook or freeform) shares one turn budget,
    # orchestrator._MAX_INVESTIGATION_TURNS (6), and every playbook in this
    # library must leave at least one turn free for the model's own
    # unrestricted concluding call (see
    # test_deepened_playbooks_stay_within_the_shared_turn_budget, which
    # covers every playbook including this one). See this playbook's own
    # description/comments in library.py for why 12 steps (one per
    # instance-wide read tool) was rejected — verified live in
    # test_orchestrator_playbooks.py.
    playbook = get_playbook("comprehensive_summary")
    assert playbook is not None
    tool_ids = [step.tool_id for step in playbook.steps]
    assert tool_ids == [
        "database.get_health",
        "database.get_blocking_sessions",
        "database.get_backup_status",
        "database.get_storage",
        "database.get_error_logs",
    ]
    assert len(tool_ids) == 5
    # Availability/workload signals (health, blocking) must come before
    # protection/config signals (backups, storage, error logs) —
    # most-urgent-first.
    workload_signals = ["database.get_health", "database.get_blocking_sessions"]
    protection_signals = [
        "database.get_backup_status", "database.get_storage", "database.get_error_logs",
    ]
    last_workload_index = max(tool_ids.index(t) for t in workload_signals)
    first_protection_index = min(tool_ids.index(t) for t in protection_signals)
    assert last_workload_index < first_protection_index


def test_comprehensive_summary_matches_its_own_broad_phrasings():
    cases = [
        "can you give me a comprehensive health check on prod-db-01",
        "we need a daily summary for CoreBanking",
        "give me a full health report",
        "what's the complete status of this server",
        "tell me everything about this server",
        "can you give a full report on postgres-local",
        "send the daily report",
        "run a comprehensive check on sqlserver-prod-01",
        "run a complete health check",
        "run a full diagnostic",
    ]
    for text in cases:
        playbook = match_playbook(text)
        assert playbook is not None, f"expected a match for {text!r}"
        assert playbook.playbook_id == "comprehensive_summary", (
            f"{text!r} matched {playbook.playbook_id!r}, expected 'comprehensive_summary'"
        )


def test_comprehensive_summary_does_not_shadow_general_healths_own_triggers():
    # general_health's own triggers ("health check", "how is", "how's",
    # "overall status", "general status", "status check", "how is it
    # doing", "how's it doing", "everything okay", "everything ok") must
    # still correctly route to general_health, not comprehensive_summary —
    # even though comprehensive_summary is checked first in match_playbook's
    # tuple order (a deliberate placement: some of comprehensive_summary's
    # own triggers, like "comprehensive health check", contain general_
    # health's "health check" as a substring, so comprehensive_summary must
    # come first for ITS OWN phrasings to route correctly — but that must
    # not come at the cost of general_health's own, narrower triggers).
    general_health = get_playbook("general_health")
    assert general_health is not None
    for trigger in general_health.triggers:
        playbook = match_playbook(trigger)
        assert playbook is not None, f"expected a match for {trigger!r}"
        assert playbook.playbook_id == "general_health", (
            f"general_health trigger {trigger!r} matched {playbook.playbook_id!r} instead"
        )


def test_comprehensive_summary_does_not_match_any_other_playbooks_existing_triggers():
    for playbook in PLAYBOOKS:
        if playbook.playbook_id == "comprehensive_summary":
            continue
        for trigger in playbook.triggers:
            matched = match_playbook(trigger)
            assert matched is not None
            assert matched.playbook_id != "comprehensive_summary", (
                f"{playbook.playbook_id}'s trigger {trigger!r} unexpectedly matched "
                "comprehensive_summary"
            )


def test_comprehensive_summarys_own_triggers_do_not_overlap_any_other_playbooks():
    comprehensive_summary = get_playbook("comprehensive_summary")
    assert comprehensive_summary is not None
    for playbook in PLAYBOOKS:
        if playbook.playbook_id == "comprehensive_summary":
            continue
        assert not (set(comprehensive_summary.triggers) & set(playbook.triggers)), (
            f"comprehensive_summary shares a literal trigger string with {playbook.playbook_id}"
        )


def test_comprehensive_summary_guidance_directs_synthesis_and_restraint_not_a_wall_of_text():
    playbook = get_playbook("comprehensive_summary")
    assert playbook is not None
    guidance = playbook.conclusion_guidance
    # Category structure mirroring the external spec's General Health Check
    # — limited to categories this playbook's own 5 fixed steps actually
    # gather evidence for (no HA/replication or configuration step, so
    # neither category — see the guidance's own explicit instruction not to
    # guess).
    for category in (
        "availability", "resource pressure", "workload/blocking",
        "protection/backups", "logs",
    ):
        assert category in guidance, f"expected {category!r} in comprehensive_summary guidance"
    # Must not invite the model to fabricate a replication or configuration
    # finding it never actually checked.
    assert "replication" in guidance.lower()
    assert "configuration" in guidance.lower()
    assert "never state or imply" in guidance or "omit those categories" in guidance
    # The restraint instruction itself — only deviations, not a clean-result
    # wall of text.
    assert "all other checks came back clean" in guidance
    assert "ONLY what deviates" in guidance or "only what deviates" in guidance.lower()


def test_comprehensive_summary_description_is_honest_about_scope():
    playbook = get_playbook("comprehensive_summary")
    assert playbook is not None
    description = playbook.description
    # Must not overclaim trend/history or multi-server sweeping — neither
    # capability exists in this system yet.
    assert "historical data store" in description or "no historical" in description.lower()
    assert "one server" in description.lower() or "single server" in description.lower() or (
        "one shot" in description.lower()
    )
    # This used to assert the word "deferred" (scheduling and multi-server
    # orchestration didn't exist yet). They do now — `agent.scheduled_report`
    # — so pinning "deferred" would pin a stale fact. The claim that still
    # matters, and that this assertion now protects, is the one that hasn't
    # changed: THIS PLAYBOOK is still one server per invocation. The
    # scheduler calls it once per server; it never sweeps a fleet itself, and
    # the description must keep saying so rather than quietly inheriting the
    # scheduler's reach.
    assert "never a fleet" in description.lower()
    assert "not something this playbook does itself" in description.lower()
    # Must also be honest about why this playbook has 5 steps, not one for
    # every instance-wide read tool — the shared investigation turn budget,
    # not an oversight.
    assert "_MAX_INVESTIGATION_TURNS" in description
    assert "orchestrator.py" in description
