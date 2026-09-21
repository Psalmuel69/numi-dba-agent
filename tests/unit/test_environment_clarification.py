"""Two live findings from real Slack usage against `handle_message`:

1. Naming a specific, registered server ("check overall health on
   postgres-local") still asked "which environment should I investigate?"
   even though a registered server has exactly one environment in
   config/servers.yaml — asking again for something already implied by the
   instance was never necessary.

2. Answering that clarification with a bare "development" (plus Slack's
   own mention markup: "development <@U0BOTID>") got classified by the
   real model as chitchat, silently discarding the DBA's answer and the
   in-progress investigation — replying with the generic help text
   instead of continuing. `_ENVIRONMENT_ANSWER_RE`'s fast path exists
   specifically so this one clarification (tracked as actual state, not
   free LLM text) can be answered without ever risking that
   misclassification — proven here by a fake LLM whose extract_intent
   raises if it's ever called for this case."""

from __future__ import annotations

import pytest

from inumi.agent.context_manager import ContextManager, InvestigationState
from inumi.agent.llm.registry import LLMRegistry
from inumi.agent.orchestrator import AgentOrchestrator
from inumi.agent.planner.actions import AskClarification, Conclude, CritiqueVerdict, IntentExtraction


class _FakeToolClient:
    def __init__(self, servers: list[dict] | None = None):
        self._servers = servers or []

    async def list_servers(self):
        return self._servers

    async def available_tools(self, channel, channel_account_id):
        return []

    async def submit(self, request):
        raise AssertionError("no tool call expected in these tests")

    async def create_investigation(self, request):
        pass

    async def update_investigation(self, investigation_id, request):
        pass

    async def get_investigation_memory(self, server_id, *, exclude_investigation_id=None, limit=3):
        return []

    async def get_cross_server_patterns(
        self, *, playbook_id, environment=None, exclude_server_id=None, limit=5
    ):
        return []


class _FakeLLM:
    """`extract_intent` raises by default — set `intent` to make it return
    something instead. Proves the fast path never calls it when unset."""

    def __init__(self, *, intent: IntentExtraction | None = None, decide_action=None):
        self._intent = intent
        self._decide_action = decide_action or Conclude(summary="Nothing wrong found.")
        self.extract_intent_calls = 0
        self.decide_calls = 0

    async def extract_intent(self, *args, **kwargs):
        self.extract_intent_calls += 1
        if self._intent is None:
            raise AssertionError("extract_intent must not be called here")
        return self._intent

    async def decide_next_action(self, **kwargs):
        self.decide_calls += 1
        return self._decide_action

    async def critique_conclusion(self, **kwargs):
        return CritiqueVerdict(sound=True)


def _orchestrator(llm, servers=None) -> tuple[AgentOrchestrator, ContextManager]:
    context = ContextManager()
    orchestrator = AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(llm),
        tool_client=_FakeToolClient(servers),
        context=context,
    )
    return orchestrator, context


@pytest.mark.asyncio
async def test_a_bare_environment_answer_continues_without_calling_extract_intent():
    llm = _FakeLLM()  # extract_intent raises if ever called
    orchestrator, context = _orchestrator(llm)
    state = context.get_or_create("conv1", "slack", "thread1", "U123")
    state.investigation = InvestigationState(
        investigation_id="inv1", problem="check overall health on postgres-local"
    )

    reply = await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv1",
        channel_thread_id="thread1",
        message="development <@U0BOTID>",  # exact live shape: answer + mention noise
    )

    assert llm.extract_intent_calls == 0
    assert state.database_context["environment"] == "development"
    # No tool call ever ran — collapses to CONCLUDED_NO_ACTION.
    assert state.investigation.status == "CONCLUDED_NO_ACTION"
    assert state.investigation.is_concluded is True
    assert reply.status == "ok"
    assert "Nothing wrong found" in reply.text


@pytest.mark.asyncio
async def test_the_fast_path_never_fires_without_a_pending_investigation():
    """A bare "development"-shaped message with no active investigation is
    just a normal message — must go through real intent extraction, not
    be silently swallowed by the fast path."""
    llm = _FakeLLM(intent=IntentExtraction(is_dba_task=False, is_greeting_or_chitchat=True))
    orchestrator, context = _orchestrator(llm)

    reply = await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv2",
        channel_thread_id="",
        message="development",
    )

    assert llm.extract_intent_calls == 1
    assert "Inumi" in reply.text


@pytest.mark.asyncio
async def test_a_resumed_investigation_never_reclassifies_even_with_environment_already_set():
    """The generalized fix, live-reproduced with a *second* clarification
    kind: environment already set, but the DBA's reply ("postgres-local")
    is answering a freeform AskClarification the LLM itself asked ("which
    server?") — this must resume directly (threaded via
    investigation.last_message) exactly like the environment-answer case,
    not just for that one hardcoded gate."""
    llm = _FakeLLM()  # extract_intent raises if ever called
    orchestrator, context = _orchestrator(llm)
    state = context.get_or_create("conv3", "slack", "", "U123")
    state.investigation = InvestigationState(investigation_id="inv1", problem="check health")
    state.database_context["environment"] = "development"

    await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv3",
        channel_thread_id="",
        message="postgres-local",
    )

    assert llm.extract_intent_calls == 0
    assert llm.decide_calls == 1
    assert state.investigation.last_message == ""  # consumed by that one decide_next_action call


@pytest.mark.asyncio
async def test_naming_a_registered_server_auto_resolves_its_environment():
    intent = IntentExtraction(
        is_dba_task=True,
        instance_hint="postgres-local",
        problem_summary="check overall health on postgres-local",
    )
    llm = _FakeLLM(intent=intent)
    servers = [{"id": "postgres-local", "aliases": ["local"], "environment": "development"}]
    orchestrator, context = _orchestrator(llm, servers=servers)

    reply = await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv4",
        channel_thread_id="",
        message="check overall health on postgres-local",
    )

    assert reply.status != "clarification"
    state = context.get_or_create("conv4", "slack", "", "U123")
    assert state.database_context["environment"] == "development"


@pytest.mark.asyncio
async def test_an_explicit_environment_is_never_overridden_by_auto_resolution():
    intent = IntentExtraction(
        is_dba_task=True,
        instance_hint="postgres-local",
        environment_hint="uat",
        problem_summary="check overall health on postgres-local uat",
    )
    llm = _FakeLLM(intent=intent)
    servers = [{"id": "postgres-local", "aliases": [], "environment": "development"}]
    orchestrator, context = _orchestrator(llm, servers=servers)

    await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv5",
        channel_thread_id="",
        message="check overall health on postgres-local uat",
    )

    state = context.get_or_create("conv5", "slack", "", "U123")
    assert state.database_context["environment"] == "uat"


@pytest.mark.asyncio
async def test_an_unregistered_instance_still_asks_for_clarification():
    intent = IntentExtraction(is_dba_task=True, problem_summary="check health on some-unknown-box")
    llm = _FakeLLM(intent=intent)
    orchestrator, context = _orchestrator(llm, servers=[])

    reply = await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv6",
        channel_thread_id="",
        message="check health on some-unknown-box",
    )

    assert reply.status == "clarification"


def test_the_dbas_last_message_is_threaded_into_the_llm_prompt():
    from inumi.agent.context_manager import InvestigationState

    investigation = InvestigationState(investigation_id="inv1", problem="check health")
    assert "postgres-local" not in AgentOrchestrator._problem_statement_for_llm(investigation)

    investigation.last_message = "postgres-local"
    problem = AgentOrchestrator._problem_statement_for_llm(investigation)
    assert "The DBA just replied" in problem
    assert "postgres-local" in problem


class _SequencedFakeLLM:
    """`extract_intent` returns one `IntentExtraction` per call, in order —
    lets a test drive several distinct fresh investigations (each starting
    from `extract_intent`) through one conversation, the way a real
    multi-turn Slack thread does. `decide_next_action` always concludes
    immediately (turn 1), since these tests are about environment
    bookkeeping across investigations, not the investigation loop itself."""

    def __init__(self, intents: list[IntentExtraction]):
        self._intents = list(intents)
        self.extract_intent_calls = 0

    async def extract_intent(self, *args, **kwargs):
        self.extract_intent_calls += 1
        return self._intents.pop(0)

    async def decide_next_action(self, **kwargs):
        return Conclude(summary="Investigated.")

    async def critique_conclusion(self, **kwargs):
        return CritiqueVerdict(sound=True)


@pytest.mark.asyncio
async def test_a_later_fresh_investigation_never_reasks_for_an_already_known_environment():
    """Live-reproduced regression: environment and instance are established
    early in a conversation (turns 1-2), two more investigations each name
    the instance again and conclude (turns 3-4), and then a plain follow-up
    that names neither ("So what database is the copy activity happening
    on?") must NOT re-trigger the environment gate just because it, alone,
    doesn't repeat what was already established earlier in this same
    conversation."""
    servers = [{"id": "postgres-local", "aliases": [], "environment": "development"}]
    llm = _SequencedFakeLLM(
        [
            IntentExtraction(is_dba_task=True, problem_summary="any blocking sessions?"),
            IntentExtraction(
                is_dba_task=True, instance_hint="postgres-local", problem_summary="general check"
            ),
            IntentExtraction(
                is_dba_task=True, instance_hint="postgres-local", problem_summary="general check"
            ),
            IntentExtraction(is_dba_task=True, problem_summary="a plain follow-up question"),
        ]
    )
    orchestrator, context = _orchestrator(llm, servers=servers)
    state = context.get_or_create("conv7", "slack", "", "U123")

    turn1 = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv7",
        channel_thread_id="", message="any blocking sessions right now across our databases?",
    )
    assert turn1.status == "clarification"

    turn2 = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv7",
        channel_thread_id="", message="development, on postgres-local",
    )
    assert turn2.status == "ok"
    assert state.investigation.status == "CONCLUDED_NO_ACTION"  # no tool call ever ran

    turn3 = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv7",
        channel_thread_id="", message="check what's happening on postgres local. it seems to be slow",
    )
    assert turn3.status == "ok"

    turn4 = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv7",
        channel_thread_id="", message="what daabase in postgres local is this issue happening on?",
    )
    assert turn4.status == "ok"

    # The exact reproduced symptom: no environment/instance wording of its
    # own, in the same ongoing conversation where both were already
    # established.
    turn5 = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv7",
        channel_thread_id="", message="So what database is the copy activity happening on? "
        "and for how long and what's the ipact?",
    )
    assert turn5.status != "clarification"
    assert "Which environment" not in turn5.text
    assert llm.extract_intent_calls == 4


@pytest.mark.asyncio
async def test_switching_to_an_unresolvable_instance_forgets_the_stale_environment():
    """The mirror image of test_instance_switch_clears_database.py's
    database case: a remembered environment belongs to whatever instance
    was previously in play too. Moving to a different, unregistered/
    ambiguous server whose environment can't be auto-resolved must not
    silently keep asserting the OLD instance's environment for this new
    one — that would be exactly the guess the spec forbids for an unnamed
    environment, it must ask instead."""
    intent = IntentExtraction(
        is_dba_task=True,
        instance_hint="some-new-unregistered-box",
        problem_summary="check health on some-new-unregistered-box",
    )
    llm = _FakeLLM(intent=intent)
    orchestrator, context = _orchestrator(llm, servers=[])
    state = context.get_or_create("conv8", "slack", "", "U123")
    state.database_context["instance"] = "postgres-local"
    state.database_context["environment"] = "development"

    reply = await orchestrator.handle_message(
        channel="slack",
        channel_account_id="U123",
        conversation_id="conv8",
        channel_thread_id="",
        message="check health on some-new-unregistered-box",
    )

    assert reply.status == "clarification"
    assert "environment" not in state.database_context


def test_environment_for_instance_matches_id_or_alias_case_insensitively():
    servers = [{"id": "postgres-local", "aliases": ["local", "mylocal"], "environment": "development"}]
    orchestrator = AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(_FakeLLM()),
        tool_client=_FakeToolClient(servers),
        context=ContextManager(),
    )
    import asyncio

    assert asyncio.run(orchestrator._environment_for_instance("Postgres-Local")) == "development"
    assert asyncio.run(orchestrator._environment_for_instance("MyLocal")) == "development"
    assert asyncio.run(orchestrator._environment_for_instance("nope")) is None


def test_canonical_server_id_normalizes_a_loosely_phrased_hint_to_the_registered_id():
    """Verified live: a real model extracted "Postgres dev 02" for the
    registered server `postgres-dev-02` — a real tool call still resolves
    that correctly via the Gateway's own independent fuzzy matching, but
    without this, `server_id` sent for investigation-memory keying would
    have been the raw, inconsistently-phrased hint instead of the stable
    id two different conversations both need to agree on."""
    servers = [{"id": "postgres-dev-02", "aliases": [], "environment": "development"}]
    orchestrator = AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(_FakeLLM()),
        tool_client=_FakeToolClient(servers),
        context=ContextManager(),
    )
    import asyncio

    assert asyncio.run(orchestrator._canonical_server_id("Postgres dev 02")) == "postgres-dev-02"
    assert asyncio.run(orchestrator._canonical_server_id("postgres-dev-02")) == "postgres-dev-02"


def test_canonical_server_id_falls_back_to_the_raw_hint_when_unmatched_or_ambiguous():
    servers = [
        {"id": "postgres-dev-01", "aliases": [], "environment": "development"},
        {"id": "postgres-dev-02", "aliases": [], "environment": "development"},
    ]
    orchestrator = AgentOrchestrator(
        llm_registry=LLMRegistry.for_testing(_FakeLLM()),
        tool_client=_FakeToolClient(servers),
        context=ContextManager(),
    )
    import asyncio

    # No registered server matches "nope" at all.
    assert asyncio.run(orchestrator._canonical_server_id("nope")) == "nope"
    # "postgres-dev" substring-matches both registered servers — ambiguous.
    assert asyncio.run(orchestrator._canonical_server_id("postgres-dev")) == "postgres-dev"


class _Turn2AsksThenConcludesLLM:
    """Live-reproduced regression, both halves in one conversation:
    `extract_intent` returns each of `intents` in order (one per FRESH
    investigation); `decide_next_action` conclude on its first call
    (turn 1's investigation), asks a clarifying question on its second
    call (turn 2's fresh, zero-entity investigation — simulating a real
    model that still asks something despite `state.database_context`
    already having resolved environment/instance, since
    `decide_next_action`'s prompt is never told that directly — see
    `_continue_investigation`), then concludes again on its third call
    (turn 2's investigation, RESUMED by turn 3's bare reply). Deliberately
    NOT the orchestrator's own hardcoded environment-question wording —
    that string exists nowhere in any LLM prompt (verified by a grep of
    every `src/inumi/agent/llm/*.py` system prompt) and a real model could
    not plausibly echo it character-for-character; this fake asks a
    plausible free-form question instead, to test the GENERAL resume
    mechanism (`investigation.last_message`), not the one hardcoded
    environment-answer fast path."""

    def __init__(self, intents: list[IntentExtraction]):
        self._intents = list(intents)
        self.extract_intent_calls = 0
        self._decide_calls = 0

    async def extract_intent(self, *args, **kwargs):
        self.extract_intent_calls += 1
        return self._intents.pop(0)

    async def decide_next_action(self, **kwargs):
        self._decide_calls += 1
        if self._decide_calls == 2:
            return AskClarification(question="Which server is the replication check for?")
        return Conclude(summary="Investigated.")

    async def critique_conclusion(self, **kwargs):
        return CritiqueVerdict(sound=True)


@pytest.mark.asyncio
async def test_the_live_three_turn_sequence_is_handled_correctly_given_one_stable_conversation():
    """Isolates whether `handle_message` itself has any remaining gap in
    the live-reported 3-turn sequence — run here with a single, constant
    `conversation_id` throughout, which is exactly what `handle_message`
    always receives from its caller. `tests/integration/
    test_channels_api.py` covers the separate, real root cause: the
    Slack webhook computing a NEW `conversation_id` for every non-threaded
    message, which is what actually broke this live (see
    `_slack_conversation_id`'s own docstring). Turn 1: "check backup
    health on postgres-local" (names the server, concludes normally).
    Turn 2: "and what about replication?" — zero named entities of its
    own; must not re-ask for the already-known environment (commit
    1f83133), but a real model can still ask something else mid-
    investigation (simulated here). Turn 3: a bare "development" — must
    resume turn 2's investigation via `investigation.last_message`
    (never reclassified by a fresh `extract_intent` call, never dropped
    into the generic chitchat fallback)."""
    servers = [{"id": "postgres-local", "aliases": [], "environment": "development"}]
    llm = _Turn2AsksThenConcludesLLM(
        [
            IntentExtraction(
                is_dba_task=True,
                instance_hint="postgres-local",
                problem_summary="check backup health on postgres-local",
            ),
            IntentExtraction(is_dba_task=True, problem_summary="check replication status"),
        ]
    )
    orchestrator, context = _orchestrator(llm, servers=servers)
    state = context.get_or_create("conv10", "slack", "", "U123")

    turn1 = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv10",
        channel_thread_id="", message="check backup health on postgres-local",
    )
    assert turn1.status == "ok"
    assert state.investigation.is_concluded is True
    assert state.database_context["environment"] == "development"

    turn2 = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv10",
        channel_thread_id="", message="and what about replication?",
    )
    # The orchestrator's own environment gate must not fire on this
    # zero-entity follow-up — environment/instance are already known.
    assert "Which environment" not in turn2.text
    assert turn2.status == "clarification"
    assert state.investigation.status == "AWAITING_CLARIFICATION"
    assert state.investigation.is_concluded is False

    turn3 = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv10",
        channel_thread_id="", message="development",
    )
    assert turn3.status == "ok"
    assert "tell me more" not in turn3.text
    assert llm.extract_intent_calls == 2  # turn 3 resumed rather than reclassifying


class _StaleClarificationThenFreshInstructionLLM:
    """Live-reproduced regression: `extract_intent` returns each of
    `intents` in order (turn 1's fresh investigation, then turn 3's
    would-be topic shift); `decide_next_action` asks a freeform
    clarification on its first call (turn 1's investigation, never
    answered) and concludes on its second (turn 3's brand-new
    investigation, once the stale one has been abandoned)."""

    def __init__(self, intents: list[IntentExtraction]):
        self._intents = list(intents)
        self.extract_intent_calls = 0
        self._decide_calls = 0

    async def extract_intent(self, *args, **kwargs):
        self.extract_intent_calls += 1
        return self._intents.pop(0)

    async def decide_next_action(self, **kwargs):
        self._decide_calls += 1
        if self._decide_calls == 1:
            return AskClarification(question="What do you mean by 'blah' / 'the thing'?")
        return Conclude(summary="Dropped the test database on postgres-local as requested.")

    async def critique_conclusion(self, **kwargs):
        return CritiqueVerdict(sound=True)


@pytest.mark.asyncio
async def test_a_fresh_fully_specified_instruction_abandons_a_stale_unanswered_clarification():
    """Live-reproduced regression: "check blah on the thing pls fix
    asap!!!" (deliberate gibberish) started an investigation that asked
    what "blah"/"the thing" meant and was never answered. 32 minutes and
    several unrelated exchanges later, "Drop the test database on
    postgres-local, it's no longer needed" — a fully-specified, unrelated
    instruction — got a reply that rambled about "blah" and "the thing"
    instead of addressing the actual request, because `handle_message`
    unconditionally treated it as a reply to the old, stale question (see
    `_problem_statement_for_llm`, which still prepends the original,
    never-updated `investigation.problem` verbatim).

    This must now: (1) recognize the second message as its own,
    self-contained instruction rather than an answer to "what do you mean
    by 'blah'", (2) mark the stale investigation CONCLUDED_UNRESOLVED
    rather than silently discard it, (3) start a genuinely fresh
    investigation whose `problem` text contains NONE of the old, unrelated
    wording, and (4) never re-ask for the environment — already known from
    turn 1's own instance_hint, exactly like `_start_fresh_investigation`
    already guarantees for any other fresh investigation in this same
    conversation."""
    servers = [{"id": "postgres-local", "aliases": [], "environment": "development"}]
    llm = _StaleClarificationThenFreshInstructionLLM(
        [
            IntentExtraction(
                is_dba_task=True,
                instance_hint="postgres-local",
                problem_summary="check blah on the thing pls fix asap!!!",
            ),
            IntentExtraction(
                is_dba_task=True,
                instance_hint="postgres-local",
                problem_summary="Drop the test database on postgres-local, it's no longer needed.",
            ),
        ]
    )
    orchestrator, context = _orchestrator(llm, servers=servers)
    state = context.get_or_create("conv_stale", "slack", "", "U123")

    turn1 = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv_stale",
        channel_thread_id="", message="check blah on the thing pls fix asap!!!",
    )
    assert turn1.status == "clarification"
    assert state.investigation.status == "AWAITING_CLARIFICATION"
    assert state.database_context["environment"] == "development"
    stale_investigation_id = state.investigation.investigation_id

    turn2 = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv_stale",
        channel_thread_id="", message="Drop the test database on postgres-local, it's no longer needed.",
    )
    # A brand-new investigation, not the old one silently resumed.
    assert state.investigation.investigation_id != stale_investigation_id
    assert "blah" not in state.investigation.problem
    assert "the thing" not in state.investigation.problem
    assert turn2.status == "ok"
    assert "blah" not in turn2.text
    assert "the thing" not in turn2.text
    # The environment gate must not re-ask — already known from turn 1.
    assert "Which environment" not in turn2.text
    assert llm.extract_intent_calls == 2  # turn 1's, plus turn 2's topic-shift check


@pytest.mark.asyncio
async def test_a_longer_reply_naming_no_target_of_its_own_still_resumes_the_stale_clarification():
    """The word-count floor alone must not be the whole gate — a longer
    reply that still doesn't name its own instance/database/environment
    (unlike the fresh-instruction case above) reads far more like an
    attempt to answer the pending question than a topic shift, so it must
    still resume the SAME investigation via `investigation.last_message`,
    not start a new one. Guards `_classify_potential_topic_shift`'s
    `names_own_target` check specifically, independent of the word-count
    floor `test_the_live_three_turn_sequence...` already covers."""
    llm = _StaleClarificationThenFreshInstructionLLM(
        [
            IntentExtraction(
                is_dba_task=True,
                instance_hint="postgres-local",
                problem_summary="check blah on the thing pls fix asap!!!",
            ),
            # Names no instance/database/environment of its own.
            IntentExtraction(is_dba_task=True, problem_summary="it's the one from this morning"),
        ]
    )
    orchestrator, context = _orchestrator(
        llm, servers=[{"id": "postgres-local", "aliases": [], "environment": "development"}]
    )
    state = context.get_or_create("conv_stale2", "slack", "", "U123")

    await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv_stale2",
        channel_thread_id="", message="check blah on the thing pls fix asap!!!",
    )
    assert state.investigation.status == "AWAITING_CLARIFICATION"
    stale_investigation_id = state.investigation.investigation_id

    turn2 = await orchestrator.handle_message(
        channel="slack", channel_account_id="U123", conversation_id="conv_stale2",
        channel_thread_id="", message="it's the one from this morning",
    )
    # SAME investigation — resumed, not abandoned, despite being 6 words long.
    assert state.investigation.investigation_id == stale_investigation_id
    assert turn2.status == "ok"
