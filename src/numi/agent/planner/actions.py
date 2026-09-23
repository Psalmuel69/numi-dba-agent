"""Structured Agent decisions (spec §35, §36).

Every decision the LLM makes is forced into one of these typed shapes and
validated with Pydantic before the Agent acts on it. There is no code path
where raw LLM text is parsed for an intent and executed directly — a
malformed or nonsensical completion fails validation and the Agent falls
back to asking the user for clarification, it never guesses.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, TypeAdapter


class AskClarification(BaseModel):
    action: Literal["ask_clarification"] = "ask_clarification"
    question: str


class ProposeToolCall(BaseModel):
    """The LLM's *proposal* only — tool_id and arguments are re-validated by
    the Gateway from scratch; this model just shapes what the Agent forwards
    as a request. The LLM cannot invent a tool_id outside the list of tools
    it was offered (see `agent.tool_client.ToolClient.available_tools`)."""

    action: Literal["propose_tool_call"] = "propose_tool_call"
    tool_id: str
    arguments: dict = Field(default_factory=dict)
    target: dict = Field(default_factory=dict)
    reason: str


class RecordObservation(BaseModel):
    action: Literal["record_observation"] = "record_observation"
    text: str


class Conclude(BaseModel):
    action: Literal["conclude"] = "conclude"
    summary: str
    likely_root_cause: str | None = None
    confidence: Literal["confirmed", "likely", "unable_to_confirm"] = "unable_to_confirm"
    recommendation: str | None = None
    recommended_tool_call: ProposeToolCall | None = None


AgentAction = Annotated[
    AskClarification | ProposeToolCall | RecordObservation | Conclude,
    Field(discriminator="action"),
]

agent_action_adapter: TypeAdapter = TypeAdapter(AgentAction)


class CritiqueVerdict(BaseModel):
    """A second, independent LLM opinion on a draft Conclude — does the
    conclusion actually follow from the evidence gathered, or is it a leap?
    Distinct from `_ungrounded_identifiers`'s regex heuristic (which only
    catches a *named* fabrication) and from `pending_verification`'s
    structural check (which only catches a missing re-check after a write)
    — this is the one check that can catch a conclusion that names nothing
    fabricated and has nothing pending, but still doesn't actually follow
    from what was found."""

    sound: bool
    issue: str | None = None


class IntentExtraction(BaseModel):
    """First-pass classification of an incoming message (spec §6, §39)."""

    is_dba_task: bool
    is_greeting_or_chitchat: bool = False
    database_hint: str | None = None
    # A Literal (not the Environment enum) so the generated JSON schema is a
    # flat {"enum": [...]}, never a $defs/$ref an Enum class would produce —
    # Gemini's function-calling schema subset supports neither. Verified
    # live: before this was constrained at all, a real model wrote "dev"
    # (never a valid Environment value) because nothing told it the field
    # was constrained — DatabaseTarget then rejects it outright as an
    # invalid enum member, not as a missing/empty field, so it read to the
    # DBA as an unexplained repeated INVALID_TARGET rather than a typo.
    environment_hint: Literal["development", "uat", "production"] | None = None
    # Whatever the DBA actually called the server — a registered id/alias,
    # or (just as often, verified live) an informal abbreviation, nickname,
    # or IP address/fragment not in the known-servers list at all. Never
    # invented out of thin air; only ever pulled from what the DBA
    # actually wrote. Carries no authority of its own — the Gateway
    # independently re-resolves and validates it (now with the same
    # fuzzy/host matching described in ServerRegistry.find_candidates), and
    # asks the DBA to disambiguate rather than guessing if that's unclear.
    instance_hint: str | None = None
    # A request for one of the agent's own utility actions, in the DBA's
    # own words — never a slash command required. `instance_hint` doubles
    # as the target server id for "catalog"/"discover" when one is named.
    # None for a genuine DBA investigation request or unrelated chitchat.
    # "models" and "approvers" were added after a live finding: two
    # legitimate free-form questions ("who can approve requests from you?",
    # "what environments and databases do you have access to?") both fell
    # through to the generic fallback because nothing recognized them as
    # meta-commands at all — see orchestrator._handle_meta_command and
    # llm/base.py's _INTENT_SYSTEM for what each one now does.
    meta_command: (
        Literal[
            "help",
            "status",
            "servers",
            "playbooks",
            "catalog",
            "discover",
            "approve",
            "reject",
            "models",
            "approvers",
        ]
        | None
    ) = None
    problem_summary: str = ""

