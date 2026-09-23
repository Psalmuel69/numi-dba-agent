"""Pins the exact keys the LLM must use inside a propose_tool_call's
`target` dict against what DatabaseTarget actually accepts.

Reproduces a live finding: the real Gemini model proposed
database.update_statistics with an empty `target` (no schema/object at
all), because _FLAT_ACTION_SCHEMA's target field had no guidance on what
keys belong there — every schema/object-scoped write tool would fail the
same way regardless of how the request was phrased. DatabaseTarget uses
`extra="forbid"` with aliases (schema_name -> "schema", object_name ->
"object") and no populate_by_name, so "schema_name"/"object_name" in the
JSON itself are rejected outright — only "schema"/"object" work."""

from __future__ import annotations

from numi.agent.llm.base import _ACTION_SYSTEM, _FLAT_ACTION_SCHEMA
from numi.common.models.target import DatabaseTarget


def test_database_target_only_accepts_the_short_alias_keys():
    # Confirms the assumption the prompt fix relies on: "schema_name" is
    # rejected (extra="forbid", no populate_by_name), "schema" is accepted.
    DatabaseTarget.model_validate(
        {"environment": "development", "schema": "Person", "object": "Person"}
    )
    try:
        DatabaseTarget.model_validate(
            {"environment": "development", "schema_name": "Person", "object_name": "Person"}
        )
    except Exception:
        pass
    else:
        raise AssertionError(
            "DatabaseTarget unexpectedly accepted schema_name/object_name — "
            "if this now passes, the prompt guidance below needs updating to match."
        )


def test_action_schema_target_description_names_the_real_keys():
    target_schema = _FLAT_ACTION_SCHEMA["properties"]["target"]
    description = target_schema["description"]
    assert "schema" in description
    assert "object" in description
    # The wrong (rejected-by-DatabaseTarget) names must never appear as
    # guidance, or the model will confidently produce an invalid target.
    assert "schema_name" not in description
    assert "object_name" not in description


def test_action_system_prompt_has_a_schema_object_scoped_example():
    assert '"schema": "Person"' in _ACTION_SYSTEM
    assert '"object": "Person"' in _ACTION_SYSTEM
    assert "database.update_statistics" in _ACTION_SYSTEM


def test_target_and_arguments_declare_real_properties_not_a_bare_object():
    """Reproduces the actual live fix: prose alone wasn't enough — a real
    model's own `reason` text said "I will now provide the required
    session_id and reason arguments" and still left `arguments: {}` every
    time. Only giving `arguments`/`target` real declared `properties` (a
    concrete template to fill in, not just a description of one) fixed it
    — confirmed by rerunning the exact failing case, first attempt, same
    model that had failed six times in a row."""
    props = _FLAT_ACTION_SCHEMA["properties"]
    assert "properties" in props["arguments"], "arguments must declare real sub-properties"
    assert "properties" in props["target"], "target must declare real sub-properties"

    for key in ("session_id", "reason", "schema", "table"):
        assert key in props["arguments"]["properties"]
    for key in ("environment", "instance", "database", "schema", "object", "session_id"):
        assert key in props["target"]["properties"]

    # environment must stay a real enum here too, not just on IntentExtraction.
    assert props["target"]["properties"]["environment"]["enum"] == [
        "development",
        "uat",
        "production",
    ]


def test_arguments_properties_cover_every_field_every_enabled_tool_actually_uses():
    """If a new tool_arguments.py model adds a field this schema doesn't
    know about, the model has no template slot for it and this test catches
    that before it becomes another live "the model understood but the
    structured output stayed empty" incident."""
    from numi.common.models import tool_arguments

    # Excluded: the raw-SQL tool args (execute_sql/execute_readonly_sql) are
    # disabled by default and out of this round's scope — deliberately not
    # giving the model an `sql` template slot to reach for.
    excluded = {"ExecuteSqlArgs", "ReadOnlySqlArgs"}
    known = set(_FLAT_ACTION_SCHEMA["properties"]["arguments"]["properties"])
    for name in dir(tool_arguments):
        if name in excluded:
            continue
        cls = getattr(tool_arguments, name)
        if isinstance(cls, type) and issubclass(cls, tool_arguments._StrictArgs):
            for field_name, field in cls.model_fields.items():
                key = field.alias if isinstance(field.alias, str) else field_name
                assert key in known, f"{cls.__name__}.{field_name} (-> {key!r}) has no schema slot"
