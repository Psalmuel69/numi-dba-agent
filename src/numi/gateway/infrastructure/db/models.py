"""Control-plane database schema (spec §58).

This is the Gateway's own PostgreSQL database — it is NOT a target database
and the Agent has no route to it. It stores identity/role bindings, the
database inventory, tool registry versions, conversations/investigations,
tool requests/executions, approvals, and the append-only audit and security
event logs.

All identifiers are ULIDs stored as text (sortable by creation time). All
timestamps are UTC.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from numi.common.ids import new_id


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Identity / authorization
# ---------------------------------------------------------------------------


class UserRecord(Base):
    """A cached, periodically-refreshed projection of an IdP identity.

    This table is a *cache* for audit/display convenience only — it is never
    the authority for whether a user is authorized. Authorization always
    re-resolves against the live `IdentityProvider` (spec §37: "context must
    never override authorization").
    """

    __tablename__ = "users"

    subject_id: Mapped[str] = mapped_column(String, primary_key=True)
    email: Mapped[str] = mapped_column(String, nullable=False, index=True)
    display_name: Mapped[str] = mapped_column(String, nullable=False)
    last_seen_groups: Mapped[list] = mapped_column(JSON, default=list)
    last_seen_roles: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class IdentityGroupRecord(Base):
    __tablename__ = "identity_groups"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("grp"))
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String, default="")


class RoleRecord(Base):
    __tablename__ = "roles"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("role"))
    name: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    description: Mapped[str] = mapped_column(String, default="")


class RoleBindingRecord(Base):
    """Maps an identity group name to a role name (mirrors config/identity.yaml
    for audit/reporting; the live authorization decision uses the config file
    / IdentityProvider directly, not this table, to avoid drift ambiguity)."""

    __tablename__ = "role_bindings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("rb"))
    group_name: Mapped[str] = mapped_column(String, nullable=False, index=True)
    role_name: Mapped[str] = mapped_column(String, nullable=False, index=True)


# ---------------------------------------------------------------------------
# Discovered catalog (populated by the discovery crawler via the Gateway)
# ---------------------------------------------------------------------------


class ServerCatalogRecord(Base):
    """One row per registered server — the latest discovered catalog, as
    JSON (`numi.common.models.catalog.ServerCatalog`). The in-memory
    `CatalogStore` is authoritative at runtime; this table lets the catalog
    survive a Gateway restart and be shared across replicas. The Agent
    never writes here."""

    __tablename__ = "server_catalogs"

    server_id: Mapped[str] = mapped_column(String, primary_key=True)
    discovered_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), default=None
    )
    engine_version: Mapped[str] = mapped_column(String, default="")
    engine_edition: Mapped[str] = mapped_column(String, default="")
    catalog: Mapped[dict] = mapped_column(JSON, default=dict)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------


class ToolRecord(Base):
    __tablename__ = "tools"

    tool_id: Mapped[str] = mapped_column(String, primary_key=True)
    latest_version: Mapped[str] = mapped_column(String, nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)


class ToolVersionRecord(Base):
    __tablename__ = "tool_versions"
    __table_args__ = (Index("ix_tool_versions_tool_id_version", "tool_id", "version", unique=True),)

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("tv"))
    tool_id: Mapped[str] = mapped_column(String, nullable=False)
    version: Mapped[str] = mapped_column(String, nullable=False)
    definition: Mapped[dict] = mapped_column(JSON, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

# ---------------------------------------------------------------------------
# Conversation / session / investigation state
# ---------------------------------------------------------------------------


class ConversationRecord(Base):
    __tablename__ = "conversations"

    conversation_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("conv")
    )
    channel: Mapped[str] = mapped_column(String, nullable=False)
    channel_thread_id: Mapped[str] = mapped_column(String, default="")
    user_subject_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class AgentSessionRecord(Base):
    __tablename__ = "agent_sessions"

    agent_session_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("asess")
    )
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.conversation_id"))
    database_context: Mapped[dict] = mapped_column(JSON, default=dict)
    investigation_id: Mapped[str | None] = mapped_column(String, default=None)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class InvestigationRecord(Base):
    __tablename__ = "investigations"

    investigation_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("inv")
    )
    conversation_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    user_subject_id: Mapped[str] = mapped_column(String, nullable=False)
    # Indexed separately from `target` (a JSON blob) so memory recall and
    # cross-server correlation can query by server directly rather than
    # scanning/filtering JSON — see migration 0003.
    server_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True, default=None)
    # Same reasoning, for cross-server correlation's other two filters (see
    # migration 0005 and gateway.domain.investigation_store.find_similar).
    # None means either freeform (no playbook matched) or predates this
    # column — never treated as "matches everything".
    playbook_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True, default=None)
    environment: Mapped[str | None] = mapped_column(String, nullable=True, index=True, default=None)
    target: Mapped[dict] = mapped_column(JSON, default=dict)
    problem: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String, default="INVESTIGATING")
    evidence: Mapped[list] = mapped_column(JSON, default=list)
    hypotheses: Mapped[list] = mapped_column(JSON, default=list)
    findings: Mapped[list] = mapped_column(JSON, default=list)
    recommendations: Mapped[list] = mapped_column(JSON, default=list)
    actions: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class InvestigationEventRecord(Base):
    __tablename__ = "investigation_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("ievt"))
    investigation_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False)  # OBSERVATION/HYPOTHESIS/...
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class LlmDecisionEventRecord(Base):
    """Durable counterpart to a handful of structured log lines (see
    gateway.domain.decision_events for exactly which ones and why not all
    of them): a conclusion rejected by grounding/verification/self-critique,
    or a cross-provider fallback substitution. Small, queryable, and meant
    to be reviewed periodically (see the /v1/decision-events/summary
    rollup) — never raw prompts or DBA content, matching the masking
    conventions everywhere else in this schema."""

    __tablename__ = "llm_decision_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("dqe"))
    conversation_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    investigation_id: Mapped[str | None] = mapped_column(String, nullable=True, index=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    provider: Mapped[str] = mapped_column(String, default="")
    model: Mapped[str] = mapped_column(String, default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

# ---------------------------------------------------------------------------
# Tool requests / executions
# ---------------------------------------------------------------------------


class ToolRequestRecord(Base):
    __tablename__ = "tool_requests"

    request_id: Mapped[str] = mapped_column(String, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    investigation_id: Mapped[str | None] = mapped_column(String, default=None)
    user_subject_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    tool_id: Mapped[str] = mapped_column(String, nullable=False)
    tool_version: Mapped[str] = mapped_column(String, nullable=False)
    target: Mapped[dict] = mapped_column(JSON, default=dict)
    arguments: Mapped[dict] = mapped_column(JSON, default=dict)
    reason: Mapped[str] = mapped_column(Text, default="")
    policy_decision: Mapped[str] = mapped_column(String, default="")
    risk: Mapped[dict] = mapped_column(JSON, default=dict)
    approval_id: Mapped[str | None] = mapped_column(String, default=None)
    status: Mapped[str] = mapped_column(String, default="RECEIVED")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

class ToolExecutionRecord(Base):
    __tablename__ = "tool_executions"

    execution_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("exec")
    )
    request_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    tool_id: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[dt.datetime] = mapped_column(default=_utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(default=None)
    success: Mapped[bool | None] = mapped_column(Boolean, default=None)
    result_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    verification_status: Mapped[str] = mapped_column(String, default="NOT_APPLICABLE")
    error_code: Mapped[str | None] = mapped_column(String, default=None)


# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------


class ApprovalRecord(Base):
    __tablename__ = "approvals"

    approval_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("appr")
    )
    request_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    actor_subject_id: Mapped[str] = mapped_column(String, nullable=False)
    tool_id: Mapped[str] = mapped_column(String, nullable=False)
    tool_version: Mapped[str] = mapped_column(String, nullable=False)
    target: Mapped[dict] = mapped_column(JSON, default=dict)
    normalized_arguments: Mapped[dict] = mapped_column(JSON, default=dict)
    risk: Mapped[dict] = mapped_column(JSON, default=dict)
    policy: Mapped[dict] = mapped_column(JSON, default=dict)
    action_hash: Mapped[str] = mapped_column(String, nullable=False)
    requires_dual_approval: Mapped[bool] = mapped_column(Boolean, default=False)
    approver_1_subject_id: Mapped[str | None] = mapped_column(String, default=None)
    approver_2_subject_id: Mapped[str | None] = mapped_column(String, default=None)
    status: Mapped[str] = mapped_column(String, default="PENDING")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    decided_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), default=None)


class ApprovalEventRecord(Base):
    __tablename__ = "approval_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("aevt"))
    approval_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    actor_subject_id: Mapped[str] = mapped_column(String, default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

# ---------------------------------------------------------------------------
# Audit / security / change management / incidents / knowledge
# ---------------------------------------------------------------------------


class AuditEventRecord(Base):
    """Append-only. No UPDATE/DELETE route is exposed anywhere in the API —
    see gateway.api.routers.audit and gateway.domain.audit."""

    __tablename__ = "audit_events"

    audit_event_id: Mapped[str] = mapped_column(
        String, primary_key=True, default=lambda: new_id("aud")
    )
    correlation_ids: Mapped[dict] = mapped_column(JSON, default=dict)
    actor_subject_id: Mapped[str] = mapped_column(String, default="")
    identity_provider: Mapped[str] = mapped_column(String, default="")
    channel: Mapped[str] = mapped_column(String, default="")
    agent_version: Mapped[str] = mapped_column(String, default="")
    tool_id: Mapped[str] = mapped_column(String, default="")
    tool_version: Mapped[str] = mapped_column(String, default="")
    target: Mapped[dict] = mapped_column(JSON, default=dict)
    arguments_hash: Mapped[str] = mapped_column(String, default="")
    policy_decision: Mapped[str] = mapped_column(String, default="")
    risk: Mapped[dict] = mapped_column(JSON, default=dict)
    approval_id: Mapped[str | None] = mapped_column(String, default=None)
    execution_result: Mapped[str] = mapped_column(String, default="")
    error_code: Mapped[str | None] = mapped_column(String, default=None)
    duration_ms: Mapped[int | None] = mapped_column(Integer, default=None)
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class ChangeRequestRecord(Base):
    __tablename__ = "change_requests"

    change_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("chg"))
    change_type: Mapped[str] = mapped_column(String, default="")
    requested_by: Mapped[str] = mapped_column(String, default="")
    approved_by: Mapped[str] = mapped_column(String, default="")
    scheduled_start: Mapped[dt.datetime | None] = mapped_column(default=None)
    scheduled_end: Mapped[dt.datetime | None] = mapped_column(default=None)
    affected_database: Mapped[str] = mapped_column(String, default="")
    affected_objects: Mapped[list] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String, default="OPEN")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class IncidentRecord(Base):
    __tablename__ = "incidents"

    incident_id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("inc"))
    investigation_id: Mapped[str | None] = mapped_column(String, default=None)
    title: Mapped[str] = mapped_column(String, default="")
    status: Mapped[str] = mapped_column(String, default="OPEN")
    severity: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class KnowledgeDocumentRecord(Base):
    __tablename__ = "knowledge_documents"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("kb"))
    doc_type: Mapped[str] = mapped_column(String, default="runbook")
    title: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

class SecurityEventRecord(Base):
    """Append-only record of every security-relevant denial/anomaly (spoofed
    identity attempts, approval tampering, rate-limit abuse, etc.), separate
    from the general audit log so security monitoring can alert on it in
    isolation."""

    __tablename__ = "security_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: new_id("sec"))
    event_type: Mapped[str] = mapped_column(String, nullable=False)
    actor_subject_id: Mapped[str] = mapped_column(String, default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    severity: Mapped[str] = mapped_column(String, default="WARNING")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
