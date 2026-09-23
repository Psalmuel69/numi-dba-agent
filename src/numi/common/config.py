"""Process-wide configuration (spec §33).

Every service imports `get_settings()` rather than reading `os.environ`
directly, so there is exactly one place that knows how configuration is
sourced. No secret ever has a literal default here beyond obviously-fake
placeholders — production values always come from the environment /
secrets manager, never from source.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Order in which a provider is auto-selected when the operator hasn't forced
# one via LLM_PROVIDER — first provider that has an API key configured wins.
LLM_PROVIDER_PREFERENCE: tuple[str, ...] = ("anthropic", "openai", "gemini", "deepseek")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    numi_env: str = "development"
    log_level: str = "INFO"

    control_db_url: str = "sqlite+aiosqlite:///./numi_dev.db"
    redis_url: str = "redis://localhost:6379/0"

    # "memory"  -> per-process fixed-window counters (tests / single-instance
    #   local dev; a rate limit means nothing shared across replicas).
    # "redis"   -> shared counters via REDIS_URL, required the moment more
    #   than one Gateway replica is running (spec §29) — see
    #   gateway.api.state.GatewayState.build.
    rate_limit_backend: str = "memory"

    secrets_provider: str = "local_dev"
    vault_addr: str = ""
    vault_token: str = ""
    aws_region: str = ""
    azure_key_vault_url: str = ""
    gcp_project_id: str = ""

    identity_provider: str = "mock"
    oidc_issuer: str = ""
    oidc_client_id: str = ""
    oidc_client_secret: str = ""

    policy_config_path: str = "./config/policy.yaml"
    servers_config_path: str = "./config/servers.yaml"
    identity_config_path: str = "./config/identity.yaml"
    rate_limit_config_path: str = "./config/rate_limits.yaml"
    # How often the discovery crawler refreshes each server's catalog.
    discovery_refresh_minutes: int = 60
    discovery_max_objects_per_database: int = 5000
    # How many recent, concluded past investigations on the same server the
    # Agent recalls as background context before starting a new one. 0
    # disables recall entirely without touching any call site.
    investigation_memory_lookback: int = Field(default=3, ge=0)
    # A second LLM opinion on a draft Conclude before it's accepted — see
    # orchestrator._self_critique_conclude. A kill switch cheaper than a
    # redeploy if this misbehaves in production (rejects too aggressively,
    # or a provider's critique calls turn out unreliable).
    self_critique_enabled: bool = True
    # Durable, queryable storage for a handful of decision-quality signals
    # (see gateway.domain.decision_events for exactly which ones) — a kill
    # switch for the same reason as the two above: never worth blocking a
    # deploy over telemetry.
    decision_event_logging_enabled: bool = True
    # Cross-server pattern correlation ("this same symptom happened on N
    # other servers recently") — see gateway.domain.investigation_memory
    # .correlate. Gateway-side kill switch (checked in the /correlate
    # route itself, the authoritative enforcement point, same posture as
    # other feature flags in this file) and a lookback window so "recently"
    # means something bounded, not an unbounded historical scan.
    cross_server_correlation_enabled: bool = True
    cross_server_correlation_lookback_days: int = Field(default=30, ge=0)
    # Crawl every registered server once at Gateway startup. Off by default:
    # the catalog also refreshes lazily on first use and via `/discover`, and
    # an eager crawl slows startup / adds load. Turn on for an always-warm
    # estate view.
    discovery_on_startup: bool = False

    # --- LLM providers (spec §34) -------------------------------------------
    # A provider becomes *selectable* the moment its API key is present. The
    # DBA can then pick a specific model per conversation via `/model`
    # (unless `llm_provider_lock` forces one).
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    gemini_api_key: str = ""
    deepseek_api_key: str = ""

    # "" / "auto" -> auto-pick the first configured provider (see
    #   LLM_PROVIDER_PREFERENCE).
    # "mock"      -> the deterministic offline planner (tests / no keys).
    # "<name>"    -> force that provider AND disable per-conversation switching.
    llm_provider: str = ""
    # Default model for the resolved provider ("" -> the provider's own
    # default; see agent.llm.registry).
    llm_model: str = ""
    # Whether the `/model` command lets a DBA switch provider/model mid-chat.
    allow_user_model_selection: bool = True
    # Task-complexity tiering (see agent.llm.registry.LLMRegistry.tier_model
    # and orchestrator._llm_for): route cheap calls (extract_intent) to
    # llm_fast_model and hard calls (decide_next_action, critique) to
    # llm_strong_model, within whatever provider is already resolved for
    # the conversation — never overriding an explicit `/model` choice.
    # Both empty by default, so a zero-config deployment sees no behavior
    # change at all.
    llm_fast_model: str = ""
    llm_strong_model: str = ""

    service_jwt_secret: str = "dev-only-insecure-secret-change-me"
    service_jwt_issuer: str = "numi-internal"

    gateway_base_url: str = "http://localhost:8001"
    gateway_port: int = 8001

    execution_base_url: str = "http://localhost:8002"
    execution_port: int = 8002
    execution_service_token_audience: str = "numi-execution"

    agent_base_url: str = "http://localhost:8000"
    agent_port: int = 8000

    channels_base_url: str = "http://localhost:8003"
    channels_port: int = 8003
    slack_signing_secret: str = ""
    slack_bot_token: str = ""
    teams_app_id: str = ""
    teams_app_password: str = ""

    # --- Daily proactive digest (agent service only) ------------------------
    # The scheduled, multi-server morning report: runs the
    # `comprehensive_summary` playbook once per registered server and posts
    # ONE combined digest. See `agent.scheduled_report` and ARCHITECTURE.md's
    # "Scheduled daily digest" section.
    #
    # Disabled by default, and disabled by exactly one switch: an empty
    # destination channel means no scheduler is constructed, no job is
    # registered, and no background task exists at all (never a scheduler
    # that wakes up daily only to discover it has nowhere to post). This is
    # deliberately the *channel*, not a separate `enable_...` boolean —
    # there is no coherent "enabled but with nowhere to send it" state, and
    # two switches would only ever be a way to get them out of sync.
    daily_report_slack_channel: str = ""
    # Hour of the day, UTC, the digest runs at (minute 0). Bounded here
    # rather than clamped at schedule time so a typo (`25`) fails the
    # process at startup, next to every other configuration mistake, instead
    # of silently running at a different hour than the operator intended.
    daily_report_hour_utc: int = Field(default=6, ge=0, le=23)
    # Comma-separated server ids/aliases to sweep. Empty (the default) means
    # every *active* registered server — a newly onboarded server is then
    # covered by the next morning's digest with no config change, which is
    # the behavior an estate-wide report should have. Set it only to
    # deliberately narrow the sweep.
    daily_report_servers: str = ""
    # The identity every tool call in a scheduled run is attributed to and
    # authorized as. This is NOT a bypass and carries no privilege of its
    # own: the Gateway independently re-resolves this (channel,
    # channel_account_id) pair through the same `IdentityProvider` and the
    # same authorization/policy pipeline as a live DBA's message (spec §62),
    # so the digest can only ever see what this account's own DBA role is
    # allowed to see, and every call it makes lands in the audit trail under
    # this account rather than anonymously. It must name a real DBA account
    # in the identity directory; if it doesn't, every call is simply DENIED
    # and each server is reported as "could not be checked" in the digest —
    # never silently skipped.
    daily_report_identity_channel: str = "dev"
    daily_report_identity_account: str = ""

    # --- Alert-triggered investigation (agent + channels services) ----------
    # An external monitoring system (Prometheus Alertmanager, Datadog, a
    # cloud provider's own alarms, ...) POSTs a threshold-breach alert to
    # Channels' `/webhooks/alerts`; Channels verifies it, then the Agent runs
    # one freeform, read-only investigation against the named server and
    # posts the result — the same architecture as the daily digest
    # (`agent.scheduled_report`), just triggered by an event instead of a
    # clock. See `agent.alert_trigger` and ARCHITECTURE.md's
    # "Alert-triggered investigation" section.
    #
    # Disabled by default, and disabled by exactly one switch for the same
    # reason the digest has exactly one: an empty secret means
    # `channels.alerts.signature` has nothing safe to verify against, so the
    # webhook route refuses every request outright rather than ever falling
    # back to "unauthenticated is fine for now."
    alert_webhook_secret: str = ""
    # Where a finished investigation is posted. Independent of
    # `daily_report_slack_channel` on purpose — an alert is an event a team
    # may want routed to a different (e.g. incident-response) channel than
    # the morning digest.
    alert_webhook_slack_channel: str = ""
    # Same non-bypass guarantee as `daily_report_identity_account`: the
    # Gateway independently re-resolves this (channel, channel_account_id)
    # pair and authorizes every call exactly as it would for a live DBA's
    # message. A separate account from the digest's, deliberately — so the
    # two features can be enabled, disabled, and audited independently.
    alert_webhook_identity_channel: str = "dev"
    alert_webhook_identity_account: str = ""
    # Per-(server, metric) cooldown: a flapping metric that re-breaches its
    # threshold every few minutes would otherwise trigger a full LLM-driven
    # investigation — and a fresh channel post — on every single firing.
    # 0 disables the cooldown entirely (every alert always investigates).
    # Shared across replicas via the same RATE_LIMIT_BACKEND setting the
    # Gateway's rate limiter uses (see agent.alert_trigger).
    alert_webhook_cooldown_seconds: int = Field(default=900, ge=0)

    enable_readonly_sql_tool: bool = False
    enable_execute_sql_tool: bool = False
    enable_restore_database_tool: bool = False
    enable_create_database_tool: bool = False
    enable_drop_database_tool: bool = False
    enable_truncate_table_tool: bool = False
    enable_bulk_delete_tool: bool = False

    # ---------------------------------------------------------------------- #

    def is_production(self) -> bool:
        return self.numi_env == "production"

    def _llm_api_keys(self) -> dict[str, str]:
        return {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "gemini": self.gemini_api_key,
            "deepseek": self.deepseek_api_key,
        }

    def configured_llm_providers(self) -> list[str]:
        """Providers that have an API key set, in preference order."""
        keys = self._llm_api_keys()
        return [p for p in LLM_PROVIDER_PREFERENCE if keys.get(p, "").strip()]

    def llm_selection_locked(self) -> bool:
        return bool(self.llm_provider) and self.llm_provider not in ("auto", "mock")

    def effective_default_llm(self) -> tuple[str, str]:
        """(provider, model) used when a conversation has no explicit choice.

        Returns provider == "mock" when there is nothing real to fall back
        to — the deterministic planner — so callers never have to special-case
        "no keys configured".
        """
        if self.llm_provider == "mock":
            return "mock", "mock-planner"
        if self.llm_selection_locked():
            return self.llm_provider, self.llm_model
        configured = self.configured_llm_providers()
        if configured:
            return configured[0], self.llm_model
        return "mock", "mock-planner"

    def validate_for_production(self) -> None:
        """Fail closed at process startup rather than at request time.

        Every "mock"/"local_dev" mode in this codebase exists to make local
        development and the test suite runnable without real credentials
        (spec §32/§65-67) — none of them are safe to run in production, and
        none of the request-time code silently tightens them back up on its
        own. This is the one place that refuses to even start the process
        if `NUMI_ENV=production` is paired with any of them, rather than
        relying on an operator remembering to flip every flag correctly.
        """
        if not self.is_production():
            return

        violations: list[str] = []

        provider, _model = self.effective_default_llm()
        if provider == "mock":
            violations.append(
                "No real LLM provider is configured — set at least one of "
                "ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY / DEEPSEEK_API_KEY "
                "(and optionally LLM_PROVIDER to force one)."
            )
        elif provider not in self.configured_llm_providers():
            violations.append(
                f"LLM_PROVIDER={provider} but no {provider.upper()}_API_KEY is set."
            )

        if self.identity_provider == "mock":
            violations.append(
                "IDENTITY_PROVIDER=mock — production must use a real enterprise "
                "identity provider (e.g. OIDC), never the config-driven mock directory."
            )
        if self.secrets_provider == "local_dev":
            violations.append(
                "SECRETS_PROVIDER=local_dev — production must use a real secrets "
                "manager (vault | aws_secrets_manager | azure_key_vault | gcp_secret_manager)."
            )
        if self.service_jwt_secret == "dev-only-insecure-secret-change-me":
            violations.append(
                "SERVICE_JWT_SECRET is still the development placeholder — set a "
                "real, unique secret for production."
            )
        if self.rate_limit_backend != "redis":
            violations.append(
                "RATE_LIMIT_BACKEND is not 'redis' — the in-memory backend counts per "
                "process, so every rate limit is silently multiplied by replica count "
                "the moment more than one Gateway instance is running."
            )

        if violations:
            raise RuntimeError(
                "Refusing to start with NUMI_ENV=production while running in a "
                "development/mock configuration:\n- " + "\n- ".join(violations)
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
