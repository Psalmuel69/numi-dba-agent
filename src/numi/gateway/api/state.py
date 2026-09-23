"""Process-wide singletons for the Gateway service.

Built once at startup from configuration; every request handler reads from
here rather than re-parsing YAML per request.
"""

from __future__ import annotations

from dataclasses import dataclass

from numi.common.config import Settings
from numi.common.identity import IdentityProvider, build_identity_provider
from numi.common.service_auth import ServiceTokenIssuer, ServiceTokenVerifier
from numi.gateway.domain.catalog import CatalogStore
from numi.gateway.domain.data_policy import DataMinimizer
from numi.gateway.domain.decision_events import DbDecisionEventStore, DecisionEventStore
from numi.gateway.domain.investigation_memory import InvestigationMemory
from numi.gateway.domain.investigation_store import DbInvestigationStore, InvestigationStore
from numi.gateway.domain.policy_engine import PolicyEngine
from numi.gateway.domain.rate_limiter import (
    InMemoryRateLimitBackend,
    RateLimitBackend,
    RateLimiter,
    RedisRateLimitBackend,
)
from numi.gateway.domain.risk_engine import RiskEngine
from numi.gateway.domain.servers import ServerRegistry
from numi.gateway.domain.target_validation import TargetValidator
from numi.gateway.domain.tool_registry import ToolRegistry
from numi.gateway.infrastructure.catalog_store import DbCatalogStore
from numi.gateway.infrastructure.db.session import Database
from numi.gateway.infrastructure.execution_client import ExecutionClient, HttpExecutionClient


def _build_rate_limit_backend(settings: Settings) -> RateLimitBackend:
    if settings.rate_limit_backend == "redis":
        import redis.asyncio as redis

        return RedisRateLimitBackend(redis.from_url(settings.redis_url))
    return InMemoryRateLimitBackend()


@dataclass
class GatewayState:
    settings: Settings
    db: Database
    identity_provider: IdentityProvider
    tool_registry: ToolRegistry
    server_registry: ServerRegistry
    catalog_store: CatalogStore
    target_validator: TargetValidator
    policy_engine: PolicyEngine
    risk_engine: RiskEngine
    rate_limiter: RateLimiter
    data_minimizer: DataMinimizer
    execution_client: ExecutionClient
    service_token_issuer: ServiceTokenIssuer
    service_token_verifier: ServiceTokenVerifier
    investigation_store: InvestigationStore
    investigation_memory: InvestigationMemory
    decision_event_store: DecisionEventStore

    @classmethod
    def build(cls, settings: Settings, *, execution_transport=None) -> GatewayState:
        """`execution_transport` lets tests point the Gateway's HTTP
        execution client at an in-process ASGI app (via
        `httpx.ASGITransport`) instead of a real network address."""
        server_registry = ServerRegistry(settings.servers_config_path)
        database = Database(settings.control_db_url)
        catalog_store: CatalogStore = DbCatalogStore(database.session_factory)
        investigation_store: InvestigationStore = DbInvestigationStore(database.session_factory)
        issuer = ServiceTokenIssuer(settings.service_jwt_secret, settings.service_jwt_issuer)
        verifier = ServiceTokenVerifier(settings.service_jwt_secret, settings.service_jwt_issuer)
        return cls(
            settings=settings,
            db=database,
            identity_provider=build_identity_provider(settings),
            tool_registry=ToolRegistry(settings),
            server_registry=server_registry,
            catalog_store=catalog_store,
            target_validator=TargetValidator(server_registry, catalog_store),
            policy_engine=PolicyEngine(settings.policy_config_path),
            risk_engine=RiskEngine(),
            rate_limiter=RateLimiter(settings.rate_limit_config_path, _build_rate_limit_backend(settings)),
            data_minimizer=DataMinimizer(),
            execution_client=HttpExecutionClient(
                settings.execution_base_url,
                issuer,
                settings.execution_service_token_audience,
                transport=execution_transport,
            ),
            service_token_issuer=issuer,
            service_token_verifier=verifier,
            investigation_store=investigation_store,
            investigation_memory=InvestigationMemory(investigation_store),
            decision_event_store=DbDecisionEventStore(database.session_factory),
        )
