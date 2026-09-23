from __future__ import annotations

import pytest
import pytest_asyncio

from numi.common.config import Settings
from numi.common.identity import MockIdentityProvider
from numi.gateway.domain.catalog import InMemoryCatalogStore
from numi.gateway.domain.policy_engine import PolicyEngine
from numi.gateway.domain.rate_limiter import InMemoryRateLimitBackend, RateLimiter
from numi.gateway.domain.servers import ServerRegistry
from numi.gateway.domain.target_validation import TargetContext, TargetValidator
from numi.gateway.domain.tool_registry import ToolRegistry
from numi.gateway.infrastructure.db.session import Database


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture
def identity_provider() -> MockIdentityProvider:
    return MockIdentityProvider("config/identity.yaml")


@pytest.fixture
def server_registry() -> ServerRegistry:
    return ServerRegistry("config/servers.yaml")


@pytest.fixture
def catalog_store() -> InMemoryCatalogStore:
    return InMemoryCatalogStore()


@pytest.fixture
def target_validator(
    server_registry: ServerRegistry, catalog_store: InMemoryCatalogStore
) -> TargetValidator:
    return TargetValidator(server_registry, catalog_store)


@pytest.fixture
def make_ctx(server_registry: ServerRegistry):
    """Factory: build a TargetContext for a registered server id (+ optional
    database, honouring any per-db override)."""

    from numi.common.models.target import DatabaseTarget

    def _make(server_id: str, database: str = "AppDB") -> TargetContext:
        server = server_registry.by_id(server_id)
        assert server is not None, f"no such server in config/servers.yaml: {server_id}"
        eff = server.effective_for(database)
        return TargetContext(
            target=DatabaseTarget(
                environment=server.environment, instance=server.id, database=database
            ),
            server=server,
            database=database,
            criticality=eff.criticality,
            classification=eff.classification,
            allowed_roles=eff.allowed_roles,
            discovered_database=None,
        )

    return _make


@pytest.fixture
def policy_engine() -> PolicyEngine:
    return PolicyEngine("config/policy.yaml")


@pytest.fixture
def tool_registry(settings: Settings) -> ToolRegistry:
    return ToolRegistry(settings)


@pytest.fixture
def rate_limiter() -> RateLimiter:
    return RateLimiter("config/rate_limits.yaml", InMemoryRateLimitBackend())


@pytest_asyncio.fixture
async def db() -> Database:
    database = Database("sqlite+aiosqlite:///:memory:")
    await database.create_all()
    yield database
    await database.dispose()
