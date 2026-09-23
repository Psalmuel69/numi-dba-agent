from __future__ import annotations

import pytest

from numi.common.config import Settings
from numi.common.models.execution import ExecutionRequest
from numi.common.models.target import Platform
from numi.execution.service import ExecutionService
from tests.canned_adapter import canned_adapter_factory


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def _request(tool_id: str, *, max_result_rows: int = 100, max_execution_time: int = 30) -> ExecutionRequest:
    return ExecutionRequest(
        execution_id="exec_test",
        tool_id=tool_id,
        tool_version="1.0.0",
        platform=Platform.SQLSERVER,
        server_id="corebanking-sqlserver-prod",
        database="CoreBanking",
        arguments={},
        max_execution_time=max_execution_time,
        max_result_rows=max_result_rows,
    )


def _service_with_canned_adapter() -> ExecutionService:
    return ExecutionService(
        _settings(), credential_provider=None, adapter_factory=canned_adapter_factory  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_dispatch_maps_tool_id_to_the_right_adapter_method():
    service = _service_with_canned_adapter()
    result = await service.execute(_request("database.get_health"))
    assert result.success is True
    assert result.rows[0]["active_sessions"] == 42


@pytest.mark.asyncio
async def test_row_cap_is_enforced_and_reported():
    service = _service_with_canned_adapter()
    result = await service.execute(_request("database.get_blocking_sessions", max_result_rows=5))
    assert result.success is True
    assert result.row_count == 5
    assert result.truncated is True


@pytest.mark.asyncio
async def test_unknown_tool_id_fails_without_touching_any_database():
    service = _service_with_canned_adapter()
    result = await service.execute(_request("database.does_not_exist"))
    assert result.success is False
    assert result.error_code == "TOOL_NOT_FOUND"


@pytest.mark.asyncio
async def test_execution_timeout_is_reported_not_hung(monkeypatch):
    import asyncio

    service = _service_with_canned_adapter()

    async def _slow_dispatch(adapter, request):
        await asyncio.sleep(10)

    monkeypatch.setattr(service, "_dispatch", _slow_dispatch)
    result = await service.execute(_request("database.get_health", max_execution_time=1))
    assert result.success is False
    assert result.error_code == "EXECUTION_TIMEOUT"


@pytest.mark.asyncio
async def test_without_configured_credentials_execution_fails_closed():
    """With no adapter_factory (i.e. the real path) and a misconfigured
    secrets provider, execution must never fall back to running anyway — it
    fails closed (spec §63)."""
    from numi.execution.credentials.provider import build_credential_provider

    settings = _settings(secrets_provider="vault")
    service = ExecutionService(settings, credential_provider=build_credential_provider(settings))
    result = await service.execute(_request("database.get_health", max_execution_time=5))
    assert result.success is False
    assert result.error_code == "EXECUTION_FAILED"


@pytest.mark.asyncio
async def test_an_unhandled_failure_is_logged_server_side_not_just_swallowed(capsys):
    """Reproduces a live finding: `error_detail` tells the caller "See
    server-side logs for detail" — but nothing was ever actually logged, so
    a real driver-level failure (a missing extension, in the live case) was
    undiagnosable without manually reproducing it. The failure itself must
    still degrade to the same generic client-facing message (never leak
    internals to the Agent/DBA), but the real cause must land in the logs."""

    class _BoomAdapter:
        async def health(self):
            raise RuntimeError("relation \"pg_stat_statements\" does not exist")

    async def factory(request):
        return _BoomAdapter(), None

    service = ExecutionService(_settings(), credential_provider=None, adapter_factory=factory)  # type: ignore[arg-type]
    result = await service.execute(_request("database.get_health"))

    assert result.success is False
    assert result.error_code == "EXECUTION_FAILED"
    assert "server-side logs" in result.error_detail

    logged = capsys.readouterr().out
    assert "execution_failed" in logged
    assert "pg_stat_statements" in logged
