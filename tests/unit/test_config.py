"""Fail-closed production configuration checks (spec §63) and LLM
provider/model resolution."""

from __future__ import annotations

import pytest

from numi.common.config import Settings


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


# --------------------------------------------------------------------------- #
# Production fail-closed
# --------------------------------------------------------------------------- #


def test_development_settings_never_raise_regardless_of_mock_flags():
    _settings(numi_env="development").validate_for_production()  # should not raise


def test_production_with_all_dev_defaults_refuses_to_start():
    with pytest.raises(RuntimeError) as exc:
        _settings(numi_env="production").validate_for_production()
    message = str(exc.value)
    assert "No real LLM provider is configured" in message
    assert "IDENTITY_PROVIDER=mock" in message
    assert "SECRETS_PROVIDER=local_dev" in message
    assert "SERVICE_JWT_SECRET" in message
    assert "RATE_LIMIT_BACKEND" in message


def test_production_with_every_flag_properly_set_does_not_raise():
    _settings(
        numi_env="production",
        anthropic_api_key="sk-ant-real-key",
        identity_provider="oidc",
        secrets_provider="vault",
        service_jwt_secret="a-real-unique-production-secret",
        rate_limit_backend="redis",
    ).validate_for_production()  # should not raise


def test_production_forced_provider_without_its_key_is_a_violation():
    with pytest.raises(RuntimeError) as exc:
        _settings(
            numi_env="production",
            llm_provider="openai",  # forced, but no OPENAI_API_KEY
            identity_provider="oidc",
            secrets_provider="vault",
            service_jwt_secret="a-real-unique-production-secret",
            rate_limit_backend="redis",
        ).validate_for_production()
    assert "no OPENAI_API_KEY is set" in str(exc.value)


def test_production_explicit_mock_llm_is_always_a_violation():
    with pytest.raises(RuntimeError) as exc:
        _settings(
            numi_env="production",
            llm_provider="mock",
            anthropic_api_key="sk-ant-real-key",
            identity_provider="oidc",
            secrets_provider="vault",
            service_jwt_secret="a-real-unique-production-secret",
            rate_limit_backend="redis",
        ).validate_for_production()
    assert "No real LLM provider is configured" in str(exc.value)


def test_production_in_memory_rate_limit_backend_is_a_violation():
    with pytest.raises(RuntimeError) as exc:
        _settings(
            numi_env="production",
            anthropic_api_key="sk-ant-real-key",
            identity_provider="oidc",
            secrets_provider="vault",
            service_jwt_secret="a-real-unique-production-secret",
            rate_limit_backend="memory",
        ).validate_for_production()
    assert "RATE_LIMIT_BACKEND is not 'redis'" in str(exc.value)


# --------------------------------------------------------------------------- #
# LLM provider/model resolution
# --------------------------------------------------------------------------- #


def test_no_keys_resolves_to_mock():
    assert _settings().effective_default_llm() == ("mock", "mock-planner")
    assert _settings().configured_llm_providers() == []


def test_first_configured_provider_is_auto_selected_in_preference_order():
    s = _settings(openai_api_key="sk-openai", gemini_api_key="g-key")
    assert s.configured_llm_providers() == ["openai", "gemini"]
    assert s.effective_default_llm()[0] == "openai"


def test_anthropic_wins_when_multiple_configured():
    s = _settings(anthropic_api_key="a", openai_api_key="b")
    assert s.effective_default_llm()[0] == "anthropic"


def test_explicit_llm_provider_locks_selection():
    s = _settings(anthropic_api_key="a", openai_api_key="b", llm_provider="openai")
    assert s.llm_selection_locked() is True
    assert s.effective_default_llm()[0] == "openai"


def test_auto_or_empty_provider_does_not_lock():
    assert _settings(anthropic_api_key="a", llm_provider="").llm_selection_locked() is False
    assert _settings(anthropic_api_key="a", llm_provider="auto").llm_selection_locked() is False
