"""Real secrets-manager credential providers (spec §19, §63).

There is no Vault server, AWS account, Azure tenant or GCP project in this
environment — and deliberately so: these tests inject a fake SDK client
into each provider's `client=` seam and assert on *this codebase's* own
behavior (request shape, response parsing, error mapping, fail-closed
defaults) without a single network call. Same pattern as
`tests/canned_adapter.py`, `FakeQueryExecutor` in
`tests/unit/test_adapters.py`, and the LLM provider tests.

The three properties every one of these four providers must hold:

  1. a valid secret parses into the same `DatabaseCredentials` the
     local-dev YAML provider would have produced;
  2. *any* SDK failure becomes `NumiError(DEPENDENCY_UNAVAILABLE)` — never
     a raw vendor exception, never a partial credential;
  3. an unconfigured backend still refuses without ever touching the SDK.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from numi.common.models.failures import FailureCode, NumiError
from numi.execution.credentials.provider import (
    AWSSecretsManagerCredentialProvider,
    AzureKeyVaultCredentialProvider,
    GCPSecretManagerCredentialProvider,
    LocalDevCredentialProvider,
    VaultCredentialProvider,
    build_credential_provider,
)

# The documented secret shape — identical for all four backends.
_SECRET = {
    "host": "db-prod-01.internal",
    "port": 5432,
    "username": "numi_diag",
    "password": "s3cr3t-value",
    "database": "postgres",
    "options": {"sslmode": "require"},
}


class _BoomError(Exception):
    """Stands in for a vendor SDK exception (auth failure, not found,
    timeout). Deliberately NOT an NumiError, so a test asserting
    `pytest.raises(NumiError)` proves the mapping actually happened rather
    than the exception merely passing through."""


def _assert_parsed(creds) -> None:
    assert creds.host == "db-prod-01.internal"
    assert creds.port == 5432
    assert creds.username == "numi_diag"
    assert creds.password.get_secret_value() == "s3cr3t-value"
    assert creds.database == "postgres"
    assert creds.options == {"sslmode": "require"}


def _assert_fails_closed(exc_info) -> None:
    assert exc_info.value.code == FailureCode.DEPENDENCY_UNAVAILABLE
    # The secret's own material must never reach the user-facing detail.
    assert "s3cr3t-value" not in exc_info.value.detail


# --------------------------------------------------------------------------- #
# Fake SDK clients — each mimics only the one call its provider makes.
# --------------------------------------------------------------------------- #


class _FakeVaultKVv2:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def read_secret_version(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeVaultClient:
    def __init__(self, response=None, error: Exception | None = None):
        self.kv_v2 = _FakeVaultKVv2(response, error)
        self.secrets = SimpleNamespace(kv=SimpleNamespace(v2=self.kv_v2))

    @property
    def calls(self) -> list[dict]:
        return self.kv_v2.calls


class _FakeAWSClient:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def get_secret_value(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


class _FakeAzureClient:
    def __init__(self, value=None, error: Exception | None = None):
        self.value = value
        self.error = error
        self.calls: list[str] = []

    def get_secret(self, name):
        self.calls.append(name)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(value=self.value)


class _FakeGCPClient:
    def __init__(self, data=None, error: Exception | None = None):
        self.data = data
        self.error = error
        self.calls: list[dict] = []

    def access_secret_version(self, request):
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(payload=SimpleNamespace(data=self.data))


# --------------------------------------------------------------------------- #
# Vault
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_vault_reads_and_parses_a_valid_kv_v2_secret():
    client = _FakeVaultClient(response={"data": {"data": dict(_SECRET)}})
    provider = VaultCredentialProvider("https://vault.internal", "tok", client=client)

    _assert_parsed(await provider.get_credentials("pg-prod-01"))
    # KV v2 path convention: <mount_point>/<path_prefix>/<server id>.
    assert client.calls[0]["path"] == "numi/db/pg-prod-01"
    assert client.calls[0]["mount_point"] == "secret"


@pytest.mark.asyncio
async def test_vault_maps_any_sdk_error_to_dependency_unavailable():
    client = _FakeVaultClient(error=_BoomError("permission denied"))
    provider = VaultCredentialProvider("https://vault.internal", "tok", client=client)

    with pytest.raises(NumiError) as exc_info:
        await provider.get_credentials("pg-prod-01")
    _assert_fails_closed(exc_info)


@pytest.mark.asyncio
async def test_vault_without_addr_or_token_fails_closed_without_touching_the_sdk():
    """The pre-existing unconfigured behavior, unchanged — and crucially it
    must not attempt a connection to an empty address first."""
    client = _FakeVaultClient(response={"data": {"data": dict(_SECRET)}})
    for addr, token in (("", "tok"), ("https://vault.internal", ""), ("", "")):
        provider = VaultCredentialProvider(addr, token, client=client)
        with pytest.raises(NumiError) as exc_info:
            await provider.get_credentials("pg-prod-01")
        assert exc_info.value.code == FailureCode.DEPENDENCY_UNAVAILABLE
        assert "not configured" in exc_info.value.detail
    assert client.calls == []


@pytest.mark.asyncio
async def test_vault_response_without_the_nested_data_object_fails_closed():
    client = _FakeVaultClient(response={"data": {}})
    provider = VaultCredentialProvider("https://vault.internal", "tok", client=client)

    with pytest.raises(NumiError) as exc_info:
        await provider.get_credentials("pg-prod-01")
    _assert_fails_closed(exc_info)


@pytest.mark.asyncio
async def test_vault_mount_point_and_prefix_are_configurable():
    client = _FakeVaultClient(response={"data": {"data": dict(_SECRET)}})
    provider = VaultCredentialProvider(
        "https://vault.internal", "tok", mount_point="kv", path_prefix="teams/dba", client=client
    )

    await provider.get_credentials("pg-prod-01")
    assert client.calls[0]["path"] == "teams/dba/pg-prod-01"
    assert client.calls[0]["mount_point"] == "kv"


# --------------------------------------------------------------------------- #
# AWS Secrets Manager
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_aws_reads_and_parses_a_valid_secret_string():
    client = _FakeAWSClient(response={"SecretString": json.dumps(_SECRET)})
    provider = AWSSecretsManagerCredentialProvider("eu-west-1", client=client)

    _assert_parsed(await provider.get_credentials("pg-prod-01"))
    assert client.calls[0]["SecretId"] == "numi/db/pg-prod-01"


@pytest.mark.asyncio
async def test_aws_accepts_a_binary_secret_carrying_the_same_json():
    client = _FakeAWSClient(response={"SecretBinary": json.dumps(_SECRET).encode("utf-8")})
    provider = AWSSecretsManagerCredentialProvider("eu-west-1", client=client)

    _assert_parsed(await provider.get_credentials("pg-prod-01"))


@pytest.mark.asyncio
async def test_aws_maps_any_sdk_error_to_dependency_unavailable():
    client = _FakeAWSClient(error=_BoomError("ResourceNotFoundException"))
    provider = AWSSecretsManagerCredentialProvider("eu-west-1", client=client)

    with pytest.raises(NumiError) as exc_info:
        await provider.get_credentials("pg-prod-01")
    _assert_fails_closed(exc_info)


@pytest.mark.asyncio
async def test_aws_without_a_region_fails_closed_without_touching_the_sdk():
    client = _FakeAWSClient(response={"SecretString": json.dumps(_SECRET)})
    provider = AWSSecretsManagerCredentialProvider("", client=client)

    with pytest.raises(NumiError) as exc_info:
        await provider.get_credentials("pg-prod-01")
    assert exc_info.value.code == FailureCode.DEPENDENCY_UNAVAILABLE
    assert "not configured" in exc_info.value.detail
    assert client.calls == []


@pytest.mark.asyncio
async def test_aws_secret_that_is_not_json_fails_closed():
    """Someone storing a bare password string instead of the documented
    object must not produce a half-built credential."""
    client = _FakeAWSClient(response={"SecretString": "s3cr3t-value"})
    provider = AWSSecretsManagerCredentialProvider("eu-west-1", client=client)

    with pytest.raises(NumiError) as exc_info:
        await provider.get_credentials("pg-prod-01")
    _assert_fails_closed(exc_info)


# --------------------------------------------------------------------------- #
# Azure Key Vault
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_azure_reads_and_parses_a_valid_secret():
    client = _FakeAzureClient(value=json.dumps(_SECRET))
    provider = AzureKeyVaultCredentialProvider("https://numi-kv.vault.azure.net", client=client)

    _assert_parsed(await provider.get_credentials("pg-prod-01"))
    assert client.calls[0] == "numi-db-pg-prod-01"


@pytest.mark.asyncio
async def test_azure_normalizes_a_server_id_into_a_legal_key_vault_name():
    """Key Vault names allow only letters/digits/dashes — a dot in a server
    id must not be sent in a URL that fails opaquely."""
    client = _FakeAzureClient(value=json.dumps(_SECRET))
    provider = AzureKeyVaultCredentialProvider("https://numi-kv.vault.azure.net", client=client)

    await provider.get_credentials("pg.prod_01")
    assert client.calls[0] == "numi-db-pg-prod-01"


@pytest.mark.asyncio
async def test_azure_maps_any_sdk_error_to_dependency_unavailable():
    client = _FakeAzureClient(error=_BoomError("ClientAuthenticationError"))
    provider = AzureKeyVaultCredentialProvider("https://numi-kv.vault.azure.net", client=client)

    with pytest.raises(NumiError) as exc_info:
        await provider.get_credentials("pg-prod-01")
    _assert_fails_closed(exc_info)


@pytest.mark.asyncio
async def test_azure_without_a_vault_url_fails_closed_without_touching_the_sdk():
    client = _FakeAzureClient(value=json.dumps(_SECRET))
    provider = AzureKeyVaultCredentialProvider("", client=client)

    with pytest.raises(NumiError) as exc_info:
        await provider.get_credentials("pg-prod-01")
    assert exc_info.value.code == FailureCode.DEPENDENCY_UNAVAILABLE
    assert "not configured" in exc_info.value.detail
    assert client.calls == []


# --------------------------------------------------------------------------- #
# GCP Secret Manager
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_gcp_reads_and_parses_a_valid_secret_version():
    client = _FakeGCPClient(data=json.dumps(_SECRET).encode("utf-8"))
    provider = GCPSecretManagerCredentialProvider("numi-prod", client=client)

    _assert_parsed(await provider.get_credentials("pg-prod-01"))
    assert (
        client.calls[0]["name"]
        == "projects/numi-prod/secrets/numi-db-pg-prod-01/versions/latest"
    )


@pytest.mark.asyncio
async def test_gcp_maps_any_sdk_error_to_dependency_unavailable():
    client = _FakeGCPClient(error=_BoomError("PermissionDenied"))
    provider = GCPSecretManagerCredentialProvider("numi-prod", client=client)

    with pytest.raises(NumiError) as exc_info:
        await provider.get_credentials("pg-prod-01")
    _assert_fails_closed(exc_info)


@pytest.mark.asyncio
async def test_gcp_without_a_project_id_fails_closed_without_touching_the_sdk():
    client = _FakeGCPClient(data=json.dumps(_SECRET).encode("utf-8"))
    provider = GCPSecretManagerCredentialProvider("", client=client)

    with pytest.raises(NumiError) as exc_info:
        await provider.get_credentials("pg-prod-01")
    assert exc_info.value.code == FailureCode.DEPENDENCY_UNAVAILABLE
    assert "not configured" in exc_info.value.detail
    assert client.calls == []


# --------------------------------------------------------------------------- #
# Secret-shape validation, shared by all four
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("missing_field", ["host", "port", "username", "password", "database"])
@pytest.mark.asyncio
async def test_a_secret_missing_any_required_field_fails_closed(missing_field):
    """Never a partial credential: a secret without a password must stop the
    execution, not connect with a blank one."""
    payload = {k: v for k, v in _SECRET.items() if k != missing_field}
    client = _FakeAWSClient(response={"SecretString": json.dumps(payload)})
    provider = AWSSecretsManagerCredentialProvider("eu-west-1", client=client)

    with pytest.raises(NumiError) as exc_info:
        await provider.get_credentials("pg-prod-01")
    assert exc_info.value.code == FailureCode.DEPENDENCY_UNAVAILABLE
    assert missing_field in exc_info.value.detail  # names the field, not its value


@pytest.mark.asyncio
async def test_a_non_integer_port_fails_closed():
    payload = dict(_SECRET, port="not-a-number")
    client = _FakeAWSClient(response={"SecretString": json.dumps(payload)})
    provider = AWSSecretsManagerCredentialProvider("eu-west-1", client=client)

    with pytest.raises(NumiError) as exc_info:
        await provider.get_credentials("pg-prod-01")
    assert exc_info.value.code == FailureCode.DEPENDENCY_UNAVAILABLE


@pytest.mark.asyncio
async def test_options_are_optional_and_default_to_empty():
    payload = {k: v for k, v in _SECRET.items() if k != "options"}
    client = _FakeAWSClient(response={"SecretString": json.dumps(payload)})
    provider = AWSSecretsManagerCredentialProvider("eu-west-1", client=client)

    creds = await provider.get_credentials("pg-prod-01")
    assert creds.options == {}


@pytest.mark.asyncio
async def test_a_retrieved_credential_never_renders_its_password():
    """The same no-leak guarantee `LocalDevCredentialProvider`'s output has,
    now on the real backends' path too."""
    client = _FakeAWSClient(response={"SecretString": json.dumps(_SECRET)})
    provider = AWSSecretsManagerCredentialProvider("eu-west-1", client=client)

    creds = await provider.get_credentials("pg-prod-01")
    assert "s3cr3t-value" not in repr(creds)
    assert "s3cr3t-value" not in str(creds)


# --------------------------------------------------------------------------- #
# build_credential_provider routing
# --------------------------------------------------------------------------- #


class _FakeSettings:
    def __init__(self, provider: str):
        self.secrets_provider = provider
        self.vault_addr = "https://vault.internal"
        self.vault_token = "tok"
        self.aws_region = "eu-west-1"
        self.azure_key_vault_url = "https://numi-kv.vault.azure.net"
        self.gcp_project_id = "numi-prod"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("local_dev", LocalDevCredentialProvider),
        ("vault", VaultCredentialProvider),
        ("aws_secrets_manager", AWSSecretsManagerCredentialProvider),
        ("azure_key_vault", AzureKeyVaultCredentialProvider),
        ("gcp_secret_manager", GCPSecretManagerCredentialProvider),
    ],
)
def test_build_credential_provider_routes_to_each_real_backend(name, expected):
    assert isinstance(build_credential_provider(_FakeSettings(name)), expected)


def test_build_credential_provider_still_defaults_to_the_dev_safe_backend():
    """`SECRETS_PROVIDER` unset means `local_dev` in `Settings` — the
    dev-safe default is unchanged by adding the real backends."""
    from numi.common.config import Settings

    settings = Settings(_env_file=None)
    assert settings.secrets_provider == "local_dev"
    assert isinstance(build_credential_provider(settings), LocalDevCredentialProvider)


def test_build_credential_provider_rejects_an_unknown_backend():
    with pytest.raises(NumiError) as exc_info:
        build_credential_provider(_FakeSettings("not-a-real-backend"))
    assert exc_info.value.code == FailureCode.DEPENDENCY_UNAVAILABLE
