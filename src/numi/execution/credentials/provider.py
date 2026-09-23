"""CredentialProvider abstraction (spec §19).

The Execution Service is the only component that ever holds a real database
credential, and even it never receives one directly from configuration in
production — it asks a `CredentialProvider`, which is backed by a real
secrets manager. Credentials are fetched just-in-time per execution and are
never logged, cached in the Gateway/Agent, or returned in any tool result
(the redaction filter in `numi.common.observability` is a defense-in-depth
backstop, not the primary control — the primary control is that credentials
simply never leave this module and the connection layer built on it).

## The secret's contents (identical shape for all four backends)

Every real backend stores one JSON object per registered server id, with
exactly the same keys `LocalDevCredentialProvider` reads out of
`config/dev_credentials.yaml` — so moving a deployment from dev to a real
secrets manager is a transport change, never a re-modelling of the data:

```json
{
  "host": "db-prod-01.internal",
  "port": 5432,
  "username": "numi_diag",
  "password": "...",
  "database": "postgres",
  "options": { "sslmode": "require" }
}
```

`options` is optional and engine-specific (see
`execution/adapters/connections.py`); every other key is required. A secret
missing a required key is a *misconfiguration*, and this module treats it
exactly like an unreachable secrets manager — `DEPENDENCY_UNAVAILABLE`,
never a partially-populated credential.

## SDK packages are optional extras, imported lazily

No deployment uses all four backends, so none of the four SDKs is a base
dependency — each lives in its own optional-dependency group in
`pyproject.toml` (`secrets-vault`, `secrets-aws`, `secrets-azure`,
`secrets-gcp`) and is imported at *first real use*, inside the provider
that needs it. This mirrors how the LLM provider SDKs and the database
drivers are already handled: selecting Vault never requires boto3 to be
installed, and a missing package surfaces as a clear, actionable
`DEPENDENCY_UNAVAILABLE` naming the package and the extra to install —
never an import-time crash of the whole service.

## Testability: an injectable client, never a hardcoded import-and-call

Each real provider takes an optional `client=` constructor argument. When
it is `None` (production), the provider builds the real SDK client lazily
on first use from its own configuration; when a test passes one in, no SDK
is imported and no network call is ever made. This is the same seam the
rest of the codebase already uses for things that don't exist in dev/test
(`tests/canned_adapter.py`, `FakeQueryExecutor`, the LLM provider tests).

## Fail closed, always

Every real `get_credentials` wraps the entire SDK interaction in a single
`try`/`except Exception` and re-raises as
`NumiError(FailureCode.DEPENDENCY_UNAVAILABLE, ...)`. A raw SDK exception
must never escape this module: it would carry vendor stack traces, secret
ARNs/paths, and sometimes the secret material itself past the failure
boundary the rest of the platform relies on (spec §28/§63). Error text
handed to the caller names the backend and the server id only — never the
secret's contents, and never a field's value.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, SecretStr

from numi.common.models.failures import FailureCode, NumiError


class DatabaseCredentials(BaseModel):
    host: str
    port: int
    username: str
    password: SecretStr
    database: str
    options: dict = {}

    def __repr__(self) -> str:  # never leak the secret via logs/repr
        return f"DatabaseCredentials(host={self.host!r}, username={self.username!r}, password=***)"

    __str__ = __repr__


class CredentialProvider(ABC):
    @abstractmethod
    async def get_credentials(self, database_id: str) -> DatabaseCredentials: ...


class LocalDevCredentialProvider(CredentialProvider):
    """Development-only provider. Reads `config/dev_credentials.yaml`, keyed
    by server id (`config/servers.yaml`) — never used in production (see
    `SECRETS_PROVIDER` env var and the Vault/AWS/Azure/GCP adapters below,
    which fail closed until properly configured)."""

    def __init__(self, config_path: str | Path):
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        self._entries = raw.get("credentials", {})

    async def get_credentials(self, database_id: str) -> DatabaseCredentials:
        entry = self._entries.get(database_id)
        if entry is None:
            raise NumiError(
                FailureCode.DEPENDENCY_UNAVAILABLE,
                f"No development credentials configured for '{database_id}'.",
            )
        return DatabaseCredentials(**entry)


# --------------------------------------------------------------------------- #
# Shared helpers for the real secrets-manager backends
# --------------------------------------------------------------------------- #

_REQUIRED_SECRET_FIELDS = ("host", "port", "username", "password", "database")

# Key Vault secret names and GCP secret ids accept only `[A-Za-z0-9-]` /
# `[A-Za-z0-9_-]`. Server ids in `config/servers.yaml` are already
# conservative, but a dot or slash in one must not produce a silently
# malformed request URL — normalize instead.
_NON_NAME_SAFE = re.compile(r"[^A-Za-z0-9-]")


def _sdk_missing_error(provider_name: str, package: str, extra: str, exc: Exception) -> NumiError:
    return NumiError(
        FailureCode.DEPENDENCY_UNAVAILABLE,
        f"The '{provider_name}' secrets manager integration requires the '{package}' "
        f"package, which is not installed in this deployment. Install it with "
        f'`pip install -e ".[{extra}]"` and restart the Execution Service.',
        internal_detail=f"ImportError importing SDK for '{provider_name}': {exc!r}",
    )


def _load_sdk(module_path: str, *, provider_name: str, package: str, extra: str) -> Any:
    """Import an optional secrets-manager SDK at first real use.

    Kept as a function (rather than a module-level `try: import`) so that a
    deployment using only one backend never needs the other three
    installed, and so an injected test client short-circuits the import
    entirely — it is only ever reached on the real, un-injected path.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        raise _sdk_missing_error(provider_name, package, extra, exc) from exc


def _backend_error(provider_name: str, database_id: str, exc: Exception) -> NumiError:
    """Map any SDK/network/auth failure onto the one code the platform
    understands, with the vendor's own text confined to `internal_detail`
    (logs/audit only — never forwarded to a channel adapter, see
    `NumiError`)."""
    return NumiError(
        FailureCode.DEPENDENCY_UNAVAILABLE,
        f"Could not retrieve the database credential for '{database_id}' from the "
        f"'{provider_name}' secrets manager. Refusing to execute rather than falling "
        "back to a less secure credential source.",
        internal_detail=f"{provider_name} get_credentials('{database_id}') failed: {exc!r}",
    )


def _malformed_secret_error(provider_name: str, database_id: str, reason: str) -> NumiError:
    """A secret that exists but doesn't carry a usable credential.

    `reason` names *fields*, never values — a malformed-secret message must
    stay safe to log and to surface, so nothing derived from the secret's
    contents is ever interpolated into it.
    """
    return NumiError(
        FailureCode.DEPENDENCY_UNAVAILABLE,
        f"The '{provider_name}' secret for '{database_id}' is not a usable database "
        f"credential ({reason}). Refusing to execute with an incomplete credential.",
        internal_detail=f"{provider_name} secret for '{database_id}' malformed: {reason}",
    )


def _decode_secret_json(raw: Any, *, provider_name: str, database_id: str) -> Any:
    """Parse a secret payload that a backend hands back as a JSON string or
    bytes (AWS `SecretString`, Azure `secret.value`, GCP `payload.data`).
    Vault already returns a decoded dict and skips this."""
    if isinstance(raw, bytes | bytearray):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _malformed_secret_error(
                provider_name, database_id, "secret is not UTF-8 text"
            ) from exc
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            # A binary/opaque secret, or someone stored a bare password
            # string. Either way it is not the documented JSON object.
            raise _malformed_secret_error(
                provider_name, database_id, "secret is not a JSON object"
            ) from exc
    return raw


def _credentials_from_payload(
    payload: Any, *, provider_name: str, database_id: str
) -> DatabaseCredentials:
    """Turn a decoded secret into `DatabaseCredentials`, the same way
    `LocalDevCredentialProvider` does — or fail closed.

    Deliberately strict: a secret missing `password`, or carrying a
    non-numeric `port`, is a misconfiguration that must stop the execution,
    not something to paper over with a default. Returning a
    partially-populated credential here would hand the connection layer
    something that fails much later, much less clearly, and (for a blank
    password against a trust-auth database) potentially succeeds when it
    should not have.
    """
    if not isinstance(payload, dict):
        raise _malformed_secret_error(
            provider_name, database_id, "secret is not a JSON object"
        )

    missing = [f for f in _REQUIRED_SECRET_FIELDS if payload.get(f) in (None, "")]
    if missing:
        raise _malformed_secret_error(
            provider_name, database_id, f"missing required field(s): {', '.join(missing)}"
        )

    options = payload.get("options") or {}
    if not isinstance(options, dict):
        raise _malformed_secret_error(
            provider_name, database_id, "'options' is present but is not an object"
        )

    try:
        return DatabaseCredentials(
            host=str(payload["host"]),
            port=int(payload["port"]),
            username=str(payload["username"]),
            password=SecretStr(str(payload["password"])),
            database=str(payload["database"]),
            options=options,
        )
    except (TypeError, ValueError) as exc:
        # Only ever raised by `int(port)` / pydantic coercion above. The
        # exception text can echo the offending value, so it is confined to
        # `internal_detail` by `_malformed_secret_error` and never
        # interpolated into the user-facing message.
        raise _malformed_secret_error(
            provider_name, database_id, "'port' is not an integer"
        ) from exc


class _UnconfiguredSecretsManagerProvider(CredentialProvider):
    """Base for real secrets-manager adapters. Fails closed with a clear,
    actionable error until the corresponding SDK/client is wired up with real
    connection details — it never silently falls back to a permissive or
    mock credential (spec §63: privileged dependencies fail closed).

    Each concrete subclass below overrides `get_credentials` with its real
    implementation, and calls back into *this* one (via `super()`) for the
    unconfigured case, so "the operator hasn't finished configuring this
    backend" keeps producing exactly the same refusal it always has — no
    connection attempt to an empty address, no vendor SDK involved at all.
    """

    name = "unconfigured"

    async def get_credentials(self, database_id: str) -> DatabaseCredentials:
        raise NumiError(
            FailureCode.DEPENDENCY_UNAVAILABLE,
            f"The '{self.name}' secrets manager integration is not configured in this "
            "deployment. Refusing to execute rather than falling back to a less secure "
            "credential source.",
        )


class VaultCredentialProvider(_UnconfiguredSecretsManagerProvider):
    """HashiCorp Vault-backed provider (KV v2).

    **Path convention.** The credential for server id `<id>` is read from
    the KV v2 secret `<mount_point>/<path_prefix>/<id>` — by default
    `secret/numi/db/<id>`, which is `secret/data/numi/db/<id>` in Vault's
    HTTP API (KV v2 interposes `data/`; `hvac`'s `kv.v2` helper adds it for
    us, so the mount point and prefix here are written without it). Both
    halves are constructor arguments, so a deployment with an existing
    mount layout points at it without a code change.

    **Secret contents.** The KV secret's `data` is the JSON object
    documented in this module's docstring (`host`, `port`, `username`,
    `password`, `database`, optional `options`). Written with, e.g.:

    ```
    vault kv put secret/numi/db/pg-prod-01 \\
        host=db-prod-01.internal port=5432 username=numi_diag \\
        password=... database=postgres
    ```

    **Auth.** `VAULT_ADDR` + `VAULT_TOKEN` (`vault_addr`/`vault_token`
    here). A deployment using Kubernetes/AppRole auth performs that login
    outside this class and injects an already-authenticated `hvac.Client`
    via `client=` — the same seam the tests use — rather than this class
    growing a login method per auth backend.

    **Dynamic credentials.** Vault's *database* secrets engine (short-lived,
    per-request credentials) returns `{username, password}` only; the
    remaining connection details are not in Vault. Point `path_prefix` at a
    KV path as above for the full object, or inject a client wrapper that
    merges the dynamic pair into the documented shape.
    """

    name = "vault"

    def __init__(
        self,
        vault_addr: str = "",
        vault_token: str = "",
        *,
        mount_point: str = "secret",
        path_prefix: str = "numi/db",
        client: Any = None,
    ):
        self._addr = vault_addr
        self._token = vault_token
        self._mount_point = mount_point
        self._path_prefix = path_prefix.strip("/")
        # `Any`, not `hvac.Client | None`, so this module stays importable
        # (and type-checkable) without the optional `hvac` package — same
        # approach `execution/adapters/connections.py` takes for the
        # optional database drivers.
        self._client: Any = client

    def secret_path(self, database_id: str) -> str:
        """The KV v2 path (without the `data/` segment hvac inserts)."""
        return f"{self._path_prefix}/{database_id}"

    def _get_client(self) -> Any:
        if self._client is None:
            hvac = _load_sdk("hvac", provider_name=self.name, package="hvac", extra="secrets-vault")
            self._client = hvac.Client(url=self._addr, token=self._token)
        return self._client

    async def get_credentials(self, database_id: str) -> DatabaseCredentials:
        if not self._addr or not self._token:
            # Exactly as before: no connection attempt to an empty address.
            return await super().get_credentials(database_id)

        client = self._get_client()
        path = self.secret_path(database_id)

        def _read() -> Any:
            # hvac is synchronous; keep the Execution Service's event loop
            # free while Vault is round-tripped.
            return client.secrets.kv.v2.read_secret_version(
                path=path, mount_point=self._mount_point, raise_on_deleted_version=True
            )

        try:
            response = await asyncio.to_thread(_read)
        except Exception as exc:  # noqa: BLE001 — every failure mode is DEPENDENCY_UNAVAILABLE
            raise _backend_error(self.name, database_id, exc) from exc

        # KV v2 nests the secret one level down: {"data": {"data": {...}}}.
        payload = (response or {}).get("data", {}).get("data") if isinstance(response, dict) else None
        if payload is None:
            raise _malformed_secret_error(
                self.name, database_id, "response had no data.data object"
            )
        return _credentials_from_payload(payload, provider_name=self.name, database_id=database_id)


class AWSSecretsManagerCredentialProvider(_UnconfiguredSecretsManagerProvider):
    """AWS Secrets Manager-backed provider.

    **Name convention.** The credential for server id `<id>` is the secret
    named `<name_prefix>/<id>` — by default `numi/db/<id>`. A full ARN
    works too if `name_prefix` is set to one; `GetSecretValue` accepts
    either. The prefix is a constructor argument so an existing naming
    scheme needs no code change.

    **Secret contents.** The secret's `SecretString` is the JSON object
    documented in this module's docstring. Created with, e.g.:

    ```
    aws secretsmanager create-secret --name numi/db/pg-prod-01 \\
        --secret-string '{"host":"...","port":5432,"username":"numi_diag",
                          "password":"...","database":"postgres"}'
    ```

    A binary-only secret (`SecretBinary`) is accepted as long as it decodes
    to that same UTF-8 JSON object.

    **Auth.** The standard boto3 credential chain (instance/task role,
    environment, shared config) — this class never takes an access key, and
    `AWS_REGION` (`region` here) is the only AWS setting it reads. IAM
    should scope the Execution Service's role to
    `secretsmanager:GetSecretValue` on `numi/db/*` and nothing else.
    """

    name = "aws_secrets_manager"

    def __init__(self, region: str = "", *, name_prefix: str = "numi/db", client: Any = None):
        self._region = region
        self._name_prefix = name_prefix.rstrip("/")
        self._client: Any = client

    def secret_name(self, database_id: str) -> str:
        return f"{self._name_prefix}/{database_id}"

    def _get_client(self) -> Any:
        if self._client is None:
            boto3 = _load_sdk(
                "boto3", provider_name=self.name, package="boto3", extra="secrets-aws"
            )
            self._client = boto3.client("secretsmanager", region_name=self._region)
        return self._client

    async def get_credentials(self, database_id: str) -> DatabaseCredentials:
        if not self._region:
            return await super().get_credentials(database_id)

        client = self._get_client()
        secret_id = self.secret_name(database_id)

        try:
            response = await asyncio.to_thread(lambda: client.get_secret_value(SecretId=secret_id))
        except Exception as exc:  # noqa: BLE001 — incl. ResourceNotFound/AccessDenied/timeouts
            raise _backend_error(self.name, database_id, exc) from exc

        if not isinstance(response, dict):
            raise _malformed_secret_error(self.name, database_id, "response was not a mapping")
        raw = response.get("SecretString")
        if raw is None:
            raw = response.get("SecretBinary")
        if raw is None:
            raise _malformed_secret_error(
                self.name, database_id, "response had neither SecretString nor SecretBinary"
            )

        payload = _decode_secret_json(raw, provider_name=self.name, database_id=database_id)
        return _credentials_from_payload(payload, provider_name=self.name, database_id=database_id)


class AzureKeyVaultCredentialProvider(_UnconfiguredSecretsManagerProvider):
    """Azure Key Vault-backed provider.

    **Name convention.** The credential for server id `<id>` is the secret
    named `<name_prefix>-<id>` — by default `numi-db-<id>`. Key Vault
    secret names allow only letters, digits and dashes, so any other
    character in a server id is normalized to a dash rather than sent in a
    URL that would fail opaquely.

    **Secret contents.** The secret's *value* is the JSON object documented
    in this module's docstring, stored as a single string:

    ```
    az keyvault secret set --vault-name numi-kv --name numi-db-pg-prod-01 \\
        --value '{"host":"...","port":5432,"username":"numi_diag",
                  "password":"...","database":"postgres"}'
    ```

    **Auth.** `DefaultAzureCredential` — managed identity in Azure,
    environment/CLI credentials elsewhere. This class never takes a client
    secret of its own; `AZURE_KEY_VAULT_URL` (`vault_url` here) is the only
    Azure setting it reads. Grant the Execution Service's identity the
    `Key Vault Secrets User` role, scoped to this vault.
    """

    name = "azure_key_vault"

    def __init__(self, vault_url: str = "", *, name_prefix: str = "numi-db", client: Any = None):
        self._vault_url = vault_url
        self._name_prefix = name_prefix
        self._client: Any = client

    def secret_name(self, database_id: str) -> str:
        return _NON_NAME_SAFE.sub("-", f"{self._name_prefix}-{database_id}")

    def _get_client(self) -> Any:
        if self._client is None:
            secrets_mod = _load_sdk(
                "azure.keyvault.secrets",
                provider_name=self.name,
                package="azure-keyvault-secrets",
                extra="secrets-azure",
            )
            identity_mod = _load_sdk(
                "azure.identity",
                provider_name=self.name,
                package="azure-identity",
                extra="secrets-azure",
            )
            self._client = secrets_mod.SecretClient(
                vault_url=self._vault_url, credential=identity_mod.DefaultAzureCredential()
            )
        return self._client

    async def get_credentials(self, database_id: str) -> DatabaseCredentials:
        if not self._vault_url:
            return await super().get_credentials(database_id)

        client = self._get_client()
        secret_name = self.secret_name(database_id)

        try:
            secret = await asyncio.to_thread(lambda: client.get_secret(secret_name))
        except Exception as exc:  # noqa: BLE001 — incl. ResourceNotFoundError/ClientAuthenticationError
            raise _backend_error(self.name, database_id, exc) from exc

        raw = getattr(secret, "value", None)
        if raw is None:
            raise _malformed_secret_error(self.name, database_id, "secret had no value")

        payload = _decode_secret_json(raw, provider_name=self.name, database_id=database_id)
        return _credentials_from_payload(payload, provider_name=self.name, database_id=database_id)


class GCPSecretManagerCredentialProvider(_UnconfiguredSecretsManagerProvider):
    """Google Cloud Secret Manager-backed provider.

    **Name convention.** The credential for server id `<id>` is read from
    `projects/<project_id>/secrets/<name_prefix>-<id>/versions/<version>` —
    by default the `latest` version of `numi-db-<id>`. GCP secret ids
    allow only letters, digits, dashes and underscores; any other character
    in a server id is normalized to a dash.

    **Secret contents.** The version's payload is the JSON object documented
    in this module's docstring, stored as UTF-8 bytes:

    ```
    printf '{"host":"...","port":5432,"username":"numi_diag",
             "password":"...","database":"postgres"}' |
      gcloud secrets create numi-db-pg-prod-01 --data-file=-
    ```

    **Auth.** Application Default Credentials (workload identity on GKE, the
    attached service account on GCE/Cloud Run). This class never takes a
    key file; `GCP_PROJECT_ID` (`project_id` here) is the only GCP setting
    it reads. Grant the Execution Service's service account
    `roles/secretmanager.secretAccessor`, scoped to these secrets.
    """

    name = "gcp_secret_manager"

    def __init__(
        self,
        project_id: str = "",
        *,
        name_prefix: str = "numi-db",
        version: str = "latest",
        client: Any = None,
    ):
        self._project_id = project_id
        self._name_prefix = name_prefix
        self._version = version
        self._client: Any = client

    def secret_version_name(self, database_id: str) -> str:
        secret_id = _NON_NAME_SAFE.sub("-", f"{self._name_prefix}-{database_id}")
        return f"projects/{self._project_id}/secrets/{secret_id}/versions/{self._version}"

    def _get_client(self) -> Any:
        if self._client is None:
            secretmanager = _load_sdk(
                "google.cloud.secretmanager",
                provider_name=self.name,
                package="google-cloud-secret-manager",
                extra="secrets-gcp",
            )
            self._client = secretmanager.SecretManagerServiceClient()
        return self._client

    async def get_credentials(self, database_id: str) -> DatabaseCredentials:
        if not self._project_id:
            return await super().get_credentials(database_id)

        client = self._get_client()
        name = self.secret_version_name(database_id)

        try:
            response = await asyncio.to_thread(
                lambda: client.access_secret_version(request={"name": name})
            )
        except Exception as exc:  # noqa: BLE001 — incl. NotFound/PermissionDenied/DeadlineExceeded
            raise _backend_error(self.name, database_id, exc) from exc

        raw = getattr(getattr(response, "payload", None), "data", None)
        if raw is None:
            raise _malformed_secret_error(
                self.name, database_id, "response had no payload.data"
            )

        payload = _decode_secret_json(raw, provider_name=self.name, database_id=database_id)
        return _credentials_from_payload(payload, provider_name=self.name, database_id=database_id)


def build_credential_provider(settings) -> CredentialProvider:
    provider = settings.secrets_provider
    if provider == "local_dev":
        return LocalDevCredentialProvider("config/dev_credentials.yaml")
    if provider == "vault":
        return VaultCredentialProvider(settings.vault_addr, settings.vault_token)
    if provider == "aws_secrets_manager":
        return AWSSecretsManagerCredentialProvider(settings.aws_region)
    if provider == "azure_key_vault":
        return AzureKeyVaultCredentialProvider(settings.azure_key_vault_url)
    if provider == "gcp_secret_manager":
        return GCPSecretManagerCredentialProvider(settings.gcp_project_id)
    raise NumiError(
        FailureCode.DEPENDENCY_UNAVAILABLE, f"Unknown secrets provider '{provider}'."
    )
