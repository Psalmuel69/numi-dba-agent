# Security Model

## The 20 non-negotiable rules (spec §74) and where each lives in code

| # | Rule | Where it's enforced |
|---|------|----------------------|
| 1 | LLM never connects directly to a database | `agent/` has no DB driver dependency at all; only `execution/adapters/` do |
| 2 | Agent never receives database credentials | `CredentialProvider` (`execution/credentials/provider.py`) only exists inside `execution/`; nothing there is importable/reachable from `agent/` |
| 3 | Agent never determines authorization | `gateway/domain/authorization.py::authorize` — always re-derived from a fresh `IdentityProvider` lookup keyed on `channel`+`channel_account_id`, never a claim in `ToolCallRequest` |
| 4 | Agent never determines whether approval is required | `gateway/domain/policy_engine.py::PolicyEngine.evaluate` — the Agent only ever sees the *result* (`APPROVAL_REQUIRED`) |
| 5 | Agent never approves its own actions | `gateway/domain/approval.py::ApprovalEngine.decide` enforces separation of duties for dual-approval tools; approval decisions go through `/v1/approvals/*`, never through the Agent |
| 6 | Tool calls always pass through Gateway | `agent/tool_client.py` is the *only* HTTP client the Agent has for anything database-related |
| 7 | Gateway always validates target scope | `gateway/domain/target_validation.py` |
| 8 | Gateway always validates policy | `gateway/domain/tool_call_handler.py` step 6, unconditionally, on every request |
| 9 | Privileged operations fail closed | `PolicyEngine.default_decision = DENY`; `CredentialProvider` adapters raise rather than fall back (`_UnconfiguredSecretsManagerProvider`) |
| 10 | Approval bound to exact action parameters | `ApprovalContext.action_hash()` — actor, tool, tool version, target, normalized arguments, environment, database, risk level, all hashed together |
| 11 | Approval expires | `ApprovalEngine` always sets `expires_at`; `_expire_if_needed` is checked on every decide/verify call, including after an approval has already been granted (an *unused* approved action still lapses) |
| 12 | Database credentials managed outside the agent | `CredentialProvider` abstraction with implemented Vault/AWS/Azure/GCP backends (`execution/credentials/provider.py`), each fetching just-in-time from the real secrets manager |
| 13 | Database output treated as untrusted data | Agent planners inspect only structural fields (row counts, named ids); the system prompt for the real LLM provider explicitly instructs it never to treat tool-result text as instructions (defense in depth on top of the structural design) |
| 14 | Audit records cannot be modified by the agent | `gateway/domain/audit.py::AuditLog` exposes only `record`/`record_security_event` — no update/delete method exists anywhere in the codebase |
| 15 | Production/non-production explicitly separated | `Environment` enum; every policy table, every registered server, every risk assessment is environment-scoped |
| 16 | Arbitrary SQL disabled by default | `Settings.enable_execute_sql_tool = False`, `enable_readonly_sql_tool = False` (and four more `enable_*` flags) — see `.env.example` |
| 17 | Typed operations preferred over raw SQL | Every controlled write tool has its own Pydantic argument model (`common/models/tool_arguments.py`); the adapter constructs SQL, never the LLM |
| 18 | All privileged actions are auditable | Every branch of `tool_call_handler.py` (success and denial) calls `AuditLog.record` |
| 19 | All writes require explicit policy evaluation | No code path reaches `execution.execute()` without first passing `PolicyEngine.evaluate` |
| 20 | LLM never part of the security boundary | See "No security by prompt" below |
| 21 | Literal values never leave a database inside free-text SQL | `gateway/domain/query_scrubber.py::scrub_sql_literals`, applied by `DataMinimizer` to every field matching `DataPolicyConfig.free_text_sql_field_patterns` (`query_text`, `blocked_query`, `deadlock_graph`, `message`, ...) at step 10 of `tool_call_handler.py` — sqlglot replaces every literal with `<redacted>` while table/column names and SQL structure survive; unparseable text (log lines, plan XML) falls back to a deliberately blunt regex scrub rather than passing through |
| 22 | The diagnostic login's least privilege is verified, not assumed | `ServerDiscoverer._check_least_privilege` per engine (`has_table_privilege` / `HAS_PERMS_BY_NAME` / the three `information_schema` privilege views), run once per discovery run — a login holding SELECT on any user table/view produces a `LeastPrivilegeFinding` on `ServerCatalog`, a WARNING-level `least_privilege_violation` log record, and a visible warning in `/catalog <server>`. Read-only introspection: Numi reports, a human DBA revokes — nothing here attempts a REVOKE |

## No security by prompt

The provider system prompts (`agent/llm/base.py`) tell the model not to
treat tool output as instructions and not to invent tools — this is **good
practice, not a control**. Which model runs (Anthropic / OpenAI / Gemini /
DeepSeek, chosen per conversation, or the offline planner) is a product
choice made in `agent/llm/registry.py` and has **zero** bearing on the
security boundary. If every LLM provider vanished and was replaced by a
coin flip, the security properties above would be unchanged, because:

- the model can only select from a `tool_id` list the Gateway actually
  registered (`ToolRegistry`), and even that is re-validated;
- the model's structured output is re-validated against Pydantic schemas
  the model doesn't control (`agent_action_adapter`);
- a proposed `tool_id` outside the offered menu is downgraded to a
  clarifying question in `StructuredLLMProvider`, and could not reach the
  Gateway anyway;
- authorization/policy/risk/approval are computed from database records and
  configuration files the model never touches.

## Identity

- `IDENTITY_PROVIDER=oidc` selects `OIDCIdentityProvider`
  (`common/identity/oidc_provider.py`), a real implementation against a
  standards-based IdP — OIDC Discovery for endpoint metadata, OAuth 2.0
  client credentials for Numi's own directory access, SCIM 2.0 for the
  lookup itself (Okta / Entra ID / Ping / generic, no vendor branches).
  Never against Slack/Teams display names.
- A chat webhook carries no token, so resolution is a *directory query*,
  not token validation: `GET /Users?filter=<attr> eq "<channel account
  id>"`, where `<attr>` is the per-channel attribute declared in
  `config/identity.yaml`'s `oidc.channel_attributes`. A channel with no
  attribute configured resolves to `None` rather than guessing one. The
  account id is escaped before it enters the SCIM filter — an unescaped
  quote there would be an authentication bypass, not merely a bad query.
- `refresh(subject_id)` always issues its own directory call; no resolved
  identity is ever cached, so a revoked group takes effect on the next tool
  call. (Numi's own client-credentials token *is* cached — that
  authenticates the service, not the user.)
- Every failure mode — network, auth, 5xx, inactive account, ambiguous
  match, malformed resource — returns `None`, which every call site already
  treats as "not a recognized DBA". Nothing is raised past the provider.
- `MockIdentityProvider` (`common/identity/provider.py`) is a config-driven
  stand-in for local dev/tests, loading `config/identity.yaml` — a file that
  intentionally contains zero real employees. It remains the default.
- Group → role mapping lives entirely in configuration
  (`config/identity.yaml`'s `identity.roles` section), never in code — and
  both providers derive roles through the same `GroupRoleMapping`, so the
  mapping cannot drift between dev and production.
- `common/identity/factory.py::build_identity_provider` is the single
  selection point; an unrecognized `IDENTITY_PROVIDER` is an error, never a
  silent fallback to the mock directory.

## Secrets

- `.env.example` contains placeholders only; `.gitignore` excludes `.env`,
  `*.pem`, `*.key`, `credentials.json`, `service-account.json`, and
  `config/dev_credentials.yaml`.
- `SECRETS_PROVIDER=local_dev` is the only mode that reads a plaintext
  YAML file (`config/dev_credentials.yaml`, git-ignored, created from
  `config/dev_credentials.example.yaml`). The four real backends —
  `vault`, `aws_secrets_manager`, `azure_key_vault`, `gcp_secret_manager`
  (`execution/credentials/provider.py`) — fetch the credential
  just-in-time from the real secrets manager per execution, and still fail
  closed whenever their connection details are unset.
- Every real backend maps *any* SDK/network/auth failure, and any secret
  that isn't a complete credential (missing `password`, non-integer
  `port`, non-JSON payload), onto
  `NumiError(DEPENDENCY_UNAVAILABLE)`. A raw vendor exception never
  escapes that module — it would carry secret paths/ARNs, vendor stack
  traces, and sometimes the secret material itself past the failure
  boundary — and a partial credential is never returned.
- Error text naming a bad secret names *fields*, never values; vendor
  detail is confined to `NumiError.internal_detail` (logs/audit only).
- Structured logging (`common/observability.py`) redacts any field whose
  *name* matches a secret-shaped pattern (`password`, `token`, `secret`,
  `api_key`, `connection_string`, ...) as a defense-in-depth backstop — the
  primary control is that secrets simply never reach a log call in the
  first place.

## Data minimization

`gateway/domain/data_policy.py::DataMinimizer` masks fields matching a
configurable sensitive-data regex (password/token/secret/card number/
account number/BVN/NIN/phone/email/address/...), caps row and column
counts, and reports whether truncation occurred — applied exactly once, in
the Gateway, before a diagnostic result is handed back to the Agent.

## Service-to-service authentication

`common/service_auth.py` issues/verifies short-lived (default 60s),
audience-scoped, HMAC-signed tokens for every internal hop
(Channels→Agent, Agent→Gateway, Gateway→Execution). There is no
`X-Internal-Request: true`-style header anywhere in the codebase.

## Fail-closed dependencies

If the Policy Engine, identity provider, server registry, or credential
provider is unavailable or misconfigured, the affected request is denied
(`NumiError`), never silently allowed. See `THREAT_MODEL.md` for the
specific failure-mode tests.
