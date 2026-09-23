from __future__ import annotations

from numi.gateway.domain.data_policy import (
    DEFAULT_SENSITIVE_FIELD_PATTERNS,
    DataMinimizer,
    DataPolicyConfig,
)


def test_masks_sensitive_fields_by_default():
    minimizer = DataMinimizer()
    rows = [{"username": "svc_app", "password_hash": "abc123", "cpu_percent": 92}]
    clean, masked, truncated = minimizer.apply(rows)
    assert clean[0]["password_hash"] == "***MASKED***"
    assert clean[0]["cpu_percent"] == 92
    assert "password_hash" in masked
    assert truncated is False


def test_truncates_rows_over_the_cap():
    minimizer = DataMinimizer()
    rows = [{"i": i} for i in range(500)]
    clean, _masked, truncated = minimizer.apply(rows, max_rows=10)
    assert len(clean) == 10
    assert truncated is True


def test_never_returns_select_star_worth_of_pii_unmasked():
    minimizer = DataMinimizer()
    rows = [
        {
            "account_number": "0123456789",
            "bvn": "12345678901",
            "email": "person@example.com",
            "phone": "+2348000000000",
        }
    ]
    clean, masked, _ = minimizer.apply(rows)
    assert all(v == "***MASKED***" for v in clean[0].values())
    assert set(masked) == {"account_number", "bvn", "email", "phone"}


# --- no-regression guard for the literal-scrubbing addition -------------------
#
# Literal scrubbing (see query_scrubber.py) added a SECOND field class to
# this layer. These pin that the FIRST one -- full masking by field name --
# behaves exactly as it did before, so the addition is strictly additive.


def test_existing_field_name_masking_is_completely_unchanged():
    """Every default sensitive pattern still fully masks, byte for byte --
    `***MASKED***`, not a scrubbed-but-present value."""
    minimizer = DataMinimizer()
    rows = [
        {
            "password": "hunter2",
            "pass_hash": "$2b$12$abc",
            "token": "eyJhbGci",
            "secret": "s3cr3t",
            "api_key": "sk-live-1",
            "card_number": "4111111111111111",
            "account_number": "0123456789",
            "bvn": "12345678901",
            "nin": "11111111111",
            "ssn": "123-45-6789",
            "phone": "+2348000000000",
            "email": "person@example.com",
            "address": "1 Main St",
            "auth_header": "Bearer x",
            "credential": "c",
            "connection_string": "Server=...;Password=...",
        }
    ]
    clean, masked, truncated = minimizer.apply(rows)

    assert all(v == "***MASKED***" for v in clean[0].values())
    assert set(masked) == set(rows[0])
    assert truncated is False
    # Sanity: the fixture really does exercise every shipped pattern.
    assert len(DEFAULT_SENSITIVE_FIELD_PATTERNS) == 16


def test_a_sensitive_name_wins_over_free_text_scrubbing():
    """A field matching BOTH lists is fully masked, never merely scrubbed --
    the ordering data_policy.py's module docstring pins."""
    out = DataMinimizer().minimize(
        [{"auth_statement": "SELECT * FROM t WHERE pw = 'p'"}]
    )
    # "auth" is a sensitive pattern; "(^|_)statement$" is a free-text one.
    assert out.rows[0]["auth_statement"] == "***MASKED***"
    assert out.masked_fields == ["auth_statement"]
    assert out.literal_scrubbed_fields == []


def test_apply_keeps_its_original_three_tuple_shape():
    """`apply()` is unchanged for every existing caller; `minimize()` is the
    richer seam the Gateway uses."""
    result = DataMinimizer().apply([{"cpu_percent": 92}])
    assert isinstance(result, tuple)
    assert len(result) == 3
    rows, masked, truncated = result
    assert rows == [{"cpu_percent": 92}]
    assert masked == []
    assert truncated is False


def test_configuring_no_free_text_patterns_scrubs_nothing():
    """`"|".join([])` is `""`, which matches every field name -- an empty
    list must mean "scrub nothing", not "scrub everything"."""
    minimizer = DataMinimizer(DataPolicyConfig(free_text_sql_field_patterns=[]))
    out = minimizer.minimize([{"query_text": "SELECT * FROM t WHERE id = 42"}])
    assert out.rows[0]["query_text"] == "SELECT * FROM t WHERE id = 42"
    assert out.literal_scrubbed_fields == []
