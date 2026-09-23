"""Generic result envelopes returned from tool execution.

Diagnostic (read) tools return tabular evidence; the Data Policy Layer
(spec §22) has already masked/limited it before it reaches this model, so by
construction nothing inside `rows` should be raw, unmasked, unbounded output.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class DiagnosticResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    columns: list[str]
    rows: list[dict[str, Any]]
    row_count: int
    truncated: bool
    max_result_rows: int
    masked_fields: list[str] = []


class WriteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    success: bool
    detail: str
    affected: dict[str, Any] = {}
    verification_status: str = "NOT_APPLICABLE"
    verification_detail: str = ""
