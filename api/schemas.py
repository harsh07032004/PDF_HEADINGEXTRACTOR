"""
api/schemas.py

Pydantic v2 models for FastAPI HTTP request/response serialization.

These are intentionally separate from extractor/models.py (dataclasses).
The dataclasses represent the internal engine contract; these Pydantic
models represent the HTTP API contract — they can evolve independently.

Separation benefits:
    - API versioning: you can add/rename fields here without touching the engine
    - OpenAPI: Pydantic v2 generates clean JSON Schema automatically
    - Validation: field-level validators catch bad inputs before engine sees them
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ── Request models ─────────────────────────────────────────────────────────────

class ExtractQueryParams(BaseModel):
    """Query parameters accepted by POST /extract and POST /extract/batch."""

    lang: str = Field(
        default="en",
        description="BCP-47 language code for heading pattern matching.",
        examples=["en", "es", "fr", "de", "ja"],
    )
    min_confidence: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score threshold (0.0–1.0).",
    )

    @field_validator("lang")
    @classmethod
    def validate_lang(cls, v: str) -> str:
        supported = {"en", "es", "fr", "de", "ja"}
        if v not in supported:
            raise ValueError(
                f"Unsupported language '{v}'. Supported: {sorted(supported)}"
            )
        return v


# ── Response models ────────────────────────────────────────────────────────────

class BoundingBoxResponse(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class HeadingResponse(BaseModel):
    level: str = Field(description="Heading level: H1, H2, or H3.")
    text: str = Field(description="Cleaned heading text.")
    page: int = Field(description="1-indexed page number.")
    confidence: float = Field(description="Detection confidence score (0–1).")
    font_name: str
    font_size: float
    bounding_box: BoundingBoxResponse
    confidence_label: str = Field(
        description="Human-readable confidence tier: high | medium | low"
    )


class ExtractionMetadataResponse(BaseModel):
    filename: str
    total_pages: int
    language: str
    toc_available: bool
    scanned_pdf_detected: bool
    processing_time_ms: float
    engine_version: str


class ExtractResponse(BaseModel):
    """Response body for POST /extract."""

    title: str
    outline: list[HeadingResponse]
    metadata: ExtractionMetadataResponse


class BatchExtractResponse(BaseModel):
    """Response body for POST /extract/batch."""

    results: list[ExtractResponse]
    total_files: int
    successful: int
    failed: int
    total_processing_time_ms: float


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = "ok"
    version: str
    engine: str = "PyMuPDF"


class LanguagesResponse(BaseModel):
    """Response body for GET /languages."""

    supported: list[str]
    default: str = "en"


class ErrorResponse(BaseModel):
    """Standard error envelope returned for 4xx / 5xx responses."""

    error: str
    detail: Optional[str] = None
    status_code: int
