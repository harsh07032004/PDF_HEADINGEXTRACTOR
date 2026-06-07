"""
extractor/models.py

Strict dataclass schema for the extraction pipeline output.

These models are the single source of truth for the shape of data flowing
through the entire system — from PyMuPDF spans → Feature Engineering →
ML Classifier → FastAPI serialization → JSON response.

Using `dataclass` here (rather than Pydantic BaseModel) keeps the core
engine dependency-free. The FastAPI layer wraps these in Pydantic models
for HTTP serialization and OpenAPI schema generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BoundingBox:
    """
    Axis-aligned bounding rectangle in PDF user-space coordinates (points).

    Origin is the top-left corner of the page.
    x0, y0 = top-left corner of the span.
    x1, y1 = bottom-right corner of the span.
    """

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    def to_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class HeadingFeatures:
    """
    Raw typographic and positional features extracted per text block.

    These feed directly into the ML classifier. Storing them on the model
    means we can re-run classification without re-parsing the PDF.

    Features are organised into four categories:

    Typography (4):
        rel_font_size, font_size_zscore, bold_percentage, is_italic

    Structure (4):
        starts_with_number, x_indent_ratio, y_position_ratio, centered_text

    Context (4):
        in_toc, is_first_page, vertical_gap_before, font_change_from_prev

    Content (5):
        line_char_count, word_count, all_caps_ratio, title_case_ratio,
        punctuation_density
    """

    # ── Typography ─────────────────────────────────────────────────────────
    rel_font_size: float          # span font_size / median body font_size
    font_size_zscore: float       # (font_size - mean) / std across all blocks
    bold_percentage: float        # ratio of bold chars to total chars (0.0–1.0)
    is_italic: bool               # detected from font flags or font name

    # ── Structure ──────────────────────────────────────────────────────────
    starts_with_number: bool      # True if line starts with "1.", "1.2", etc.
    x_indent_ratio: float         # x0 / page_width  — normalised indentation
    y_position_ratio: float       # y0 / page_height — normalised vertical pos
    centered_text: float          # 0.0–1.0 how centered the text is on page

    # ── Context ────────────────────────────────────────────────────────────
    in_toc: bool                  # matched against the PDF's internal TOC
    is_first_page: bool           # True if the block is on page 1
    vertical_gap_before: float    # normalised gap from previous block
    font_change_from_prev: bool   # True if font differs from previous block

    # ── Content ────────────────────────────────────────────────────────────
    line_char_count: int          # number of characters in this text line
    word_count: int               # number of whitespace-separated tokens
    all_caps_ratio: float         # uppercase_chars / total_alpha_chars  (0–1)
    title_case_ratio: float       # fraction of words that are Title Case
    punctuation_density: float    # punctuation_chars / total_chars


@dataclass
class Heading:
    """
    A single detected heading in the document.

    ``confidence`` encodes the strength of the detection signal on [0, 1].
    Values ≥ 0.8 are high-confidence (font + TOC agreement).
    Values in [0.5, 0.8) are medium (typography only or TOC only).
    """

    text: str
    level: str                    # "H1" | "H2" | "H3"
    page: int                     # 1-indexed page number
    confidence: float             # hybrid score in [0.0, 1.0]
    font_name: str
    font_size: float
    bounding_box: BoundingBox
    features: Optional[HeadingFeatures] = None  # populated during extraction

    def to_dict(self) -> dict:
        """Serialise to plain dict for JSON output (CLI / legacy compat)."""
        return {
            "level": self.level,
            "text": self.text,
            "page": self.page,
            "confidence": self.confidence,
            "font_name": self.font_name,
            "font_size": self.font_size,
            "bounding_box": self.bounding_box.to_tuple(),
        }


@dataclass
class ExtractionMetadata:
    """
    Provenance metadata attached to every DocumentOutline.

    Useful for debugging, performance monitoring, and the evaluation harness.
    """

    engine_version: str = "0.1.0"
    language: str = "en"
    toc_available: bool = False      # True if the PDF embedded a TOC
    total_spans_processed: int = 0
    scanned_pdf_detected: bool = False  # True if text content is very sparse
    processing_time_ms: float = 0.0


@dataclass
class DocumentOutline:
    """
    Top-level output of the extraction pipeline for a single PDF.

    This object is serialised to JSON by the CLI and returned directly
    from the FastAPI ``/extract`` endpoint (via a Pydantic wrapper).
    """

    filename: str
    title: str
    total_pages: int
    headings: list[Heading] = field(default_factory=list)
    metadata: ExtractionMetadata = field(default_factory=ExtractionMetadata)

    def to_dict(self) -> dict:
        """Produce the canonical JSON output format."""
        return {
            "title": self.title,
            "outline": [h.to_dict() for h in self.headings],
            "metadata": {
                "filename": self.filename,
                "total_pages": self.total_pages,
                "language": self.metadata.language,
                "toc_available": self.metadata.toc_available,
                "scanned_pdf_detected": self.metadata.scanned_pdf_detected,
                "processing_time_ms": self.metadata.processing_time_ms,
            },
        }
