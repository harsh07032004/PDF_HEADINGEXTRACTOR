"""
extractor/core.py

Production-grade PDF heading extraction engine.

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │  ExtractorEngine.process()                           │
    │                                                      │
    │  1. _extract_toc()        — internal TOC baseline    │
    │  2. _extract_spans()      — character-level parsing  │
    │  3. _build_blocks()       — group spans into lines   │
    │  4. _detect_title()       — largest-font first block │
    │  5. _cluster_font_sizes() — body vs heading sizes    │
    │  6. _build_features()     — 8 ML-ready features      │
    │  7. _compute_confidence() — hybrid TOC + typography  │
    │  8. _classify_level()     — H1 / H2 / H3 assignment  │
    │  9. _deduplicate()        — remove near-duplicates   │
    └──────────────────────────────────────────────────────┘

This module intentionally has NO FastAPI dependency so it can be used
as a standalone library, as a CLI tool, and as a backend service equally.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import fitz  # PyMuPDF

from extractor.models import (
    BoundingBox,
    DocumentOutline,
    ExtractionMetadata,
    Heading,
    HeadingFeatures,
)

# ── Constants ──────────────────────────────────────────────────────────────────

# Minimum ratio of alpha characters for a span to be considered text
_MIN_ALPHA_RATIO: float = 0.30

# Minimum confidence score to promote a block to a heading candidate
_CONFIDENCE_THRESHOLD: float = 0.20

# Number (top) heading levels to retain from font-size clustering
_MAX_HEADING_LEVELS: int = 4

# Path to the language patterns file, resolved relative to this module
_LANGUAGES_FILE: Path = Path(__file__).parent.parent / "languages.json"

# Regex for detecting numbered section labels like "1.", "1.2", "1.2.3"
_NUMBER_PREFIX_RE = re.compile(r"^\d+(\.\d+)*[\.\s]")

# Scanned PDF heuristic: if fewer than this many characters per page → warn
_SCANNED_CHARS_PER_PAGE_THRESHOLD: int = 50


# ── Internal block type (intermediate representation) ─────────────────────────

class _TextBlock:
    """
    Intermediate representation of a text line assembled from PDF spans.

    Not part of the public API — kept internal to avoid polluting models.py
    with processing-stage artifacts.
    """

    __slots__ = (
        "text", "font_size", "font_name", "bold_percentage", "is_light_gray",
        "page", "x0", "y0", "x1", "y1", "page_width", "page_height",
    )

    def __init__(
        self,
        text: str,
        font_size: float,
        font_name: str,
        bold_percentage: float,
        is_light_gray: bool,
        page: int,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        page_width: float,
        page_height: float,
    ) -> None:
        self.text = text
        self.font_size = font_size
        self.font_name = font_name
        self.bold_percentage = bold_percentage
        self.is_light_gray = is_light_gray
        self.page = page
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.page_width = page_width
        self.page_height = page_height


# ── Main Engine ────────────────────────────────────────────────────────────────

class ExtractorEngine:
    """
    Stateful extraction engine for a single PDF document.

    Usage::

        engine = ExtractorEngine("path/to/doc.pdf", lang="en")
        outline = engine.process()
        print(outline.to_dict())

        # With ML classifier:
        engine = ExtractorEngine("path/to/doc.pdf", use_ml=True)
        outline = engine.process()

    The engine is not thread-safe. Create one instance per file in concurrent
    settings (e.g., FastAPI background tasks or asyncio.to_thread).
    """

    def __init__(
        self,
        file_path: str | Path,
        lang: str = "en",
        use_ml: bool = False,
    ) -> None:
        self.file_path = Path(file_path)
        self.lang = lang
        self.use_ml = use_ml
        self._doc: fitz.Document = fitz.open(str(self.file_path))
        self._lang_patterns: list[re.Pattern[str]] = self._load_lang_patterns(lang)

        # Load ML classifier if requested
        self._ml_classifier = None
        if use_ml:
            try:
                from extractor.ml_classifier import get_classifier
                self._ml_classifier = get_classifier()
            except ImportError:
                pass  # scikit-learn not installed

    # ── Public interface ───────────────────────────────────────────────────────

    def process(self) -> DocumentOutline:
        """
        Run the full extraction pipeline and return a ``DocumentOutline``.

        Steps:
            1. Extract the PDF's internal TOC for cross-validation.
            2. Parse character-level spans from every page.
            3. Group spans into logical text lines (blocks).
            4. Detect the document title (largest font on first page).
            5. Cluster font sizes to identify body vs heading tiers.
            6. Build HeadingFeatures for every candidate block.
            7. Compute a hybrid confidence score.
            8. Classify H1 / H2 / H3 using font clusters + rules.
            9. Deduplicate near-identical headings.
        """
        start_ts = time.perf_counter()

        toc_titles = self._extract_toc()
        toc_available = len(toc_titles) > 0

        all_spans = self._extract_spans()
        total_spans = len(all_spans)

        blocks = self._build_blocks(all_spans)

        # Multi-dimensional pre-filter: strip watermarks, boilerplate, light-gray artifacts
        blocks = self._filter_boilerplate(blocks)

        # Scanned PDF guard
        total_chars = sum(len(b.text) for b in blocks)
        scanned = (
            self._doc.page_count > 0
            and (total_chars / self._doc.page_count) < _SCANNED_CHARS_PER_PAGE_THRESHOLD
        )

        title, body_blocks = self._detect_title(blocks)

        if not body_blocks:
            elapsed = (time.perf_counter() - start_ts) * 1000
            meta = ExtractionMetadata(
                language=self.lang,
                toc_available=toc_available,
                total_spans_processed=total_spans,
                scanned_pdf_detected=scanned,
                processing_time_ms=round(elapsed, 2),
            )
            return DocumentOutline(
                filename=self.file_path.name,
                title=title,
                total_pages=self._doc.page_count,
                headings=[],
                metadata=meta,
            )

        heading_font_sizes, body_size = self._cluster_font_sizes(body_blocks)

        # Choose ML or heuristic path
        if self.use_ml and self._ml_classifier is not None and self._ml_classifier.is_fitted:
            headings = self._extract_headings_ml(
                body_blocks, heading_font_sizes, body_size, toc_titles
            )
        else:
            headings = self._extract_headings(
                body_blocks, heading_font_sizes, body_size, toc_titles
            )
        headings = self._deduplicate_headings(headings)

        elapsed = (time.perf_counter() - start_ts) * 1000
        meta = ExtractionMetadata(
            language=self.lang,
            toc_available=toc_available,
            total_spans_processed=total_spans,
            scanned_pdf_detected=scanned,
            processing_time_ms=round(elapsed, 2),
        )

        return DocumentOutline(
            filename=self.file_path.name,
            title=title,
            total_pages=self._doc.page_count,
            headings=headings,
            metadata=meta,
        )

    def close(self) -> None:
        """Release the underlying PyMuPDF document handle."""
        self._doc.close()

    def __enter__(self) -> "ExtractorEngine":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── Step 1: TOC extraction ─────────────────────────────────────────────────

    def _extract_toc(self) -> set[str]:
        """
        Pull the PDF's embedded Table of Contents.

        Returns a set of normalised title strings for O(1) membership
        testing during confidence scoring.

        PyMuPDF returns TOC entries as:
            [level, title, page, *dest]
        """
        toc = self._doc.get_toc(simple=True)
        return {entry[1].strip().lower() for entry in toc if len(entry) >= 2}

    # ── Step 2: Span extraction ────────────────────────────────────────────────

    def _extract_spans(self) -> list[dict[str, Any]]:
        """
        Extract character-level span metadata from every page.

        Each span is a dict with keys: text, font_size, font_name, bold,
        page, bbox, page_width, page_height.

        We iterate at the span level (not block/line) because spans give
        us the finest-grained font metadata, which is essential for
        accurately computing relative font sizes.
        """
        spans: list[dict[str, Any]] = []

        for page_idx in range(self._doc.page_count):
            page = self._doc.load_page(page_idx)
            page_rect = page.rect
            page_width = page_rect.width
            page_height = page_rect.height

            text_dict: dict[str, Any] = page.get_text("dict")
            for block in text_dict.get("blocks", []):
                if block.get("type") != 0:  # 0 = text block
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        raw_text: str = span.get("text", "").strip()
                        if not raw_text:
                            continue

                        alpha_count = sum(c.isalpha() for c in raw_text)
                        if alpha_count / max(len(raw_text), 1) < _MIN_ALPHA_RATIO:
                            continue

                        font_flags: int = span.get("flags", 0)
                        font_name: str = span.get("font", "")

                        # Bold detection: PyMuPDF flag bit 4 (value 16)
                        # and font name heuristic for robustness
                        is_bold_span: bool = bool(font_flags & 16) or (
                            "bold" in font_name.lower()
                        )
                        char_count = len(raw_text.strip())
                        bold_char_count = char_count if is_bold_span else 0

                        # Color extraction: PyMuPDF colors are packed sRGB ints
                        color_int: int = span.get("color", 0)
                        r = (color_int >> 16) & 255
                        g = (color_int >> 8) & 255
                        b = color_int & 255
                        # Light gray / watermark threshold: all channels > 200
                        color_is_light: bool = (r > 200 and g > 200 and b > 200)

                        bbox: tuple[float, ...] = span.get("bbox", (0, 0, 0, 0))
                        spans.append({
                            "text": raw_text,
                            "font_size": round(float(span.get("size", 0)), 2),
                            "font_name": font_name,
                            "char_count": char_count,
                            "bold_char_count": bold_char_count,
                            "color_is_light": color_is_light,
                            "page": page_idx + 1,  # 1-indexed
                            "bbox": bbox,
                            "page_width": page_width,
                            "page_height": page_height,
                        })

        return spans

    # ── Step 3: Block assembly ─────────────────────────────────────────────────

    def _build_blocks(self, spans: list[dict[str, Any]]) -> list[_TextBlock]:
        """
        Group spans that share the same page and vertical Y-position
        into logical text lines (_TextBlock instances).

        Two spans are on the same line when their rounded y0 values are
        within 1 point of each other — matching PyMuPDF's internal line
        grouping without duplicating its work.
        """
        # Bucket by (page, rounded_y0)
        buckets: dict[tuple[int, float], list[dict[str, Any]]] = {}
        for span in spans:
            y0 = round(float(span["bbox"][1]), 1)
            key = (span["page"], y0)
            buckets.setdefault(key, []).append(span)

        blocks: list[_TextBlock] = []
        for (page, y0), line_spans in buckets.items():
            line_spans.sort(key=lambda s: s["bbox"][0])  # left → right

            text = " ".join(s["text"] for s in line_spans)
            # Take the dominant (maximum) font size in the line
            font_size = max(s["font_size"] for s in line_spans)
            dominant_span = max(line_spans, key=lambda s: s["font_size"])
            font_name = dominant_span["font_name"]

            # Aggregate bold percentage: bold_chars / total_chars across spans
            total_chars = sum(s["char_count"] for s in line_spans)
            total_bold_chars = sum(s["bold_char_count"] for s in line_spans)
            bold_percentage = (
                round(total_bold_chars / total_chars, 4) if total_chars > 0 else 0.0
            )

            # Light gray flag: if ANY span on the line is light-colored, the
            # whole line is considered a watermark candidate
            is_light_gray = any(s["color_is_light"] for s in line_spans)

            x0 = min(float(s["bbox"][0]) for s in line_spans)
            y0_exact = min(float(s["bbox"][1]) for s in line_spans)
            x1 = max(float(s["bbox"][2]) for s in line_spans)
            y1 = max(float(s["bbox"][3]) for s in line_spans)
            page_width = line_spans[0]["page_width"]
            page_height = line_spans[0]["page_height"]

            blocks.append(_TextBlock(
                text=text,
                font_size=font_size,
                font_name=font_name,
                bold_percentage=bold_percentage,
                is_light_gray=is_light_gray,
                page=page,
                x0=x0,
                y0=y0_exact,
                x1=x1,
                y1=y1,
                page_width=page_width,
                page_height=page_height,
            ))

        blocks.sort(key=lambda b: (b.page, b.y0, b.x0))
        return blocks

    # ── Step 3b: Boilerplate / Watermark Pre-Filters ──────────────────────────

    def _filter_boilerplate(self, blocks: list[_TextBlock]) -> list[_TextBlock]:
        """
        Multi-dimensional pre-filter that removes persistent watermarks,
        headers, footers, and other non-heading artifacts BEFORE they reach
        the ML model or heuristic scorer.

        Filters applied in order:
            1. Color: drop light-gray text (RGB channels all > 200).
            2. Frequency: drop text that appears on > 80% of pages at the
               same rounded font size (classic watermark / page-number pattern).
            3. Spatial + density: drop blocks whose y0 ratio puts them in the
               center band of the page (25%–75%) AND whose character-to-area
               density is very low (large font, few chars) — the NPTEL
               watermark geometry.
        """
        if not blocks:
            return blocks

        total_pages = self._doc.page_count
        # ---------- Filter 1: Light gray (opacity/watermark) ----------
        blocks = [b for b in blocks if not b.is_light_gray]

        # ---------- Filter 2: Document-wide frequency ----------
        # Hash (normalised_text, rounded_font_size) → set of pages it appears on
        freq: dict[tuple[str, int], set[int]] = {}
        for b in blocks:
            key = (b.text.strip().lower(), round(b.font_size))
            freq.setdefault(key, set()).add(b.page)

        boilerplate_keys: set[tuple[str, int]] = {
            k for k, pages in freq.items()
            if total_pages > 0 and len(pages) / total_pages > 0.80
        }
        blocks = [b for b in blocks if
                  (b.text.strip().lower(), round(b.font_size)) not in boilerplate_keys]

        # ---------- Filter 3: Spatial density (watermark geometry) ----------
        def _char_area_density(b: _TextBlock) -> float:
            """Characters per square point of bounding box area."""
            area = max((b.x1 - b.x0) * (b.y1 - b.y0), 1.0)
            return len(b.text.strip()) / area

        filtered: list[_TextBlock] = []
        for b in blocks:
            y_ratio = b.y0 / max(b.page_height, 1.0)
            in_center_band = 0.25 < y_ratio < 0.75
            low_density = _char_area_density(b) < 0.05   # < 1 char per 20 sq pts
            # Only suppress if ALSO unusually large font (likely watermark, not body)
            large_font = b.font_size > 20.0
            if in_center_band and low_density and large_font:
                continue  # watermark geometry: skip
            filtered.append(b)

        return filtered

    # ── Step 4: Title detection ────────────────────────────────────────────────

    def _detect_title(
        self, blocks: list[_TextBlock]
    ) -> tuple[str, list[_TextBlock]]:
        """
        Identify the document title as the block(s) with the largest
        font size on the first page, then return the remaining blocks.

        Strategy:
            - Find the global maximum font size.
            - Locate the earliest block at that size (anchor).
            - Collect all blocks on the same page near the anchor with
              font sizes within [1×, 2×] of max (title can span lines).
            - Everything else becomes the body for heading classification.
        """
        if not blocks:
            return "", []

        max_font = max(b.font_size for b in blocks)
        anchor_candidates = [b for b in blocks if b.font_size == max_font]
        anchor_candidates.sort(key=lambda b: (b.page, b.y0))
        anchor = anchor_candidates[0]
        title_page = anchor.page

        title_blocks = [
            b for b in blocks
            if b.page == title_page
            and b.y0 >= anchor.y0
            and max_font <= b.font_size <= 2.0 * max_font
        ]
        title_blocks.sort(key=lambda b: b.y0)
        title_text = self._clean_text(" ".join(b.text for b in title_blocks))

        # Body = everything after the anchor block
        body_blocks = [
            b for b in blocks
            if b.page > title_page
            or (b.page == title_page and b.y0 > anchor.y0)
        ]
        return title_text, body_blocks

    # ── Step 5: Font size clustering ───────────────────────────────────────────

    def _cluster_font_sizes(
        self, blocks: list[_TextBlock]
    ) -> tuple[list[float], float]:
        """
        Determine the body font size and heading font size tiers.

        The body size is the most frequently occurring rounded font size
        (the modal size). Heading sizes are any sizes strictly above the
        body, returned in descending order (largest = H1).

        Returns:
            heading_sizes: up to ``_MAX_HEADING_LEVELS`` sizes, desc order
            body_size:     the modal (body) font size
        """
        size_counts: Counter[int] = Counter(
            round(b.font_size) for b in blocks
        )
        if not size_counts:
            return [], 12.0

        body_size = float(size_counts.most_common(1)[0][0])
        heading_sizes = sorted(
            {float(s) for s in size_counts if float(s) > body_size},
            reverse=True,
        )[:_MAX_HEADING_LEVELS]
        return heading_sizes, body_size

    # ── Step 6–8 (ML path): ML-based heading extraction ────────────────────

    def _extract_headings_ml(
        self,
        blocks: list[_TextBlock],
        heading_sizes: list[float],
        body_size: float,
        toc_titles: set[str],
    ) -> list[Heading]:
        """
        ML-based heading extraction.

        Uses the trained RandomForest classifier to predict heading level
        and confidence for every text block, instead of the heuristic rules.

        The ML classifier provides:
            - Better generalisation across different PDF styles
            - Learned feature interactions (e.g., bold + indentation together)
            - Calibrated probability scores as confidence values
        """
        headings: list[Heading] = []
        clf = self._ml_classifier

        if clf is None or not clf.is_fitted:
            return self._extract_headings(
                blocks, heading_sizes, body_size, toc_titles
            )

        # Precompute font statistics for z-score feature
        import numpy as np
        sizes = [b.font_size for b in blocks]
        font_stats = (float(np.mean(sizes)), float(np.std(sizes))) if sizes else (0.0, 1.0)

        # Build features for all blocks with contextual info
        all_features: list[HeadingFeatures] = []
        prev_block: Optional[_TextBlock] = None
        for block in blocks:
            features = self._build_features(
                block, body_size, toc_titles,
                prev_block=prev_block, font_stats=font_stats,
            )
            all_features.append(features)
            prev_block = block

        # Batch predict
        predictions = clf.predict_batch(all_features)

        for block, features, (level, ml_confidence) in zip(
            blocks, all_features, predictions
        ):
            if level is None:  # predicted as body text
                continue

            # Combine ML confidence with heuristic confidence (ensemble)
            heuristic_conf = self._compute_confidence(
                features, block, heading_sizes
            )
            # Weighted average: 70% ML, 30% heuristic
            combined_confidence = 0.70 * ml_confidence + 0.30 * heuristic_conf

            if combined_confidence < 0.25:
                continue

            headings.append(Heading(
                text=self._clean_text(block.text),
                level=level,
                page=block.page,
                confidence=round(combined_confidence, 2),
                font_name=block.font_name,
                font_size=round(block.font_size, 2),
                bounding_box=BoundingBox(
                    x0=round(block.x0, 2),
                    y0=round(block.y0, 2),
                    x1=round(block.x1, 2),
                    y1=round(block.y1, 2),
                ),
                features=features,
            ))

        return headings

    # ── Step 6–8 (Heuristic path): Feature engineering, confidence & classification

    def _extract_headings(
        self,
        blocks: list[_TextBlock],
        heading_sizes: list[float],
        body_size: float,
        toc_titles: set[str],
    ) -> list[Heading]:
        """
        Run steps 6, 7, and 8 in sequence for every block:
            6. Build HeadingFeatures
            7. Compute hybrid confidence score
            8. Classify heading level
        """
        headings: list[Heading] = []
        seen_h1 = False

        # Precompute font statistics for z-score feature
        import numpy as np
        sizes = [b.font_size for b in blocks]
        font_stats = (float(np.mean(sizes)), float(np.std(sizes))) if sizes else (0.0, 1.0)

        prev_block: Optional[_TextBlock] = None
        for block in blocks:
            features = self._build_features(
                block, body_size, toc_titles,
                prev_block=prev_block, font_stats=font_stats,
            )
            confidence = self._compute_confidence(features, block, heading_sizes)

            if confidence < _CONFIDENCE_THRESHOLD:
                continue

            level = self._classify_level(
                block, heading_sizes, body_size, features, seen_h1
            )
            if level is None:
                continue

            if level == "H1":
                seen_h1 = True
            # NOTE: H2/H3 are allowed even without a preceding H1
            # Many documents use H2 as the top-level heading style

            headings.append(Heading(
                text=self._clean_text(block.text),
                level=level,
                page=block.page,
                confidence=round(confidence, 2),
                font_name=block.font_name,
                font_size=round(block.font_size, 2),
                bounding_box=BoundingBox(
                    x0=round(block.x0, 2),
                    y0=round(block.y0, 2),
                    x1=round(block.x1, 2),
                    y1=round(block.y1, 2),
                ),
                features=features,
            ))
            prev_block = block

        return headings

    def _build_features(
        self,
        block: _TextBlock,
        body_size: float,
        toc_titles: set[str],
        *,
        prev_block: Optional[_TextBlock] = None,
        font_stats: Optional[tuple[float, float]] = None,
    ) -> HeadingFeatures:
        """
        Compute 18 typographic, structural, contextual, and content
        features for a single text block.

        Args:
            block: the text block to featurise
            body_size: the modal (body) font size for this document
            toc_titles: set of normalised TOC entry strings
            prev_block: the preceding text block (for gap / font-change)
            font_stats: (mean_font_size, std_font_size) across all blocks
        """
        text = block.text

        # ── Typography ─────────────────────────────────────────────────────
        rel_font_size = block.font_size / max(body_size, 1.0)

        # Font z-score: how many std-devs above the mean font size
        if font_stats and font_stats[1] > 0:
            font_size_zscore = (block.font_size - font_stats[0]) / font_stats[1]
        else:
            font_size_zscore = 0.0

        bold_percentage = block.bold_percentage

        # Italic detection: PyMuPDF flag bit 1 (value 2) or name heuristic
        is_italic = "italic" in block.font_name.lower() or "oblique" in block.font_name.lower()

        # ── Structure ──────────────────────────────────────────────────────
        starts_with_number = bool(_NUMBER_PREFIX_RE.match(text))

        x_indent_ratio = (
            block.x0 / block.page_width if block.page_width > 0 else 0.0
        )
        y_position_ratio = (
            block.y0 / block.page_height if block.page_height > 0 else 0.0
        )

        # Centered text: how symmetric is the text horizontally on the page
        if block.page_width > 0:
            text_width = block.x1 - block.x0
            left_margin = block.x0
            right_margin = block.page_width - block.x1
            margin_diff = abs(left_margin - right_margin)
            # 1.0 = perfectly centered, 0.0 = fully left/right aligned
            centered_text = max(0.0, 1.0 - margin_diff / (block.page_width * 0.5))
        else:
            centered_text = 0.0

        # ── Context ────────────────────────────────────────────────────────
        in_toc = text.strip().lower() in toc_titles
        is_first_page = block.page == 1

        # Vertical gap: normalised distance from previous block
        if prev_block is not None and prev_block.page == block.page:
            raw_gap = block.y0 - prev_block.y1
            vertical_gap_before = max(0.0, raw_gap) / max(block.page_height, 1.0)
        else:
            vertical_gap_before = 0.1  # default for first block / new page

        # Font change: does this block use a different font than the previous?
        if prev_block is not None:
            font_change_from_prev = (
                round(block.font_size) != round(prev_block.font_size)
                or block.font_name != prev_block.font_name
            )
        else:
            font_change_from_prev = False

        # ── Content ────────────────────────────────────────────────────────
        line_char_count = len(text)
        word_count = len(text.split())

        alpha_chars = [c for c in text if c.isalpha()]
        all_caps_ratio = (
            sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if alpha_chars else 0.0
        )

        # Title case ratio: fraction of words that start with uppercase
        words = text.split()
        if words:
            title_words = sum(1 for w in words if w and w[0].isupper())
            title_case_ratio = title_words / len(words)
        else:
            title_case_ratio = 0.0

        # Punctuation density: commas, periods, semicolons relative to length
        if line_char_count > 0:
            punct_count = sum(1 for c in text if c in ".,;:!?()-\"'")
            punctuation_density = punct_count / line_char_count
        else:
            punctuation_density = 0.0

        return HeadingFeatures(
            rel_font_size=round(rel_font_size, 4),
            font_size_zscore=round(font_size_zscore, 4),
            bold_percentage=round(bold_percentage, 4),
            is_italic=is_italic,
            starts_with_number=starts_with_number,
            x_indent_ratio=round(x_indent_ratio, 4),
            y_position_ratio=round(y_position_ratio, 4),
            centered_text=round(centered_text, 4),
            in_toc=in_toc,
            is_first_page=is_first_page,
            vertical_gap_before=round(vertical_gap_before, 4),
            font_change_from_prev=font_change_from_prev,
            line_char_count=line_char_count,
            word_count=word_count,
            all_caps_ratio=round(all_caps_ratio, 4),
            title_case_ratio=round(title_case_ratio, 4),
            punctuation_density=round(punctuation_density, 4),
        )

    def _compute_confidence(
        self,
        features: HeadingFeatures,
        block: _TextBlock,
        heading_sizes: list[float],
    ) -> float:
        """
        Hybrid confidence score combining:
            * TOC cross-validation  (+0.40)
            * Font size above body  (+0.30 scaled by tier)
            * Boldness              (+0.15)
            * Language regex match  (+0.10)
            * Numbered prefix       (+0.05)
            * All-caps signal       (+0.05)

        Penalties:
            * Very long lines (>120 chars):  -0.20
            * Single character lines:        -0.30
        """
        score = 0.0

        # TOC cross-validation is the strongest signal
        if features.in_toc:
            score += 0.40

        # Font size signal
        rounded_size = round(block.font_size)
        heading_rounded = [round(s) for s in heading_sizes]
        if rounded_size in heading_rounded:
            tier_idx = heading_rounded.index(rounded_size)
            score += 0.30 * max(0.0, 1.0 - tier_idx * 0.10)
        elif block.font_size > 0 and features.rel_font_size >= 1.10:
            # Slightly above body but not in a named cluster — partial credit
            score += 0.10

        # Boldness: continuous signal — a fully bold heading scores full 0.20,
        # a partially bold paragraph scores proportionally less.
        # Extra bonus when block is predominantly bold AND short (classic heading)
        score += 0.20 * features.bold_percentage
        if features.bold_percentage >= 0.80 and features.word_count <= 15:
            score += 0.10  # strong heading-specific boldness bonus

        # Language pattern match
        for pattern in self._lang_patterns:
            if pattern.match(block.text):
                score += 0.10
                break

        # Numbered section (e.g., "2.3 Background")
        if features.starts_with_number:
            score += 0.05

        # All-caps heading (e.g., "INTRODUCTION")
        if features.all_caps_ratio > 0.80 and features.word_count <= 6:
            score += 0.05

        # Penalties
        if features.line_char_count > 120:
            score -= 0.20
        if features.line_char_count <= 1:
            score -= 0.30

        return max(0.0, min(score, 1.0))

    def _classify_level(
        self,
        block: _TextBlock,
        heading_sizes: list[float],
        body_size: float,
        features: HeadingFeatures,
        seen_h1: bool,
    ) -> Optional[str]:
        """
        Assign a heading level (H1/H2/H3) or return None (not a heading).

        Primary rule: map font size cluster tier → heading level.
            Tier 0 (largest) → H1
            Tier 1           → H2
            Tier 2           → H3
            Tier 3+          → H3  (collapse deep tiers into H3)

        Fallback rule: if the block is bold + moderately larger than body
        and not already matched, assign H3.

        Language regex match always assigns at least H2.
        """
        rounded_size = round(block.font_size)
        heading_rounded = [round(s) for s in heading_sizes]

        # Primary: font-size cluster membership
        if rounded_size in heading_rounded:
            tier = heading_rounded.index(rounded_size)
            if tier == 0:
                return "H1"
            elif tier == 1:
                return "H2"
            else:
                return "H3"

        # Fallback: predominantly bold + larger than body + reasonable indentation
        if (
            block.bold_percentage >= 0.80
            and block.font_size > body_size
            and features.x_indent_ratio < 0.30
            and features.word_count <= 15
        ):
            return "H3"

        # Fallback: bold + same size as body ("Boldness Trap" catch for enterprise PDFs)
        if (
            block.bold_percentage >= 0.80
            and block.font_size >= body_size
            and features.word_count <= 12
            and features.y_position_ratio < 0.85  # not a footer
        ):
            return "H3"

        # Fallback: language regex match
        for pattern in self._lang_patterns:
            if pattern.match(block.text):
                return "H2"

        return None

    # ── Step 9: Deduplication ──────────────────────────────────────────────────

    def _deduplicate_headings(self, headings: list[Heading]) -> list[Heading]:
        """
        Remove duplicate headings that appear on the same page at nearly
        the same vertical position (within 2.0 points in y0).

        When duplicates exist, keep the one with the higher confidence score.
        This handles layered PDF rendering where the same text span is drawn
        twice (e.g., shadow effects or watermarks).
        """
        result: list[Heading] = []
        seen: list[Heading] = []

        for heading in headings:
            is_dup = False
            for existing in seen:
                same_page = heading.page == existing.page
                same_text = heading.text == existing.text
                close_y = abs(heading.bounding_box.y0 - existing.bounding_box.y0) < 2.0
                if same_page and same_text and close_y:
                    # Keep higher confidence
                    if heading.confidence > existing.confidence:
                        seen.remove(existing)
                        result.remove(existing)
                        seen.append(heading)
                        result.append(heading)
                    is_dup = True
                    break

            if not is_dup:
                seen.append(heading)
                result.append(heading)

        return result

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Normalise extracted text by collapsing whitespace and removing
        artefacts caused by repeated character rendering in some PDFs.

        Example artefact: "IInnttrroodduuccttiioonn" → "Introduction"
        """
        # Collapse consecutive duplicate characters that are alphanumeric
        chars = list(text)
        result: list[str] = []
        prev = ""
        for c in chars:
            if c != prev or not c.isalnum():
                result.append(c)
            prev = c
        cleaned = "".join(result)
        # Normalise whitespace
        return " ".join(cleaned.split())

    @staticmethod
    def _load_lang_patterns(lang: str) -> list[re.Pattern[str]]:
        """
        Load heading regex patterns for the given language from
        ``languages.json``.  Falls back to English on any error.
        """
        try:
            with open(_LANGUAGES_FILE, encoding="utf-8") as fh:
                lang_data: dict[str, Any] = json.load(fh)
            raw_patterns: list[str] = (
                lang_data.get(lang, {}).get("heading_patterns", [])
                or lang_data.get("en", {}).get("heading_patterns", [])
            )
            return [re.compile(p, re.IGNORECASE) for p in raw_patterns]
        except Exception as exc:
            import warnings
            warnings.warn(
                f"Could not load language patterns for '{lang}': {exc}",
                stacklevel=2,
            )
            return []
