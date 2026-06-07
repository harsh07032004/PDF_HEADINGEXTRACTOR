"""
tests/test_extractor.py

Unit and integration tests for the extractor package.

Test categories (marked with pytest.mark):
    unit        — pure function tests, no file I/O
    integration — tests that open real PDFs from sample_datasets/
    slow        — integration tests that process all 5 sample PDFs
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from extractor.models import BoundingBox, DocumentOutline, Heading, HeadingFeatures
from extractor.utils import (
    clean_text,
    compute_prf,
    describe_confidence,
    file_sha256,
    headings_match,
    is_valid_pdf,
    normalise_for_comparison,
)


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — extractor/utils.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestCleanText:
    @pytest.mark.unit
    def test_removes_duplicate_adjacent_chars(self):
        assert clean_text("IInnttrroo") == "Intro"

    @pytest.mark.unit
    def test_preserves_non_duplicated_text(self):
        assert clean_text("Introduction") == "Introduction"

    @pytest.mark.unit
    def test_collapses_whitespace(self):
        assert clean_text("Hello   World") == "Hello World"

    @pytest.mark.unit
    def test_strips_leading_trailing(self):
        assert clean_text("  Chapter 1  ") == "Chapter 1"

    @pytest.mark.unit
    def test_empty_string(self):
        assert clean_text("") == ""

    @pytest.mark.unit
    def test_preserves_numbers(self):
        # Numbers are not alpha so consecutive duplicates are kept
        assert clean_text("1.1 Introduction") == "1.1 Introduction"


class TestNormaliseForComparison:
    @pytest.mark.unit
    def test_lowercases(self):
        assert normalise_for_comparison("HELLO") == "hello"

    @pytest.mark.unit
    def test_strips_punctuation(self):
        assert normalise_for_comparison("Hello, World!") == "hello world"

    @pytest.mark.unit
    def test_collapses_whitespace(self):
        assert normalise_for_comparison("  A   B  ") == "a b"


class TestDescribeConfidence:
    @pytest.mark.unit
    @pytest.mark.parametrize("score,expected", [
        (0.90, "high"),
        (0.80, "high"),
        (0.79, "medium"),
        (0.55, "medium"),
        (0.54, "low"),
        (0.00, "low"),
    ])
    def test_tiers(self, score, expected):
        assert describe_confidence(score) == expected


class TestHeadingsMatch:
    @pytest.mark.unit
    def test_identical_strings_match(self):
        assert headings_match("Introduction", "Introduction")

    @pytest.mark.unit
    def test_case_insensitive(self):
        assert headings_match("introduction", "INTRODUCTION")

    @pytest.mark.unit
    def test_high_overlap_matches(self):
        assert headings_match("Chapter 1 Introduction", "Chapter 1: Introduction")

    @pytest.mark.unit
    def test_low_overlap_no_match(self):
        assert not headings_match("Conclusion", "References and Bibliography")

    @pytest.mark.unit
    def test_empty_strings(self):
        assert headings_match("", "")

    @pytest.mark.unit
    def test_threshold_respected(self):
        # Exact same → should match at any threshold
        assert headings_match("Test", "Test", threshold=1.0)


class TestComputePRF:
    @pytest.mark.unit
    def test_perfect_match(self):
        preds = ["A", "B", "C"]
        gts   = ["A", "B", "C"]
        result = compute_prf(preds, gts)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    @pytest.mark.unit
    def test_zero_predictions(self):
        result = compute_prf([], ["A", "B"])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    @pytest.mark.unit
    def test_no_ground_truth(self):
        result = compute_prf(["A"], [])
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    @pytest.mark.unit
    def test_partial_match(self):
        preds = ["Introduction", "Methodology", "Conclusion"]
        gts   = ["Introduction", "Background", "Methodology", "Results"]
        result = compute_prf(preds, gts)
        # 2 TPs out of 3 preds → P = 0.667
        # 2 TPs out of 4 GTs  → R = 0.500
        assert result["precision"] == pytest.approx(2/3, abs=0.01)
        assert result["recall"]    == pytest.approx(0.50, abs=0.01)


class TestIsValidPdf:
    @pytest.mark.unit
    def test_valid_pdf(self, minimal_pdf_path: Path):
        assert is_valid_pdf(minimal_pdf_path) is True

    @pytest.mark.unit
    def test_non_pdf_file(self, tmp_path: Path):
        bad = tmp_path / "notapdf.pdf"
        bad.write_bytes(b"This is not a PDF file at all.")
        assert is_valid_pdf(bad) is False

    @pytest.mark.unit
    def test_missing_file(self, tmp_path: Path):
        assert is_valid_pdf(tmp_path / "ghost.pdf") is False


class TestFileSha256:
    @pytest.mark.unit
    def test_deterministic(self, minimal_pdf_path: Path):
        h1 = file_sha256(minimal_pdf_path)
        h2 = file_sha256(minimal_pdf_path)
        assert h1 == h2

    @pytest.mark.unit
    def test_different_files_different_hashes(self, tmp_path: Path):
        f1 = tmp_path / "a.pdf"
        f2 = tmp_path / "b.pdf"
        f1.write_bytes(b"AAA")
        f2.write_bytes(b"BBB")
        assert file_sha256(f1) != file_sha256(f2)

    @pytest.mark.unit
    def test_hash_length(self, minimal_pdf_path: Path):
        assert len(file_sha256(minimal_pdf_path)) == 64  # hex SHA-256


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS — extractor/models.py
# ═══════════════════════════════════════════════════════════════════════════════

class TestBoundingBox:
    @pytest.mark.unit
    def test_width_and_height(self):
        bb = BoundingBox(10.0, 20.0, 110.0, 50.0)
        assert bb.width == pytest.approx(100.0)
        assert bb.height == pytest.approx(30.0)

    @pytest.mark.unit
    def test_to_tuple(self):
        bb = BoundingBox(1.0, 2.0, 3.0, 4.0)
        assert bb.to_tuple() == (1.0, 2.0, 3.0, 4.0)


class TestDocumentOutline:
    @pytest.mark.unit
    def test_to_dict_structure(self):
        heading = Heading(
            text="Introduction",
            level="H1",
            page=1,
            confidence=0.90,
            font_name="Helvetica-Bold",
            font_size=18.0,
            bounding_box=BoundingBox(50.0, 100.0, 300.0, 130.0),
        )
        outline = DocumentOutline(
            filename="test.pdf",
            title="Test Document",
            total_pages=5,
            headings=[heading],
        )
        d = outline.to_dict()
        assert d["title"] == "Test Document"
        assert len(d["outline"]) == 1
        assert d["outline"][0]["level"] == "H1"
        assert d["outline"][0]["text"] == "Introduction"
        assert d["metadata"]["total_pages"] == 5


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION TESTS — ExtractorEngine on real PDFs
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractorEngine:
    @pytest.mark.integration
    def test_returns_document_outline(self, first_sample_pdf: Path):
        from extractor.core import ExtractorEngine
        with ExtractorEngine(first_sample_pdf, lang="en") as engine:
            result = engine.process()
        assert isinstance(result, DocumentOutline)

    @pytest.mark.integration
    def test_filename_matches(self, first_sample_pdf: Path):
        from extractor.core import ExtractorEngine
        with ExtractorEngine(first_sample_pdf, lang="en") as engine:
            result = engine.process()
        assert result.filename == first_sample_pdf.name

    @pytest.mark.integration
    def test_total_pages_positive(self, first_sample_pdf: Path):
        from extractor.core import ExtractorEngine
        with ExtractorEngine(first_sample_pdf, lang="en") as engine:
            result = engine.process()
        assert result.total_pages >= 1

    @pytest.mark.integration
    def test_processing_time_recorded(self, first_sample_pdf: Path):
        from extractor.core import ExtractorEngine
        with ExtractorEngine(first_sample_pdf, lang="en") as engine:
            result = engine.process()
        assert result.metadata.processing_time_ms > 0

    @pytest.mark.integration
    def test_headings_have_valid_levels(self, first_sample_pdf: Path):
        from extractor.core import ExtractorEngine
        with ExtractorEngine(first_sample_pdf, lang="en") as engine:
            result = engine.process()
        valid_levels = {"H1", "H2", "H3"}
        for h in result.headings:
            assert h.level in valid_levels, f"Invalid level: {h.level}"

    @pytest.mark.integration
    def test_confidence_in_range(self, first_sample_pdf: Path):
        from extractor.core import ExtractorEngine
        with ExtractorEngine(first_sample_pdf, lang="en") as engine:
            result = engine.process()
        for h in result.headings:
            assert 0.0 <= h.confidence <= 1.0, (
                f"Out-of-range confidence {h.confidence} for '{h.text}'"
            )

    @pytest.mark.integration
    def test_page_numbers_positive(self, first_sample_pdf: Path):
        from extractor.core import ExtractorEngine
        with ExtractorEngine(first_sample_pdf, lang="en") as engine:
            result = engine.process()
        for h in result.headings:
            assert h.page >= 1

    @pytest.mark.integration
    def test_heading_text_not_empty(self, first_sample_pdf: Path):
        from extractor.core import ExtractorEngine
        with ExtractorEngine(first_sample_pdf, lang="en") as engine:
            result = engine.process()
        for h in result.headings:
            assert h.text.strip(), f"Empty heading text on page {h.page}"

    @pytest.mark.integration
    def test_context_manager_closes_doc(self, first_sample_pdf: Path):
        """Engine should not raise when used as a context manager."""
        from extractor.core import ExtractorEngine
        with ExtractorEngine(first_sample_pdf) as engine:
            result = engine.process()
        # After __exit__, doc is closed — accessing it should be handled
        assert result is not None

    @pytest.mark.integration
    def test_to_dict_is_json_serialisable(self, first_sample_pdf: Path):
        from extractor.core import ExtractorEngine
        with ExtractorEngine(first_sample_pdf, lang="en") as engine:
            result = engine.process()
        serialised = json.dumps(result.to_dict())
        parsed = json.loads(serialised)
        assert "title" in parsed
        assert "outline" in parsed
        assert "metadata" in parsed

    @pytest.mark.slow
    def test_all_sample_pdfs_process_without_error(self, sample_pdf_paths: list[Path]):
        from extractor.core import ExtractorEngine
        for pdf in sample_pdf_paths:
            with ExtractorEngine(pdf, lang="en") as engine:
                result = engine.process()
            assert isinstance(result, DocumentOutline), f"Failed on {pdf.name}"

    @pytest.mark.slow
    def test_against_ground_truth_has_headings(
        self,
        sample_pdf_paths: list[Path],
        sample_ground_truth: dict[str, dict],
    ):
        """
        For PDFs with non-empty ground truth, the extractor should
        detect at least one heading.
        """
        from extractor.core import ExtractorEngine
        for pdf in sample_pdf_paths:
            gt = sample_ground_truth.get(pdf.stem, {})
            if not gt.get("outline"):
                continue  # ground truth has no headings — skip
            with ExtractorEngine(pdf, lang="en") as engine:
                result = engine.process()
            assert len(result.headings) > 0, (
                f"{pdf.name} has GT headings but extractor found none"
            )
