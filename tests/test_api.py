"""
tests/test_api.py

FastAPI endpoint integration tests.

Uses FastAPI's TestClient (built on httpx + ASGI transport) so no
real server is needed — tests run in-process.

Coverage:
    GET  /health
    GET  /languages
    POST /extract    (valid PDF, invalid file, size limit, unsupported lang)
    POST /extract/batch
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def pdf_upload(client, pdf_path: Path, lang: str = "en", min_conf: float = 0.45):
    """Helper: POST /extract with a PDF file."""
    with open(pdf_path, "rb") as fh:
        return client.post(
            "/extract",
            files={"file": (pdf_path.name, fh, "application/pdf")},
            params={"lang": lang, "min_confidence": min_conf},
        )


# ═══════════════════════════════════════════════════════════════════════════════
# GET /health
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealth:
    def test_returns_200(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200

    def test_status_ok(self, api_client):
        data = api_client.get("/health").json()
        assert data["status"] == "ok"

    def test_has_version(self, api_client):
        data = api_client.get("/health").json()
        assert "version" in data
        assert data["version"]

    def test_has_engine(self, api_client):
        data = api_client.get("/health").json()
        assert data.get("engine") == "PyMuPDF"


# ═══════════════════════════════════════════════════════════════════════════════
# GET /languages
# ═══════════════════════════════════════════════════════════════════════════════

class TestLanguages:
    def test_returns_200(self, api_client):
        assert api_client.get("/languages").status_code == 200

    def test_has_supported_list(self, api_client):
        data = api_client.get("/languages").json()
        assert "supported" in data
        assert isinstance(data["supported"], list)
        assert len(data["supported"]) >= 2

    def test_english_included(self, api_client):
        data = api_client.get("/languages").json()
        assert "en" in data["supported"]


# ═══════════════════════════════════════════════════════════════════════════════
# POST /extract — valid PDF
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractValid:
    def test_returns_200(self, api_client, first_sample_pdf: Path):
        resp = pdf_upload(api_client, first_sample_pdf)
        assert resp.status_code == 200, resp.text

    def test_response_has_title(self, api_client, first_sample_pdf: Path):
        data = pdf_upload(api_client, first_sample_pdf).json()
        assert "title" in data

    def test_response_has_outline_list(self, api_client, first_sample_pdf: Path):
        data = pdf_upload(api_client, first_sample_pdf).json()
        assert "outline" in data
        assert isinstance(data["outline"], list)

    def test_response_has_metadata(self, api_client, first_sample_pdf: Path):
        data = pdf_upload(api_client, first_sample_pdf).json()
        assert "metadata" in data
        meta = data["metadata"]
        assert "processing_time_ms" in meta
        assert "total_pages" in meta
        assert "toc_available" in meta

    def test_heading_has_required_fields(self, api_client, first_sample_pdf: Path):
        data = pdf_upload(api_client, first_sample_pdf).json()
        for h in data["outline"]:
            assert "level" in h
            assert "text" in h
            assert "page" in h
            assert "confidence" in h
            assert "confidence_label" in h

    def test_heading_levels_valid(self, api_client, first_sample_pdf: Path):
        data = pdf_upload(api_client, first_sample_pdf).json()
        valid = {"H1", "H2", "H3"}
        for h in data["outline"]:
            assert h["level"] in valid

    def test_confidence_label_valid(self, api_client, first_sample_pdf: Path):
        data = pdf_upload(api_client, first_sample_pdf).json()
        valid_labels = {"high", "medium", "low"}
        for h in data["outline"]:
            assert h["confidence_label"] in valid_labels

    def test_confidence_in_range(self, api_client, first_sample_pdf: Path):
        data = pdf_upload(api_client, first_sample_pdf).json()
        for h in data["outline"]:
            assert 0.0 <= h["confidence"] <= 1.0

    def test_min_confidence_filter(self, api_client, first_sample_pdf: Path):
        """Headings with confidence below threshold must not appear in output."""
        high_thresh = pdf_upload(api_client, first_sample_pdf, min_conf=0.9).json()
        low_thresh  = pdf_upload(api_client, first_sample_pdf, min_conf=0.1).json()
        assert len(high_thresh["outline"]) <= len(low_thresh["outline"])

    @pytest.mark.parametrize("lang", ["en", "es", "fr", "de"])
    def test_supported_languages(self, api_client, first_sample_pdf: Path, lang: str):
        resp = pdf_upload(api_client, first_sample_pdf, lang=lang)
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# POST /extract — error cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractErrors:
    def test_invalid_file_returns_400(self, api_client, tmp_path: Path):
        bad = tmp_path / "bad.pdf"
        bad.write_bytes(b"Not a PDF at all")
        with open(bad, "rb") as fh:
            resp = api_client.post(
                "/extract",
                files={"file": ("bad.pdf", fh, "application/pdf")},
            )
        assert resp.status_code == 400

    def test_unsupported_language_returns_400(self, api_client, first_sample_pdf: Path):
        with open(first_sample_pdf, "rb") as fh:
            resp = api_client.post(
                "/extract",
                files={"file": (first_sample_pdf.name, fh, "application/pdf")},
                params={"lang": "xx_invalid"},
            )
        assert resp.status_code == 400

    def test_missing_file_returns_422(self, api_client):
        resp = api_client.post("/extract")
        assert resp.status_code == 422

    def test_confidence_out_of_range_returns_422(self, api_client, first_sample_pdf: Path):
        with open(first_sample_pdf, "rb") as fh:
            resp = api_client.post(
                "/extract",
                files={"file": (first_sample_pdf.name, fh, "application/pdf")},
                params={"min_confidence": 1.5},
            )
        assert resp.status_code == 422


# ═══════════════════════════════════════════════════════════════════════════════
# POST /extract/batch
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtractBatch:
    def test_batch_single_file(self, api_client, first_sample_pdf: Path):
        with open(first_sample_pdf, "rb") as fh:
            resp = api_client.post(
                "/extract/batch",
                files=[("files", (first_sample_pdf.name, fh, "application/pdf"))],
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_files"] == 1
        assert data["successful"] >= 0

    def test_batch_response_structure(self, api_client, first_sample_pdf: Path):
        with open(first_sample_pdf, "rb") as fh:
            data = api_client.post(
                "/extract/batch",
                files=[("files", (first_sample_pdf.name, fh, "application/pdf"))],
            ).json()
        assert "results" in data
        assert "total_files" in data
        assert "successful" in data
        assert "failed" in data
        assert "total_processing_time_ms" in data

    def test_batch_too_many_files_returns_400(self, api_client, first_sample_pdf: Path):
        files = []
        for i in range(11):  # 11 > max of 10
            with open(first_sample_pdf, "rb") as fh:
                files.append(("files", (f"file{i}.pdf", fh.read(), "application/pdf")))
        resp = api_client.post("/extract/batch", files=files)
        assert resp.status_code == 400
