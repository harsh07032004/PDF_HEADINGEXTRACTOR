"""
tests/conftest.py

Shared pytest fixtures for the entire test suite.

Fixtures are organised by scope:
    session  — expensive objects created once for the whole test run
    module   — per test file
    function — per test (default)
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ── Paths ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root (Challenge_1a/)."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_pdfs_dir(project_root: Path) -> Path:
    return project_root / "sample_datasets" / "pdfs"


@pytest.fixture(scope="session")
def sample_outputs_dir(project_root: Path) -> Path:
    return project_root / "sample_datasets" / "outputs"


@pytest.fixture(scope="session")
def languages_file(project_root: Path) -> Path:
    return project_root / "languages.json"


# ── Sample PDF fixtures ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_pdf_paths(sample_pdfs_dir: Path) -> list[Path]:
    """All PDFs in the sample dataset, sorted."""
    pdfs = sorted(sample_pdfs_dir.glob("*.pdf"))
    if not pdfs:
        pytest.skip(f"No PDFs found in {sample_pdfs_dir}")
    return pdfs


@pytest.fixture(scope="session")
def first_sample_pdf(sample_pdf_paths: list[Path]) -> Path:
    """The first sample PDF — used for fast single-file tests."""
    return sample_pdf_paths[0]


@pytest.fixture(scope="session")
def sample_ground_truth(sample_outputs_dir: Path) -> dict[str, dict]:
    """
    Load all ground-truth JSON files into a dict keyed by stem.

    Example: {"file01": {"title": "...", "outline": [...]}}
    """
    import json
    data: dict[str, dict] = {}
    for json_file in sorted(sample_outputs_dir.glob("*.json")):
        with open(json_file, encoding="utf-8") as fh:
            data[json_file.stem] = json.load(fh)
    return data


# ── FastAPI test client ────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def api_client():
    """
    Create a TestClient for the FastAPI application.

    Uses httpx's ASGITransport so no real network is involved.
    Scoped to session because FastAPI startup is inexpensive.
    """
    try:
        from fastapi.testclient import TestClient
        from api.main import app
        return TestClient(app)
    except ImportError:
        pytest.skip("FastAPI or httpx not installed — skipping API tests.")


# ── Tiny fake PDF (in-memory) ──────────────────────────────────────────────────

@pytest.fixture(scope="session")
def minimal_pdf_bytes() -> bytes:
    """
    A minimal hand-crafted valid PDF byte string for unit tests that
    need a PDF without touching the filesystem.

    Contains one page with one text object.
    """
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 595 842]/Parent 2 0 R"
        b"/Resources<</Font<</F1<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>>>>>"
        b"/Contents 4 0 R>>endobj\n"
        b"4 0 obj<</Length 44>>\nstream\n"
        b"BT /F1 24 Tf 100 700 Td (Introduction) Tj ET\n"
        b"endstream\nendobj\n"
        b"xref\n0 5\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000306 00000 n \n"
        b"trailer<</Size 5/Root 1 0 R>>\n"
        b"startxref\n401\n%%EOF"
    )


@pytest.fixture
def minimal_pdf_path(tmp_path: Path, minimal_pdf_bytes: bytes) -> Path:
    """Write the minimal PDF to a temp file and return its path."""
    pdf_path = tmp_path / "minimal.pdf"
    pdf_path.write_bytes(minimal_pdf_bytes)
    return pdf_path
