# 📄 PDF Heading Extractor

> A **production-grade, multilingual PDF document-structure extraction pipeline** built with PyMuPDF, FastAPI, and a confidence-scoring engine. Exposes a REST API, ships with a drag-and-drop Web UI, and includes a full evaluation harness for measurable precision/recall benchmarking.

[![CI](https://github.com/harsh07032004/PDF_HEADINGEXTRACTOR/actions/workflows/ci.yml/badge.svg)](https://github.com/harsh07032004/PDF_HEADINGEXTRACTOR/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-009688.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](Dockerfile)

---

## ✨ Features

| Feature | Description |
|---|---|
| **Hybrid heading detection** | Font-size clustering + boldness + indentation + TOC cross-validation |
| **Confidence scoring** | Every heading gets a `0.0–1.0` confidence score from 6 weighted signals |
| **Multilingual** | English, Spanish, French, German, Japanese regex patterns |
| **REST API** | FastAPI with `/extract`, `/extract/batch`, `/health`, `/languages` |
| **Web UI** | Drag-and-drop interface with live tree view + JSON download |
| **Evaluation harness** | Precision / Recall / F1 per heading level against ground truth |
| **Docker ready** | Multi-stage build, non-root user, health check |
| **CI/CD** | GitHub Actions: lint → test (3× Python matrix) → Docker smoke test |
| **60+ tests** | Unit + integration tests with coverage reporting |

---

## 🏗️ System Architecture

```
[ Web UI / Client ]
       │
       ▼
[ FastAPI Layer ]  ──────── POST /extract
  (Async REST)              POST /extract/batch
       │                    GET  /health
       ▼                    GET  /languages
[ ExtractorEngine ]
  ├── TOC cross-validation   (strongest signal: +0.40)
  ├── Font-size clustering   (body vs H1/H2/H3 tiers)
  ├── Span-level parsing     (character → line → block)
  ├── Feature engineering    (8 typographic features)
  ├── Confidence scoring     (hybrid weighted model)
  └── Level classification   (H1 / H2 / H3)
       │
       ▼
[ DocumentOutline ]  ──────► JSON response / file
  title, headings[], metadata{}
```

---

## 📁 Project Structure

```
.
├── extractor/               # Core package
│   ├── __init__.py          # Public API surface
│   ├── models.py            # Dataclass schema (Heading, DocumentOutline, …)
│   ├── core.py              # ExtractorEngine (9-step pipeline)
│   ├── utils.py             # Text cleaning, PDF validation, PRF metrics
│   └── cli.py               # CLI entry point
├── api/                     # FastAPI application
│   ├── main.py              # Routes + middleware
│   └── schemas.py           # Pydantic HTTP models
├── web/                     # Web UI (no framework, vanilla JS)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── tests/                   # Test suite (pytest)
│   ├── conftest.py          # Shared fixtures
│   ├── test_extractor.py    # 30+ unit & integration tests
│   └── test_api.py          # 35 API endpoint tests
├── sample_datasets/
│   ├── pdfs/                # 5 sample input PDFs
│   └── outputs/             # Ground-truth JSON files
├── evaluate.py              # Evaluation harness (P/R/F1)
├── process_pdfs.py          # Legacy CLI wrapper (backwards compat)
├── languages.json           # Per-language heading regex patterns
├── pyproject.toml           # Build system + Ruff + Mypy + Pytest config
├── requirements.txt
├── Dockerfile               # Multi-stage Docker build
└── docker-compose.yml
```

---

## 🚀 Quick Start

### 1. Install

```bash
# Clone
git clone https://github.com/harsh07032004/PDF_HEADINGEXTRACTOR.git
cd PDF_HEADINGEXTRACTOR

# Create venv and install
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Editable install (enables `extract-pdf` CLI command)
pip install -e .
```

### 2. Run the API Server

```bash
uvicorn api.main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.  
Open **http://localhost:8000/ui** for the drag-and-drop Web UI.

### 3. CLI Usage

```bash
# Basic extraction (English)
extract-pdf sample_datasets/pdfs/ output/

# Spanish PDFs with high confidence threshold
extract-pdf input/ output/ --lang es --min-confidence 0.6

# Verbose output showing per-heading details
extract-pdf input/ output/ --verbose
```

### 4. Docker

```bash
# Build
docker build -t pdf-heading-extractor .

# Run API server
docker run -p 8000:8000 pdf-heading-extractor

# Or with docker-compose
docker compose up

# CLI batch processing
docker compose --profile cli up cli
```

---

## 📡 API Reference

### `POST /extract`

Extract headings from a single PDF.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | PDF file (max 20 MB) |
| `lang` | query | `en` | Language code: en, es, fr, de, ja |
| `min_confidence` | query | `0.45` | Minimum confidence threshold |

**Response:**
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "1. Introduction",
      "page": 2,
      "confidence": 0.87,
      "confidence_label": "high",
      "font_name": "Helvetica-Bold",
      "font_size": 18.0,
      "bounding_box": { "x0": 72.0, "y0": 120.5, "x1": 320.0, "y1": 142.0 }
    }
  ],
  "metadata": {
    "filename": "document.pdf",
    "total_pages": 12,
    "language": "en",
    "toc_available": true,
    "scanned_pdf_detected": false,
    "processing_time_ms": 145.3
  }
}
```

### `POST /extract/batch`

Upload up to 10 PDFs, process all, get combined results.

### `GET /health`

```json
{ "status": "ok", "version": "0.1.0", "engine": "PyMuPDF" }
```

### `GET /languages`

```json
{ "supported": ["en", "es", "fr", "de", "ja"], "default": "en" }
```

---

## 🧠 Confidence Scoring

Each heading receives a hybrid confidence score from 6 signals:

| Signal | Weight | Description |
|---|---|---|
| TOC match | +0.40 | Heading text found in PDF's embedded TOC |
| Font size tier | +0.25 | Font is in the heading size cluster |
| Boldness | +0.15 | Font is bold |
| Language regex | +0.10 | Matches heading pattern (e.g. "Chapter 1") |
| Numbered section | +0.05 | Starts with "1.", "1.2", etc. |
| All-caps | +0.05 | >80% uppercase characters |
| Long line penalty | −0.20 | >120 characters (likely body text) |

Scores are clamped to `[0.0, 1.0]`. Labels: `≥ 0.80 = high`, `≥ 0.55 = medium`, `< 0.55 = low`.

---

## 📊 Evaluation

Run the evaluation harness against the 5 labeled sample PDFs:

```bash
python evaluate.py --verbose
python evaluate.py --output eval_report.json
```

Sample output:
```
  PDF Heading Extractor — Evaluation Report
  5 document(s) evaluated  (0 error(s))
  Title exact match: 3/5  │  Fuzzy: 4/5

  Level    Precision      Recall          F1      TP  GT Total
  ──────────────────────────────────────────────────────────────
  H1          0.9000      0.8182      0.8571       9        11
  H2          0.7500      0.7059      0.7273      24        34
  H3          0.6667      0.6667      0.6667       4         6
  ──────────────────────────────────────────────────────────────
  Overall     0.7843      0.7451      0.7642      37        51
```

---

## 🧪 Testing

```bash
# Run all tests
pytest

# Run only fast unit tests
pytest -m unit

# Run integration tests
pytest -m integration

# With coverage
pytest --cov=extractor --cov=api --cov-report=html
open htmlcov/index.html

# Run the slow full-suite (all 5 PDFs)
pytest -m slow
```

---

## 🛠️ Development

```bash
# Lint (Ruff replaces Black + Flake8 + isort)
ruff check .

# Auto-fix lint issues
ruff check --fix .

# Format
ruff format .

# Type check
mypy extractor/ api/
```

---

## 🌐 Deployment

### Render (Recommended for free hosting)

1. Connect your GitHub repo to [Render](https://render.com)
2. New → Web Service → Choose repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
5. Your live URL: `https://your-app.onrender.com`

### Railway

```bash
railway init
railway up
```

### AWS EC2

```bash
# On EC2 instance (Amazon Linux 2)
git clone https://github.com/harsh07032004/PDF_HEADINGEXTRACTOR.git
cd PDF_HEADINGEXTRACTOR
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 2
```

---

## 📝 License

MIT © 2024 Harshita
