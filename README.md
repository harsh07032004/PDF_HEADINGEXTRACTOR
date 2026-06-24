# PDF Heading Extractor

Extracts structured headings (H1, H2, H3) from PDF documents using font analysis and machine learning classification.

Designed to work with PDFs that lack a Table of Contents or structural metadata by analyzing text properties like font size, weight, and position.

---

## Problem Statement

PDFs don't store semantic structure like HTML (`<h1>`, `<p>` tags). They only store X/Y coordinates and character glyphs. 

This tool reconstructs document hierarchy by analyzing:
- Font size and weight (bold, italic)
- Text position on the page
- Text length and capitalization
- Spacing between text blocks

A trained machine learning classifier (scikit-learn) then predicts whether each text block is a heading or body text.

---

## System Architecture

```
PDF File
   ↓
[PyMuPDF] Extract text spans, bounding boxes, font metadata
   ↓
[Block Clustering] Group text spans into lines based on position
   ↓
[Font Size Analysis] Identify which font sizes are headings vs body text
   ↓
[Feature Engineering] Convert each block to 18 numerical features
   ↓
[ML Classifier] scikit-learn model predicts: heading or body text?
   ↓
[Hierarchy Mapping] Assign H1, H2, H3 based on font sizes
   ↓
JSON Output (headings with confidence scores and position data)
```

**Backend:** FastAPI REST API with `/extract` endpoint
**Frontend:** Vanilla JavaScript + HTML (served by FastAPI)
**Core Logic:** Pure Python library (no dependencies on the API)

---

## Extraction Pipeline

### Step 1: Extract Text & Metadata (PyMuPDF)
Reads the PDF binary using PyMuPDF (`fitz` library):
- Extracts text spans with bounding box coordinates (x0, y0, x1, y1)
- Captures font name, size, and bold/italic flags
- Detects if the PDF has no text layer (scanned image)

### Step 2: Block Clustering
Groups individual text spans into lines:
- Spans sharing the same vertical baseline (y-coordinate) are merged
- Adjacent horizontal spans are combined into cohesive text blocks

### Step 3: Font Size Analysis
Identifies which font sizes correspond to headings vs body text:
- Calculates the median body text font size
- Groups blocks by their font sizes
- Largest font size → H1, second largest → H2, etc.

### Step 4: Feature Extraction
For each text block, extracts 18 features:

**Typography (4 features):**
- `relative_font_size` — font size / median body text size
- `font_size_zscore` — how many standard deviations from average
- `bold_percentage` — how many characters are bold (0-1)
- `is_italic` — 1 if italic, 0 otherwise

**Structure (4 features):**
- `starts_with_number` — 1 if block starts with "1.", "1.2", etc.
- `x_indent_ratio` — left margin position (detects sub-bullets)
- `y_position_ratio` — position from top of page (0-1)
- `centered_text` — 1 if text is centered

**Context (4 features):**
- `in_toc` — 1 if block appears in Table of Contents
- `is_first_page` — 1 if on page 1
- `vertical_gap_before` — space above the block
- `font_change_from_prev` — 1 if font changed from previous block

**Content (6 features):**
- `char_count` — number of characters
- `word_count` — number of words
- `all_caps_ratio` — percentage of uppercase letters (0-1)
- `title_case_ratio` — percentage of title-cased words (0-1)
- `punctuation_density` — punctuation / total characters
- `short_line` — 1 if less than 60 characters

### Step 5: Machine Learning Classification
Feeds the 18-feature vector into a scikit-learn classifier:
- Model: **GradientBoostingClassifier** (with RandomForest as fallback)
- Output: Probability score that the block is a heading (0.0 to 1.0)
- Blocks exceeding the confidence threshold (default 0.45) are labeled as headings

### Step 6: Hierarchy Assignment
Assigns H1, H2, H3 based on font size clustering:
- Headings are sorted by font size
- Largest → H1, second largest → H2, third largest → H3
- This dynamic scaling works for any document regardless of its font sizes

---

## Performance & Accuracy

**Accuracy:** The system achieves approximately 92% F1-Score on test documents, measured using a ground-truth evaluation script that compares predicted headings against manually annotated documents.

**Speed:** Most PDFs process in 50-200ms depending on document complexity.

**Optimization techniques:**
- Font-size clustering filters out ~40% of blocks before ML classification
- Scikit-learn models run efficiently on CPU (no GPU required)
- Simple block clustering reduces the feature extraction workload

---

## Design Decisions

**Why scikit-learn instead of deep learning?**
Deep learning models require GPUs and massive memory. This is a tabular classification problem (18 numbers → heading or not). Traditional ML is simpler, faster, and easier to debug.

**Why FastAPI?**
It's lightweight and has built-in support for file uploads, automatic documentation, and easy deployment.

**How do you handle scanned PDFs?**
PyMuPDF detects if a page has no text layer. The API returns a metadata flag `scanned_pdf_detected: true` so the frontend can notify the user.

---

## API Documentation

### `POST /extract`
Uploads a PDF and returns extracted headings.

**Query Parameters:**
- `min_confidence` (float, default 0.45) — Classification confidence threshold
- `use_ml` (bool, default true) — Use ML model (false uses font size heuristic only)

**Request:**
`multipart/form-data` with a PDF file

**Response:**
```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "1. Introduction",
      "page": 1,
      "confidence": 0.94,
      "font_size": 18.5,
      "bounding_box": {"x0": 72.0, "y0": 100.5, "x1": 250.0, "y1": 120.0}
    }
  ],
  "metadata": {
    "filename": "sample.pdf",
    "total_pages": 12,
    "scanned_pdf_detected": false,
    "processing_time_ms": 145
  }
}
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### Install

```bash
git clone https://github.com/harsh07032004/PDF_HEADINGEXTRACTOR.git
cd PDF_HEADINGEXTRACTOR
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser. Upload a PDF to extract its headings.

### Run Evaluation

```bash
python scripts/evaluate_metrics.py
```

This benchmarks the extractor against ground-truth heading annotations.

---

## Running the Evaluation

```bash
python scripts/evaluate_metrics.py
```

This script tests the extractor against labeled ground-truth documents and outputs Precision, Recall, and F1-Score.
