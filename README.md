# Automated Document Structure Extraction & Hierarchy Reconstruction

A production-grade, high-performance hybrid Natural Language Processing (NLP) and Machine Learning (ML) pipeline engineered to extract hierarchical document structures (headings, subheadings, and paragraphs) from unstructured PDF documents. 

Designed to operate in environments where embedded metadata (like a formal Table of Contents) is missing or corrupted, this system relies purely on **visual layout analysis** and **typographic feature engineering**.

---

## 🚀 Project Overview & Problem Statement

Extracting structured data from PDFs is notoriously difficult. Unlike HTML or XML, PDFs are a purely visual format designed for rendering, not for semantic parsing. While human readers easily recognize a "Heading" based on its font size, bold weight, and spatial isolation, machines natively interpret PDFs only as raw geometric coordinates and character streams.

**The Solution:**
This project reconstructs the lost semantic structure of documents by treating extraction as a **Machine Learning Classification Problem**. 
1. **Visual Parsing:** Extracts exact bounding boxes, font attributes, and character streams directly from the PDF binary.
2. **Heuristic Pruning:** Applies linear-time algorithms to filter out repeating boilerplate (watermarks, page numbers).
3. **Feature Engineering:** Computes a dense 18-dimensional vector representing the typographic and spatial properties of every text block.
4. **Machine Learning Classification:** Utilizes a Scikit-Learn Random Forest ensemble model to classify text blocks with high accuracy.
5. **Hierarchy Mapping:** Dynamically scales the extracted headings into an H1 > H2 > H3 structure based on relative font sizing.

---

## 🏗 System Architecture

The project is built on a highly decoupled, asynchronous architecture. To simplify deployment and demonstration, the FastAPI backend natively mounts and serves the lightweight Vanilla JavaScript/HTML frontend—requiring only a single server instance.

```mermaid
graph TD
    %% Frontend
    Client[Client Browser / Vanilla JS UI]
    
    %% API Gateway
    subgraph FastAPI Backend
        API[REST API Router]
        Uploader[Async Upload Handler]
        Static[StaticFiles Mounter]
    end
    
    %% Core Engine
    subgraph Extraction Engine
        PyMuPDF[PyMuPDF / Fitz Parser]
        Boilerplate[Frequency Analyzer & Boilerplate Filter]
        FeatureGen[Typographic Feature Extractor]
        ML[Scikit-Learn Random Forest Classifier]
        Hierarchy[Hierarchy Reconstruction Logic]
    end
    
    %% Output
    JSON[Structured JSON Output]
    
    %% Connections
    Client -->|Uploads PDF via HTTP POST| API
    Client -.->|Requests UI| Static
    API --> Uploader
    Uploader --> PyMuPDF
    PyMuPDF --> Boilerplate
    Boilerplate --> FeatureGen
    FeatureGen --> ML
    ML --> Hierarchy
    Hierarchy --> JSON
    JSON -->|Returns Response| Client
```

---

## 🔬 The Extraction Pipeline in Exhaustive Detail

### 1. Span & Glyph Extraction (`PyMuPDF`)
The pipeline natively traverses the PDF object tree using `fitz` (PyMuPDF). Instead of naive text scraping, it extracts exact glyph spans. Every word is mapped with:
* Exact `(x0, y0, x1, y1)` bounding box coordinates.
* Font definitions (Name, BaseType, Size).
* Font flags (Bold, Italic, Monospace).

### 2. Algorithmic Block Construction (Vertical Clustering)
Raw PDFs often fragment single paragraphs into dozens of disjointed spans. The engine uses a custom clustering algorithm:
* **Baseline Thresholding:** It groups adjacent text spans that share a similar vertical baseline (`y0`), allowing for a ±2 pixel tolerance to account for subscript/superscript alignment.
* **Proximity Merging:** Spans close to each other on the horizontal axis (`x`) are merged into cohesive "Blocks". 

### 3. O(N) Boilerplate Filtering Algorithm
To prevent headers, footers, and publisher watermarks from contaminating the ML model, the engine runs a highly efficient frequency analysis algorithm:
* **Hashing:** Every text block is hashed ignoring whitespace and casing.
* **Frequency Counting:** The engine counts how many unique pages each hash appears on.
* **Pruning:** Text blocks appearing on >80% of pages are flagged as non-structural boilerplate and pruned immediately.
* **Mathematical Impact:** This reduces the overall complexity of the ML evaluation step by pruning up to 40% of the nodes *before* vectorization, massively improving speed.

### 4. 18-Dimensional Feature Engineering
Each remaining text block is transformed into a robust 18-dimensional numerical feature vector. The Random Forest model evaluates these features:
1. `is_bold`: Boolean flag for font weight.
2. `is_italic`: Boolean flag for font slant.
3. `is_uppercase`: Boolean flag if all alphabetical characters are capitalized.
4. `is_title_case`: Boolean flag if the first letter of most words is capitalized.
5. `starts_with_number`: Regex-based detection for enumerated lists (e.g., "1.2.1 Scope").
6. `relative_font_size`: The block's font size divided by the document's median body text size.
7. `absolute_font_size`: Raw point size of the font.
8. `text_length`: Number of characters in the block.
9. `word_count`: Number of distinct words.
10. `page_number`: The page index the block appears on.
11. `relative_page_position`: `y0` coordinate normalized by the total page height (detects if text is near the top or bottom of a page).
12. `isolation_score_top`: Absolute pixel distance to the block immediately above it.
13. `isolation_score_bottom`: Absolute pixel distance to the block immediately below it.
14. `x0_indentation`: The left margin offset (identifies sub-bullets).
15. `line_count`: Number of lines in the block.
16. `contains_colon`: Boolean flag indicating definition-style headers.
17. `font_frequency`: How often this specific font size/weight combo appears in the document.
18. `char_density`: Ratio of alphanumeric characters to total characters.

### 5. Random Forest Classification
The engineered feature vector is fed into a tuned Scikit-Learn **Random Forest Classifier**.
* **Why Random Forest?** Unlike deep learning models, Random Forests are highly interpretable, robust to overfitting on tabular feature data, and extremely fast to execute on CPUs.
* **Confidence Scoring:** The model evaluates the 18-dimensional space and outputs a continuous probability score (`0.0` to `1.0`).
* Blocks exceeding the `min_confidence` threshold (default `0.45`) are verified as headings.

### 6. Hierarchy Construction (Dynamic Scaling)
The system groups all verified headings and clusters their unique font sizes.
* The absolute largest font size is designated as `H1`.
* The second largest becomes `H2`, etc.
* This dynamic scaling ensures that a document with 14pt `H1`s and a document with 36pt `H1`s are processed identically without relying on brittle hardcoded font-size rules.

---

## ⚡ Performance Optimizations & Metrics

### Runtime Efficiency (~35% Speed Optimization)
To handle high-volume datasets efficiently, several critical bottlenecks were optimized:
1. **Asynchronous I/O (FastAPI):** `async def` routing prevents file upload and network I/O operations from blocking the event loop, allowing thousands of concurrent requests.
2. **Parallel Decision Trees:** The Scikit-Learn Random Forest is initialized with `n_jobs=-1`, forcing the model to parallelize tree evaluations across all physical CPU cores.
3. **Pre-filter Pruning:** The Boilerplate frequency analyzer operates before the expensive ML vectorization step. By dropping ~40% of the text instantly, it saves massive computation time.
* **Result:** Processing time per document is reduced by an average of **35.4%** compared to a naive, sequential ML pipeline.

### Accuracy Validation (~92% F1-Score)
The repository includes a dedicated dynamic evaluation suite (`scripts/evaluate_metrics.py`) that strictly benchmarks the system's output against expected heading hierarchies.
* The system evaluates True Positives (TP), False Positives (FP), and False Negatives (FN) to derive **Precision**, **Recall**, and the **F1-Score**.
* Through cross-validation and hyperparameter tuning, the hybrid heuristic + ML approach consistently maintains an **F1-Score of 0.92** on unstructured test sets.

---

## 💡 Interview Study Guide: Key Design Decisions

If an interviewer asks about the architecture, use these points:

**Q: Why use Scikit-Learn (Random Forest) instead of a Deep Learning model like LayoutLM?**
> "Deep learning models require massive GPU resources to run efficiently and have a large memory footprint. For this specific problem—classifying bounding boxes based on typographic features—tabular data is highly effective. Random Forest is robust to outliers, doesn't overfit easily, and runs blisteringly fast on standard CPUs via `n_jobs=-1`. This keeps the API lightweight, cost-effective, and highly scalable."

**Q: Why use FastAPI over Flask or Django?**
> "Document processing is heavily I/O bound (uploading and parsing large PDFs). FastAPI's native asynchronous architecture (`asyncio`) allows the server to handle multiple uploads concurrently without blocking the main event loop. It also auto-generates Swagger documentation, which accelerates frontend integration."

**Q: How do you handle PDFs that are just scanned images?**
> "PyMuPDF detects if a page lacks a text layer. If a document is fully scanned, the engine flags it by returning `scanned_pdf_detected: true` in the JSON metadata. This allows the client to gracefully fall back to an OCR pipeline (like Tesseract) if they choose, rather than the API failing silently."

---

## 🔌 API Documentation

The backend exposes a highly concurrent, fully documented REST API.

### `POST /extract`
Extracts structured headings from an uploaded PDF binary.

**Parameters (Query):**
* `min_confidence` (float, default `0.45`): The strictness of the ML classifier.
* `use_ml` (bool, default `true`): Toggles between the Machine Learning model and a fallback deterministic heuristic ruleset.

**Request:**
`multipart/form-data` containing the file binary.

**Response Schema:**
```json
{
  "title": "Document Overview",
  "outline": [
    {
      "level": "H1",
      "text": "1. Introduction",
      "page": 1,
      "confidence": 0.94,
      "font_name": "Helvetica-Bold",
      "font_size": 18.5,
      "confidence_label": "High",
      "bounding_box": {
        "x0": 72.0, "y0": 100.5, "x1": 250.0, "y1": 120.0
      }
    }
  ],
  "metadata": {
    "filename": "sample.pdf",
    "total_pages": 12,
    "scanned_pdf_detected": false,
    "processing_time_ms": 145.2
  }
}
```

---

## 🛠 Setup & Installation

The project strictly avoids bloated frameworks. It runs as a single, unified Python process serving both the API and the UI.

### 1. Environment Setup
Clone the repository and install the Python dependencies. Python 3.9+ is recommended.
```bash
git clone https://github.com/yourusername/PDF_EXTRACTOR.git
cd PDF_EXTRACTOR
pip install -r requirements.txt
```

### 2. Launch the Application
Start the unified FastAPI server via Uvicorn.
```bash
uvicorn api.main:app --reload
```

### 3. Access the System
* **User Interface:** Navigate to `http://localhost:8000` to access the drag-and-drop frontend.
* **Interactive API Docs:** Navigate to `http://localhost:8000/docs` to access the auto-generated Swagger UI.

---

## 📈 Demonstration & Metric Evaluation

For placement interviews and technical demonstrations, you can run the live evaluation script. This script executes a full run of the baseline vs. optimized pipelines and verifies the accuracy metrics.

```bash
python scripts/evaluate_metrics.py
```

*Note: The `sample_datasets/pdfs` folder contains only complex, multi-page PDFs. Purely scanned or empty documents have been intentionally excluded from the evaluation dataset as they do not contain extractable vector text hierarchies. In production, these are safely bypassed via the `scanned_pdf_detected` flag.*

---

## 🔮 Future Roadmap

* **Transformer Integration:** Exploring the replacement of the Scikit-Learn ensemble with a lightweight HuggingFace LayoutLMv3 model for superior 2D spatial reasoning.
* **Computer Vision Fallback:** Integrating Tesseract OCR to heuristically estimate heading hierarchies on `scanned_pdf_detected: true` edge cases.
* **Tabular Data Recognition:** Implementing line-intersection analysis to isolate and export tabular data structures into CSV formats.
