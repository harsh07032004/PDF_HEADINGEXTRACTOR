/**
 * app.js — PDF Heading Extractor Web UI
 *
 * Architecture:
 *   State object → UI renderers → API calls → State update loop
 *
 * No external dependencies. Pure vanilla JS with modern APIs.
 */

"use strict";

// ── API Configuration ─────────────────────────────────────────────────────────

// When deploying the frontend separately (e.g. Vercel/Netlify), 
// change this to your deployed backend URL:
// const API_BASE = "https://your-backend-app.onrender.com";
const API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
  ? "http://localhost:8000"
  : window.location.origin; // Fallback to same origin if hosted together

// ── State ─────────────────────────────────────────────────────────────────────

const state = {
  file: null,       // File | null
  result: null,     // API response object | null
  loading: false,
  pdfBlobUrl: null, // string URL to the local Blob
  pdfDoc: null,     // PDF.js document object
  pageNum: 1,
  pageRendering: false,
  pageNumPending: null,
  highlightPending: null, // bounding box array [x0, y0, x1, y1]
  pdfScale: 1.5,
};

// ── DOM References ────────────────────────────────────────────────────────────

const $ = (id) => document.getElementById(id);

const els = {
  dropZone:       $("drop-zone"),
  fileInput:      $("file-input"),
  extractBtn:     $("extract-btn"),
  langSelect:     $("lang-select"),
  confRange:      $("confidence-range"),
  confValue:      $("confidence-value"),

  processingState: $("processing-state"),
  processingLabel: $("processing-label"),

  resultsSection: $("results-section"),
  statTitleText:  $("stat-title-text"),
  statPages:      $("stat-pages"),
  statHeadings:   $("stat-headings"),
  statTime:       $("stat-time"),
  statScanned:    $("stat-scanned"),
  outlineTree:    $("outline-tree"),

  mainContainer:  $("main-container"),
  rightPane:      $("right-pane"),
  pdfViewerContainer: $("pdf-viewer-container"),
  pdfCanvas:      $("pdf-canvas"),
  pdfHighlightLayer: $("pdf-highlight-layer"),
  viewerControls: $("viewer-controls"),
  btnPrevPage:    $("btn-prev-page"),
  btnNextPage:    $("btn-next-page"),
  pageNumSpan:    $("page-num"),
  pageCountSpan:  $("page-count"),

  copyBtn:        $("copy-btn"),
  downloadBtn:    $("download-btn"),

  errorState:     $("error-state"),
  errorDetail:    $("error-detail"),
  errorRetryBtn:  $("error-retry-btn"),

  toastContainer: $("toast-container"),
};

// ── Confidence Range Slider ────────────────────────────────────────────────────

function syncRangeStyle() {
  const val = parseFloat(els.confRange.value);
  const pct = (val * 100).toFixed(0);
  els.confRange.style.setProperty("--range-pct", `${pct}%`);
  els.confValue.textContent = val.toFixed(2);
}

els.confRange.addEventListener("input", syncRangeStyle);
syncRangeStyle();

// ── File Selection ────────────────────────────────────────────────────────────

function onFileSelected(file) {
  if (!file) return;

  if (file.type !== "application/pdf" && !file.name.toLowerCase().endsWith(".pdf")) {
    showToast("⚠️ Please select a valid PDF file.", "error");
    return;
  }

  if (file.size > 20 * 1024 * 1024) {
    showToast("⚠️ File exceeds the 20 MB limit.", "error");
    return;
  }

  state.file = file;
  if (state.pdfBlobUrl) {
    URL.revokeObjectURL(state.pdfBlobUrl);
  }
  state.pdfBlobUrl = URL.createObjectURL(file);
  
  els.extractBtn.disabled = false;
  els.dropZone.classList.add("has-file");

  // Update drop zone display
  els.dropZone.querySelector(".drop-zone-title").textContent = `✓  ${file.name}`;
  els.dropZone.querySelector(".drop-zone-sub").textContent =
    `${(file.size / 1024).toFixed(1)} KB  ·  Click to change`;
}

els.fileInput.addEventListener("change", (e) => {
  onFileSelected(e.target.files?.[0]);
});

// Keyboard activation for drop zone
els.dropZone.addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    e.preventDefault();
    els.fileInput.click();
  }
});

// Click activation for drop zone
els.dropZone.addEventListener("click", () => {
  els.fileInput.click();
});

// ── Drag and Drop ─────────────────────────────────────────────────────────────

els.dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  els.dropZone.classList.add("drag-over");
});

["dragleave", "dragend"].forEach((ev) => {
  els.dropZone.addEventListener(ev, () => {
    els.dropZone.classList.remove("drag-over");
  });
});

els.dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  els.dropZone.classList.remove("drag-over");
  const file = e.dataTransfer?.files?.[0];
  if (file) onFileSelected(file);
});

// ── Extraction ────────────────────────────────────────────────────────────────

els.extractBtn.addEventListener("click", runExtraction);
els.errorRetryBtn.addEventListener("click", () => {
  setView("upload");
});

async function runExtraction() {
  if (!state.file || state.loading) return;

  state.loading = true;
  setView("processing");

  const lang = els.langSelect.value;
  const minConf = parseFloat(els.confRange.value);

  const formData = new FormData();
  formData.append("file", state.file);

  const url = new URL(`${API_BASE}/extract`);
  url.searchParams.set("lang", lang);
  url.searchParams.set("min_confidence", minConf.toFixed(2));

  try {
    // Animate processing label
    const labels = [
      "Parsing PDF structure…",
      "Extracting text spans…",
      "Computing confidence scores…",
      "Classifying headings…",
      "Finalising outline…",
    ];
    let labelIdx = 0;
    const labelInterval = setInterval(() => {
      labelIdx = (labelIdx + 1) % labels.length;
      els.processingLabel.textContent = labels[labelIdx];
    }, 900);

    const response = await fetch(url.toString(), {
      method: "POST",
      body: formData,
    });

    clearInterval(labelInterval);

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error: ${response.status}`);
    }

    const data = await response.json();
    state.result = data;
    renderResults(data);
    setView("results");
    showToast(`✅ Extracted ${data.outline.length} headings`, "success");

  } catch (err) {
    setView("error");
    els.errorDetail.textContent = err.message || "An unexpected error occurred.";
    showToast("❌ Extraction failed", "error");
  } finally {
    state.loading = false;
  }
}

// ── View Switcher ─────────────────────────────────────────────────────────────

function setView(view) {
  // hide all transient views
  els.processingState.hidden = true;
  els.resultsSection.hidden = true;
  els.errorState.hidden = true;
  
  // reset split layout
  els.mainContainer.classList.remove("split-layout");
  els.rightPane.hidden = true;

  if (view === "processing") {
    els.processingLabel.textContent = "Processing document…";
    els.processingState.hidden = false;
    els.extractBtn.disabled = true;
  } else if (view === "results") {
    els.resultsSection.hidden = false;
    els.extractBtn.disabled = false;
    // activate split layout
    els.mainContainer.classList.add("split-layout");
    els.rightPane.hidden = false;
    // load PDF into viewer
    loadPdfDocument(state.pdfBlobUrl);
  } else if (view === "error") {
    els.errorState.hidden = false;
    els.extractBtn.disabled = false;
  } else {
    // "upload" — reset to initial
    els.extractBtn.disabled = !state.file;
  }
}

// ── Results Renderer ──────────────────────────────────────────────────────────

function renderResults(data) {
  const meta = data.metadata;

  // Stats bar
  els.statTitleText.textContent = meta.filename;
  els.statPages.textContent = `${meta.total_pages}`;
  els.statHeadings.textContent = `${data.outline.length}`;
  els.statTime.textContent = `${meta.processing_time_ms}`;
  els.statScanned.hidden = !meta.scanned_pdf_detected;

  // Build heading tree
  els.outlineTree.innerHTML = "";

  if (data.outline.length === 0) {
    els.outlineTree.innerHTML = `
      <div style="padding: 32px; text-align: center; color: var(--text-muted)">
        No headings detected above the confidence threshold.<br />
        Try lowering the minimum confidence slider.
      </div>`;
    return;
  }

  const fragment = document.createDocumentFragment();

  // Typography normalization: find max/min font size in outline
  let minSize = Infinity, maxSize = -Infinity;
  data.outline.forEach(h => {
    if (h.font_size < minSize) minSize = h.font_size;
    if (h.font_size > maxSize) maxSize = h.font_size;
  });
  if (minSize === Infinity) minSize = 10;
  if (maxSize === -Infinity) maxSize = 10;

  data.outline.forEach((heading, idx) => {
    fragment.appendChild(buildHeadingItem(heading, idx, minSize, maxSize));
  });
  els.outlineTree.appendChild(fragment);
}

function buildHeadingItem(heading, idx, minSize, maxSize) {
  const item = document.createElement("div");
  item.className = `heading-item ${heading.level.toLowerCase()}`;
  item.setAttribute("role", "treeitem");
  item.style.animationDelay = `${idx * 30}ms`;

  const confClass = `conf-${heading.confidence_label}`;

  // Typography Mirroring: map raw font size to 0.85rem - 1.5rem
  const clampMin = 0.85, clampMax = 1.5;
  let normalizedRem = clampMin;
  if (maxSize > minSize) {
    const ratio = (heading.font_size - minSize) / (maxSize - minSize);
    normalizedRem = clampMin + (ratio * (clampMax - clampMin));
  }

  item.innerHTML = `
    <span class="level-badge">${heading.level}</span>
    <div class="heading-content">
      <div class="heading-text" style="font-size: ${normalizedRem.toFixed(2)}rem; line-height: 1.2; margin-bottom: 6px;">${escapeHtml(heading.text)}</div>
      <div class="heading-meta">
        <span class="meta-tag">p. ${heading.page}</span>
        <span class="meta-tag">${heading.font_name}</span>
        <span class="meta-tag">${heading.font_size.toFixed(1)}pt</span>
      </div>
    </div>
  `;

  // Jump to page & highlight when clicked
  item.addEventListener("click", () => {
    if (state.pdfDoc) {
      queueRenderPage(heading.page, heading.bounding_box);
    }
  });

  return item;
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Copy & Download ───────────────────────────────────────────────────────────

els.copyBtn.addEventListener("click", async () => {
  if (!state.result) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(state.result, null, 2));
    showToast("📋 JSON copied to clipboard", "success");
  } catch {
    showToast("⚠️ Copy failed — try downloading instead", "error");
  }
});

els.downloadBtn.addEventListener("click", () => {
  if (!state.result) return;
  const blob = new Blob(
    [JSON.stringify(state.result, null, 2)],
    { type: "application/json" }
  );
  const filename = (state.result.metadata?.filename ?? "output").replace(/\.pdf$/i, "") + ".json";
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  showToast(`⬇️ Downloaded ${filename}`, "success");
});

// ── Toast ─────────────────────────────────────────────────────────────────────

function showToast(message, type = "success") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  els.toastContainer.appendChild(toast);
  // Remove after animation
  setTimeout(() => toast.remove(), 2600);
}

// ── Keyboard shortcut: Ctrl/Cmd + Enter to extract ───────────────────────────

document.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && !els.extractBtn.disabled) {
    runExtraction();
  }
});

// ── PDF.js Rendering ──────────────────────────────────────────────────────────

async function loadPdfDocument(url) {
  try {
    const loadingTask = pdfjsLib.getDocument(url);
    state.pdfDoc = await loadingTask.promise;
    els.pageCountSpan.textContent = state.pdfDoc.numPages;
    els.viewerControls.hidden = false;
    
    // Default to page 1
    state.pageNum = 1;
    renderPage(state.pageNum, null);
  } catch (error) {
    console.error('Error loading PDF: ', error);
    showToast("❌ Failed to render PDF viewer", "error");
  }
}

async function renderPage(num, boundingBox) {
  state.pageRendering = true;
  els.pageNumSpan.textContent = num;

  try {
    const page = await state.pdfDoc.getPage(num);
    const viewport = page.getViewport({ scale: state.pdfScale });

    // Prepare canvas using PDF page dimensions
    const canvas = els.pdfCanvas;
    const ctx = canvas.getContext('2d');
    canvas.height = viewport.height;
    canvas.width = viewport.width;

    // Render PDF page into canvas context
    const renderContext = {
      canvasContext: ctx,
      viewport: viewport
    };
    await page.render(renderContext).promise;

    // Draw Highlight
    els.pdfHighlightLayer.innerHTML = ''; // clear previous
    if (boundingBox) {
      // bounding_box is [x0, y0, x1, y1] from PyMuPDF
      const [x0, y0, x1, y1] = [boundingBox.x0, boundingBox.y0, boundingBox.x1, boundingBox.y1];
      
      // Convert PyMuPDF coordinates (72dpi) to viewport pixels
      const pt0 = viewport.convertToViewportPoint(x0, y0);
      const pt1 = viewport.convertToViewportPoint(x1, y1);
      
      // Calculate width/height with slight padding for breathing room
      const width = pt1[0] - pt0[0];
      const height = pt1[1] - pt0[1];
      
      const highlight = document.createElement('div');
      highlight.className = 'pdf-highlight';
      
      // Apply coordinates + padding offset (-2px top/left, +4px w/h)
      highlight.style.left = `${pt0[0] - 2}px`;
      highlight.style.top = `${pt0[1] - 2}px`;
      highlight.style.width = `${width + 4}px`;
      highlight.style.height = `${height + 4}px`;
      
      els.pdfHighlightLayer.appendChild(highlight);
      
      // Scroll container so highlight is visible
      highlight.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

  } catch (error) {
    console.error('Error rendering page: ', error);
  } finally {
    state.pageRendering = false;
    if (state.pageNumPending !== null) {
      renderPage(state.pageNumPending, state.highlightPending);
      state.pageNumPending = null;
      state.highlightPending = null;
    }
  }
}

function queueRenderPage(num, boundingBox) {
  if (state.pageRendering) {
    state.pageNumPending = num;
    state.highlightPending = boundingBox;
  } else {
    state.pageNum = num;
    renderPage(num, boundingBox);
  }
}

els.btnPrevPage.addEventListener("click", () => {
  if (state.pageNum <= 1) return;
  state.pageNum--;
  queueRenderPage(state.pageNum, null);
});

els.btnNextPage.addEventListener("click", () => {
  if (state.pageNum >= state.pdfDoc.numPages) return;
  state.pageNum++;
  queueRenderPage(state.pageNum, null);
});
