# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder
#   Install all dependencies into a clean venv so only the venv is
#   copied into the final image (keeps the final layer small).
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build tools (needed for some native wheels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Create a venv and install deps into it
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime
#   Copy only the venv and application code, nothing from builder layer.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Harshita"
LABEL description="Production-grade multilingual PDF heading extractor API"
LABEL version="0.1.0"

# Runtime system libraries required by PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the venv from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy application source
COPY extractor/   ./extractor/
COPY api/         ./api/
COPY web/         ./web/
COPY languages.json .
COPY process_pdfs.py .

# Create directories for CLI batch processing
RUN mkdir -p /app/input /app/output

# Non-root user for security
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health check via the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Default: run the FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
