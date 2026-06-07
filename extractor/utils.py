"""
extractor/utils.py

Shared utility functions for the PDF extraction pipeline.

Kept separate from core.py to allow isolated unit testing and
reuse by the evaluation script and CLI without importing the
full ExtractorEngine (which opens a file handle).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


# ── Text normalisation ─────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalise a heading string for display and comparison.

    - Removes duplicate adjacent characters caused by layered PDF rendering
      (e.g. "IInnttrroo" → "Intro")
    - Collapses internal whitespace
    - Strips leading / trailing whitespace
    """
    chars = list(text)
    result: list[str] = []
    prev = ""
    for c in chars:
        if c != prev or not c.isalnum():
            result.append(c)
        prev = c
    return " ".join("".join(result).split())


def normalise_for_comparison(text: str) -> str:
    """
    Produce a normalised key for fuzzy heading comparison.

    Used by the evaluation harness to match predicted headings against
    ground-truth labels despite minor whitespace / capitalisation differences.

    Steps:
        1. Lower-case
        2. Strip punctuation
        3. Collapse whitespace
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


# ── PDF validation helpers ─────────────────────────────────────────────────────

def is_valid_pdf(path: Path) -> bool:
    """
    Quick magic-bytes check: a valid PDF starts with ``%PDF-``.

    This is faster than attempting a full open() and catching errors
    in bulk processing scenarios.
    """
    try:
        with open(path, "rb") as fh:
            header = fh.read(5)
        return header == b"%PDF-"
    except OSError:
        return False


def file_sha256(path: Path) -> str:
    """
    Compute the SHA-256 digest of a file in streaming chunks.

    Used to detect duplicate uploads in the FastAPI layer and to
    provide a deterministic cache key for results.
    """
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


# ── Confidence scoring helpers ─────────────────────────────────────────────────

def describe_confidence(score: float) -> str:
    """
    Human-readable label for a confidence score.

    Tier bands:
        ≥ 0.80  → "high"
        ≥ 0.55  → "medium"
        < 0.55  → "low"

    Used by the Web UI to colour confidence badges.
    """
    if score >= 0.80:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


# ── Heading comparison (for evaluation) ───────────────────────────────────────

def headings_match(predicted: str, ground_truth: str, threshold: float = 0.85) -> bool:
    """
    Determine whether a predicted heading matches a ground-truth heading.

    Uses normalised token-overlap (Jaccard similarity) rather than exact
    string equality, which is more robust to minor OCR / formatting differences.

    Args:
        predicted:    heading text from the extractor
        ground_truth: heading text from the labelled dataset
        threshold:    minimum Jaccard similarity to count as a match

    Returns:
        True if the headings are sufficiently similar.
    """
    p_tokens = set(normalise_for_comparison(predicted).split())
    g_tokens = set(normalise_for_comparison(ground_truth).split())

    if not p_tokens or not g_tokens:
        return predicted.strip().lower() == ground_truth.strip().lower()

    intersection = len(p_tokens & g_tokens)
    union = len(p_tokens | g_tokens)
    jaccard = intersection / union if union > 0 else 0.0
    return jaccard >= threshold


def compute_prf(
    predicted: list[str],
    ground_truth: list[str],
    threshold: float = 0.85,
) -> dict[str, float]:
    """
    Compute Precision, Recall, and F1-Score for a list of headings.

    Each predicted heading is matched against the ground truth list using
    ``headings_match()``. A match is consumed (removed from consideration)
    once used, preventing double-counting.

    Returns:
        dict with keys: "precision", "recall", "f1"
    """
    gt_remaining = list(ground_truth)
    true_positives = 0

    for pred in predicted:
        for i, gt in enumerate(gt_remaining):
            if headings_match(pred, gt, threshold):
                true_positives += 1
                gt_remaining.pop(i)
                break

    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(ground_truth) if ground_truth else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }
