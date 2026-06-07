"""
ml/train_pipeline.py

End-to-end ML training pipeline for the heading classifier.

Steps:
    1. Generate synthetic training PDFs (if not already present)
    2. Extract features from all training PDFs using ExtractorEngine internals
    3. Label each text block by matching against ground truth
    4. Train the RandomForest classifier with cross-validation
    5. Print evaluation metrics (accuracy, F1, confusion matrix)
    6. Save the trained model to ml/models/heading_clf.pkl
    7. Re-evaluate on the original 5 sample PDFs to show improvement

Usage:
    python -m ml.train_pipeline
    python -m ml.train_pipeline --skip-generate   # if PDFs already exist
    python -m ml.train_pipeline --cv 10            # 10-fold cross-validation
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ── Project imports ────────────────────────────────────────────────────────────

from extractor.core import ExtractorEngine, _TextBlock, _CONFIDENCE_THRESHOLD
from extractor.models import HeadingFeatures
from extractor.ml_classifier import (
    HeadingMLClassifier,
    features_to_vector,
    FEATURE_NAMES,
    LEVEL_TO_INT,
    LABEL_MAP,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
TRAINING_DIR = PROJECT_ROOT / "ml" / "training_data"
TRAINING_PDFS = TRAINING_DIR / "pdfs"
TRAINING_GT = TRAINING_DIR / "ground_truth"
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
MODEL_PATH = MODEL_DIR / "heading_clf.pkl"
SAMPLE_PDFS = PROJECT_ROOT / "sample_datasets" / "pdfs"
SAMPLE_GT = PROJECT_ROOT / "sample_datasets" / "outputs"


# ── Feature extraction from a PDF ─────────────────────────────────────────────

def extract_all_blocks_with_features(
    pdf_path: Path, lang: str = "en"
) -> list[tuple[_TextBlock, HeadingFeatures]]:
    """
    Open a PDF, run span extraction and block assembly, compute features
    for every text block (not just heading candidates).

    Returns a list of (block, features) tuples.
    """
    import numpy as np

    with ExtractorEngine(pdf_path, lang=lang) as engine:
        toc_titles = engine._extract_toc()
        spans = engine._extract_spans()
        blocks = engine._build_blocks(spans)

        if not blocks:
            return []

        _, body_blocks = engine._detect_title(blocks)
        if not body_blocks:
            body_blocks = blocks

        _, body_size = engine._cluster_font_sizes(body_blocks)

        # Precompute font statistics for z-score feature
        sizes = [b.font_size for b in body_blocks]
        font_stats = (float(np.mean(sizes)), float(np.std(sizes))) if sizes else (0.0, 1.0)

        result = []
        prev_block = None
        for block in body_blocks:
            features = engine._build_features(
                block, body_size, toc_titles,
                prev_block=prev_block, font_stats=font_stats,
            )
            result.append((block, features))
            prev_block = block

    return result


def match_block_to_gt(
    block: _TextBlock,
    gt_headings: list[dict],
    threshold: float = 0.45,
) -> str | None:
    """
    Try to match a text block to a ground-truth heading entry.

    Uses normalised text overlap (Jaccard on words) and page matching.
    Returns the GT level (H1/H2/H3) or None if no match.
    """
    block_words = set(block.text.strip().lower().split())
    if not block_words:
        return None

    best_score = 0.0
    best_level = None

    for gt in gt_headings:
        gt_words = set(gt["text"].strip().lower().split())
        if not gt_words:
            continue

        # Jaccard similarity
        intersection = len(block_words & gt_words)
        union = len(block_words | gt_words)
        jaccard = intersection / union if union > 0 else 0.0

        # Page proximity bonus (same page or adjacent)
        page_match = abs(block.page - gt["page"]) <= 1

        score = jaccard * (1.0 if page_match else 0.7)

        if score > best_score:
            best_score = score
            best_level = gt["level"]

    if best_score >= threshold:
        return best_level
    return None


# ── Dataset construction ───────────────────────────────────────────────────────

def build_dataset(
    pdf_dir: Path,
    gt_dir: Path,
    label: str = "training",
) -> tuple[list[list[float]], list[int], dict[str, int]]:
    """
    Build feature matrix X and label vector y from a directory of PDFs
    with matching ground-truth JSONs.

    Returns:
        X: list of feature vectors
        y: list of integer labels (0=body, 1=H1, 2=H2, 3=H3)
        stats: count per class
    """
    X: list[list[float]] = []
    y: list[int] = []
    stats: dict[str, int] = {"body": 0, "H1": 0, "H2": 0, "H3": 0}

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"  Warning: No PDFs found in {pdf_dir}")
        return X, y, stats

    for pdf_path in pdfs:
        gt_path = gt_dir / f"{pdf_path.stem}.json"
        if not gt_path.exists():
            continue

        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)

        gt_headings = gt.get("outline", [])

        try:
            blocks_features = extract_all_blocks_with_features(pdf_path)
        except Exception as e:
            print(f"  [ERROR] {pdf_path.name}: {e}")
            continue

        matched_count = 0
        for block, features in blocks_features:
            vec = features_to_vector(features)
            level = match_block_to_gt(block, gt_headings)

            if level is not None:
                label_int = LEVEL_TO_INT.get(level, 0)
                matched_count += 1
            else:
                label_int = 0  # body

            X.append(vec)
            y.append(label_int)
            class_name = LABEL_MAP.get(label_int) or "body"
            stats[class_name] = stats.get(class_name, 0) + 1

        print(f"  {pdf_path.name}: {len(blocks_features)} blocks, "
              f"{matched_count} matched to GT headings")

    return X, y, stats


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the heading ML classifier."
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip synthetic PDF generation (use existing files)."
    )
    parser.add_argument(
        "--cv", type=int, default=5,
        help="Number of cross-validation folds (default: 5)."
    )
    parser.add_argument(
        "--n-estimators", type=int, default=300,
        help="Number of trees in the Random Forest (default: 300)."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  PDF Heading Extractor -- ML Training Pipeline")
    print("=" * 70)

    # ── Step 1: Generate synthetic PDFs ────────────────────────────────────────
    if not args.skip_generate:
        print("\n[Step 1] Generating synthetic training PDFs...")
        from ml.generate_pdfs import main as gen_main
        gen_main()
    else:
        print("\n[Step 1] Skipping generation (--skip-generate)")

    # ── Step 2: Build training dataset ────────────────────────────────────────
    print("\n[Step 2] Building training dataset from synthetic PDFs...")
    X_synth, y_synth, stats_synth = build_dataset(
        TRAINING_PDFS, TRAINING_GT, label="synthetic"
    )
    print(f"  Synthetic dataset: {len(X_synth)} samples")
    print(f"  Class distribution: {stats_synth}")

    # Also include real sample PDFs in training
    print("\n  Adding real sample PDFs to training set...")
    X_real, y_real, stats_real = build_dataset(
        SAMPLE_PDFS, SAMPLE_GT, label="real"
    )
    print(f"  Real dataset: {len(X_real)} samples")
    print(f"  Class distribution: {stats_real}")

    # Combine datasets
    X = X_synth + X_real
    y = y_synth + y_real
    total_stats = {
        k: stats_synth.get(k, 0) + stats_real.get(k, 0)
        for k in ["body", "H1", "H2", "H3"]
    }
    print(f"\n  Combined dataset: {len(X)} samples")
    print(f"  Combined distribution: {total_stats}")

    if len(X) < 20:
        print("\n  ERROR: Not enough training samples. Aborting.")
        return 1

    # ── Step 3: Cross-validation ──────────────────────────────────────────────
    print(f"\n[Step 3] Running {args.cv}-fold stratified cross-validation...")
    clf = HeadingMLClassifier()
    cv_report = clf.cross_validate_report(X, y, cv=args.cv)

    print(f"\n  Cross-Validation Results ({args.cv} folds):")
    print(f"    Accuracy:   {cv_report['accuracy_mean']:.4f} "
          f"(+/- {cv_report['accuracy_std']:.4f})")
    print(f"    F1 (macro): {cv_report['f1_macro_mean']:.4f} "
          f"(+/- {cv_report['f1_macro_std']:.4f})")
    print(f"\n  Per-class F1:")
    for cls, f1 in cv_report["per_class_f1"].items():
        print(f"    {cls:>6}: {f1:.4f}")

    print(f"\n  Confusion Matrix:")
    labels = cv_report["confusion_matrix_labels"]
    header = "          " + "  ".join(f"{l:>7}" for l in labels)
    print(header)
    for i, row in enumerate(cv_report["confusion_matrix"]):
        row_str = "  ".join(f"{v:>7}" for v in row)
        print(f"    {labels[i]:>6}  {row_str}")

    # ── Step 4: Train final model on ALL data ─────────────────────────────────
    print(f"\n[Step 4] Training final model on all {len(X)} samples "
          f"(n_estimators={args.n_estimators})...")
    start = time.perf_counter()
    clf.train(X, y, n_estimators=args.n_estimators)
    elapsed = time.perf_counter() - start
    print(f"  Training completed in {elapsed:.2f}s")

    # Feature importances
    print(f"\n  Feature Importances:")
    for entry in clf.feature_importance_report():
        bar = "#" * int(entry["importance"] * 50)
        print(f"    {entry['feature']:>20}: {entry['importance']:.4f}  {bar}")

    # ── Step 5: Save model ────────────────────────────────────────────────────
    clf.save(MODEL_PATH)
    print(f"\n[Step 5] Model saved to: {MODEL_PATH}")
    print(f"  Model file size: {MODEL_PATH.stat().st_size / 1024:.1f} KB")

    # ── Step 6: Evaluate on synthetic test set ────────────────────────────────
    print("\n[Step 6] Quick evaluation on training data (in-sample)...")
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    preds = clf._pipeline.predict(X_arr)
    in_sample_acc = float(np.mean(preds == y_arr))
    print(f"  In-sample accuracy: {in_sample_acc:.4f}")

    # Per-class breakdown
    from sklearn.metrics import classification_report
    target_names = ["body", "H1", "H2", "H3"]
    present_labels = sorted(set(y))
    present_names = [target_names[i] for i in present_labels]
    report = classification_report(
        y_arr, preds,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    )
    print(report)

    print("=" * 70)
    print("  Training pipeline complete!")
    print(f"  Model ready at: {MODEL_PATH}")
    print("  Run `python evaluate.py --use-ml` to evaluate with ML.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
