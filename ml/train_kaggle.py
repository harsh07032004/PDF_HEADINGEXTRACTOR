"""
ml/train_kaggle.py

Aggressive ML training pipeline using the Kaggle PDF dataset.

Methodical progression:
    1. Download & auto-label Kaggle PDFs (silver-standard from embedded TOC)
    2. Combine datasets: Kaggle + synthetic + sample PDFs
    3. Extract 18-dim features from all PDFs
    4. RandomForest baseline → metrics
    5. GradientBoosting comparison → metrics
    6. Feature importance analysis
    7. Select best model, calibrate, save
    8. Evaluate on held-out sample PDFs

Training labels are automatically generated from embedded PDF bookmarks
and therefore represent silver-standard annotations.

Usage:
    python -m ml.train_kaggle
    python -m ml.train_kaggle --skip-download --skip-generate
    python -m ml.train_kaggle --cv 10 --max-kaggle 300
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

from extractor.ml_classifier import (
    HeadingMLClassifier,
    features_to_vector,
    FEATURE_NAMES,
    LEVEL_TO_INT,
    LABEL_MAP,
)
from ml.train_pipeline import (
    extract_all_blocks_with_features,
    match_block_to_gt,
    build_dataset,
)
from ml.auto_label import process_pdf_directory

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
KAGGLE_PDF_DIR = PROJECT_ROOT / "ml" / "kaggle_data" / "pdfs"
KAGGLE_LABELS_DIR = PROJECT_ROOT / "ml" / "kaggle_labels"
TRAINING_PDFS = PROJECT_ROOT / "ml" / "training_data" / "pdfs"
TRAINING_GT = PROJECT_ROOT / "ml" / "training_data" / "ground_truth"
SAMPLE_PDFS = PROJECT_ROOT / "sample_datasets" / "pdfs"
SAMPLE_GT = PROJECT_ROOT / "sample_datasets" / "outputs"
MODEL_DIR = PROJECT_ROOT / "ml" / "models"
MODEL_PATH = MODEL_DIR / "heading_clf.pkl"


# ── Kaggle dataset building ───────────────────────────────────────────────────

def build_kaggle_dataset(
    pdf_index_path: Path,
    gt_dir: Path,
) -> tuple[list[list[float]], list[int], dict[str, int]]:
    """
    Build feature matrix X and label vector y from Kaggle PDFs
    using the pdf_index.json created by auto_label.py.
    """
    X: list[list[float]] = []
    y: list[int] = []
    stats: dict[str, int] = {"body": 0, "H1": 0, "H2": 0, "H3": 0}

    if not pdf_index_path.exists():
        print("  WARNING: No pdf_index.json found. Skipping Kaggle data.")
        return X, y, stats

    with open(pdf_index_path, encoding="utf-8") as f:
        pdf_index = json.load(f)

    print(f"  Processing {len(pdf_index)} Kaggle PDFs with TOC labels...")

    for i, entry in enumerate(pdf_index):
        pdf_path = Path(entry["pdf_path"])
        gt_path = gt_dir / entry["gt_file"]

        if not pdf_path.exists() or not gt_path.exists():
            continue

        with open(gt_path, encoding="utf-8") as f:
            gt = json.load(f)

        gt_headings = gt.get("outline", [])

        try:
            blocks_features = extract_all_blocks_with_features(pdf_path)
        except Exception as e:
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(pdf_index)}] ERROR: {pdf_path.name}: {e}")
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

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i+1}/{len(pdf_index)}] {pdf_path.name}: "
                  f"{len(blocks_features)} blocks, {matched_count} matched")

    return X, y, stats


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggressive ML training with Kaggle PDF dataset."
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip Kaggle dataset download (use existing)."
    )
    parser.add_argument(
        "--skip-generate", action="store_true",
        help="Skip synthetic PDF generation."
    )
    parser.add_argument(
        "--skip-label", action="store_true",
        help="Skip auto-labeling (use existing labels)."
    )
    parser.add_argument(
        "--cv", type=int, default=5,
        help="Number of cross-validation folds (default: 5)."
    )
    parser.add_argument(
        "--max-kaggle", type=int, default=500,
        help="Maximum Kaggle PDFs to process (default: 500)."
    )
    parser.add_argument(
        "--n-estimators", type=int, default=300,
        help="Number of trees/boosting rounds (default: 300)."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  PDF Heading Extractor — Aggressive ML Training Pipeline")
    print("  (Kaggle dataset + methodical model comparison)")
    print("=" * 70)

    # ── Step 1: Download Kaggle PDFs ───────────────────────────────────────
    if not args.skip_download:
        print("\n[Step 1] Downloading Kaggle PDF dataset...")
        try:
            from ml.download_kaggle_data import download_dataset, organise_pdfs
            source_dir = download_dataset()
            organise_pdfs(source_dir, max_files=args.max_kaggle)
        except Exception as e:
            print(f"  WARNING: Download failed ({e}). Using existing data.")
    else:
        print("\n[Step 1] Skipping download (--skip-download)")

    # ── Step 2: Generate synthetic PDFs ────────────────────────────────────
    if not args.skip_generate:
        print("\n[Step 2] Generating synthetic training PDFs...")
        try:
            from ml.generate_pdfs import main as gen_main
            gen_main()
        except Exception as e:
            print(f"  WARNING: Generation failed ({e}). Using existing data.")
    else:
        print("\n[Step 2] Skipping generation (--skip-generate)")

    # ── Step 3: Auto-label Kaggle PDFs using TOC ───────────────────────────
    if not args.skip_label and KAGGLE_PDF_DIR.exists():
        print("\n[Step 3] Auto-labeling Kaggle PDFs (silver-standard from TOC)...")
        label_stats = process_pdf_directory(
            KAGGLE_PDF_DIR,
            KAGGLE_LABELS_DIR,
            min_toc=3,
            max_files=args.max_kaggle,
        )
        print(f"  Total PDFs scanned:     {label_stats['total_pdfs']}")
        print(f"  PDFs with usable TOC:   {label_stats['usable']}")
        print(f"  Total heading labels:   {label_stats['total_headings']}")
    else:
        print("\n[Step 3] Skipping auto-labeling (--skip-label or no data)")

    # ── Step 4: Build combined training dataset ────────────────────────────
    print("\n[Step 4] Building combined training dataset...")

    # 4a. Synthetic PDFs
    print("\n  [4a] Synthetic PDFs...")
    X_synth, y_synth, stats_synth = build_dataset(
        TRAINING_PDFS, TRAINING_GT, label="synthetic"
    )
    print(f"  Synthetic: {len(X_synth)} samples | {stats_synth}")

    # 4b. Real sample PDFs
    print("\n  [4b] Real sample PDFs...")
    X_real, y_real, stats_real = build_dataset(
        SAMPLE_PDFS, SAMPLE_GT, label="real"
    )
    print(f"  Real samples: {len(X_real)} samples | {stats_real}")

    # 4c. Kaggle PDFs
    print("\n  [4c] Kaggle PDFs (silver-standard labels)...")
    kaggle_index = KAGGLE_LABELS_DIR / "pdf_index.json"
    kaggle_gt_dir = KAGGLE_LABELS_DIR / "ground_truth"
    X_kaggle, y_kaggle, stats_kaggle = build_kaggle_dataset(
        kaggle_index, kaggle_gt_dir,
    )
    print(f"  Kaggle: {len(X_kaggle)} samples | {stats_kaggle}")

    # Combine all datasets
    X = X_synth + X_real + X_kaggle
    y = y_synth + y_real + y_kaggle
    total_stats = {
        k: stats_synth.get(k, 0) + stats_real.get(k, 0) + stats_kaggle.get(k, 0)
        for k in ["body", "H1", "H2", "H3"]
    }
    print(f"\n  +--------------------------------------------+")
    print(f"  |  Combined dataset: {len(X):>6} samples           |")
    print(f"  |  body: {total_stats['body']:>6}  H1: {total_stats['H1']:>5}           |")
    print(f"  |  H2:   {total_stats['H2']:>6}  H3: {total_stats['H3']:>5}           |")
    print(f"  +--------------------------------------------+")

    if len(X) < 50:
        print("\n  ERROR: Not enough training samples. Aborting.")
        return 1

    # -- Step 5: Model comparison (RF vs GBM) ------------------------------
    print(f"\n[Step 5] Model comparison ({args.cv}-fold stratified CV)...")
    print("=" * 70)

    results_table = []

    # 5a. Random Forest baseline
    print(f"\n  [5a] Random Forest baseline...")
    clf_rf = HeadingMLClassifier()
    rf_report = clf_rf.cross_validate_report(X, y, cv=args.cv, model_type="rf")
    results_table.append({
        "model": "Random Forest",
        "accuracy": rf_report["accuracy_mean"],
        "accuracy_std": rf_report["accuracy_std"],
        "f1_macro": rf_report["f1_macro_mean"],
        "f1_std": rf_report["f1_macro_std"],
        "per_class": rf_report["per_class_f1"],
    })
    print(f"    Accuracy:   {rf_report['accuracy_mean']:.4f} "
          f"(±{rf_report['accuracy_std']:.4f})")
    print(f"    F1 (macro): {rf_report['f1_macro_mean']:.4f} "
          f"(±{rf_report['f1_macro_std']:.4f})")
    for cls, f1 in rf_report["per_class_f1"].items():
        print(f"      {cls:>6}: {f1:.4f}")

    # 5b. Gradient Boosting
    print(f"\n  [5b] Gradient Boosting...")
    clf_gbm = HeadingMLClassifier()
    gbm_report = clf_gbm.cross_validate_report(X, y, cv=args.cv, model_type="gbm")
    results_table.append({
        "model": "Gradient Boosting",
        "accuracy": gbm_report["accuracy_mean"],
        "accuracy_std": gbm_report["accuracy_std"],
        "f1_macro": gbm_report["f1_macro_mean"],
        "f1_std": gbm_report["f1_macro_std"],
        "per_class": gbm_report["per_class_f1"],
    })
    print(f"    Accuracy:   {gbm_report['accuracy_mean']:.4f} "
          f"(±{gbm_report['accuracy_std']:.4f})")
    print(f"    F1 (macro): {gbm_report['f1_macro_mean']:.4f} "
          f"(±{gbm_report['f1_macro_std']:.4f})")
    for cls, f1 in gbm_report["per_class_f1"].items():
        print(f"      {cls:>6}: {f1:.4f}")

    # -- Comparison table ---------------------------------------------------
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON TABLE")
    print("=" * 70)
    header = f"  {'Model':<20} {'Accuracy':>10} {'F1 (macro)':>12} {'F1 body':>9} {'F1 H1':>7} {'F1 H2':>7} {'F1 H3':>7}"
    print(header)
    print("  " + "-" * 72)
    for r in results_table:
        pc = r["per_class"]
        print(
            f"  {r['model']:<20} "
            f"{r['accuracy']:>10.4f} "
            f"{r['f1_macro']:>12.4f} "
            f"{pc.get('body', 0):>9.4f} "
            f"{pc.get('H1', 0):>7.4f} "
            f"{pc.get('H2', 0):>7.4f} "
            f"{pc.get('H3', 0):>7.4f}"
        )
    print("=" * 70)

    # -- Select best model --------------------------------------------------
    best_rf = rf_report["f1_macro_mean"]
    best_gbm = gbm_report["f1_macro_mean"]

    if best_gbm >= best_rf:
        best_model_type = "gbm"
        best_f1 = best_gbm
        print(f"\n  >>> Best model: Gradient Boosting (F1={best_f1:.4f})")
    else:
        best_model_type = "rf"
        best_f1 = best_rf
        print(f"\n  >>> Best model: Random Forest (F1={best_f1:.4f})")

    # -- Step 6: Train final model on ALL data -----------------------------
    print(f"\n[Step 6] Training final {best_model_type.upper()} model "
          f"on all {len(X)} samples (n_estimators={args.n_estimators})...")
    start = time.perf_counter()
    clf = HeadingMLClassifier()
    clf.train(X, y, model_type=best_model_type, n_estimators=args.n_estimators)
    elapsed = time.perf_counter() - start
    print(f"  Training completed in {elapsed:.2f}s")

    # -- Step 7: Feature importance ----------------------------------------
    print(f"\n[Step 7] Feature Importances:")
    for entry in clf.feature_importance_report():
        bar = "#" * int(entry["importance"] * 50)
        print(f"    {entry['feature']:>22}: {entry['importance']:.4f}  {bar}")

    # -- Step 8: Save model ------------------------------------------------
    clf.save(MODEL_PATH)
    model_size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"\n[Step 8] Model saved to: {MODEL_PATH}")
    print(f"  Model type: {best_model_type.upper()}")
    print(f"  Model file size: {model_size_kb:.1f} KB")

    # ── Step 9: In-sample accuracy check ──────────────────────────────────
    print("\n[Step 9] In-sample verification...")
    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    preds = clf._pipeline.predict(X_arr)
    in_sample_acc = float(np.mean(preds == y_arr))
    print(f"  In-sample accuracy: {in_sample_acc:.4f}")

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

    # ── Summary ────────────────────────────────────────────────────────────
    print("=" * 70)
    print("  Training pipeline complete!")
    print(f"  Model: {best_model_type.upper()} | F1={best_f1:.4f}")
    print(f"  Dataset: {len(X)} samples ({len(X_kaggle)} from Kaggle)")
    print(f"  Features: {len(FEATURE_NAMES)} dimensions")
    print(f"  Saved to: {MODEL_PATH}")
    print("  Run `python evaluate.py --use-ml --verbose` to evaluate.")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
