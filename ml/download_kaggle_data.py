"""
ml/download_kaggle_data.py

Download the Kaggle "dataset-of-pdf-files" and organise PDFs for training.

Uses kagglehub for authenticated download. The downloaded PDFs are
copied into ``ml/kaggle_data/pdfs/`` so the training pipeline can
find them without hard-coding external cache paths.

Usage:
    python -m ml.download_kaggle_data
    python -m ml.download_kaggle_data --max-files 300
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
KAGGLE_DATA_DIR = PROJECT_ROOT / "ml" / "kaggle_data"
KAGGLE_PDF_DIR = KAGGLE_DATA_DIR / "pdfs"


def download_dataset() -> Path:
    """Download the Kaggle dataset and return the local cache path."""
    import kagglehub

    print("Downloading Kaggle dataset 'manisha717/dataset-of-pdf-files'...")
    path = kagglehub.dataset_download("manisha717/dataset-of-pdf-files")
    print(f"  Downloaded to: {path}")
    return Path(path)


def discover_pdfs(root: Path) -> list[Path]:
    """Recursively find all PDF files under a directory."""
    pdfs: list[Path] = []
    for ext in ("*.pdf", "*.PDF"):
        pdfs.extend(root.rglob(ext))
    return sorted(pdfs)


def organise_pdfs(
    source_dir: Path,
    max_files: int = 500,
) -> dict:
    """
    Copy PDFs from the Kaggle cache into ml/kaggle_data/pdfs/.

    Returns stats dict with counts and sizes.
    """
    KAGGLE_PDF_DIR.mkdir(parents=True, exist_ok=True)

    pdfs = discover_pdfs(source_dir)
    pdfs = pdfs[:max_files]

    stats = {
        "total_discovered": len(pdfs),
        "copied": 0,
        "skipped_exists": 0,
        "total_size_mb": 0.0,
        "errors": 0,
    }

    for i, pdf_path in enumerate(pdfs):
        # Use a clean filename: kaggle_001_originalname.pdf
        safe_name = pdf_path.stem.replace(" ", "_")[:60]
        dest_name = f"kaggle_{i+1:03d}_{safe_name}.pdf"
        dest_path = KAGGLE_PDF_DIR / dest_name

        if dest_path.exists():
            stats["skipped_exists"] += 1
            stats["total_size_mb"] += dest_path.stat().st_size / (1024 * 1024)
            continue

        try:
            shutil.copy2(str(pdf_path), str(dest_path))
            stats["copied"] += 1
            stats["total_size_mb"] += dest_path.stat().st_size / (1024 * 1024)
        except Exception as e:
            print(f"  [ERROR] Could not copy {pdf_path.name}: {e}")
            stats["errors"] += 1

        if (i + 1) % 50 == 0:
            print(f"  Organised {i+1}/{len(pdfs)} PDFs...")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Download Kaggle PDF dataset and organise for training."
    )
    parser.add_argument(
        "--max-files", type=int, default=500,
        help="Maximum number of PDFs to copy (default: 500)."
    )
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip download, use existing cache."
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Kaggle PDF Dataset — Download & Organise")
    print("=" * 60)

    if not args.skip_download:
        source_dir = download_dataset()
    else:
        # Try to find existing cache
        source_dir = KAGGLE_PDF_DIR
        if not source_dir.exists():
            print("ERROR: No existing data found. Run without --skip-download.")
            return 1

    all_pdfs = discover_pdfs(source_dir)
    print(f"\n  Found {len(all_pdfs)} PDF files in dataset")

    if not all_pdfs:
        print("  WARNING: No PDFs found in downloaded dataset!")
        return 1

    # Show some stats about the source
    total_size = sum(p.stat().st_size for p in all_pdfs) / (1024 * 1024)
    print(f"  Total size: {total_size:.1f} MB")

    print(f"\n  Organising up to {args.max_files} PDFs into {KAGGLE_PDF_DIR}/...")
    stats = organise_pdfs(source_dir, max_files=args.max_files)

    print(f"\n  Results:")
    print(f"    PDFs discovered:    {stats['total_discovered']}")
    print(f"    PDFs copied:        {stats['copied']}")
    print(f"    Already existed:    {stats['skipped_exists']}")
    print(f"    Errors:             {stats['errors']}")
    print(f"    Total size:         {stats['total_size_mb']:.1f} MB")
    print(f"    Output directory:   {KAGGLE_PDF_DIR}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
