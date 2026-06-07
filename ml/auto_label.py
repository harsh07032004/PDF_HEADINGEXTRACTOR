"""
ml/auto_label.py

Auto-labeling pipeline for real-world PDFs.

Strategy (self-supervised):
    Real PDFs often have an embedded Table of Contents (TOC) stored as
    PDF bookmarks. We treat these TOC entries as "silver-standard" labels:
        - TOC level 1 -> H1
        - TOC level 2 -> H2
        - TOC level 3 -> H3

    For each text block in the PDF, we check whether it matches any TOC
    entry (by fuzzy text similarity + page proximity). If it does, we
    label it as the corresponding heading level. Otherwise it's body text.

    This lets us train on hundreds of real PDFs without manual labeling.

Usage:
    python -m ml.auto_label --pdf-dir path/to/pdfs --out ml/kaggle_labels
    python -m ml.auto_label --pdf-dir path/to/pdfs --out ml/kaggle_labels --min-toc 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF


def extract_toc_as_ground_truth(pdf_path: Path) -> dict | None:
    """
    Open a PDF and extract its embedded TOC (bookmarks) as ground truth.

    Returns None if the PDF has fewer than min_entries TOC entries,
    meaning it's not useful for training.

    Returns:
        {
            "title": "...",
            "outline": [
                {"level": "H1", "text": "...", "page": 1},
                ...
            ],
            "toc_count": 15,
        }
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception:
        return None

    try:
        toc = doc.get_toc(simple=True)
        if not toc:
            doc.close()
            return None

        # Extract title from metadata or first page largest text
        title = ""
        meta = doc.metadata
        if meta and meta.get("title"):
            title = meta["title"].strip()

        if not title and doc.page_count > 0:
            # Fallback: largest font text on page 1
            page = doc.load_page(0)
            blocks = page.get_text("dict").get("blocks", [])
            max_size = 0
            for block in blocks:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if span.get("size", 0) > max_size:
                            text = span.get("text", "").strip()
                            if len(text) > 3:
                                max_size = span["size"]
                                title = text

        # Convert TOC entries to our heading format
        outline = []
        for entry in toc:
            if len(entry) < 3:
                continue
            toc_level, toc_title, toc_page = entry[0], entry[1], entry[2]

            # Map TOC depth to our heading levels
            if toc_level == 1:
                level = "H1"
            elif toc_level == 2:
                level = "H2"
            elif toc_level >= 3:
                level = "H3"
            else:
                continue

            text = toc_title.strip()
            if not text or len(text) < 2:
                continue

            outline.append({
                "level": level,
                "text": text,
                "page": max(1, toc_page),  # ensure positive page
            })

        doc.close()

        if not outline:
            return None

        return {
            "title": title,
            "outline": outline,
            "toc_count": len(outline),
        }

    except Exception:
        doc.close()
        return None


def process_pdf_directory(
    pdf_dir: Path,
    out_dir: Path,
    min_toc: int = 3,
    max_files: int = 200,
) -> dict:
    """
    Scan a directory for PDFs, extract TOC-based ground truth, save as JSONs.

    Returns stats dict.
    """
    gt_dir = out_dir / "ground_truth"
    gt_dir.mkdir(parents=True, exist_ok=True)

    # Also create a symlink/copy-free reference to the PDF directory
    pdf_index_file = out_dir / "pdf_index.json"

    stats = {
        "total_pdfs": 0,
        "with_toc": 0,
        "usable": 0,
        "skipped_few_toc": 0,
        "skipped_error": 0,
        "total_headings": 0,
    }

    pdf_files = []
    for ext in ("*.pdf", "*.PDF"):
        pdf_files.extend(pdf_dir.rglob(ext))
    pdf_files = sorted(pdf_files)[:max_files]

    stats["total_pdfs"] = len(pdf_files)
    pdf_index = []

    for i, pdf_path in enumerate(pdf_files):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(pdf_files)}...")

        result = extract_toc_as_ground_truth(pdf_path)

        if result is None:
            stats["skipped_error"] += 1
            continue

        stats["with_toc"] += 1

        if result["toc_count"] < min_toc:
            stats["skipped_few_toc"] += 1
            continue

        stats["usable"] += 1
        stats["total_headings"] += result["toc_count"]

        # Save ground truth
        safe_name = pdf_path.stem.replace(" ", "_")[:50]
        gt_filename = f"kaggle_{stats['usable']:03d}_{safe_name}.json"
        gt_path = gt_dir / gt_filename

        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump({
                "title": result["title"],
                "outline": result["outline"],
            }, f, indent=2, ensure_ascii=False)

        pdf_index.append({
            "pdf_path": str(pdf_path.resolve()),
            "gt_file": gt_filename,
            "stem": f"kaggle_{stats['usable']:03d}_{safe_name}",
            "toc_entries": result["toc_count"],
        })

    # Save PDF index
    with open(pdf_index_file, "w", encoding="utf-8") as f:
        json.dump(pdf_index, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Auto-label PDFs using their embedded TOC."
    )
    parser.add_argument(
        "--pdf-dir", type=Path, required=True,
        help="Directory containing PDF files to process."
    )
    parser.add_argument(
        "--out", type=Path, default=Path("ml/kaggle_labels"),
        help="Output directory for ground-truth JSONs."
    )
    parser.add_argument(
        "--min-toc", type=int, default=3,
        help="Minimum TOC entries for a PDF to be usable (default: 3)."
    )
    parser.add_argument(
        "--max-files", type=int, default=200,
        help="Maximum number of PDFs to process (default: 200)."
    )
    args = parser.parse_args()

    print(f"Auto-labeling PDFs from: {args.pdf_dir}")
    print(f"Output: {args.out}")
    print(f"Min TOC entries: {args.min_toc}")
    print()

    stats = process_pdf_directory(
        args.pdf_dir, args.out,
        min_toc=args.min_toc,
        max_files=args.max_files,
    )

    print(f"\n  Auto-Labeling Results:")
    print(f"    Total PDFs scanned:     {stats['total_pdfs']}")
    print(f"    PDFs with TOC:          {stats['with_toc']}")
    print(f"    Usable (>= {args.min_toc} entries): {stats['usable']}")
    print(f"    Skipped (few entries):  {stats['skipped_few_toc']}")
    print(f"    Skipped (error):        {stats['skipped_error']}")
    print(f"    Total heading labels:   {stats['total_headings']}")
    print(f"\n  Ground truth saved to: {args.out / 'ground_truth'}")


if __name__ == "__main__":
    main()
