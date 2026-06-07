"""
extractor/cli.py

Command-line interface for the PDF heading extractor.

This module is the entry point registered in pyproject.toml:

    [project.scripts]
    extract-pdf = "extractor.cli:main"

After `pip install -e .`, you can run:

    extract-pdf input/ output/ --lang en
    extract-pdf --help
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from extractor.core import ExtractorEngine
from extractor.utils import is_valid_pdf


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extract-pdf",
        description=(
            "Production-grade multilingual PDF heading extractor.\n\n"
            "Processes every *.pdf in INPUT_DIR and writes a JSON outline\n"
            "for each file to OUTPUT_DIR."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  extract-pdf ./input ./output\n"
            "  extract-pdf ./input ./output --lang es\n"
            "  extract-pdf ./input ./output --lang en --min-confidence 0.6\n"
        ),
    )
    parser.add_argument(
        "input_dir",
        metavar="INPUT_DIR",
        type=Path,
        help="Directory containing PDF files to process.",
    )
    parser.add_argument(
        "output_dir",
        metavar="OUTPUT_DIR",
        type=Path,
        help="Directory where JSON output files will be written.",
    )
    parser.add_argument(
        "--lang",
        default="en",
        metavar="LANG",
        help="Language code for heading pattern matching (default: en).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.45,
        dest="min_confidence",
        metavar="SCORE",
        help=(
            "Minimum confidence threshold for including a heading "
            "in the output (0.0–1.0, default: 0.45)."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: true).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-heading details during processing.",
    )
    return parser


def process_single(
    pdf_path: Path,
    output_dir: Path,
    lang: str,
    min_confidence: float,
    pretty: bool,
    verbose: bool,
) -> bool:
    """
    Extract headings from ``pdf_path`` and write JSON to ``output_dir``.

    Returns True on success, False on any handled error.
    """
    if not is_valid_pdf(pdf_path):
        print(f"  [SKIP] {pdf_path.name} — not a valid PDF file.", file=sys.stderr)
        return False

    try:
        with ExtractorEngine(pdf_path, lang=lang) as engine:
            outline = engine.process()
    except Exception as exc:
        print(f"  [ERROR] {pdf_path.name} — {exc}", file=sys.stderr)
        return False

    # Apply confidence filter
    outline.headings = [
        h for h in outline.headings if h.confidence >= min_confidence
    ]

    if outline.metadata.scanned_pdf_detected:
        print(f"  [WARN] {pdf_path.name} — scanned PDF detected, text may be sparse.")

    result = outline.to_dict()
    out_path = output_dir / f"{pdf_path.stem}.json"
    indent = 2 if pretty else None
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=indent)

    if verbose:
        for h in outline.headings:
            print(
                f"    [{h.level}] p{h.page:02d} conf={h.confidence:.2f}  {h.text}"
            )

    n_headings = len(outline.headings)
    ms = outline.metadata.processing_time_ms
    print(f"  ✓ {pdf_path.name}  →  {n_headings} headings  ({ms:.0f} ms)")
    return True


def main(argv: list[str] | None = None) -> int:
    """
    Entry point for the CLI.  Returns an exit code (0 = success).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.is_dir():
        print(f"Error: INPUT_DIR '{input_dir}' does not exist.", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in '{input_dir}'.", file=sys.stderr)
        return 1

    print(f"Processing {len(pdfs)} PDF(s)  [lang={args.lang}]\n")
    successes = 0
    for pdf in pdfs:
        print(f"→ {pdf.name}")
        ok = process_single(
            pdf_path=pdf,
            output_dir=output_dir,
            lang=args.lang,
            min_confidence=args.min_confidence,
            pretty=args.pretty,
            verbose=args.verbose,
        )
        if ok:
            successes += 1

    print(f"\nDone — {successes}/{len(pdfs)} files processed successfully.")
    return 0 if successes == len(pdfs) else 1


if __name__ == "__main__":
    sys.exit(main())
