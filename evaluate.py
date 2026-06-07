#!/usr/bin/env python3
"""
evaluate.py — Evaluation harness for the PDF heading extractor.

Compares the extractor's output against ground-truth JSON files and
reports Precision, Recall, and F1-Score per heading level (H1/H2/H3)
plus overall, and title extraction quality (exact / fuzzy match).

Usage:
    python evaluate.py                              # uses sample_datasets/
    python evaluate.py --pdfs path/to/pdfs --gts path/to/gts
    python evaluate.py --output eval_report.json   # save full report

Ground-truth format (matches the output JSON schema):
    {
      "title": "Document Title",
      "outline": [
        {"level": "H1", "text": "Introduction", "page": 1},
        ...
      ]
    }

Output:
    ┌─────────────────────────────────────────────────────────────┐
    │  PDF Heading Extractor — Evaluation Report                  │
    │  5 documents evaluated                                      │
    ├────────┬───────────┬────────┬────────┬────────┬────────────┤
    │ Level  │ Precision │ Recall │   F1   │   TP   │  GT Total  │
    ├────────┼───────────┼────────┼────────┼────────┼────────────┤
    │   H1   │   0.9000  │ 0.8182 │ 0.8571 │   9    │     11     │
    │   H2   │   0.7500  │ 0.7059 │ 0.7273 │  24    │     34     │
    │   H3   │   0.6667  │ 0.6667 │ 0.6667 │   4    │      6     │
    │ Overall│   0.7843  │ 0.7451 │ 0.7642 │  37    │     51     │
    └────────┴───────────┴────────┴────────┴────────┴────────────┘
    Title match:  exact=3/5  fuzzy=4/5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from extractor.core import ExtractorEngine
from extractor.utils import compute_prf, headings_match, normalise_for_comparison

# ── Defaults ───────────────────────────────────────────────────────────────────

DEFAULT_PDFS = Path("sample_datasets/pdfs")
DEFAULT_GTS  = Path("sample_datasets/outputs")
LEVELS       = ["H1", "H2", "H3"]


# ── Per-file evaluation ────────────────────────────────────────────────────────

def evaluate_file(
    pdf_path: Path,
    gt_path: Path,
    lang: str,
    min_confidence: float,
    verbose: bool,
    use_ml: bool = False,
) -> dict:
    """
    Evaluate a single PDF against its ground truth.

    Returns a dict with keys:
        filename, title_exact, title_fuzzy, level_metrics
    """
    # Load ground truth
    with open(gt_path, encoding="utf-8") as fh:
        gt = json.load(fh)

    gt_title: str = gt.get("title", "")
    gt_outline: list[dict] = gt.get("outline", [])

    # Run extractor
    try:
        with ExtractorEngine(pdf_path, lang=lang, use_ml=use_ml) as engine:
            result = engine.process()
    except Exception as exc:
        print(f"  [ERROR] {pdf_path.name}: {exc}", file=sys.stderr)
        return {
            "filename": pdf_path.name,
            "error": str(exc),
            "title_exact": False,
            "title_fuzzy": False,
            "level_metrics": {},
        }

    pred_headings = [h for h in result.headings if h.confidence >= min_confidence]

    # Title evaluation
    pred_title = result.title
    title_exact = normalise_for_comparison(pred_title) == normalise_for_comparison(gt_title)
    title_fuzzy = headings_match(pred_title, gt_title, threshold=0.75) if gt_title else False

    if verbose:
        print(f"\n  {pdf_path.name}")
        print(f"    Title pred:  {pred_title!r}")
        print(f"    Title truth: {gt_title!r}")
        print(f"    Exact={title_exact}  Fuzzy={title_fuzzy}")

    # Per-level metrics
    level_metrics: dict[str, dict] = {}
    for level in LEVELS:
        pred_texts = [h.text for h in pred_headings if h.level == level]
        gt_texts   = [e["text"] for e in gt_outline if e.get("level") == level]

        if not gt_texts and not pred_texts:
            # Skip levels with no data in either set
            continue

        metrics = compute_prf(pred_texts, gt_texts)
        metrics["tp"] = round(metrics["precision"] * len(pred_texts))
        metrics["pred_count"] = len(pred_texts)
        metrics["gt_count"] = len(gt_texts)
        level_metrics[level] = metrics

        if verbose:
            print(
                f"    {level}: P={metrics['precision']:.3f}  "
                f"R={metrics['recall']:.3f}  F1={metrics['f1']:.3f}  "
                f"({metrics['tp']}/{len(gt_texts)} GT)"
            )

    return {
        "filename": pdf_path.name,
        "title_exact": title_exact,
        "title_fuzzy": title_fuzzy,
        "level_metrics": level_metrics,
    }


# ── Aggregate across files ─────────────────────────────────────────────────────

def aggregate(per_file: list[dict]) -> dict:
    """
    Macro-average metrics across all files and levels.

    Uses macro-averaging: compute P/R/F1 per level per file, then average.
    Also computes micro-aggregate by summing TP and GT counts.
    """
    level_accum: dict[str, dict[str, list]] = {
        lvl: {"precision": [], "recall": [], "f1": [], "tp": [], "gt": []}
        for lvl in LEVELS
    }

    title_exact = sum(1 for r in per_file if r.get("title_exact", False))
    title_fuzzy = sum(1 for r in per_file if r.get("title_fuzzy", False))
    valid = [r for r in per_file if "error" not in r]

    for result in valid:
        for level, metrics in result.get("level_metrics", {}).items():
            if level in level_accum:
                level_accum[level]["precision"].append(metrics["precision"])
                level_accum[level]["recall"].append(metrics["recall"])
                level_accum[level]["f1"].append(metrics["f1"])
                level_accum[level]["tp"].append(metrics.get("tp", 0))
                level_accum[level]["gt"].append(metrics.get("gt_count", 0))

    summary: dict[str, dict] = {}
    all_tp, all_gt, all_pred = 0, 0, 0

    for level in LEVELS:
        data = level_accum[level]
        if not data["precision"]:
            continue
        avg_p = sum(data["precision"]) / len(data["precision"])
        avg_r = sum(data["recall"]) / len(data["recall"])
        avg_f1 = sum(data["f1"]) / len(data["f1"])
        tp_sum = sum(data["tp"])
        gt_sum = sum(data["gt"])

        summary[level] = {
            "precision": round(avg_p, 4),
            "recall": round(avg_r, 4),
            "f1": round(avg_f1, 4),
            "tp_total": tp_sum,
            "gt_total": gt_sum,
        }
        all_tp += tp_sum
        all_gt += gt_sum

    # Overall micro F1 — sum TP across all levels, compute simple P and R
    total_pred = sum(
        sum(level_accum[l]["tp"]) for l in LEVELS if level_accum[l]["precision"]
    )
    overall_p = all_tp / total_pred if total_pred > 0 else 0.0
    overall_r = all_tp / all_gt if all_gt > 0 else 0.0
    overall_f1 = (
        2 * overall_p * overall_r / (overall_p + overall_r)
        if (overall_p + overall_r) > 0 else 0.0
    )

    return {
        "documents_evaluated": len(valid),
        "documents_errored": len(per_file) - len(valid),
        "title_exact_match": title_exact,
        "title_fuzzy_match": title_fuzzy,
        "title_total": len(valid),
        "level_summary": summary,
        "overall": {
            "precision": round(overall_p, 4),
            "recall": round(overall_r, 4),
            "f1": round(overall_f1, 4),
            "tp_total": all_tp,
            "gt_total": all_gt,
        },
    }


# ── Console Table Printer ──────────────────────────────────────────────────────

def print_report(agg: dict) -> None:
    W = 13  # column width
    sep = "-" * (8 + W * 4 + 10)

    print()
    print("  PDF Heading Extractor -- Evaluation Report")
    print(f"  {agg['documents_evaluated']} document(s) evaluated  "
          f"({agg['documents_errored']} error(s))")
    print(f"  Title exact match: {agg['title_exact_match']}/{agg['title_total']}  "
          f"|  Fuzzy: {agg['title_fuzzy_match']}/{agg['title_total']}")
    print()
    print(f"  {'Level':<8} {'Precision':>{W}} {'Recall':>{W}} {'F1':>{W}} "
          f"{'TP':>6} {'GT Total':>9}")
    print("  " + sep)

    for level in LEVELS:
        m = agg["level_summary"].get(level)
        if not m:
            continue
        print(
            f"  {level:<8} {m['precision']:>{W}.4f} {m['recall']:>{W}.4f} "
            f"{m['f1']:>{W}.4f} {m['tp_total']:>6} {m['gt_total']:>9}"
        )

    o = agg["overall"]
    print("  " + sep)
    print(
        f"  {'Overall':<8} {o['precision']:>{W}.4f} {o['recall']:>{W}.4f} "
        f"{o['f1']:>{W}.4f} {o['tp_total']:>6} {o['gt_total']:>9}"
    )
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evaluate",
        description="Evaluate the PDF heading extractor against ground-truth labels.",
    )
    p.add_argument("--pdfs", type=Path, default=DEFAULT_PDFS, metavar="DIR",
                   help="Directory of input PDFs (default: sample_datasets/pdfs).")
    p.add_argument("--gts", type=Path, default=DEFAULT_GTS, metavar="DIR",
                   help="Directory of ground-truth JSONs (default: sample_datasets/outputs).")
    p.add_argument("--lang", default="en", metavar="LANG",
                   help="Language code for pattern matching (default: en).")
    p.add_argument("--min-confidence", type=float, default=0.45, dest="min_confidence",
                   metavar="SCORE", help="Confidence threshold (default: 0.45).")
    p.add_argument("--output", type=Path, default=None, metavar="FILE",
                   help="Save full JSON report to this path.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-file per-level details.")
    p.add_argument("--use-ml", action="store_true", dest="use_ml",
                   help="Use the trained ML classifier instead of heuristics.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    pdf_dir: Path = args.pdfs
    gt_dir: Path  = args.gts

    if not pdf_dir.is_dir():
        print(f"Error: PDFs directory '{pdf_dir}' not found.", file=sys.stderr)
        return 1
    if not gt_dir.is_dir():
        print(f"Error: Ground-truth directory '{gt_dir}' not found.", file=sys.stderr)
        return 1

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in '{pdf_dir}'.", file=sys.stderr)
        return 1

    print(f"\nEvaluating {len(pdfs)} PDF(s) against ground truth in '{gt_dir}'…")

    per_file_results: list[dict] = []
    for pdf in pdfs:
        gt_path = gt_dir / f"{pdf.stem}.json"
        if not gt_path.exists():
            print(f"  [SKIP] {pdf.name} — no ground truth at '{gt_path}'")
            continue
        print(f"  Evaluating {pdf.name}…", end=" ", flush=True)
        result = evaluate_file(
            pdf_path=pdf,
            gt_path=gt_path,
            lang=args.lang,
            min_confidence=args.min_confidence,
            verbose=args.verbose,
            use_ml=args.use_ml,
        )
        per_file_results.append(result)
        status = "OK" if "error" not in result else f"ERROR: {result['error']}"
        print(status)

    if not per_file_results:
        print("No files evaluated. Check that ground-truth JSONs match PDF stems.")
        return 1

    agg = aggregate(per_file_results)
    print_report(agg)

    if args.output:
        full_report = {"summary": agg, "per_file": per_file_results}
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(full_report, fh, indent=2, ensure_ascii=False)
        print(f"  Full report saved to: {args.output}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
