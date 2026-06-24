import sys
import time
import json
from pathlib import Path
import Levenshtein

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractor.core import ExtractorEngine

def is_fuzzy_match(str1: str, str2: str, threshold: float = 0.8) -> bool:
    """Returns True if str1 and str2 are similar enough (Levenshtein ratio >= threshold)."""
    if not str1 or not str2:
        return False
    return Levenshtein.ratio(str1.lower().strip(), str2.lower().strip()) >= threshold

def run_evaluation():
    print("================================================")
    print("   Actual Document Structure Extraction Evaluation")
    print("================================================\n")
    
    pdfs_dir = Path("sample_datasets/pdfs_test")
    gt_dir = Path("sample_datasets/ground_truth")
    
    if not pdfs_dir.exists() or not gt_dir.exists():
        print("Error: Test datasets not found. Please run scripts/generate_test_data.py first.")
        return
        
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No test PDFs found.")
        return
        
    print(f"Found {len(pdf_files)} test documents.")
    
    total_expected = 0
    total_extracted = 0
    
    tp_total = 0
    fp_total = 0
    fn_total = 0
    
    baseline_times = []
    optimized_times = []

    print("Evaluating models...")
    for pdf_path in pdf_files:
        gt_path = gt_dir / pdf_path.with_suffix('.json').name
        
        if not gt_path.exists():
            continue
            
        with open(gt_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
            
        # Baseline processing (Simulate ML=False)
        start_base = time.perf_counter()
        with ExtractorEngine(str(pdf_path), use_ml=False) as engine:
            engine.process()
        baseline_times.append(time.perf_counter() - start_base)
            
        # Optimized processing (ML=True)
        start_opt = time.perf_counter()
        with ExtractorEngine(str(pdf_path), use_ml=True) as engine:
            outline = engine.process()
        optimized_times.append(time.perf_counter() - start_opt)
        
        extracted_headings = [{"level": h.level, "text": h.text} for h in outline.headings]
        
        # Match Extracted against Ground Truth
        # We'll do a simple greedy matching for True Positives
        matched_gt_indices = set()
        matched_ext_indices = set()
        
        for i, ext in enumerate(extracted_headings):
            best_match_idx = -1
            best_score = 0.0
            
            for j, gt in enumerate(ground_truth):
                if j in matched_gt_indices:
                    continue
                    
                # Match level exactly, fuzz-match text
                if ext["level"] == gt["level"]:
                    score = Levenshtein.ratio(ext["text"].lower().strip(), gt["text"].lower().strip())
                    if score > best_score and score >= 0.8:
                        best_score = score
                        best_match_idx = j
                        
            if best_match_idx != -1:
                matched_gt_indices.add(best_match_idx)
                matched_ext_indices.add(i)
                
        tp = len(matched_gt_indices)
        fp = len(extracted_headings) - len(matched_ext_indices)
        fn = len(ground_truth) - len(matched_gt_indices)
        
        tp_total += tp
        fp_total += fp
        fn_total += fn
        
        total_expected += len(ground_truth)
        total_extracted += len(extracted_headings)
        
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    print(f"\nDataset Size       : {len(pdf_files)} documents")
    print(f"Expected Headings  : {total_expected}")
    print(f"Extracted Headings : {total_extracted}\n")
    
    print("--- Accuracy Metrics (Actual) ---")
    print(f"True Positives     : {tp_total}")
    print(f"False Positives    : {fp_total}")
    print(f"False Negatives    : {fn_total}")
    print(f"Precision          : {precision:.2f}")
    print(f"Recall             : {recall:.2f}")
    print(f"F1 Score           : {f1:.2f}\n")
    
    print("--- Performance Benchmarking (Actual) ---")
    total_baseline = sum(baseline_times)
    total_optimized = sum(optimized_times)
    speedup = ((total_baseline - total_optimized) / total_baseline) * 100 if total_baseline > 0 else 0.0
    
    print(f"Baseline Runtime   : {total_baseline:.2f} sec")
    print(f"Optimized Runtime  : {total_optimized:.2f} sec")
    print(f"Speed Improvement  : {speedup:.1f} %\n")
    
    print("Evaluation Complete.")

if __name__ == "__main__":
    run_evaluation()
