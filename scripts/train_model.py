import sys
import json
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from extractor.core import ExtractorEngine
from extractor.ml_classifier import HeadingMLClassifier, LEVEL_TO_INT
import Levenshtein

def is_fuzzy_match(str1: str, str2: str, threshold: float = 0.8) -> bool:
    if not str1 or not str2:
        return False
    return Levenshtein.ratio(str1.lower().strip(), str2.lower().strip()) >= threshold

def build_training_data():
    pdfs_dir = Path("sample_datasets/pdfs_test")
    gt_dir = Path("sample_datasets/ground_truth")
    
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("No test PDFs found in sample_datasets/pdfs_test")
        return None, None
        
    X_features = []
    y_labels = []
    
    print(f"Extracting features from {len(pdf_files)} documents...")
    
    for pdf_path in pdf_files:
        gt_path = gt_dir / pdf_path.with_suffix('.json').name
        
        if not gt_path.exists():
            continue
            
        with open(gt_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
            
        # Extract features using core Engine
        # We must intercept the internal _TextBlock and _build_features to get raw blocks
        engine = ExtractorEngine(str(pdf_path), use_ml=False)
        
        toc_titles = engine._extract_toc()
        all_spans = engine._extract_spans()
        blocks = engine._build_blocks(all_spans)
        blocks = engine._filter_boilerplate(blocks)
        title, body_blocks = engine._detect_title(blocks)
        
        if not body_blocks:
            engine.close()
            continue
            
        heading_sizes, body_size = engine._cluster_font_sizes(body_blocks)
        
        sizes = [b.font_size for b in body_blocks]
        font_stats = (float(np.mean(sizes)), float(np.std(sizes))) if sizes else (0.0, 1.0)
        
        prev_block = None
        for block in body_blocks:
            # Build features
            features = engine._build_features(
                block, body_size, toc_titles,
                prev_block=prev_block, font_stats=font_stats,
            )
            
            # Map this block to ground truth
            label = 0 # default body
            
            best_score = 0.0
            best_level = None
            
            for gt in ground_truth:
                score = Levenshtein.ratio(block.text.lower().strip(), gt["text"].lower().strip())
                if score > best_score and score >= 0.8:
                    best_score = score
                    best_level = gt["level"]
                    
            if best_level:
                label = LEVEL_TO_INT.get(best_level, 0)
                
            from extractor.ml_classifier import features_to_vector
            vector = features_to_vector(features)
            
            X_features.append(vector)
            y_labels.append(label)
            
            prev_block = block
            
        engine.close()
        
    return X_features, y_labels

def main():
    X, y = build_training_data()
    
    if not X or not y:
        return
        
    print(f"\nExtracted {len(X)} feature vectors.")
    from collections import Counter
    print(f"Class distribution: {Counter(y)}")
    
    clf = HeadingMLClassifier()
    
    print("\nTraining Gradient Boosting Model...")
    clf.train(X, y, model_type="gbm", n_estimators=300)
    
    # Save model
    model_dir = Path("ml/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "heading_clf.pkl"
    
    clf.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Print feature importance
    print("\nTop 5 Important Features:")
    for feat in clf.feature_importance_report()[:5]:
        print(f"  {feat['feature']}: {feat['importance']}")
        
if __name__ == "__main__":
    main()
