"""
extractor/ml_classifier.py

Scikit-learn heading classifier for the PDF extraction pipeline.

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │  HeadingMLClassifier                                     │
    │                                                          │
    │  Feature vector (18 dims per text block):                │
    │                                                          │
    │  Typography (4):                                         │
    │    [0]  rel_font_size       — font / median body font    │
    │    [1]  font_size_zscore    — (size - mean) / std        │
    │    [2]  bold_percentage     — 0.0–1.0                    │
    │    [3]  is_italic           — 0 or 1                     │
    │                                                          │
    │  Structure (4):                                          │
    │    [4]  starts_with_number  — 0 or 1                     │
    │    [5]  x_indent_ratio      — 0.0–1.0                   │
    │    [6]  y_position_ratio    — 0.0–1.0                   │
    │    [7]  centered_text       — 0.0–1.0                   │
    │                                                          │
    │  Context (4):                                            │
    │    [8]  in_toc              — 0 or 1                     │
    │    [9]  is_first_page       — 0 or 1                     │
    │    [10] vertical_gap_before — 0.0–1.0                   │
    │    [11] font_change_from_prev — 0 or 1                  │
    │                                                          │
    │  Content (6):                                            │
    │    [12] char_count_norm     — len / 100 (capped at 3)    │
    │    [13] word_count_norm     — words / 20 (capped at 3)   │
    │    [14] all_caps_ratio      — 0.0–1.0                   │
    │    [15] title_case_ratio    — 0.0–1.0                   │
    │    [16] punctuation_density — 0.0–1.0                   │
    │    [17] short_line          — 0 or 1 (< 60 chars)       │
    │                                                          │
    │  Labels: 0=body  1=H1  2=H2  3=H3                       │
    │                                                          │
    │  Model: StandardScaler → GradientBoostingClassifier      │
    │         (or RandomForest baseline for comparison)         │
    └──────────────────────────────────────────────────────────┘

Usage:
    from extractor.ml_classifier import HeadingMLClassifier

    clf = HeadingMLClassifier()
    clf.load("ml/models/heading_clf.pkl")
    level, confidence = clf.predict_level(features)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from extractor.models import HeadingFeatures

# ── Constants ──────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    # Typography
    "rel_font_size",
    "font_size_zscore",
    "bold_percentage",
    "is_italic",
    # Structure
    "starts_with_number",
    "x_indent_ratio",
    "y_position_ratio",
    "centered_text",
    # Context
    "in_toc",
    "is_first_page",
    "vertical_gap_before",
    "font_change_from_prev",
    # Content
    "char_count_norm",
    "word_count_norm",
    "all_caps_ratio",
    "title_case_ratio",
    "punctuation_density",
    "short_line",
]

# Integer label → heading level string
LABEL_MAP = {0: None, 1: "H1", 2: "H2", 3: "H3"}
# Reverse: level string → integer label
LEVEL_TO_INT = {"body": 0, "H1": 1, "H2": 2, "H3": 3}

# Default saved model path (relative to project root)
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "ml" / "models" / "heading_clf.pkl"


# ── Feature extraction ─────────────────────────────────────────────────────────

def features_to_vector(f: HeadingFeatures) -> list[float]:
    """
    Convert a HeadingFeatures dataclass into an 18-dim float vector.

    All values are normalised to reasonable ranges so that StandardScaler
    can centre and scale them effectively.
    """
    return [
        # Typography
        float(f.rel_font_size),                     # usually 0.5–4.0
        float(f.font_size_zscore),                  # usually -2 to +4
        float(f.bold_percentage),                   # 0.0–1.0 (continuous)
        float(f.is_italic),                         # 0 or 1
        # Structure
        float(f.starts_with_number),                # 0 or 1
        float(f.x_indent_ratio),                    # 0.0–1.0
        float(f.y_position_ratio),                  # 0.0–1.0
        float(f.centered_text),                     # 0.0–1.0
        # Context
        float(f.in_toc),                            # 0 or 1
        float(f.is_first_page),                     # 0 or 1
        float(f.vertical_gap_before),               # 0.0–1.0
        float(f.font_change_from_prev),             # 0 or 1
        # Content
        min(f.line_char_count / 100.0, 3.0),        # cap at 300 chars
        min(f.word_count / 20.0, 3.0),              # cap at 60 words
        float(f.all_caps_ratio),                    # 0.0–1.0
        float(f.title_case_ratio),                  # 0.0–1.0
        float(f.punctuation_density),               # 0.0–1.0
        float(f.line_char_count < 60),              # short_line: 0 or 1
    ]


# ── Classifier ─────────────────────────────────────────────────────────────────

class HeadingMLClassifier:
    """
    Scikit-learn pipeline wrapping a StandardScaler + classifier.

    The classifier distinguishes 4 classes:
        0 = body text (not a heading)
        1 = H1
        2 = H2
        3 = H3

    Supports both RandomForest and GradientBoosting for methodical
    comparison. The best-performing model is selected during training.
    """

    def __init__(self, model_path: Optional[Path | str] = None) -> None:
        self._pipeline = None   # sklearn Pipeline, set after train() or load()
        self.is_fitted: bool = False
        self.feature_importances_: Optional[np.ndarray] = None
        self.model_type: str = "unknown"  # "rf" or "gbm"

        if model_path is not None:
            path = Path(model_path)
            if path.exists():
                self.load(path)

    # ── Training ───────────────────────────────────────────────────────────────

    def train(
        self,
        X: list[list[float]],
        y: list[int],
        *,
        model_type: str = "gbm",
        n_estimators: int = 300,
        max_depth: Optional[int] = None,
        class_weight: str = "balanced",
        learning_rate: float = 0.1,
    ) -> None:
        """
        Fit the classifier on a labelled feature matrix.

        Args:
            X: list of feature vectors (one per text block)
            y: integer labels (0=body, 1=H1, 2=H2, 3=H3)
            model_type: "rf" for RandomForest, "gbm" for GradientBoosting
            n_estimators: number of trees/boosting rounds
            max_depth: max tree depth (None = unlimited for RF, 5 for GBM)
            class_weight: 'balanced' corrects for class imbalance (RF only)
            learning_rate: step size for GBM (ignored for RF)
        """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int32)

        if model_type == "gbm":
            from sklearn.ensemble import GradientBoostingClassifier
            clf = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth or 5,
                learning_rate=learning_rate,
                subsample=0.8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
            )
        else:
            from sklearn.ensemble import RandomForestClassifier
            clf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                class_weight=class_weight,
                random_state=42,
                n_jobs=-1,
            )

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", clf),
        ])
        self._pipeline.fit(X_arr, y_arr)
        self.is_fitted = True
        self.model_type = model_type

        # Store feature importances for explainability
        self.feature_importances_ = self._pipeline.named_steps["clf"].feature_importances_

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict_level(
        self, features: HeadingFeatures
    ) -> tuple[Optional[str], float]:
        """
        Predict heading level for a single text block.

        Returns:
            (level, confidence)
            level is None if the block is predicted to be body text.
            confidence is the predicted class probability (0.0–1.0).
        """
        if not self.is_fitted or self._pipeline is None:
            return None, 0.0

        X = np.array([features_to_vector(features)], dtype=np.float32)
        label_int = int(self._pipeline.predict(X)[0])
        proba = float(self._pipeline.predict_proba(X)[0][label_int])

        return LABEL_MAP[label_int], proba

    def predict_batch(
        self, feature_list: list[HeadingFeatures]
    ) -> list[tuple[Optional[str], float]]:
        """
        Predict heading levels for a batch of features.

        More efficient than calling predict_level() in a loop because
        the whole batch goes through the pipeline at once.
        """
        if not self.is_fitted or self._pipeline is None:
            return [(None, 0.0)] * len(feature_list)

        X = np.array(
            [features_to_vector(f) for f in feature_list], dtype=np.float32
        )
        labels = self._pipeline.predict(X)
        probas = self._pipeline.predict_proba(X)

        return [
            (LABEL_MAP[int(labels[i])], float(probas[i][labels[i]]))
            for i in range(len(feature_list))
        ]

    # ── Cross-validation ───────────────────────────────────────────────────────

    def cross_validate_report(
        self,
        X: list[list[float]],
        y: list[int],
        cv: int = 5,
        model_type: str = "rf",
    ) -> dict:
        """
        Run stratified k-fold cross-validation and return a metrics dict.

        Returns:
            {
                "cv_folds": 5,
                "model_type": "rf",
                "accuracy_mean": 0.92,
                "accuracy_std": 0.03,
                "f1_macro_mean": 0.88,
                "f1_macro_std": 0.04,
                "per_class_f1": {"body": 0.95, "H1": 0.91, "H2": 0.85, "H3": 0.80},
                "confusion_matrix": [[...], ...],
            }
        """
        from sklearn.metrics import (
            confusion_matrix,
            f1_score,
        )
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int32)

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        fold_accs: list[float] = []
        fold_f1s: list[float] = []
        all_true: list[int] = []
        all_pred: list[int] = []

        for train_idx, val_idx in skf.split(X_arr, y_arr):
            X_tr, X_val = X_arr[train_idx], X_arr[val_idx]
            y_tr, y_val = y_arr[train_idx], y_arr[val_idx]

            if model_type == "gbm":
                from sklearn.ensemble import GradientBoostingClassifier
                clf_obj = GradientBoostingClassifier(
                    n_estimators=300, max_depth=5, learning_rate=0.1,
                    subsample=0.8, min_samples_split=10,
                    random_state=42,
                )
            else:
                from sklearn.ensemble import RandomForestClassifier
                clf_obj = RandomForestClassifier(
                    n_estimators=300, class_weight="balanced",
                    random_state=42, n_jobs=-1,
                )

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf_obj),
            ])
            pipe.fit(X_tr, y_tr)
            preds = pipe.predict(X_val)

            fold_accs.append(float(np.mean(preds == y_val)))
            fold_f1s.append(
                float(f1_score(y_val, preds, average="macro", zero_division=0))
            )
            all_true.extend(y_val.tolist())
            all_pred.extend(preds.tolist())

        # Per-class F1 on all folds combined
        present_labels = sorted(set(all_true))
        class_f1 = f1_score(
            all_true, all_pred,
            labels=present_labels,
            average=None,
            zero_division=0,
        )
        per_class = {
            LABEL_MAP.get(lbl, f"class_{lbl}") or "body": round(float(f1), 4)
            for lbl, f1 in zip(present_labels, class_f1)
        }

        cm = confusion_matrix(all_true, all_pred, labels=present_labels).tolist()

        return {
            "cv_folds": cv,
            "model_type": model_type,
            "accuracy_mean": round(float(np.mean(fold_accs)), 4),
            "accuracy_std":  round(float(np.std(fold_accs)), 4),
            "f1_macro_mean": round(float(np.mean(fold_f1s)), 4),
            "f1_macro_std":  round(float(np.std(fold_f1s)), 4),
            "per_class_f1":  per_class,
            "confusion_matrix_labels": [
                LABEL_MAP.get(l, "body") or "body" for l in present_labels
            ],
            "confusion_matrix": cm,
        }

    # ── Feature importance report ──────────────────────────────────────────────

    def feature_importance_report(self) -> list[dict]:
        """
        Return feature importances sorted descending.

        Only available after train().
        """
        if self.feature_importances_ is None:
            return []
        pairs = sorted(
            zip(FEATURE_NAMES, self.feature_importances_.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        return [{"feature": name, "importance": round(imp, 4)} for name, imp in pairs]

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: Path | str) -> None:
        """Pickle the fitted pipeline to disk."""
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted classifier.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "pipeline": self._pipeline,
                "feature_importances": self.feature_importances_,
                "model_type": self.model_type,
            }, fh)

    def load(self, path: Path | str) -> None:
        """Load a previously saved pipeline from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        self._pipeline = data["pipeline"]
        self.feature_importances_ = data.get("feature_importances")
        self.model_type = data.get("model_type", "unknown")
        self.is_fitted = True


# ── Singleton loader ───────────────────────────────────────────────────────────

_GLOBAL_CLASSIFIER: Optional[HeadingMLClassifier] = None


def get_classifier() -> Optional[HeadingMLClassifier]:
    """
    Return the globally loaded ML classifier, or None if not available.

    Called by ExtractorEngine. The classifier is loaded once at module import
    if the default model file exists, avoiding per-request disk I/O.
    """
    global _GLOBAL_CLASSIFIER
    if _GLOBAL_CLASSIFIER is None and DEFAULT_MODEL_PATH.exists():
        try:
            _GLOBAL_CLASSIFIER = HeadingMLClassifier(DEFAULT_MODEL_PATH)
        except Exception:
            _GLOBAL_CLASSIFIER = None
    return _GLOBAL_CLASSIFIER
