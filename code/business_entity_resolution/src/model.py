"""
Precision-Heavy Entity Matching Model & F_0.5 Metric Optimization.
Supports XGBoost and HistGradientBoosting classifiers.
Calibrated specifically to maximize macro F_0.5 and protect singletons.
"""
import os
import pickle
from typing import Dict, List, Set, Tuple, Any, Optional
import numpy as np
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier

def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, float]:
    """
    Compute macro-averaged F_0.5 score according to competition rules.
    - F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    - Singletons: If GT is empty and Pred is empty -> 1.0; if Pred is non-empty -> 0.0.
    """
    entity_scores = []
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"macro_f05": 0.0, "precision": 0.0, "recall": 0.0}

    macro_precision = []
    macro_recall = []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        # Singleton evaluation
        if not true_set:
            if not pred_set:
                score = 1.0
                macro_precision.append(1.0)
                macro_recall.append(1.0)
            else:
                score = 0.0
                macro_precision.append(0.0)
                macro_recall.append(1.0)
            entity_scores.append(score)
            continue

        # Non-empty true match set
        if not pred_set:
            score = 0.0
            macro_precision.append(0.0)
            macro_recall.append(0.0)
            entity_scores.append(score)
            continue

        tp = len(true_set.intersection(pred_set))
        p = tp / len(pred_set)
        r = tp / len(true_set)
        macro_precision.append(p)
        macro_recall.append(r)

        denom = (0.25 * p) + r
        if denom > 0:
            f05 = (1.25 * p * r) / denom
        else:
            f05 = 0.0
        entity_scores.append(f05)

    return {
        "macro_f05": float(np.mean(entity_scores)),
        "precision": float(np.mean(macro_precision)),
        "recall": float(np.mean(macro_recall))
    }


class EntityMatcher:
    """
    Entity matching classifier wrapper with probability calibration and threshold optimization.
    Automatically leverages CUDA GPU acceleration on Kaggle / high-end GPUs with CPU fallback.
    """
    def __init__(self, model_type: str = "xgb"):
        self.model_type = model_type
        if model_type == "xgb":
            # Auto-detect CUDA GPU availability
            device = "cpu"
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
            except ImportError:
                pass

            self.clf = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=7,
                learning_rate=0.07,
                subsample=0.85,
                colsample_bytree=0.85,
                tree_method="hist",
                device=device,
                scale_pos_weight=1.0,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        else:
            self.clf = HistGradientBoostingClassifier(
                max_iter=200,
                max_depth=7,
                learning_rate=0.07,
                random_state=42
            )
        self.optimal_threshold: float = 0.70

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit the matching classifier."""
        self.clf.fit(X, y)
        if self.model_type == "xgb":
            try:
                self.clf.set_params(device="cpu")
            except Exception:
                pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return match probability for each pair."""
        if len(X) == 0:
            return np.array([], dtype=np.float32)
        return self.clf.predict_proba(X)[:, 1]

    def optimize_threshold(
        self,
        candidate_pairs: List[Tuple[str, str]],
        probabilities: np.ndarray,
        ground_truth: Dict[str, Set[str]],
        threshold_range: Tuple[float, float, float] = (0.40, 0.95, 0.02)
    ) -> float:
        """
        Grid search for the probability threshold that maximizes Macro F_0.5 on validation data.
        """
        best_threshold = 0.70
        best_score = -1.0
        start, stop, step = threshold_range
        thresholds = np.arange(start, stop + step, step)

        print("Optimizing probability threshold for Macro F_0.5...")
        for tau in thresholds:
            preds: Dict[str, Set[str]] = {s1_id: set() for s1_id in ground_truth.keys()}
            for (s1_id, cand_id), prob in zip(candidate_pairs, probabilities):
                if prob >= tau:
                    preds[s1_id].add(cand_id)

            metrics = compute_macro_f05(ground_truth, preds)
            score = metrics["macro_f05"]
            if score > best_score:
                best_score = score
                best_threshold = float(tau)

        print(f"Optimal Threshold: {best_threshold:.2f} (Validation Macro F_0.5: {best_score:.4f})")
        self.optimal_threshold = best_threshold
        return best_threshold

    def predict_entity_matches(
        self,
        candidate_ids: List[str],
        probabilities: np.ndarray,
        anchor_threshold: float = 0.72,
        expansion_threshold: float = 0.65
    ) -> List[str]:
        """
        Intentional Singleton & Match Prediction Gate:
        1. If candidate list is empty, returns [] (singleton score: 1.0).
        2. If max probability < anchor_threshold (0.72), the entity fails the high-confidence
           anchor test and is classified as a singleton -> returns [] (protects singleton from 0.0 penalty).
        3. If anchor is verified (>= 0.72), all candidates passing expansion_threshold (0.65)
           are included as genuine cluster matches.
        """
        if len(candidate_ids) == 0 or len(probabilities) == 0:
            return []

        max_prob = float(np.max(probabilities))
        # Intentional Singleton Verification Gate
        if max_prob < anchor_threshold:
            return []

        # Anchor confirmed: retain all cluster members meeting expansion threshold
        matched = [
            cid for cid, p in zip(candidate_ids, probabilities)
            if p >= expansion_threshold
        ]
        return matched

    def save(self, filepath: str):
        """Save model artifact to disk."""
        with open(filepath, "wb") as f:
            pickle.dump({"clf": self.clf, "threshold": self.optimal_threshold, "type": self.model_type}, f)

    def load(self, filepath: str):
        """Load model artifact from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.clf = data["clf"]
            self.optimal_threshold = data["threshold"]
            self.model_type = data.get("type", "xgb")
