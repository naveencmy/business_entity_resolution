#!/usr/bin/env python3
"""
Evaluate Entity Resolution Pipeline Metrics (Macro F_0.5, Macro F1, Precision, Recall, Singleton Accuracy)
Runs on validation ground truth data to display the full diagnostic score report.
"""

import csv
import os
import sys
from pathlib import Path
from typing import Dict, Set, List
import numpy as np

def compute_detailed_metrics(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]]) -> Dict[str, float]:
    f05_scores = []
    f1_scores = []
    precisions = []
    recalls = []
    
    singleton_total = 0
    singleton_correct = 0
    matched_total = 0
    matched_correct = 0

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())

        # Singleton evaluation (true_set is empty)
        if not true_set:
            singleton_total += 1
            if not pred_set:
                singleton_correct += 1
                f05_scores.append(1.0)
                f1_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            else:
                f05_scores.append(0.0)
                f1_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(1.0)
            continue

        # Matched entity evaluation
        matched_total += 1
        if not pred_set:
            f05_scores.append(0.0)
            f1_scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        tp = len(true_set.intersection(pred_set))
        p = tp / len(pred_set)
        r = tp / len(true_set)
        precisions.append(p)
        recalls.append(r)

        # F0.5 (beta = 0.5): (1.25 * P * R) / (0.25 * P + R)
        denom_05 = (0.25 * p) + r
        f05 = (1.25 * p * r) / denom_05 if denom_05 > 0 else 0.0
        f05_scores.append(f05)

        # F1 (beta = 1.0): (2 * P * R) / (P + R)
        denom_1 = p + r
        f1 = (2 * p * r) / denom_1 if denom_1 > 0 else 0.0
        f1_scores.append(f1)

        if tp == len(true_set) and len(pred_set) == len(true_set):
            matched_correct += 1

    return {
        "macro_f05": float(np.mean(f05_scores)),
        "macro_f1": float(np.mean(f1_scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "singleton_accuracy": (singleton_correct / singleton_total) if singleton_total > 0 else 1.0,
        "singleton_count": singleton_total,
        "matched_count": matched_total,
        "total_evaluated": len(ground_truth)
    }

def print_metrics_dashboard(metrics: Dict[str, float]):
    print("=" * 65)
    print("🏆 BUSINESS ENTITY RESOLUTION — PERFORMANCE SCORE DASHBOARD")
    print("=" * 65)
    print(f"Total Evaluated Entities:    {int(metrics['total_evaluated']):,}")
    print(f"  - Non-Singleton Entities:  {int(metrics['matched_count']):,}")
    print(f"  - Singleton Entities:      {int(metrics['singleton_count']):,}")
    print("-" * 65)
    print(f"🎯 Competition Metric (Macro F_0.5):  {metrics['macro_f05']:.4f}  ({metrics['macro_f05']*100:.2f}%)")
    print(f"⭐ Standard Balanced F1 (Macro F1):   {metrics['macro_f1']:.4f}  ({metrics['macro_f1']*100:.2f}%)")
    print(f"🔍 Macro Precision:                   {metrics['macro_precision']:.4f}  ({metrics['macro_precision']*100:.2f}%)")
    print(f"📡 Macro Recall:                      {metrics['macro_recall']:.4f}  ({metrics['macro_recall']*100:.2f}%)")
    print(f"🛡️ Singleton Preservation Accuracy:  {metrics['singleton_accuracy']:.4f}  ({metrics['singleton_accuracy']*100:.2f}%)")
    print("=" * 65)
    if metrics['macro_f05'] >= 0.985:
        print("🚀 STATUS: TOP 1% LEADERBOARD PERFORMANCE CALIBRATED!")
    print("=" * 65)

if __name__ == "__main__":
    # Test quick demonstration with sample data
    gt_sample = {
        "S1-001": {"S2-101", "S3-201"},
        "S1-002": {"S2-102"},
        "S1-003": set(),  # singleton
        "S1-004": {"S2-104", "S3-204"},
        "S1-005": set(),  # singleton
    }
    pred_sample = {
        "S1-001": {"S2-101", "S3-201"},
        "S1-002": {"S2-102"},
        "S1-003": set(),
        "S1-004": {"S2-104", "S3-204"},
        "S1-005": set(),
    }
    metrics = compute_detailed_metrics(gt_sample, pred_sample)
    print_metrics_dashboard(metrics)
