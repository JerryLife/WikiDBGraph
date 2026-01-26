#!/usr/bin/env python
"""
String Match Evaluator: A naive baseline for column similarity.

Computes Jaccard similarity between normalized column name sets of databases.
Uses the same evaluation framework as the embedding-based evaluator for
fair comparison.

Usage:
    python -m baseline.string_match.string_match_evaluator \
        --schema-dir data/schema \
        --test-triplets out/graph_full_ss3_neg6/triplets/triplets_test_seed0.jsonl \
        --output-dir out/baseline_string_match/test_results
"""

import os
import re
import sys
import csv
import json
import random
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm


def normalize_column_name(name: str) -> str:
    """
    Normalize column name: lowercase + keep only alphanumeric characters.
    
    Args:
        name: Original column name
    
    Returns:
        Normalized column name
    """
    return re.sub(r'[^a-z0-9]', '', name.lower())


def load_schema_columns(schema_dir: str) -> Dict[str, Set[str]]:
    """
    Load all database schemas and extract normalized column names.
    
    Args:
        schema_dir: Directory containing schema JSON files
    
    Returns:
        Dict mapping db_id to set of normalized column names
    """
    db_columns = {}
    schema_path = Path(schema_dir)
    
    for schema_file in tqdm(list(schema_path.glob("*.json")), desc="Loading schemas"):
        # Extract db_id from filename (e.g., "00001_xxx.json" -> "00001")
        db_id = schema_file.stem.split("_")[0]
        
        with open(schema_file, 'r') as f:
            schema = json.load(f)
        
        # Extract all column names from all tables
        columns = set()
        for table in schema.get("tables", []):
            for col in table.get("columns", []):
                col_name = col.get("column_name", "")
                if col_name:
                    normalized = normalize_column_name(col_name)
                    if normalized:  # Skip empty after normalization
                        columns.add(normalized)
        
        db_columns[db_id] = columns
    
    return db_columns


def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Compute Jaccard similarity between two sets.
    
    Args:
        set_a: First set
        set_b: Second set
    
    Returns:
        Jaccard similarity score (0 to 1)
    """
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def evaluate_single_seed(
    triplets: List[Dict],
    db_columns: Dict[str, Set[str]],
    seed: int
) -> Tuple[List[int], List[float], List[Tuple], int]:
    """
    Evaluate string matching on test triplets with 1:1 ratio using a specific seed.
    
    For each triplet, samples 1 random negative (controlled by seed) to achieve 1:1 ratio.
    
    Returns:
        Tuple of (y_true, y_scores, records, skipped)
    """
    random.seed(seed)
    
    y_true = []
    y_scores = []
    records = []
    skipped = 0
    
    for item in triplets:
        anchor_id = item["anchor"]
        pos_id = item["positive"]
        neg_ids = item["negatives"]
        
        # Sample 1 random negative for 1:1 ratio
        neg_id = random.choice(neg_ids)
        
        # Check if all IDs exist in schemas
        if anchor_id not in db_columns or pos_id not in db_columns or neg_id not in db_columns:
            skipped += 1
            continue
        
        anchor_cols = db_columns[anchor_id]
        pos_cols = db_columns[pos_id]
        neg_cols = db_columns[neg_id]
        
        # Positive pair
        sim_pos = jaccard_similarity(anchor_cols, pos_cols)
        y_true.append(1)
        y_scores.append(sim_pos)
        records.append((anchor_id, pos_id, sim_pos, 1))
        
        # Negative pair (1:1 ratio)
        sim_neg = jaccard_similarity(anchor_cols, neg_cols)
        y_true.append(0)
        y_scores.append(sim_neg)
        records.append((anchor_id, neg_id, sim_neg, 0))
    
    return y_true, y_scores, records, skipped


def compute_metrics_at_threshold(y_true, y_scores, threshold):
    """Compute precision, recall, f1, accuracy at a specific threshold."""
    y_pred = (np.array(y_scores) >= threshold).astype(int)
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'accuracy': accuracy_score(y_true, y_pred)
    }


def evaluate_string_match(
    schema_dir: str,
    test_triplets_path: str,
    output_dir: str,
    seeds: List[int] = None
) -> Dict:
    """
    Evaluate string matching baseline on test triplets.
    
    Args:
        schema_dir: Directory containing schema JSON files
        test_triplets_path: Path to test triplets JSONL file
        output_dir: Directory to save results
        seeds: List of random seeds for negative sampling (default: [0,1,2,3,4])
    
    Returns:
        Dict with aggregated metrics (mean ± std)
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load schema columns
    print(f"📂 Loading schemas from {schema_dir}")
    db_columns = load_schema_columns(schema_dir)
    print(f"   Loaded {len(db_columns)} database schemas")
    
    # Load test triplets
    print(f"📂 Loading test triplets from {test_triplets_path}")
    triplets = []
    with open(test_triplets_path, "r") as f:
        for line in f:
            triplets.append(json.loads(line))
    print(f"   Loaded {len(triplets)} triplets")
    print(f"   Using 1:1 positive:negative ratio with {len(seeds)} seeds")
    
    # Evaluate across all seeds
    all_seed_metrics = []
    all_aucs = []
    all_thresholds = []
    
    for seed in tqdm(seeds, desc="Evaluating seeds"):
        y_true, y_scores, records, skipped = evaluate_single_seed(
            triplets, db_columns, seed
        )
        
        if not y_true:
            print(f"⚠️  Seed {seed}: No valid triplets found!")
            continue
        
        # Compute ROC + AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Find optimal threshold (Youden's J)
        youden_j = tpr - fpr
        best_idx = youden_j.argmax()
        best_threshold = thresholds[best_idx]
        
        # Compute metrics at optimal threshold
        metrics = compute_metrics_at_threshold(y_true, y_scores, best_threshold)
        metrics['auc'] = roc_auc
        metrics['threshold'] = best_threshold
        metrics['best_tpr'] = tpr[best_idx]
        metrics['best_fpr'] = fpr[best_idx]
        
        all_seed_metrics.append(metrics)
        all_aucs.append(roc_auc)
        all_thresholds.append(best_threshold)
        
        # Save predictions for this seed
        seed_csv_path = os.path.join(output_dir, f"predictions_seed{seed}.csv")
        with open(seed_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["anchor_id", "target_id", "similarity", "label"])
            writer.writerows(records)
    
    if not all_seed_metrics:
        print("❌ No valid results from any seed!")
        return {}
    
    # Aggregate metrics across seeds
    aggregated = {}
    metric_names = ['auc', 'accuracy', 'precision', 'recall', 'f1', 'threshold']
    for metric in metric_names:
        values = [m[metric] for m in all_seed_metrics]
        aggregated[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    # Use the last seed's data for ROC curve plot
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {aggregated["auc"]["mean"]:.4f}±{aggregated["auc"]["std"]:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.scatter(all_seed_metrics[-1]['best_fpr'], all_seed_metrics[-1]['best_tpr'], 
                color='red', label=f'Optimal threshold = {aggregated["threshold"]["mean"]:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('String Match Baseline ROC (1:1 Balanced)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()
    
    # Compute class distribution (should be 1:1)
    num_positive = sum(y_true)
    num_negative = len(y_true) - num_positive
    pos_ratio = num_positive / len(y_true) * 100
    
    # Save summary
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"AUC: {aggregated['auc']['mean']:.4f}\n")
        f.write(f"AUC_std: {aggregated['auc']['std']:.4f}\n")
        f.write(f"Best Threshold: {aggregated['threshold']['mean']:.4f}\n")
        f.write(f"Threshold_std: {aggregated['threshold']['std']:.4f}\n")
        f.write(f"Precision: {aggregated['precision']['mean']:.4f}\n")
        f.write(f"Precision_std: {aggregated['precision']['std']:.4f}\n")
        f.write(f"Recall: {aggregated['recall']['mean']:.4f}\n")
        f.write(f"Recall_std: {aggregated['recall']['std']:.4f}\n")
        f.write(f"F1: {aggregated['f1']['mean']:.4f}\n")
        f.write(f"F1_std: {aggregated['f1']['std']:.4f}\n")
        f.write(f"Accuracy: {aggregated['accuracy']['mean']:.4f}\n")
        f.write(f"Accuracy_std: {aggregated['accuracy']['std']:.4f}\n")
        f.write(f"Total triplets: {len(triplets)}\n")
        f.write(f"Seeds used: {seeds}\n")
        f.write(f"Positive pairs: {num_positive} ({pos_ratio:.1f}%)\n")
        f.write(f"Negative pairs: {num_negative} ({100-pos_ratio:.1f}%)\n")
        f.write(f"Pos:Neg ratio: 1:1\n")
        f.write(f"Method: Jaccard similarity on normalized column names\n")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"📊 String Match Baseline Results (1:1 ratio, {len(seeds)} seeds)")
    print(f"{'='*60}")
    print(f"AUC-ROC:    {aggregated['auc']['mean']:.4f} ± {aggregated['auc']['std']:.4f}")
    print(f"Precision:  {aggregated['precision']['mean']:.4f} ± {aggregated['precision']['std']:.4f}")
    print(f"Recall:     {aggregated['recall']['mean']:.4f} ± {aggregated['recall']['std']:.4f}")
    print(f"F1 Score:   {aggregated['f1']['mean']:.4f} ± {aggregated['f1']['std']:.4f}")
    print(f"Accuracy:   {aggregated['accuracy']['mean']:.4f} ± {aggregated['accuracy']['std']:.4f}")
    print(f"Threshold:  {aggregated['threshold']['mean']:.4f} ± {aggregated['threshold']['std']:.4f}")
    print(f"{'='*60}")
    print(f"📊 Class distribution: {num_positive} positive (50%), {num_negative} negative (50%)")
    print(f"💾 Results saved in: {output_dir}")
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate string matching baseline on test triplets"
    )
    parser.add_argument(
        "--schema-dir",
        type=str,
        default="data/schema",
        help="Directory containing schema JSON files"
    )
    parser.add_argument(
        "--test-triplets",
        type=str,
        required=True,
        help="Path to test triplets JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs='+',
        default=[0, 1, 2, 3, 4],
        help="Random seeds for negative sampling (default: 0 1 2 3 4)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.schema_dir):
        print(f"Error: Schema directory not found: {args.schema_dir}")
        sys.exit(1)
    if not os.path.exists(args.test_triplets):
        print(f"Error: Test triplets file not found: {args.test_triplets}")
        sys.exit(1)
    
    evaluate_string_match(
        schema_dir=args.schema_dir,
        test_triplets_path=args.test_triplets,
        output_dir=args.output_dir,
        seeds=args.seeds
    )


if __name__ == "__main__":
    main()
