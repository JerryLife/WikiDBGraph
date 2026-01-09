#!/usr/bin/env python
"""
Evaluate SANTOS baseline on test triplets.

Evaluates using 1:1 balanced positive:negative ratio with multiple seeds
for robustness, matching the preprocess evaluator output format.

Usage:
    python -m src.baseline.santos.evaluate_auc \
        --scores out/santos_eval/scores.csv \
        --triplets out/graph_full_ss3_neg6/triplets/triplets_test_seed0.jsonl \
        --output-dir out/santos_eval \
        --seeds 0 1 2 3 4
"""

import os
import json
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, precision_score, recall_score


def compute_metrics_at_threshold(y_true, y_scores, threshold):
    """Compute precision, recall, f1 at a specific threshold."""
    y_pred = (np.array(y_scores) >= threshold).astype(int)
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


def evaluate_single_seed(df_scores: pd.DataFrame, triplets: list, seed: int) -> tuple:
    """
    Evaluate with 1:1 balanced ratio using a specific seed.
    
    For each triplet, samples 1 random negative (controlled by seed).
    
    Returns:
        Tuple of (y_true, y_scores, skipped)
    """
    random.seed(seed)
    
    # Build lookup: (anchor, target) -> score
    score_lookup = {}
    for _, row in df_scores.iterrows():
        key = (str(row['db1']), str(row['db2']))
        score_lookup[key] = row['score']
    
    y_true = []
    y_scores = []
    skipped = 0
    
    for item in triplets:
        anchor_id = str(item["anchor"])
        pos_id = str(item["positive"])
        neg_ids = [str(n) for n in item["negatives"]]
        
        # Sample 1 random negative for 1:1 ratio
        neg_id = random.choice(neg_ids)
        
        # Look up scores
        pos_key = (anchor_id, pos_id)
        neg_key = (anchor_id, neg_id)
        
        if pos_key not in score_lookup or neg_key not in score_lookup:
            skipped += 1
            continue
        
        # Positive pair
        y_true.append(1)
        y_scores.append(score_lookup[pos_key])
        
        # Negative pair (1:1 ratio)
        y_true.append(0)
        y_scores.append(score_lookup[neg_key])
    
    return y_true, y_scores, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SANTOS baseline (1:1 balanced, multi-seed)"
    )
    parser.add_argument("--scores", type=str, required=True, 
                        help="Input scores CSV from score_pairs.py")
    parser.add_argument("--triplets", type=str, default=None,
                        help="Original triplets JSONL for 1:1 sampling (if not provided, uses all pairs)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--seeds", type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help="Random seeds for negative sampling (default: 0 1 2 3 4)")
    parser.add_argument("--plot", type=str, default=None, 
                        help="Output ROC plot path")
    args = parser.parse_args()
    
    # Load scores
    df = pd.read_csv(args.scores)
    if len(df) == 0:
        print("No scores found.")
        return
    
    # If no triplets file, fall back to legacy single-evaluation mode
    if args.triplets is None:
        print("Warning: No --triplets file provided, using legacy single-evaluation mode")
        y_true = df["label"].values
        y_scores = df["score"].values
        
        try:
            auc_score = roc_auc_score(y_true, y_scores)
            print(f"ROC AUC: {auc_score:.4f}")
            
            if args.plot:
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                plt.figure()
                plt.plot(fpr, tpr, color='darkorange', lw=2, 
                         label=f'ROC curve (area = {auc_score:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")
                plt.savefig(args.plot)
                print(f"Saved ROC plot to {args.plot}")
                
        except Exception as e:
            print(f"Error calculating AUC: {e}")
            print(f"Labels distribution: {df['label'].value_counts()}")
        return
    
    # Load triplets for balanced evaluation
    print(f"📂 Loading triplets from {args.triplets}")
    triplets = []
    with open(args.triplets, "r") as f:
        for line in f:
            triplets.append(json.loads(line))
    print(f"   Loaded {len(triplets)} triplets")
    print(f"   Using 1:1 positive:negative ratio with {len(args.seeds)} seeds")
    
    # Evaluate across all seeds
    all_seed_metrics = []
    last_fpr, last_tpr = None, None
    
    for seed in args.seeds:
        y_true, y_scores, skipped = evaluate_single_seed(df, triplets, seed)
        
        if not y_true:
            print(f"⚠️  Seed {seed}: No valid pairs found!")
            continue
        
        if skipped > 0:
            print(f"   Seed {seed}: Skipped {skipped} triplets (missing scores)")
        
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
        last_fpr, last_tpr = fpr, tpr
    
    if not all_seed_metrics:
        print("❌ No valid results from any seed!")
        return
    
    # Aggregate metrics across seeds
    aggregated = {}
    metric_names = ['auc', 'precision', 'recall', 'f1', 'threshold']
    for metric in metric_names:
        values = [m[metric] for m in all_seed_metrics]
        aggregated[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
    
    # Create output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.dirname(args.scores)
    os.makedirs(output_dir, exist_ok=True)
    
    # Save ROC plot
    plot_path = args.plot or os.path.join(output_dir, "roc_curve.png")
    if last_fpr is not None:
        plt.figure()
        plt.plot(last_fpr, last_tpr, 
                 label=f'ROC curve (AUC = {aggregated["auc"]["mean"]:.4f}±{aggregated["auc"]["std"]:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.scatter(all_seed_metrics[-1]['best_fpr'], all_seed_metrics[-1]['best_tpr'],
                    color='red', label=f'Optimal threshold = {aggregated["threshold"]["mean"]:.4f}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (1:1 Balanced)')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()
    
    # Save summary (matching preprocess evaluator format)
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
        f.write(f"Total triplets: {len(triplets)}\n")
        f.write(f"Seeds used: {args.seeds}\n")
        f.write(f"Pos:Neg ratio: 1:1\n")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"📊 SANTOS Results (1:1 ratio, {len(args.seeds)} seeds, optimal threshold)")
    print(f"{'='*60}")
    print(f"AUC-ROC:    {aggregated['auc']['mean']:.4f} ± {aggregated['auc']['std']:.4f}")
    print(f"Precision:  {aggregated['precision']['mean']:.4f} ± {aggregated['precision']['std']:.4f}")
    print(f"Recall:     {aggregated['recall']['mean']:.4f} ± {aggregated['recall']['std']:.4f}")
    print(f"F1 Score:   {aggregated['f1']['mean']:.4f} ± {aggregated['f1']['std']:.4f}")
    print(f"Threshold:  {aggregated['threshold']['mean']:.4f} ± {aggregated['threshold']['std']:.4f}")
    print(f"{'='*60}")
    print(f"💾 Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
