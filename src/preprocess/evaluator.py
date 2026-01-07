#!/usr/bin/env python
"""
Evaluator for trained BGE-M3 embedding model.

Evaluates the model on test triplets using precomputed embeddings.
Uses 1:1 positive/negative ratio with multiple seeds for robustness.
Outputs summary.txt and predictions.csv for downstream analysis.

Usage:
    python -m preprocess.evaluator \
        --embedding-path out/database_embeddings.pt \
        --test-triplets out/triplets/triplets_test.jsonl \
        --output-dir out/test_results
"""

import os
import sys
import csv
import json
import random
import argparse
import numpy as np

# Parse --gpu argument early, before importing torch
def parse_gpu_arg():
    """Parse --gpu argument early, before importing torch."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU device ID(s) to use (e.g., '0', '1', '0,1')")
    args, _ = parser.parse_known_args()
    return args.gpu


# Set GPU before importing torch
gpu_id = parse_gpu_arg()
if gpu_id is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    print(f"Setting CUDA_VISIBLE_DEVICES={gpu_id}")

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_single_seed(
    triplets: list,
    embedding_matrix: torch.Tensor,
    db_id_to_index: dict,
    seed: int
) -> tuple:
    """
    Evaluate embeddings on test triplets with 1:1 ratio using a specific seed.
    
    For each triplet, samples 1 random negative (controlled by seed) to achieve 1:1 ratio.
    
    Returns:
        Tuple of (y_true, y_scores, records, skipped)
    """
    random.seed(seed)
    
    y_true = []
    y_scores = []
    records = []
    skipped = 0
    
    with torch.no_grad():
        for item in triplets:
            anchor_id = item["anchor"]
            pos_id = item["positive"]
            neg_ids = item["negatives"]
            
            # Sample 1 random negative for 1:1 ratio
            neg_id = random.choice(neg_ids)
            
            # Check if all IDs exist in embeddings
            if anchor_id not in db_id_to_index or pos_id not in db_id_to_index or neg_id not in db_id_to_index:
                skipped += 1
                continue
            
            emb_anchor = embedding_matrix[db_id_to_index[anchor_id]].unsqueeze(0)
            emb_pos = embedding_matrix[db_id_to_index[pos_id]].unsqueeze(0)
            emb_neg = embedding_matrix[db_id_to_index[neg_id]].unsqueeze(0)
            
            # Positive pair
            sim_pos = F.cosine_similarity(emb_anchor, emb_pos).item()
            y_true.append(1)
            y_scores.append(sim_pos)
            records.append((anchor_id, pos_id, sim_pos, 1))
            
            # Negative pair (1:1 ratio)
            sim_neg = F.cosine_similarity(emb_anchor, emb_neg).item()
            y_true.append(0)
            y_scores.append(sim_neg)
            records.append((anchor_id, neg_id, sim_neg, 0))
    
    return y_true, y_scores, records, skipped


def compute_metrics_at_threshold(y_true, y_scores, threshold):
    """Compute precision, recall, f1 at a specific threshold."""
    y_pred = (np.array(y_scores) >= threshold).astype(int)
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


def evaluate_embeddings(
    embedding_path: str,
    test_triplets_path: str,
    output_dir: str,
    seeds: list = None
) -> dict:
    """
    Evaluate embeddings on test triplets with 1:1 ratio across multiple seeds.
    
    Args:
        embedding_path: Path to precomputed embeddings .pt file
        test_triplets_path: Path to test triplets JSONL file
        output_dir: Directory to save results
        seeds: List of random seeds for negative sampling (default: [0,1,2,3,4])
    
    Returns:
        Dict with aggregated metrics (mean ± std)
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load embeddings
    print(f"📂 Loading precomputed embeddings from {embedding_path}")
    saved_data = torch.load(embedding_path, map_location=device, weights_only=True)
    db_id_to_index = saved_data["db_id_to_index"]
    embedding_matrix = saved_data["embeddings"].to(device)
    print(f"   Loaded {len(db_id_to_index)} embeddings")
    
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
            triplets, embedding_matrix, db_id_to_index, seed
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
    metric_names = ['auc', 'precision', 'recall', 'f1', 'threshold']
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
    plt.title('Receiver Operating Characteristic (1:1 Balanced)')
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
        f.write(f"Total triplets: {len(triplets)}\n")
        f.write(f"Seeds used: {seeds}\n")
        f.write(f"Positive pairs: {num_positive} ({pos_ratio:.1f}%)\n")
        f.write(f"Negative pairs: {num_negative} ({100-pos_ratio:.1f}%)\n")
        f.write(f"Pos:Neg ratio: 1:1\n")
    
    # Print results
    print(f"\n{'='*60}")
    print(f"📊 Results (1:1 ratio, {len(seeds)} seeds, optimal threshold)")
    print(f"{'='*60}")
    print(f"AUC-ROC:    {aggregated['auc']['mean']:.4f} ± {aggregated['auc']['std']:.4f}")
    print(f"Precision:  {aggregated['precision']['mean']:.4f} ± {aggregated['precision']['std']:.4f}")
    print(f"Recall:     {aggregated['recall']['mean']:.4f} ± {aggregated['recall']['std']:.4f}")
    print(f"F1 Score:   {aggregated['f1']['mean']:.4f} ± {aggregated['f1']['std']:.4f}")
    print(f"Threshold:  {aggregated['threshold']['mean']:.4f} ± {aggregated['threshold']['std']:.4f}")
    print(f"{'='*60}")
    print(f"📊 Class distribution: {num_positive} positive (50%), {num_negative} negative (50%)")
    print(f"💾 Results saved in: {output_dir}")
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate BGE-M3 embeddings on test triplets (1:1 balanced)"
    )
    parser.add_argument(
        "--embedding-path",
        type=str,
        required=True,
        help="Path to precomputed embeddings .pt file"
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
    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="GPU device ID to use"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.embedding_path):
        print(f"Error: Embedding file not found: {args.embedding_path}")
        sys.exit(1)
    if not os.path.exists(args.test_triplets):
        print(f"Error: Test triplets file not found: {args.test_triplets}")
        sys.exit(1)
    
    evaluate_embeddings(
        embedding_path=args.embedding_path,
        test_triplets_path=args.test_triplets,
        output_dir=args.output_dir,
        seeds=args.seeds
    )


if __name__ == "__main__":
    main()
