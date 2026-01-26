#!/usr/bin/env python
"""
GitTables Evaluator

Evaluator for pre-serialized GitTables triplets.
Computes embeddings on-the-fly and evaluates similarity metrics.

Usage:
    python -m preprocess.GitTables.evaluator_gittables \
        --test-triplets triplets/triplets_test.jsonl \
        --model-path out/model/best \
        --output-dir out/test_results
"""

import argparse
import os
import sys
import json
import random
import numpy as np


def parse_gpu_arg():
    """Parse --gpu argument early, before importing torch."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU device ID(s) to use")
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
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model.BGEEmbedder import BGEEmbedder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_triplets(path: str) -> list:
    """Load triplets from JSONL file."""
    triplets = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                triplets.append(json.loads(line))
    return triplets


def evaluate_single_seed(
    triplets: list,
    embedder: BGEEmbedder,
    seed: int,
    batch_size: int = 32,
):
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
    
    for triplet in tqdm(triplets, desc=f"Evaluating (seed={seed})"):
        anchor_text = triplet['anchor']
        pos_text = triplet['positive']
        neg_texts = triplet['negatives']
        
        if not neg_texts:
            skipped += 1
            continue
        
        # Sample 1 random negative
        neg_text = random.choice(neg_texts)
        
        # Compute embeddings
        texts = [anchor_text, pos_text, neg_text]
        embs = embedder.get_embedding(texts, batch_size=batch_size)
        
        anchor_emb = embs[0:1]
        pos_emb = embs[1:2]
        neg_emb = embs[2:3]
        
        # Compute similarities
        pos_sim = F.cosine_similarity(anchor_emb, pos_emb).item()
        neg_sim = F.cosine_similarity(anchor_emb, neg_emb).item()
        
        # Record positive pair
        y_true.append(1)
        y_scores.append(pos_sim)
        records.append({
            'anchor': triplet.get('metadata', {}).get('table_id', 'unknown'),
            'target': 'positive',
            'similarity': pos_sim,
            'label': 1,
            'split_type': triplet.get('metadata', {}).get('split_type', 'unknown'),
        })
        
        # Record negative pair
        y_true.append(0)
        y_scores.append(neg_sim)
        records.append({
            'anchor': triplet.get('metadata', {}).get('table_id', 'unknown'),
            'target': 'negative',
            'similarity': neg_sim,
            'label': 0,
            'split_type': triplet.get('metadata', {}).get('split_type', 'unknown'),
        })
    
    return y_true, y_scores, records, skipped


def compute_metrics_at_threshold(y_true, y_scores, threshold):
    """Compute precision, recall, f1, accuracy at a specific threshold."""
    y_pred = [1 if s >= threshold else 0 for s in y_scores]
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    return precision, recall, f1, accuracy


def evaluate_gittables(
    test_triplets_path: str,
    model_path: str,
    output_dir: str,
    model_type: str = "bge-m3",
    seeds: list = None,
    batch_size: int = 32,
):
    """
    Evaluate model on GitTables test triplets.
    
    Args:
        test_triplets_path: Path to test triplets JSONL
        model_path: Path to trained model
        output_dir: Output directory for results
        model_type: Model type
        seeds: List of random seeds for evaluation
        batch_size: Batch size for embedding
        
    Returns:
        Dict with aggregated metrics
    """
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("GitTables Model Evaluation")
    print("=" * 60)
    print(f"Test triplets: {test_triplets_path}")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Seeds: {seeds}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    embedder = BGEEmbedder(model_type=model_type, model_path=model_path)
    embedder.model.eval()
    
    # Load triplets
    print("Loading test triplets...")
    triplets = load_triplets(test_triplets_path)
    print(f"Loaded {len(triplets)} triplets")
    
    # Evaluate across seeds
    all_seed_metrics = []
    
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        
        with torch.no_grad():
            y_true, y_scores, records, skipped = evaluate_single_seed(
                triplets, embedder, seed, batch_size
            )
        
        if not y_true:
            print(f"No valid triplets for seed {seed}")
            continue
        
        # ROC + AUC
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Best threshold (Youden's J)
        youden_j = tpr - fpr
        best_idx = youden_j.argmax()
        best_threshold = thresholds[best_idx]
        
        # Metrics at best threshold
        precision, recall, f1, accuracy = compute_metrics_at_threshold(
            y_true, y_scores, best_threshold
        )
        
        seed_metrics = {
            'seed': seed,
            'auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'best_threshold': best_threshold,
            'skipped': skipped,
        }
        all_seed_metrics.append(seed_metrics)
        
        print(f"AUC: {roc_auc:.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}")
    
    # Aggregate metrics
    if all_seed_metrics:
        metrics_summary = {}
        for key in ['auc', 'precision', 'recall', 'f1', 'accuracy']:
            values = [m[key] for m in all_seed_metrics]
            metrics_summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'values': values,
            }
        
        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY (Mean ± Std)")
        print("=" * 60)
        for key in ['auc', 'precision', 'recall', 'f1', 'accuracy']:
            mean = metrics_summary[key]['mean']
            std = metrics_summary[key]['std']
            print(f"{key.upper():12s}: {mean:.4f} ± {std:.4f}")
        
        # Save summary
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, 'w') as f:
            f.write("GitTables Evaluation Summary\n")
            f.write("=" * 40 + "\n")
            for key in ['auc', 'precision', 'recall', 'f1', 'accuracy']:
                mean = metrics_summary[key]['mean']
                std = metrics_summary[key]['std']
                f.write(f"{key.upper():12s}: {mean:.4f} ± {std:.4f}\n")
        
        # Save detailed results
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump({
                'summary': {k: {'mean': v['mean'], 'std': v['std']} 
                           for k, v in metrics_summary.items()},
                'per_seed': all_seed_metrics,
            }, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        
        return metrics_summary
    
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model on GitTables test triplets"
    )
    
    parser.add_argument("--test-triplets", type=str, required=True,
                        help="Path to test triplets JSONL file")
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to trained model (default: pretrained)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--model-type", type=str, default="bge-m3",
                        help="Model type (default: bge-m3)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="Random seeds for evaluation (default: 0 1 2 3 4)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--gpu", type=str, default=None,
                        help="GPU device ID")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if not os.path.exists(args.test_triplets):
        print(f"Error: Test triplets not found: {args.test_triplets}")
        sys.exit(1)
    
    evaluate_gittables(
        test_triplets_path=args.test_triplets,
        model_path=args.model_path,
        output_dir=args.output_dir,
        model_type=args.model_type,
        seeds=args.seeds,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
