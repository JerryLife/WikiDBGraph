"""
Plot ROC curve comparison between original and finetuned embedding models.

Reads evaluation results from log files and generates comparison plots.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_roc_data(roc_data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse ROC data from CSV file.
    
    Args:
        roc_data_path: Path to roc_data.csv file
    
    Returns:
        Tuple of (fpr, tpr) arrays
    """
    df = pd.read_csv(roc_data_path)
    return df['fpr'].values, df['tpr'].values


def parse_summary(summary_path: str) -> Dict[str, float]:
    """
    Parse summary file for metrics.
    
    Args:
        summary_path: Path to summary.txt file
    
    Returns:
        Dict with AUC, best_threshold, best_tpr, best_fpr
    """
    metrics = {}
    with open(summary_path, 'r') as f:
        content = f.read()
    
    # Parse AUC
    auc_match = re.search(r'AUC:\s*([0-9.]+)', content)
    if auc_match:
        metrics['auc'] = float(auc_match.group(1))
    
    # Parse best threshold
    thresh_match = re.search(r'Best Threshold:\s*([0-9.]+)', content)
    if thresh_match:
        metrics['best_threshold'] = float(thresh_match.group(1))
    
    # Parse TPR/FPR
    tpr_match = re.search(r'Best TPR:\s*([0-9.]+)', content)
    if tpr_match:
        metrics['best_tpr'] = float(tpr_match.group(1))
    
    fpr_match = re.search(r'Best FPR:\s*([0-9.]+)', content)
    if fpr_match:
        metrics['best_fpr'] = float(fpr_match.group(1))
    
    return metrics


def collect_seed_results(results_dir: str, pattern: str = "test_results_seed*") -> List[Dict]:
    """
    Collect results from multiple seed runs.
    
    Args:
        results_dir: Directory containing test results
        pattern: Glob pattern for seed directories
    
    Returns:
        List of result dictionaries
    """
    results = []
    results_path = Path(results_dir)
    
    for seed_dir in sorted(results_path.glob(pattern)):
        summary_path = seed_dir / "summary.txt"
        roc_path = seed_dir / "roc_data.csv"
        
        if summary_path.exists():
            result = parse_summary(str(summary_path))
            result['seed_dir'] = str(seed_dir)
            
            if roc_path.exists():
                fpr, tpr = parse_roc_data(str(roc_path))
                result['fpr'] = fpr
                result['tpr'] = tpr
            
            results.append(result)
    
    return results


def aggregate_metrics(results: List[Dict]) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate metrics across seeds.
    
    Args:
        results: List of result dictionaries
    
    Returns:
        Dict mapping metric name to (mean, std) tuple
    """
    metrics = {}
    metric_names = ['auc', 'best_threshold', 'best_tpr', 'best_fpr']
    
    for name in metric_names:
        values = [r[name] for r in results if name in r]
        if values:
            metrics[name] = (np.mean(values), np.std(values))
    
    return metrics


def plot_roc_comparison(
    original_results_dir: str,
    finetuned_results_dir: str,
    output_path: str,
    title: str = "ROC Curve Comparison",
    figsize: Tuple[int, int] = (8, 8)
) -> str:
    """
    Plot ROC curve comparison between original and finetuned models.
    
    Args:
        original_results_dir: Directory with original model results
        finetuned_results_dir: Directory with finetuned model results
        output_path: Output path for the figure
        title: Plot title
        figsize: Figure size
    
    Returns:
        Path to saved figure
    """
    # Collect results
    original_results = collect_seed_results(original_results_dir)
    finetuned_results = collect_seed_results(finetuned_results_dir)
    
    if not original_results and not finetuned_results:
        raise ValueError("No results found in either directory")
    
    # Aggregate metrics
    original_metrics = aggregate_metrics(original_results) if original_results else {}
    finetuned_metrics = aggregate_metrics(finetuned_results) if finetuned_results else {}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot original model ROC (use first seed for curve, show mean AUC in legend)
    if original_results and 'fpr' in original_results[0]:
        fpr = original_results[0]['fpr']
        tpr = original_results[0]['tpr']
        auc_mean = original_metrics.get('auc', (0, 0))[0]
        ax.plot(fpr, tpr, color='#4f9d8e', linewidth=2,
                label=f'Original (AUC={auc_mean:.4f})')
    
    # Plot finetuned model ROC
    if finetuned_results and 'fpr' in finetuned_results[0]:
        fpr = finetuned_results[0]['fpr']
        tpr = finetuned_results[0]['tpr']
        auc_mean = finetuned_metrics.get('auc', (0, 0))[0]
        ax.plot(fpr, tpr, color='#9d4f8e', linewidth=2,
                label=f'Fine-tuned (AUC={auc_mean:.4f})')
    
    # Plot diagonal
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    
    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if title:
        ax.set_title(title)
    
    # Save figure
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.tight_layout()
    
    # Save both PNG and PDF
    base_path = os.path.splitext(output_path)[0]
    plt.savefig(f"{base_path}.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{base_path}.pdf", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved ROC comparison to {base_path}.png and {base_path}.pdf")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ROC curve comparison")
    parser.add_argument("--original", type=str, required=True,
                        help="Directory with original model results")
    parser.add_argument("--finetuned", type=str, required=True,
                        help="Directory with finetuned model results")
    parser.add_argument("--output", type=str, default="fig/roc_comparison.png",
                        help="Output path for figure")
    parser.add_argument("--title", type=str, default="",
                        help="Plot title (empty for no title)")
    
    args = parser.parse_args()
    
    plot_roc_comparison(
        original_results_dir=args.original,
        finetuned_results_dir=args.finetuned,
        output_path=args.output,
        title=args.title
    )
