"""
Generate plot for number of negatives ablation study.

Shows AUC-ROC vs number of negatives with error bars.
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple


def parse_summary(summary_path: str) -> Dict[str, float]:
    """Parse summary file for AUC metric."""
    metrics = {}
    with open(summary_path, 'r') as f:
        content = f.read()
    
    auc_match = re.search(r'AUC:\s*([0-9.]+)', content)
    if auc_match:
        metrics['auc'] = float(auc_match.group(1))
    
    return metrics


def collect_neg_results(results_dir: str, num_neg: int) -> Tuple[float, float]:
    """Collect and aggregate results for a number of negatives across seeds."""
    # Look for results in out/graph_full_ss3_neg{num}/ structure
    neg_dir = Path(results_dir) / f"graph_full_ss3_neg{num_neg}"
    
    # Fallback to old structure if new one doesn't exist
    if not neg_dir.exists():
        neg_dir = Path(results_dir) / f"neg_{num_neg}"
    
    auc_values = []
    for seed_dir in sorted(neg_dir.glob("test_results_seed*")):
        summary_path = seed_dir / "summary.txt"
        if summary_path.exists():
            metrics = parse_summary(str(summary_path))
            if 'auc' in metrics:
                auc_values.append(metrics['auc'])
    
    if auc_values:
        return np.mean(auc_values), np.std(auc_values)
    return None, None


def plot_ablation_num_negatives(
    results_dir: str,
    output_path: str,
    num_negatives: List[int] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> str:
    """Generate plot for number of negatives ablation."""
    
    if num_negatives is None:
        num_negatives = [2, 4, 6, 10, 15]
    
    # Collect results
    means = []
    stds = []
    valid_nums = []
    
    for num in num_negatives:
        mean, std = collect_neg_results(results_dir, num)
        if mean is not None:
            means.append(mean)
            stds.append(std)
            valid_nums.append(num)
    
    if not valid_nums:
        print("No results found!")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.errorbar(valid_nums, means, yerr=stds,
                marker='s', markersize=8, capsize=5,
                linewidth=2, color='#28A745')
    
    # Highlight default (6)
    if 6 in valid_nums:
        idx = valid_nums.index(6)
        ax.scatter([6], [means[idx]], s=150, c='#E94F37',
                   zorder=5, label='Default (6)')
    
    ax.set_xlabel('Number of Negatives per Triplet', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('Effect of Negative Samples on Contrastive Learning', fontsize=14)
    ax.set_xticks(valid_nums)
    ax.grid(True, alpha=0.3)
    
    if 6 in valid_nums:
        ax.legend(loc='lower right')
    
    # Y-axis formatting
    if means:
        ax.set_ylim([min(means) - 0.02, max(means) + 0.02])
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved num negatives ablation plot to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate num negatives ablation plot")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory with ablation results")
    parser.add_argument("--output", type=str, default="fig/ablation_num_negatives.png",
                        help="Output path for plot")
    parser.add_argument("--nums", type=int, nargs='+', default=[2, 4, 6, 10, 15],
                        help="Number of negatives to plot")
    
    args = parser.parse_args()
    
    plot_ablation_num_negatives(
        results_dir=args.results_dir,
        output_path=args.output,
        num_negatives=args.nums
    )
