"""
Generate plot for sample size ablation study.

Shows AUC-ROC vs sample size with error bars.
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


def collect_size_results(results_dir: str, size: int) -> Tuple[float, float]:
    """Collect and aggregate results for a sample size across seeds."""
    # Look for results in out/graph_full_ss{size}_neg6/ structure
    size_dir = Path(results_dir) / f"graph_full_ss{size}_neg6"
    
    # Fallback to old structure if new one doesn't exist
    if not size_dir.exists():
        size_dir = Path(results_dir) / f"size_{size}"
    
    auc_values = []
    for seed_dir in sorted(size_dir.glob("test_results_seed*")):
        summary_path = seed_dir / "summary.txt"
        if summary_path.exists():
            metrics = parse_summary(str(summary_path))
            if 'auc' in metrics:
                auc_values.append(metrics['auc'])
    
    if auc_values:
        return np.mean(auc_values), np.std(auc_values)
    return None, None


def plot_ablation_sample_size(
    results_dir: str,
    output_path: str,
    sizes: List[int] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> str:
    """Generate plot for sample size ablation."""
    
    if sizes is None:
        sizes = [1, 3, 5, 10]
    
    # Collect results
    means = []
    stds = []
    valid_sizes = []
    
    for size in sizes:
        mean, std = collect_size_results(results_dir, size)
        if mean is not None:
            means.append(mean)
            stds.append(std)
            valid_sizes.append(size)
    
    if not valid_sizes:
        print("No results found!")
        return None
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.errorbar(valid_sizes, means, yerr=stds, 
                marker='o', markersize=8, capsize=5,
                linewidth=2, color='#2E86AB')
    
    # Highlight default (3)
    if 3 in valid_sizes:
        idx = valid_sizes.index(3)
        ax.scatter([3], [means[idx]], s=150, c='#E94F37', 
                   zorder=5, label='Default (3)')
    
    ax.set_xlabel('Sample Size (values per column)', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('Effect of Sample Size on Embedding Quality', fontsize=14)
    ax.set_xticks(valid_sizes)
    ax.grid(True, alpha=0.3)
    
    if 3 in valid_sizes:
        ax.legend(loc='lower right')
    
    # Y-axis formatting
    ax.set_ylim([min(means) - 0.02, max(means) + 0.02])
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved sample size ablation plot to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample size ablation plot")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory with ablation results")
    parser.add_argument("--output", type=str, default="fig/ablation_sample_size.png",
                        help="Output path for plot")
    parser.add_argument("--sizes", type=int, nargs='+', default=[1, 3, 5, 10],
                        help="Sample sizes to plot")
    
    args = parser.parse_args()
    
    plot_ablation_sample_size(
        results_dir=args.results_dir,
        output_path=args.output,
        sizes=args.sizes
    )
