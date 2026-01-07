"""
Generate professional plot for sample size ablation study.

Shows AUC-ROC vs sample size with error bars.
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_summary(summary_path: str) -> Dict[str, float]:
    """Parse summary file for AUC metric."""
    metrics = {}
    with open(summary_path, 'r') as f:
        content = f.read()
    
    auc_match = re.search(r'AUC:\s*([0-9.]+)', content)
    if auc_match:
        metrics['auc'] = float(auc_match.group(1))
    
    thresh_match = re.search(r'Best Threshold:\s*([0-9.]+)', content)
    if thresh_match:
        metrics['threshold'] = float(thresh_match.group(1))
    
    return metrics


def collect_size_results(results_dir: str, size: int) -> Tuple[Optional[float], Optional[float]]:
    """Collect and aggregate results for a sample size across seeds."""
    # Look for results in out/graph_full_ss{size}_neg6/ structure
    size_dir = Path(results_dir) / f"graph_full_ss{size}_neg6"
    
    # Fallback to old structure if new one doesn't exist
    if not size_dir.exists():
        size_dir = Path(results_dir) / f"size_{size}"
    
    if not size_dir.exists():
        return None, None
    
    auc_values = []
    
    # Try new structure: test_results/summary.txt (single run)
    single_summary = size_dir / "test_results" / "summary.txt"
    if single_summary.exists():
        metrics = parse_summary(str(single_summary))
        if 'auc' in metrics:
            auc_values.append(metrics['auc'])
    
    # Also try old structure: test_results_seed*/summary.txt (multi-seed)
    for seed_dir in sorted(size_dir.glob("test_results_seed*")):
        summary_path = seed_dir / "summary.txt"
        if summary_path.exists():
            metrics = parse_summary(str(summary_path))
            if 'auc' in metrics:
                auc_values.append(metrics['auc'])
    
    if auc_values:
        return np.mean(auc_values), np.std(auc_values) if len(auc_values) > 1 else 0.0
    return None, None


def plot_ablation_sample_size(
    results_dir: str,
    output_path: str,
    sizes: List[int] = None,
    figsize: Tuple[float, float] = (8, 6)
) -> Optional[str]:
    """Generate professional plot for sample size ablation."""
    
    if sizes is None:
        sizes = [1, 3, 5, 10]
    
    # Collect results
    means = []
    stds = []
    valid_sizes = []
    
    print("=" * 60)
    print("Sample Size Ablation Results")
    print("=" * 60)
    print(f"{'Sample Size':<15} {'AUC':<12} {'Std':<12}")
    print("-" * 40)
    
    for size in sizes:
        mean, std = collect_size_results(results_dir, size)
        if mean is not None:
            means.append(mean)
            stds.append(std)
            valid_sizes.append(size)
            print(f"{size:<15} {mean:<12.4f} {std:<12.4f}")
        else:
            print(f"{size:<15} {'N/A':<12} {'N/A':<12}")
    
    print("=" * 60)
    
    if not valid_sizes:
        print("❌ No results found!")
        return None
    
    # Create professional plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Main line with markers and error bars
    line_color = '#2E86AB'
    ax.errorbar(valid_sizes, means, yerr=stds,
                marker='o', markersize=10, capsize=6, capthick=2,
                linewidth=2.5, color=line_color, ecolor=line_color,
                markerfacecolor='white', markeredgewidth=2.5,
                label='AUC-ROC')
    
    # Highlight default (3) with a different marker
    if 3 in valid_sizes:
        idx = valid_sizes.index(3)
        ax.scatter([3], [means[idx]], s=200, c='#E94F37', 
                   marker='*', zorder=5, label='Default (3)',
                   edgecolors='white', linewidths=1.5)
    
    # Find best result and annotate
    best_idx = np.argmax(means)
    best_size = valid_sizes[best_idx]
    best_auc = means[best_idx]
    
    # Labels and title
    ax.set_xlabel('Sample Size (values per column)', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUC-ROC', fontsize=14, fontweight='bold')
    ax.set_title('Effect of Sample Size on Embedding Quality', fontsize=16, fontweight='bold', pad=15)
    
    # X-axis formatting
    ax.set_xticks(valid_sizes)
    ax.set_xticklabels([str(s) for s in valid_sizes], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Y-axis formatting - expand range slightly
    if means:
        y_min = min(means) - max(stds) * 1.5 if stds else min(means) - 0.02
        y_max = max(means) + max(stds) * 1.5 if stds else max(means) + 0.02
        y_margin = (y_max - y_min) * 0.1
        ax.set_ylim([max(0, y_min - y_margin), min(1.0, y_max + y_margin)])
    
    # Grid
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='lower right', fontsize=11, framealpha=0.95)
    
    # Add subtle annotations for best result
    if len(valid_sizes) > 1:
        ax.annotate(f'Best: {best_auc:.4f}', 
                    xy=(best_size, best_auc), 
                    xytext=(best_size + 0.5, best_auc + 0.01),
                    fontsize=10, color='#333333',
                    arrowprops=dict(arrowstyle='->', color='#666666', lw=1))
    
    # Tight layout and save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
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
