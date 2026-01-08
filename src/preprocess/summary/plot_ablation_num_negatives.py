"""
Generate professional plot for number of negatives ablation study.

Shows both AUC-ROC and F1-Score vs number of negatives with error bars.
"""

import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_summary(summary_path: str) -> Dict[str, float]:
    """Parse summary file for F1 and AUC metrics with std."""
    metrics = {}
    with open(summary_path, 'r') as f:
        content = f.read()
    
    # Parse F1 and F1_std
    f1_match = re.search(r'F1:\s*([0-9.]+)', content)
    if f1_match:
        metrics['f1'] = float(f1_match.group(1))
    
    f1_std_match = re.search(r'F1_std:\s*([0-9.]+)', content)
    if f1_std_match:
        metrics['f1_std'] = float(f1_std_match.group(1))
    
    # Parse AUC and AUC_std
    auc_match = re.search(r'AUC:\s*([0-9.]+)', content)
    if auc_match:
        metrics['auc'] = float(auc_match.group(1))
    
    auc_std_match = re.search(r'AUC_std:\s*([0-9.]+)', content)
    if auc_std_match:
        metrics['auc_std'] = float(auc_std_match.group(1))
    
    return metrics


def collect_neg_results(results_dir: str, num_neg: int) -> Dict[str, Optional[float]]:
    """Collect results for a number of negatives."""
    neg_dir = Path(results_dir) / f"graph_full_ss3_neg{num_neg}"
    
    if not neg_dir.exists():
        return {}
    
    single_summary = neg_dir / "test_results" / "summary.txt"
    if single_summary.exists():
        return parse_summary(str(single_summary))
    
    return {}


def plot_ablation_num_negatives(
    results_dir: str,
    output_path: str,
    num_negatives: List[int] = None,
    figsize: Tuple[float, float] = (8, 6),
    include_f1: bool = False
) -> Optional[str]:
    """Generate professional plot for number of negatives ablation.
    
    Args:
        include_f1: If True, plot both AUC and F1. If False (default), only plot AUC.
    """
    
    if num_negatives is None:
        num_negatives = [2, 4, 6, 10]
    
    # Collect results
    auc_means, auc_stds = [], []
    f1_means, f1_stds = [], []
    valid_nums = []
    
    print("=" * 70)
    print("Number of Negatives Ablation Results")
    print("=" * 70)
    print(f"{'Num Neg':<10} {'AUC':<10} {'AUC_std':<10} {'F1':<10} {'F1_std':<10}")
    print("-" * 70)
    
    for num in num_negatives:
        metrics = collect_neg_results(results_dir, num)
        if 'auc' in metrics:
            auc_means.append(metrics['auc'])
            auc_stds.append(metrics.get('auc_std', 0.0))
            f1_means.append(metrics.get('f1', metrics['auc']))  # Fallback to AUC
            f1_stds.append(metrics.get('f1_std', 0.0))
            valid_nums.append(num)
            print(f"{num:<10} {metrics['auc']:<10.4f} {metrics.get('auc_std', 0.0):<10.4f} "
                  f"{metrics.get('f1', 'N/A'):<10} {metrics.get('f1_std', 0.0):<10.4f}")
        else:
            print(f"{num:<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    print("=" * 70)
    
    if not valid_nums:
        print("❌ No results found!")
        return None
    
    # Load baseline (original pretrained model) results
    baseline_path = Path(results_dir) / "original_bge-m3" / "test_results" / "summary.txt"
    baseline_auc, baseline_f1 = None, None
    if baseline_path.exists():
        baseline_metrics = parse_summary(str(baseline_path))
        baseline_auc = baseline_metrics.get('auc')
        baseline_f1 = baseline_metrics.get('f1')
        print(f"Baseline (original): AUC={baseline_auc:.4f}, F1={baseline_f1:.4f}" if baseline_f1 else f"Baseline: AUC={baseline_auc:.4f}")
    
    # Create professional plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=figsize)
    
    # Baseline horizontal lines (dotted)
    if baseline_auc is not None:
        ax.axhline(y=baseline_auc, color='#666666', linestyle=':', linewidth=2, 
                   label=f'Baseline AUC ({baseline_auc:.4f})', alpha=0.8)
    if include_f1 and baseline_f1 is not None:
        ax.axhline(y=baseline_f1, color='#999999', linestyle=':', linewidth=2,
                   label=f'Baseline F1 ({baseline_f1:.4f})', alpha=0.8)
    
    # AUC line (green)
    ax.errorbar(valid_nums, auc_means, yerr=auc_stds,
                marker='o', markersize=10, capsize=6, capthick=2,
                linewidth=2.5, color='#28A745', ecolor='#28A745',
                markerfacecolor='white', markeredgewidth=2.5,
                label='AUC-ROC (finetuned)')
    
    # F1 line (blue) - only if include_f1 is True
    if include_f1:
        ax.errorbar(valid_nums, f1_means, yerr=f1_stds,
                    marker='s', markersize=9, capsize=6, capthick=2,
                    linewidth=2.5, color='#2196F3', ecolor='#2196F3',
                    markerfacecolor='white', markeredgewidth=2.5,
                    label='F1-Score (finetuned)')
    
    # Labels and title
    ax.set_xlabel('Number of Negatives per Triplet', fontsize=14, fontweight='bold')
    ax.set_ylabel('AUC-ROC' if not include_f1 else 'Score', fontsize=14, fontweight='bold')
    ax.set_title('Effect of Negative Samples on Contrastive Learning', fontsize=16, fontweight='bold', pad=15)
    
    # X-axis formatting
    ax.set_xticks(valid_nums)
    ax.set_xticklabels([str(n) for n in valid_nums], fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    
    # Y-axis formatting - include baseline in range
    y_min = 0.95 if baseline_auc and baseline_auc < 0.97 else 0.96
    ax.set_ylim([y_min, 1.0])
    
    # Grid
    ax.grid(True, alpha=0.4, linestyle='--')
    ax.set_axisbelow(True)
    
    # Legend (show baseline and optionally F1)
    ax.legend(loc='lower right', fontsize=10, framealpha=0.95)
    
    # Tight layout and save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✅ Saved num negatives ablation plot to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate num negatives ablation plot")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory with ablation results")
    parser.add_argument("--output", type=str, default="fig/ablation_num_negatives.png",
                        help="Output path for plot")
    parser.add_argument("--nums", type=int, nargs='+', default=[2, 4, 6, 10],
                        help="Number of negatives to plot")
    parser.add_argument("--include-f1", action="store_true",
                        help="Include F1-Score in addition to AUC (default: AUC only)")
    
    args = parser.parse_args()
    
    plot_ablation_num_negatives(
        results_dir=args.results_dir,
        output_path=args.output,
        num_negatives=args.nums,
        include_f1=args.include_f1
    )
