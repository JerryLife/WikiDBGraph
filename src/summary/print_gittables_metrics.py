#!/usr/bin/env python3
"""
GitTables Prediction Metrics Comparison

Compares pretrained (BAAI/bge-m3) vs fine-tuned (contrastive) model results
and generates a LaTeX table.

Usage:
    python src/summary/print_gittables_metrics.py
    python src/summary/print_gittables_metrics.py --results-dir out/gittables/both_full_ss3_neg2
"""

import argparse
import os
import re
from typing import Dict, Optional, Tuple


def parse_summary_file(filepath: str) -> Dict[str, Tuple[float, float]]:
    """Parse summary.txt file and extract metrics (mean ± std)."""
    metrics = {}
    
    if not os.path.exists(filepath):
        return metrics
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' not in line or '±' not in line:
                continue
            
            # Parse lines like "AUC         : 0.8297 ± 0.0008"
            match = re.match(r'(\w+)\s*:\s*(\d+\.\d+)\s*±\s*(\d+\.\d+)', line)
            if match:
                metric_name = match.group(1).lower()
                mean = float(match.group(2))
                std = float(match.group(3))
                metrics[metric_name] = (mean, std)
    
    return metrics


def format_metric(mean: Optional[float], std: Optional[float]) -> str:
    """Format metric as mean±std."""
    if mean is None:
        return "N/A"
    if std is not None and std > 0:
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.4f}"


def format_pct(value: float) -> str:
    """Format percentage for LaTeX."""
    return f"{value:.1f}\\%"


def generate_latex_table(
    pretrained: Dict[str, Tuple[float, float]],
    finetuned: Dict[str, Tuple[float, float]],
    output_path: Optional[str] = None
) -> str:
    """Generate LaTeX table comparing pretrained vs fine-tuned."""
    
    metrics_order = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    metrics_display = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1',
        'auc': 'AUC-ROC'
    }
    
    latex = r"""\begin{table}[tb]
    \centering
    \caption{GitTables table partition matching: Pretrained (BAAI/bge-m3) vs Contrastive fine-tuned. Task: identify partitions from the same original table via synthetic vertical/horizontal splits. Best in \textbf{bold}.}
    \label{tab:gittables-results}
    \begin{tabular}{lcc}
        \toprule
        \textbf{Metric} & \textbf{Pretrained} & \textbf{Contrastive} \\
        \midrule
"""
    
    for metric in metrics_order:
        display_name = metrics_display.get(metric, metric.capitalize())
        
        pre_val = pretrained.get(metric)
        fin_val = finetuned.get(metric)
        
        pre_str = format_metric(pre_val[0], pre_val[1]) if pre_val else "N/A"
        fin_str = format_metric(fin_val[0], fin_val[1]) if fin_val else "N/A"
        
        # Bold the higher value
        if pre_val and fin_val:
            if fin_val[0] > pre_val[0]:
                fin_str = f"\\textbf{{{fin_str}}}"
            elif pre_val[0] > fin_val[0]:
                pre_str = f"\\textbf{{{pre_str}}}"
        
        latex += f"        {display_name} & {pre_str} & {fin_str} \\\\\n"
    
    latex += r"""        \bottomrule
    \end{tabular}
\end{table}
"""
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"\n✅ Saved LaTeX table to {output_path}")
    
    return latex


def main():
    parser = argparse.ArgumentParser(description='GitTables Metrics Comparison')
    parser.add_argument('--results-dir', default='out/gittables/both_full_ss3_neg2',
                       help='Base directory containing test results')
    parser.add_argument('--output', default=None,
                       help='Output path for LaTeX table')
    args = parser.parse_args()
    
    # Find result directories
    pretrained_dir = os.path.join(args.results_dir, 'test_results_pretrained')
    finetuned_dir = os.path.join(args.results_dir, 'test_results_finetuned')
    
    # Also check legacy location
    if not os.path.exists(pretrained_dir):
        pretrained_dir = os.path.join(args.results_dir, 'test_results')
    
    pretrained_summary = os.path.join(pretrained_dir, 'summary.txt')
    finetuned_summary = os.path.join(finetuned_dir, 'summary.txt')
    
    print("=" * 70)
    print("GITTABLES METRICS COMPARISON: PRETRAINED vs CONTRASTIVE")
    print("=" * 70)
    print(f"Results directory: {args.results_dir}")
    print(f"Pretrained summary: {pretrained_summary}")
    print(f"Fine-tuned summary: {finetuned_summary}")
    print("=" * 70)
    
    # Parse results
    pretrained = parse_summary_file(pretrained_summary)
    finetuned = parse_summary_file(finetuned_summary)
    
    if not pretrained:
        print(f"\n⚠️  Warning: No pretrained results found at {pretrained_summary}")
    if not finetuned:
        print(f"\n⚠️  Warning: No fine-tuned results found at {finetuned_summary}")
    
    if not pretrained and not finetuned:
        print("\n❌ No results found!")
        return 1
    
    # Print comparison table
    print("\n" + "-" * 70)
    print(f"{'Metric':<15} {'Pretrained':<25} {'Contrastive':<25} {'Δ':>10}")
    print("-" * 70)
    
    metrics_order = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    
    for metric in metrics_order:
        pre_val = pretrained.get(metric)
        fin_val = finetuned.get(metric)
        
        pre_str = format_metric(pre_val[0], pre_val[1]) if pre_val else "N/A"
        fin_str = format_metric(fin_val[0], fin_val[1]) if fin_val else "N/A"
        
        if pre_val and fin_val:
            delta = fin_val[0] - pre_val[0]
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "N/A"
        
        print(f"{metric.upper():<15} {pre_str:<25} {fin_str:<25} {delta_str:>10}")
    
    print("-" * 70)
    
    # Generate LaTeX
    output_path = args.output or os.path.join(args.results_dir, 'comparison_table.tex')
    latex = generate_latex_table(pretrained, finetuned, output_path)
    
    print("\nGenerated LaTeX:")
    print("-" * 40)
    print(latex)
    
    return 0


if __name__ == "__main__":
    exit(main())
