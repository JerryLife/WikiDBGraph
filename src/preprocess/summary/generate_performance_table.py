"""
Generate LaTeX performance comparison table from evaluation logs.

Reads evaluation results from log files and generates formatted LaTeX tables.
Uses optimal threshold (Youden's J) and mean±std across seeds.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def parse_summary(summary_path: str) -> Dict[str, float]:
    """
    Parse summary file for metrics with mean and std.
    
    Args:
        summary_path: Path to summary.txt
    
    Returns:
        Dict with metrics (auc, auc_std, precision, precision_std, etc.)
    """
    metrics = {}
    with open(summary_path, 'r') as f:
        content = f.read()
    
    # Parse AUC (with optional std)
    auc_match = re.search(r'AUC:\s*([0-9.]+)', content)
    if auc_match:
        metrics['auc'] = float(auc_match.group(1))
    auc_std_match = re.search(r'AUC_std:\s*([0-9.]+)', content)
    if auc_std_match:
        metrics['auc_std'] = float(auc_std_match.group(1))
    
    # Parse threshold
    thresh_match = re.search(r'Best Threshold:\s*([0-9.]+)', content)
    if thresh_match:
        metrics['best_threshold'] = float(thresh_match.group(1))
    thresh_std_match = re.search(r'Threshold_std:\s*([0-9.]+)', content)
    if thresh_std_match:
        metrics['threshold_std'] = float(thresh_std_match.group(1))
    
    # Parse Precision
    prec_match = re.search(r'Precision:\s*([0-9.]+)', content)
    if prec_match:
        metrics['precision'] = float(prec_match.group(1))
    prec_std_match = re.search(r'Precision_std:\s*([0-9.]+)', content)
    if prec_std_match:
        metrics['precision_std'] = float(prec_std_match.group(1))
    
    # Parse Recall
    rec_match = re.search(r'Recall:\s*([0-9.]+)', content)
    if rec_match:
        metrics['recall'] = float(rec_match.group(1))
    rec_std_match = re.search(r'Recall_std:\s*([0-9.]+)', content)
    if rec_std_match:
        metrics['recall_std'] = float(rec_std_match.group(1))
    
    # Parse F1
    f1_match = re.search(r'F1:\s*([0-9.]+)', content)
    if f1_match:
        metrics['f1'] = float(f1_match.group(1))
    f1_std_match = re.search(r'F1_std:\s*([0-9.]+)', content)
    if f1_std_match:
        metrics['f1_std'] = float(f1_std_match.group(1))
    
    return metrics


def collect_model_results(results_dir: str) -> Dict[str, float]:
    """
    Collect results from a model's test_results directory.
    
    Args:
        results_dir: Base directory for model results (e.g., out/graph_full_ss3_neg6)
    
    Returns:
        Dict with all metrics
    """
    results_path = Path(results_dir)
    
    # Try test_results/ directory
    test_results = results_path / "test_results"
    if test_results.exists():
        summary_path = test_results / "summary.txt"
        if summary_path.exists():
            return parse_summary(str(summary_path))
    
    return {}


def format_metric(metrics: Dict, key: str) -> str:
    """Format metric as mean±std."""
    if key not in metrics:
        return "N/A"
    
    mean = metrics[key]
    std_key = f"{key}_std"
    
    if std_key in metrics and metrics[std_key] > 0:
        std = metrics[std_key]
        return f"{mean:.4f}±{std:.4f}"
    else:
        return f"{mean:.4f}"


def generate_performance_table(
    original_results_dir: str,
    finetuned_results_dir: str,
    output_path: str,
    caption: str = "Performance comparison (1:1 balanced, optimal threshold, 5 seeds)",
    label: str = "tab:embedding-model-performance"
) -> str:
    """
    Generate LaTeX table comparing original and finetuned model performance.
    
    Uses optimal threshold (Youden's J) and reports mean±std across seeds.
    """
    # Collect results
    original = collect_model_results(original_results_dir)
    finetuned = collect_model_results(finetuned_results_dir)
    
    def format_bold(orig: Dict, ft: Dict, key: str) -> Tuple[str, str]:
        """Format metrics with bold for better model."""
        orig_str = format_metric(orig, key)
        ft_str = format_metric(ft, key)
        
        if key in orig and key in ft:
            if ft[key] > orig[key]:
                ft_str = f"\\textbf{{{ft_str}}}"
            elif orig[key] > ft[key]:
                orig_str = f"\\textbf{{{orig_str}}}"
        
        return orig_str, ft_str
    
    # Get threshold info for caption
    ft_thresh = format_metric(finetuned, 'best_threshold')
    
    # Generate LaTeX
    latex = f"""\\begin{{table}}[tb]
    \\centering
    \\caption{{{caption} (threshold: {ft_thresh}).}}
    \\label{{{label}}}
    \\resizebox{{\\columnwidth}}{{!}}{{
    \\setlength{{\\tabcolsep}}{{3pt}}
    \\begin{{tabular}}{{lcccc}}
        \\toprule
        \\multirow{{2}}{{*}}{{\\textbf{{Model}}}} & \\multicolumn{{4}}{{c}}{{\\textbf{{Performance}}}} \\\\
        \\cmidrule(lr){{2-5}}
        & \\textbf{{AUC-ROC}} & \\textbf{{F1}} & \\textbf{{Precision}} & \\textbf{{Recall}} \\\\
        \\midrule
"""
    
    # Get formatted metrics with bold for better values
    auc_orig, auc_ft = format_bold(original, finetuned, 'auc')
    f1_orig, f1_ft = format_bold(original, finetuned, 'f1')
    prec_orig, prec_ft = format_bold(original, finetuned, 'precision')
    rec_orig, rec_ft = format_bold(original, finetuned, 'recall')
    
    latex += f"Original & {auc_orig} & {f1_orig} & {prec_orig} & {rec_orig} \\\\\n"
    latex += f"Fine-tuned & {auc_ft} & {f1_ft} & {prec_ft} & {rec_ft} \\\\\n"
    
    latex += """        \\bottomrule
    \\end{tabular}}
\\end{table}
"""
    
    # Save to file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"✅ Saved performance table to {output_path}")
    
    # Also print to console
    print("\n" + "=" * 70)
    print("Performance Comparison (1:1 balanced, optimal threshold)")
    print("=" * 70)
    print(f"{'Model':<15} {'AUC-ROC':<20} {'F1':<20} {'Precision':<20} {'Recall':<20}")
    print("-" * 95)
    print(f"{'Original':<15} {format_metric(original, 'auc'):<20} {format_metric(original, 'f1'):<20} {format_metric(original, 'precision'):<20} {format_metric(original, 'recall'):<20}")
    print(f"{'Fine-tuned':<15} {format_metric(finetuned, 'auc'):<20} {format_metric(finetuned, 'f1'):<20} {format_metric(finetuned, 'precision'):<20} {format_metric(finetuned, 'recall'):<20}")
    print("=" * 70)
    
    print("\nGenerated LaTeX:")
    print("-" * 40)
    print(latex)
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate performance comparison table")
    parser.add_argument("--original", type=str, default="out/original_bge-m3",
                        help="Directory with original model results")
    parser.add_argument("--finetuned", type=str, default="out/graph_full_ss3_neg6",
                        help="Directory with finetuned model results")
    parser.add_argument("--output", type=str, default="tables/performance_table.tex",
                        help="Output path for LaTeX table")
    parser.add_argument("--caption", type=str,
                        default="Performance comparison (1:1 balanced, optimal threshold, 5 seeds)",
                        help="Table caption")
    
    args = parser.parse_args()
    
    generate_performance_table(
        original_results_dir=args.original,
        finetuned_results_dir=args.finetuned,
        output_path=args.output,
        caption=args.caption
    )
