"""
Generate LaTeX table for encoder model ablation study.

Compares BGE-M3 vs sentence-transformers/all-mpnet-base-v2 (both original and fine-tuned).
"""

import os
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def parse_summary(summary_path: str) -> Dict[str, float]:
    """Parse summary file for all metrics (new format with mean±std)."""
    metrics = {}
    with open(summary_path, 'r') as f:
        content = f.read()
    
    # Parse AUC
    auc_match = re.search(r'AUC:\s*([0-9.]+)', content)
    if auc_match:
        metrics['auc'] = float(auc_match.group(1))
    auc_std_match = re.search(r'AUC_std:\s*([0-9.]+)', content)
    if auc_std_match:
        metrics['auc_std'] = float(auc_std_match.group(1))
    
    # Parse threshold
    thresh_match = re.search(r'Best Threshold:\s*([0-9.]+)', content)
    if thresh_match:
        metrics['threshold'] = float(thresh_match.group(1))
    
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


def collect_results(results_dir: str, model_key: str, is_original: bool) -> Dict[str, float]:
    """Collect results for a model (original or fine-tuned)."""
    if is_original:
        # out/original_{model}/test_results/
        model_dir = Path(results_dir) / f"original_{model_key}"
    else:
        # out/graph_full_ss3_neg6_{model}/test_results/ or out/graph_full_ss3_neg6/test_results/
        if model_key == "bge-m3":
            model_dir = Path(results_dir) / "graph_full_ss3_neg6"
        else:
            model_dir = Path(results_dir) / f"graph_full_ss3_neg6_{model_key}"
    
    if not model_dir.exists():
        return {}
    
    summary_path = model_dir / "test_results" / "summary.txt"
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


def generate_ablation_table(
    results_dir: str,
    output_path: str
) -> Optional[str]:
    """Generate LaTeX table comparing encoder models (original and fine-tuned)."""
    
    models = {
        "bge-m3": "BGE-M3",
        "mpnet": "all-mpnet-base-v2"
    }
    
    # Collect results for both original and fine-tuned
    all_results = {}
    for model_key in models.keys():
        all_results[f"{model_key}_orig"] = collect_results(results_dir, model_key, is_original=True)
        all_results[f"{model_key}_ft"] = collect_results(results_dir, model_key, is_original=False)
    
    # Print summary first
    print()
    print("=" * 100)
    print("Encoder Model Ablation Results")
    print("=" * 100)
    print(f"{'Model':<30} {'Type':<12} {'AUC':<18} {'F1':<18} {'Precision':<18} {'Recall':<18}")
    print("-" * 100)
    
    for model_key, model_name in models.items():
        # Original
        orig = all_results[f"{model_key}_orig"]
        print(f"{model_name:<30} {'Original':<12} {format_metric(orig, 'auc'):<18} {format_metric(orig, 'f1'):<18} {format_metric(orig, 'precision'):<18} {format_metric(orig, 'recall'):<18}")
        
        # Fine-tuned
        ft = all_results[f"{model_key}_ft"]
        print(f"{'':<30} {'Fine-tuned':<12} {format_metric(ft, 'auc'):<18} {format_metric(ft, 'f1'):<18} {format_metric(ft, 'precision'):<18} {format_metric(ft, 'recall'):<18}")
    
    print("=" * 100)
    
    # Find best model for each metric (considering all variants)
    def get_best(metric: str) -> Optional[str]:
        best = None
        best_val = -1
        for key, results in all_results.items():
            if metric in results:
                val = results[metric]
                if val > best_val:
                    best_val = val
                    best = key
        return best
    
    def format_bold(key: str, metric: str) -> str:
        """Format metric, bold if best."""
        val = format_metric(all_results[key], metric)
        best = get_best(metric)
        if best == key and val != "N/A":
            return f"\\textbf{{{val}}}"
        return val
    
    # Check if any results exist
    has_results = any(all_results[k] for k in all_results.keys())
    if not has_results:
        print("\n❌ No results found!")
        return None
    
    # Generate LaTeX
    latex = """\\begin{table}[tb]
    \\centering
    \\caption{Ablation study: Encoder model comparison (optimal threshold, 5 seeds).}
    \\label{tab:ablation-encoder-model}
    \\resizebox{\\columnwidth}{!}{
    \\setlength{\\tabcolsep}{3pt}
    \\begin{tabular}{llcccc}
        \\toprule
        \\textbf{Encoder Model} & \\textbf{Type} & \\textbf{AUC-ROC} & \\textbf{F1} & \\textbf{Precision} & \\textbf{Recall} \\\\
        \\midrule
"""
    
    for model_key, model_name in models.items():
        # Original
        orig_key = f"{model_key}_orig"
        latex += f"        {model_name} & Original & {format_bold(orig_key, 'auc')} & {format_bold(orig_key, 'f1')} & {format_bold(orig_key, 'precision')} & {format_bold(orig_key, 'recall')} \\\\\n"
        
        # Fine-tuned
        ft_key = f"{model_key}_ft"
        latex += f"         & Fine-tuned & {format_bold(ft_key, 'auc')} & {format_bold(ft_key, 'f1')} & {format_bold(ft_key, 'precision')} & {format_bold(ft_key, 'recall')} \\\\\n"
        latex += "        \\midrule\n"
    
    # Remove last midrule and add bottomrule
    latex = latex.rsplit("\\midrule\n", 1)[0]
    latex += """        \\bottomrule
    \\end{tabular}}
\\end{table}
"""
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"\n✅ Saved ablation table to {output_path}")
    print("\nGenerated LaTeX:")
    print("-" * 40)
    print(latex)
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate encoder model ablation table")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory with ablation results")
    parser.add_argument("--output", type=str, default="fig/ablation_encoder_model.tex",
                        help="Output path for LaTeX table")
    
    args = parser.parse_args()
    
    generate_ablation_table(
        results_dir=args.results_dir,
        output_path=args.output
    )
