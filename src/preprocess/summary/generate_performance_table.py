"""
Generate LaTeX performance comparison table from evaluation logs.

Reads evaluation results from log files and generates formatted LaTeX tables.
No hard-coded data - everything is read from actual results.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import f1_score, precision_score, recall_score


def parse_predictions(predictions_path: str, threshold: float = 0.5) -> Dict[str, float]:
    """
    Parse predictions CSV and compute metrics at given threshold.
    
    Args:
        predictions_path: Path to predictions.csv
        threshold: Similarity threshold for classification
    
    Returns:
        Dict with precision, recall, f1 metrics
    """
    df = pd.read_csv(predictions_path)
    
    y_true = df['label'].values
    y_scores = df['similarity'].values
    y_pred = (y_scores >= threshold).astype(int)
    
    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    return metrics


def parse_summary(summary_path: str) -> Dict[str, float]:
    """
    Parse summary file for AUC and threshold.
    
    Args:
        summary_path: Path to summary.txt
    
    Returns:
        Dict with auc, best_threshold
    """
    metrics = {}
    with open(summary_path, 'r') as f:
        content = f.read()
    
    auc_match = re.search(r'AUC:\s*([0-9.]+)', content)
    if auc_match:
        metrics['auc'] = float(auc_match.group(1))
    
    thresh_match = re.search(r'Best Threshold:\s*([0-9.]+)', content)
    if thresh_match:
        metrics['best_threshold'] = float(thresh_match.group(1))
    
    return metrics


def collect_model_results(
    results_dir: str,
    threshold: float = 0.5,
    pattern: str = "test_results_seed*"
) -> List[Dict[str, float]]:
    """
    Collect results from multiple seed runs for a model.
    
    Args:
        results_dir: Base directory for model results
        threshold: Threshold for classification metrics
        pattern: Glob pattern for seed directories
    
    Returns:
        List of metric dictionaries per seed
    """
    results = []
    results_path = Path(results_dir)
    
    for seed_dir in sorted(results_path.glob(pattern)):
        summary_path = seed_dir / "summary.txt"
        predictions_path = seed_dir / "predictions.csv"
        
        if summary_path.exists():
            result = parse_summary(str(summary_path))
            
            if predictions_path.exists():
                pred_metrics = parse_predictions(str(predictions_path), threshold)
                result.update(pred_metrics)
            
            results.append(result)
    
    return results


def aggregate_results(results: List[Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
    """
    Aggregate results across seeds.
    
    Args:
        results: List of metric dictionaries
    
    Returns:
        Dict mapping metric name to (mean, std) tuple
    """
    if not results:
        return {}
    
    aggregated = {}
    all_keys = set()
    for r in results:
        all_keys.update(r.keys())
    
    for key in all_keys:
        values = [r[key] for r in results if key in r]
        if values:
            aggregated[key] = (np.mean(values), np.std(values))
    
    return aggregated


def generate_performance_table(
    original_results_dir: str,
    finetuned_results_dir: str,
    output_path: str,
    threshold: float = 0.5,
    caption: str = "Performance of the embedding model (BGE-M3) on the test set",
    label: str = "tab:embedding-model-performance"
) -> str:
    """
    Generate LaTeX table comparing original and finetuned model performance.
    
    Args:
        original_results_dir: Directory with original model results
        finetuned_results_dir: Directory with finetuned model results
        output_path: Output path for LaTeX file
        threshold: Threshold for classification metrics
        caption: Table caption
        label: Table label
    
    Returns:
        Path to saved LaTeX file
    """
    # Collect results
    original_results = collect_model_results(original_results_dir, threshold)
    finetuned_results = collect_model_results(finetuned_results_dir, threshold)
    
    # Aggregate
    original_agg = aggregate_results(original_results)
    finetuned_agg = aggregate_results(finetuned_results)
    
    def format_metric(agg: Dict, key: str) -> str:
        """Format metric as mean±std with bold if better."""
        if key not in agg:
            return "N/A"
        mean, std = agg[key]
        return f"{mean:.4f}±{std:.4f}"
    
    def format_metric_bold(orig_agg: Dict, ft_agg: Dict, key: str, higher_better: bool = True) -> Tuple[str, str]:
        """Format metrics with bold for better model."""
        orig_str = format_metric(orig_agg, key)
        ft_str = format_metric(ft_agg, key)
        
        if key in orig_agg and key in ft_agg:
            orig_mean = orig_agg[key][0]
            ft_mean = ft_agg[key][0]
            
            if higher_better:
                if ft_mean > orig_mean:
                    ft_str = f"\\textbf{{{ft_str}}}"
                elif orig_mean > ft_mean:
                    orig_str = f"\\textbf{{{orig_str}}}"
            else:
                if ft_mean < orig_mean:
                    ft_str = f"\\textbf{{{ft_str}}}"
                elif orig_mean < ft_mean:
                    orig_str = f"\\textbf{{{orig_str}}}"
        
        return orig_str, ft_str
    
    # Generate LaTeX
    latex = f"""\\begin{{table}}[tb]
    \\centering
    \\caption{{{caption} with threshold {threshold}.}}
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
    
    # Original model row
    auc_orig, auc_ft = format_metric_bold(original_agg, finetuned_agg, 'auc')
    f1_orig, f1_ft = format_metric_bold(original_agg, finetuned_agg, 'f1')
    prec_orig, prec_ft = format_metric_bold(original_agg, finetuned_agg, 'precision')
    rec_orig, rec_ft = format_metric_bold(original_agg, finetuned_agg, 'recall')
    
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
    print("\n" + "=" * 60)
    print("Generated LaTeX Table:")
    print("=" * 60)
    print(latex)
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate performance comparison table")
    parser.add_argument("--original", type=str, required=True,
                        help="Directory with original model results")
    parser.add_argument("--finetuned", type=str, required=True,
                        help="Directory with finetuned model results")
    parser.add_argument("--output", type=str, default="tables/performance_table.tex",
                        help="Output path for LaTeX table")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold")
    parser.add_argument("--caption", type=str,
                        default="Performance of the embedding model (BGE-M3) on the test set",
                        help="Table caption")
    
    args = parser.parse_args()
    
    generate_performance_table(
        original_results_dir=args.original,
        finetuned_results_dir=args.finetuned,
        output_path=args.output,
        threshold=args.threshold,
        caption=args.caption
    )
