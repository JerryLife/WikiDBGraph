"""
Generate LaTeX table for serialization mode ablation study.

Compares schema_only, data_only, and full modes.
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
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


def parse_predictions(predictions_path: str, threshold: float = 0.5) -> Dict[str, float]:
    """Parse predictions and compute F1, precision, recall."""
    from sklearn.metrics import f1_score, precision_score, recall_score
    
    df = pd.read_csv(predictions_path)
    y_true = df['label'].values
    y_scores = df['similarity'].values
    y_pred = (y_scores >= threshold).astype(int)
    
    return {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }


def collect_mode_results(results_dir: str, mode: str, threshold: float = 0.5) -> Dict[str, Tuple[float, float]]:
    """Collect and aggregate results for a mode across seeds."""
    # Look for results in out/graph_{mode}_ss3_neg6/ structure
    mode_dir = Path(results_dir) / f"graph_{mode}_ss3_neg6"
    
    # Fallback to old structure if new one doesn't exist
    if not mode_dir.exists():
        mode_dir = Path(results_dir) / mode
    
    all_results = []
    for seed_dir in sorted(mode_dir.glob("test_results_seed*")):
        summary_path = seed_dir / "summary.txt"
        predictions_path = seed_dir / "predictions.csv"
        
        if summary_path.exists():
            result = parse_summary(str(summary_path))
            if predictions_path.exists():
                result.update(parse_predictions(str(predictions_path), threshold))
            all_results.append(result)
    
    # Aggregate
    aggregated = {}
    if all_results:
        for key in all_results[0].keys():
            values = [r[key] for r in all_results if key in r]
            if values:
                aggregated[key] = (np.mean(values), np.std(values))
    
    return aggregated


def generate_ablation_table(
    results_dir: str,
    output_path: str,
    threshold: float = 0.5
) -> str:
    """Generate LaTeX table comparing serialization modes."""
    
    modes = ["schema_only", "data_only", "full"]
    mode_labels = {
        "schema_only": "Schema Only",
        "data_only": "Data Only", 
        "full": "Full (Combined)"
    }
    
    # Collect results
    all_results = {}
    for mode in modes:
        all_results[mode] = collect_mode_results(results_dir, mode, threshold)
    
    def format_metric(results: Dict, key: str, best_mode: str = None) -> str:
        if key not in results:
            return "N/A"
        mean, std = results[key]
        text = f"{mean:.4f}±{std:.4f}"
        return text
    
    # Find best mode for each metric
    def get_best_mode(metric: str) -> str:
        best = None
        best_val = -1
        for mode in modes:
            if metric in all_results[mode]:
                val = all_results[mode][metric][0]
                if val > best_val:
                    best_val = val
                    best = mode
        return best
    
    # Generate LaTeX
    latex = """\\begin{table}[tb]
    \\centering
    \\caption{Ablation study: Serialization mode comparison (threshold """ + f"{threshold}" + """).}
    \\label{tab:ablation-serialization-mode}
    \\resizebox{\\columnwidth}{!}{
    \\setlength{\\tabcolsep}{3pt}
    \\begin{tabular}{lcccc}
        \\toprule
        \\textbf{Mode} & \\textbf{AUC-ROC} & \\textbf{F1} & \\textbf{Precision} & \\textbf{Recall} \\\\
        \\midrule
"""
    
    for mode in modes:
        results = all_results[mode]
        label = mode_labels[mode]
        
        # Format each metric, bold if best
        metrics = []
        for metric in ['auc', 'f1', 'precision', 'recall']:
            val = format_metric(results, metric)
            best = get_best_mode(metric)
            if best == mode and val != "N/A":
                val = f"\\textbf{{{val}}}"
            metrics.append(val)
        
        latex += f"        {label} & {metrics[0]} & {metrics[1]} & {metrics[2]} & {metrics[3]} \\\\\n"
    
    latex += """        \\bottomrule
    \\end{tabular}}
\\end{table}
"""
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"✅ Saved ablation table to {output_path}")
    print("\n" + latex)
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate serialization mode ablation table")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory with ablation results")
    parser.add_argument("--output", type=str, default="fig/ablation_serialization_mode.tex",
                        help="Output path for LaTeX table")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold")
    
    args = parser.parse_args()
    
    generate_ablation_table(
        results_dir=args.results_dir,
        output_path=args.output,
        threshold=args.threshold
    )
