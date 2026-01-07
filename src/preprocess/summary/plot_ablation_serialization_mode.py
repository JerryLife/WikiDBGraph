"""
Generate LaTeX table for serialization mode ablation study.

Compares schema_only, data_only, and full modes.
"""

import os
import re
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional


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


def collect_mode_results(results_dir: str, mode: str) -> Dict[str, float]:
    """Collect results for a mode from test_results/summary.txt."""
    # Look for results in out/graph_{mode}_ss3_neg6/ structure
    mode_dir = Path(results_dir) / f"graph_{mode}_ss3_neg6"
    
    # Fallback to old structure if new one doesn't exist
    if not mode_dir.exists():
        mode_dir = Path(results_dir) / mode
    
    if not mode_dir.exists():
        return {}
    
    # Try test_results/summary.txt
    summary_path = mode_dir / "test_results" / "summary.txt"
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
        all_results[mode] = collect_mode_results(results_dir, mode)
    
    # Print summary first
    print()
    print("=" * 80)
    print("Serialization Mode Ablation Results")
    print("=" * 80)
    print(f"{'Mode':<20} {'AUC':<18} {'F1':<18} {'Precision':<18} {'Recall':<18}")
    print("-" * 80)
    
    for mode in modes:
        results = all_results[mode]
        label = mode_labels[mode]
        
        auc_str = format_metric(results, 'auc')
        f1_str = format_metric(results, 'f1')
        prec_str = format_metric(results, 'precision')
        rec_str = format_metric(results, 'recall')
        
        print(f"{label:<20} {auc_str:<18} {f1_str:<18} {prec_str:<18} {rec_str:<18}")
    
    print("=" * 80)
    
    # Find best mode for each metric
    def get_best_mode(metric: str) -> Optional[str]:
        best = None
        best_val = -1
        for mode in modes:
            if metric in all_results[mode]:
                val = all_results[mode][metric]
                if val > best_val:
                    best_val = val
                    best = mode
        return best
    
    def format_bold(mode: str, metric: str) -> str:
        """Format metric, bold if best."""
        val = format_metric(all_results[mode], metric)
        best = get_best_mode(metric)
        if best == mode and val != "N/A":
            return f"\\textbf{{{val}}}"
        return val
    
    # Check if any results exist
    has_results = any(all_results[mode] for mode in modes)
    if not has_results:
        print("\n❌ No results found!")
        return None
    
    # Get threshold info
    thresh_str = "optimal threshold"
    for mode in modes:
        if 'threshold' in all_results[mode]:
            thresh_str = f"threshold {all_results[mode]['threshold']:.4f}"
            break
    
    # Generate LaTeX
    latex = f"""\\begin{{table}}[tb]
    \\centering
    \\caption{{Ablation study: Serialization mode comparison ({thresh_str}).}}
    \\label{{tab:ablation-serialization-mode}}
    \\resizebox{{\\columnwidth}}{{!}}{{
    \\setlength{{\\tabcolsep}}{{3pt}}
    \\begin{{tabular}}{{lcccc}}
        \\toprule
        \\textbf{{Mode}} & \\textbf{{AUC-ROC}} & \\textbf{{F1}} & \\textbf{{Precision}} & \\textbf{{Recall}} \\\\
        \\midrule
"""
    
    for mode in modes:
        label = mode_labels[mode]
        
        auc = format_bold(mode, 'auc')
        f1 = format_bold(mode, 'f1')
        prec = format_bold(mode, 'precision')
        rec = format_bold(mode, 'recall')
        
        latex += f"        {label} & {auc} & {f1} & {prec} & {rec} \\\\\n"
    
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
    parser = argparse.ArgumentParser(description="Generate serialization mode ablation table")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory with ablation results")
    parser.add_argument("--output", type=str, default="fig/ablation_serialization_mode.tex",
                        help="Output path for LaTeX table")
    
    args = parser.parse_args()
    
    generate_ablation_table(
        results_dir=args.results_dir,
        output_path=args.output
    )
