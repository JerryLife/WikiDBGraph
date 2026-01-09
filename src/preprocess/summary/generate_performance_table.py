"""
Generate LaTeX performance comparison table from evaluation logs.

Combines default performance, encoder ablation, and serialization mode results.
Supports bold for best and underline for second-best in each metric.
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
    
    # Parse Accuracy
    acc_match = re.search(r'Accuracy:\s*([0-9.]+)', content)
    if acc_match:
        metrics['accuracy'] = float(acc_match.group(1))
    acc_std_match = re.search(r'Accuracy_std:\s*([0-9.]+)', content)
    if acc_std_match:
        metrics['accuracy_std'] = float(acc_std_match.group(1))
    
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


def collect_encoder_results(results_dir: str, model_key: str, is_original: bool) -> Dict[str, float]:
    """Collect results for an encoder model (original or fine-tuned)."""
    results_path = Path(results_dir)
    
    if is_original:
        model_dir = results_path / f"original_{model_key}"
    else:
        if model_key == "bge-m3":
            model_dir = results_path / "graph_full_ss3_neg6"
        else:
            model_dir = results_path / f"graph_full_ss3_neg6_{model_key}"
    
    if not model_dir.exists():
        return {}
    
    summary_path = model_dir / "test_results" / "summary.txt"
    if summary_path.exists():
        return parse_summary(str(summary_path))
    
    return {}


def collect_serialization_results(results_dir: str, mode: str) -> Dict[str, float]:
    """Collect results for a serialization mode."""
    results_path = Path(results_dir)
    
    mode_dir = results_path / f"graph_{mode}_ss3_neg6"
    if not mode_dir.exists():
        mode_dir = results_path / mode
    
    if not mode_dir.exists():
        return {}
    
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


def get_ranking(all_results: Dict[str, Dict], key: str) -> Tuple[Optional[str], Optional[str]]:
    """Get best and second-best row keys for a metric."""
    values = []
    for row_key, metrics in all_results.items():
        if key in metrics:
            values.append((row_key, metrics[key]))
    
    if not values:
        return None, None
    
    values.sort(key=lambda x: x[1], reverse=True)
    best = values[0][0] if values else None
    second = values[1][0] if len(values) > 1 else None
    return best, second


def format_ranked(all_results: Dict[str, Dict], row_key: str, metric: str) -> str:
    """Format metric: bold if best, underline if second-best."""
    val = format_metric(all_results[row_key], metric)
    if val == "N/A":
        return val
    
    best, second = get_ranking(all_results, metric)
    if row_key == best:
        return f"\\textbf{{{val}}}"
    elif row_key == second:
        return f"\\underline{{{val}}}"
    return val


def generate_combined_performance_table(
    results_dir: str,
    output_path: str,
    include_encoder_ablation: bool = True,
    include_serialization_ablation: bool = True,
    caption: str = "Performance comparison (1:1 balanced, optimal threshold, 5 seeds)",
    label: str = "tab:embedding-model-performance"
) -> str:
    """
    Generate LaTeX table combining all ablation results.
    
    Rows are ordered: ablation studies first, default at bottom.
    Best is bold, second-best is underlined.
    """
    results_path = Path(results_dir)
    all_results = {}
    row_order = []
    row_labels = {}
    
    # --- Encoder Model Ablation ---
    if include_encoder_ablation:
        encoder_models = {
            "bge-m3": "BGE-M3"
        }
        for model_key, model_name in encoder_models.items():
            # Original
            orig_key = f"encoder_{model_key}_orig"
            orig_results = collect_encoder_results(results_dir, model_key, is_original=True)
            if orig_results:
                all_results[orig_key] = orig_results
                row_order.append(orig_key)
                row_labels[orig_key] = (model_name, "Original")
    
    # --- Serialization Mode Ablation ---
    if include_serialization_ablation:
        modes = ["schema_only", "data_only"]  # 'full' is the default
        mode_labels = {
            "schema_only": "Schema Only",
            "data_only": "Data Only"
        }
        for mode in modes:
            mode_key = f"serial_{mode}"
            mode_results = collect_serialization_results(results_dir, mode)
            if mode_results:
                all_results[mode_key] = mode_results
                row_order.append(mode_key)
                row_labels[mode_key] = (mode_labels[mode], "Contrast.")
    
    # --- Default (BGE-M3 + Full, Fine-tuned) ---
    default_key = "default"
    default_dir = results_path / "graph_full_ss3_neg6"
    default_results = collect_model_results(str(default_dir))
    if default_results:
        all_results[default_key] = default_results
        row_order.append(default_key)
        row_labels[default_key] = ("BGE-M3 + Full", "Contrast.")

    
    # Print summary
    print()
    print("=" * 110)
    print("Combined Performance Table")
    print("=" * 110)
    print(f"{'Row':<35} {'Method':<12} {'Accuracy':<18} {'AUC':<18} {'F1':<18}")

    print("-" * 110)
    
    for row_key in row_order:
        metrics = all_results[row_key]
        label_tuple = row_labels[row_key]
        print(f"{label_tuple[0]:<35} {label_tuple[1]:<12} {format_metric(metrics, 'accuracy'):<18} {format_metric(metrics, 'auc'):<18} {format_metric(metrics, 'f1'):<18}")
    
    print("=" * 110)
    
    if not all_results:
        print("\n❌ No results found!")
        return None
    
    # Generate LaTeX
    latex = f"""\\begin{{table}}[tb]
    \\centering
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\resizebox{{\\columnwidth}}{{!}}{{
    \\setlength{{\\tabcolsep}}{{3pt}}
    \\begin{{tabular}}{{llccc}}
        \\toprule
        \\textbf{{Configuration}} & \\textbf{{Method}} & \\textbf{{Accuracy}} & \\textbf{{AUC-ROC}} & \\textbf{{F1}} \\\\
        \\midrule
"""
    
    prev_section = None
    for i, row_key in enumerate(row_order):
        label_tuple = row_labels[row_key]
        config_name = label_tuple[0]
        type_name = label_tuple[1]
        
        # Add section separators
        current_section = row_key.split("_")[0]
        if prev_section and prev_section != current_section:
            latex += "        \\midrule\n"
        prev_section = current_section
        
        acc = format_ranked(all_results, row_key, 'accuracy')
        auc = format_ranked(all_results, row_key, 'auc')
        f1 = format_ranked(all_results, row_key, 'f1')
        
        latex += f"        {config_name} & {type_name} & {acc} & {auc} & {f1} \\\\\n"
    
    latex += """        \\bottomrule
    \\end{tabular}}
\\end{table}
"""
    
    # Save to file
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(latex)
    
    print(f"\n✅ Saved performance table to {output_path}")
    print("\nGenerated LaTeX:")
    print("-" * 40)
    print(latex)
    
    return output_path


# Keep the old function for backward compatibility
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
    (Legacy function - use generate_combined_performance_table for new code)
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
    subparsers = parser.add_subparsers(dest="mode", help="Mode of operation")
    
    # Combined mode (new)
    combined_parser = subparsers.add_parser("combined", help="Generate combined ablation table")
    combined_parser.add_argument("--results-dir", type=str, default="out",
                                  help="Directory with all results")
    combined_parser.add_argument("--output", type=str, default="tables/combined_performance.tex",
                                  help="Output path for LaTeX table")
    combined_parser.add_argument("--no-encoder", action="store_true",
                                  help="Exclude encoder model ablation")
    combined_parser.add_argument("--no-serialization", action="store_true",
                                  help="Exclude serialization mode ablation")
    combined_parser.add_argument("--caption", type=str,
                                  default="Performance comparison (1:1 balanced, optimal threshold, 5 seeds)",
                                  help="Table caption")
    
    # Legacy mode
    legacy_parser = subparsers.add_parser("legacy", help="Generate original vs finetuned table (legacy)")
    legacy_parser.add_argument("--original", type=str, default="out/original_bge-m3",
                                help="Directory with original model results")
    legacy_parser.add_argument("--finetuned", type=str, default="out/graph_full_ss3_neg6",
                                help="Directory with finetuned model results")
    legacy_parser.add_argument("--output", type=str, default="tables/performance_table.tex",
                                help="Output path for LaTeX table")
    legacy_parser.add_argument("--caption", type=str,
                                default="Performance comparison (1:1 balanced, optimal threshold, 5 seeds)",
                                help="Table caption")
    
    # Default mode (for backward compatibility)
    parser.add_argument("--original", type=str, default="out/original_bge-m3",
                        help="Directory with original model results (legacy)")
    parser.add_argument("--finetuned", type=str, default="out/graph_full_ss3_neg6",
                        help="Directory with finetuned model results (legacy)")
    parser.add_argument("--results-dir", type=str, default="out",
                        help="Directory with all results (combined mode)")
    parser.add_argument("--output", type=str, default="tables/performance_table.tex",
                        help="Output path for LaTeX table")
    parser.add_argument("--caption", type=str,
                        default="Performance comparison (1:1 balanced, optimal threshold, 5 seeds)",
                        help="Table caption")
    parser.add_argument("--combined", action="store_true",
                        help="Use combined mode (encoder + serialization ablation)")
    
    args = parser.parse_args()
    
    if args.mode == "legacy":
        # Legacy mode (original vs finetuned only)
        generate_performance_table(
            original_results_dir=args.original,
            finetuned_results_dir=args.finetuned,
            output_path=args.output,
            caption=args.caption
        )
    else:
        # Default: combined mode
        results_dir = getattr(args, 'results_dir', 'out')
        generate_combined_performance_table(
            results_dir=results_dir,
            output_path=args.output,
            include_encoder_ablation=not getattr(args, 'no_encoder', False),
            include_serialization_ablation=not getattr(args, 'no_serialization', False),
            caption=args.caption
        )
