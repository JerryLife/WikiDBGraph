#!/usr/bin/env python3
"""
Compute accuracy, F1, and AUC from prediction CSVs and generate LaTeX table.

Usage:
    python src/summary/print_predictions_metrics.py --predictions path/to/pred.csv
    python src/summary/print_predictions_metrics.py --pred_dir results/fedgnn/01318-15832
    python src/summary/print_predictions_metrics.py --pred_dir results/fedgnn/01318-15832 --latex
"""

import argparse
import os
import re
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _infer_scores(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[int]]:
    if "prob_pos" in df.columns:
        return df["prob_pos"].to_numpy(), 2
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    if not prob_cols:
        return None, None
    def _col_key(name: str) -> int:
        suffix = name.split("_", 1)[-1]
        return int(suffix) if suffix.isdigit() else 0
    prob_cols = sorted(prob_cols, key=_col_key)
    return df[prob_cols].to_numpy(), len(prob_cols)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray, num_classes: int) -> Optional[float]:
    try:
        if len(np.unique(y_true)) < 2:
            return None
        if num_classes == 2:
            return roc_auc_score(y_true, y_score)
        return roc_auc_score(y_true, y_score, multi_class="ovr", average="macro")
    except Exception:
        return None


def _compute_metrics(df: pd.DataFrame) -> Tuple[float, float, Optional[float]]:
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    y_score, num_classes = _infer_scores(df)

    accuracy = accuracy_score(y_true, y_pred)
    if num_classes == 2:
        f1 = f1_score(y_true, y_pred)
    else:
        f1 = f1_score(y_true, y_pred, average="macro")

    auc = None
    if y_score is not None and num_classes is not None:
        auc = _safe_auc(y_true, y_score, num_classes)
    return accuracy, f1, auc


def _format_metric(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.4f}"


def _format_metric_std(mean: Optional[float], std: Optional[float]) -> str:
    """Format metric as mean±std."""
    if mean is None:
        return "N/A"
    if std is not None and std > 0:
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.4f}"


def _collect_files(predictions: str, pred_dir: str) -> List[str]:
    files = []
    if predictions:
        files.extend([p.strip() for p in predictions.split(",") if p.strip()])
    if pred_dir:
        for name in os.listdir(pred_dir):
            if name.startswith("predictions_") and name.endswith(".csv"):
                files.append(os.path.join(pred_dir, name))
    return sorted(set(files))


def _parse_method_from_filename(filename: str) -> Optional[str]:
    """Extract method name from prediction filename.
    
    Example: predictions_fedgnn_both_seed0_r10_e2_h64_lr0p01_c4_pmall_drop0.csv -> fedgnn_both
    """
    basename = os.path.basename(filename)
    # Remove prefix and extension
    if not basename.startswith("predictions_"):
        return None
    name = basename[len("predictions_"):-len(".csv")]
    
    # Find where seed starts
    seed_match = re.search(r"_seed\d+", name)
    if seed_match:
        method = name[:seed_match.start()]
        return method
    return name


def _get_ranking(all_results: Dict[str, Dict[str, float]], metric: str) -> Tuple[Optional[str], Optional[str]]:
    """Get best and second-best method keys for a metric."""
    values = []
    for key, results in all_results.items():
        if metric in results and results[metric] is not None:
            values.append((key, results[metric]))
    
    if not values:
        return None, None
    
    values.sort(key=lambda x: x[1], reverse=True)
    best = values[0][0] if values else None
    second = values[1][0] if len(values) > 1 else None
    return best, second


def _generate_latex_table(
    all_results: Dict[str, Dict[str, float]],
    output_path: Optional[str] = None
) -> str:
    """Generate LaTeX table from aggregated results."""
    
    # Method display names and ordering
    method_order = ["solo", "fedavg", "fedgnn_none", "fedgnn_node_only", "fedgnn_edge_only", "fedgnn_both"]
    method_names = {
        "solo": "Solo (No Federation)",
        "fedavg": "FedAvg",
        "fedgnn_none": "FedGNN (No Properties)",
        "fedgnn_node_only": "FedGNN (Node Only)",
        "fedgnn_edge_only": "FedGNN (Edge Only)",
        "fedgnn_both": "FedGNN (Both)",
    }
    
    def format_ranked(method: str, metric: str) -> str:
        """Format metric: bold if best, underline if second-best."""
        if method not in all_results:
            return "N/A"
        results = all_results[method]
        mean = results.get(metric)
        std = results.get(f"{metric}_std")
        val = _format_metric_std(mean, std)
        if val == "N/A":
            return val
        best, second = _get_ranking(all_results, metric)
        if method == best:
            return f"\\textbf{{{val}}}"
        elif method == second:
            return f"\\underline{{{val}}}"
        return val
    
    # Check which methods have results
    available_methods = [m for m in method_order if m in all_results]
    
    # Generate LaTeX
    latex = """\\begin{table}[tb]
    \\centering
    \\caption{FedGNN experiment results (aggregated across seeds). Best in \\textbf{bold}, second-best \\underline{underlined}.}
    \\label{tab:fedgnn-results}
    \\resizebox{\\columnwidth}{!}{
    \\setlength{\\tabcolsep}{4pt}
    \\begin{tabular}{lccc}
        \\toprule
        \\textbf{Method} & \\textbf{Accuracy} & \\textbf{F1} & \\textbf{AUC-ROC} \\\\
        \\midrule
"""
    
    for method in available_methods:
        display_name = method_names.get(method, method)
        acc = format_ranked(method, "accuracy")
        f1 = format_ranked(method, "f1")
        auc = format_ranked(method, "auc")
        latex += f"        {display_name} & {acc} & {f1} & {auc} \\\\\n"
    
    latex += """        \\bottomrule
    \\end{tabular}}
\\end{table}
"""
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"\n✅ Saved LaTeX table to {output_path}")
    
    return latex


def main() -> None:
    parser = argparse.ArgumentParser(description="Print metrics from prediction CSVs")
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Comma-separated prediction CSV paths"
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        help="Directory containing prediction CSVs"
    )
    parser.add_argument(
        "--per_db",
        action="store_true",
        help="Print metrics per db_id in addition to overall"
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Generate LaTeX table output"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for LaTeX table (default: <pred_dir>/results_table.tex)"
    )
    args = parser.parse_args()

    files = _collect_files(args.predictions, args.pred_dir)
    if not files:
        print("Error: No prediction files provided.")
        return

    # Group files by method and collect metrics
    method_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    
    for path in files:
        if not os.path.exists(path):
            print(f"Warning: Missing file {path}")
            continue
        df = pd.read_csv(path)
        if not {"y_true", "y_pred"}.issubset(df.columns):
            print(f"Warning: Missing y_true/y_pred in {path}")
            continue

        acc, f1, auc = _compute_metrics(df)
        method = _parse_method_from_filename(path)
        
        if not args.latex:
            # Original behavior: print per-file metrics
            print(f"{path}: accuracy={_format_metric(acc)} f1={_format_metric(f1)} auc={_format_metric(auc)}")

            if args.per_db and "db_id" in df.columns:
                for db_id in sorted(df["db_id"].unique()):
                    db_df = df[df["db_id"] == db_id]
                    db_acc, db_f1, db_auc = _compute_metrics(db_df)
                    print(
                        f"  DB{db_id}: accuracy={_format_metric(db_acc)} "
                        f"f1={_format_metric(db_f1)} auc={_format_metric(db_auc)}"
                    )
        
        # Collect for aggregation
        if method:
            method_metrics[method]["accuracy"].append(acc)
            method_metrics[method]["f1"].append(f1)
            if auc is not None:
                method_metrics[method]["auc"].append(auc)

    if args.latex:
        # Aggregate metrics (mean ± std)
        all_results: Dict[str, Dict[str, float]] = {}
        
        for method, metrics in method_metrics.items():
            all_results[method] = {}
            for metric_name, values in metrics.items():
                if values:
                    all_results[method][metric_name] = np.mean(values)
                    all_results[method][f"{metric_name}_std"] = np.std(values)
        
        # Print summary table
        print()
        print("=" * 100)
        print("FedGNN Experiment Results (Aggregated)")
        print("=" * 100)
        print(f"{'Method':<30} {'Accuracy':<20} {'F1':<20} {'AUC':<20}")
        print("-" * 100)
        
        method_order = ["solo", "fedavg", "fedgnn_none", "fedgnn_node_only", "fedgnn_edge_only", "fedgnn_both"]
        for method in method_order:
            if method not in all_results:
                continue
            results = all_results[method]
            acc_str = _format_metric_std(results.get("accuracy"), results.get("accuracy_std"))
            f1_str = _format_metric_std(results.get("f1"), results.get("f1_std"))
            auc_str = _format_metric_std(results.get("auc"), results.get("auc_std"))
            print(f"{method:<30} {acc_str:<20} {f1_str:<20} {auc_str:<20}")
        
        print("=" * 100)
        
        # Generate LaTeX
        output_path = args.output
        if output_path is None and args.pred_dir:
            output_path = os.path.join(args.pred_dir, "results_table.tex")
        
        latex = _generate_latex_table(all_results, output_path)
        print("\nGenerated LaTeX:")
        print("-" * 40)
        print(latex)


if __name__ == "__main__":
    main()
