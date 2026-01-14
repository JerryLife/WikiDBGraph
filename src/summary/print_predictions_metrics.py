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


# Default prediction directory (matches train_fedgnn.py default db_ids)
DEFAULT_PRED_DIR = "results/fedgnn/01318-15832-26192-34036-52953-67222-79114-84208"


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


def _compute_per_db_metrics(df: pd.DataFrame) -> Dict[str, Dict[str, Optional[float]]]:
    """Compute metrics per DB, returning dict of db_id -> metrics."""
    result = {}
    if "db_id" not in df.columns:
        acc, f1, auc = _compute_metrics(df)
        result["all"] = {"accuracy": acc, "f1": f1, "auc": auc}
        return result

    for db_id, db_df in df.groupby("db_id"):
        acc, f1, auc = _compute_metrics(db_df)
        result[db_id] = {"accuracy": acc, "f1": f1, "auc": auc}
    return result


def _compute_per_db_average_metrics(df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Compute metrics per DB, then average across DBs (unweighted)."""
    per_db = _compute_per_db_metrics(df)
    acc_vals = [m["accuracy"] for m in per_db.values() if m["accuracy"] is not None]
    f1_vals = [m["f1"] for m in per_db.values() if m["f1"] is not None]
    auc_vals = [m["auc"] for m in per_db.values() if m["auc"] is not None]
    return {
        "accuracy": float(np.mean(acc_vals)) if acc_vals else None,
        "f1": float(np.mean(f1_vals)) if f1_vals else None,
        "auc": float(np.mean(auc_vals)) if auc_vals else None
    }


def _format_metric(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.4f}"


def _format_metric_std(mean: Optional[float], std: Optional[float]) -> str:
    """Format metric as mean±std."""
    if mean is None:
        return "N/A"
    if std is not None and std > 0:
        return f"{mean:.4f}±{std:.4f}"
    return f"{mean:.4f}"


def _format_pct(value: Optional[float]) -> str:
    """Format percentage value."""
    if value is None:
        return "N/A"
    return f"{value:.1f}\\%"


def _collect_files(predictions: str, pred_dir: str) -> List[str]:
    files = []
    if predictions:
        files.extend([p.strip() for p in predictions.split(",") if p.strip()])
    if pred_dir:
        if not os.path.exists(pred_dir):
            return files
        for name in os.listdir(pred_dir):
            if name.startswith("predictions_") and name.endswith(".csv"):
                files.append(os.path.join(pred_dir, name))
    return sorted(set(files))


def _parse_method_from_filename(filename: str) -> Optional[str]:
    """Extract method name from prediction filename."""
    basename = os.path.basename(filename)
    if not basename.startswith("predictions_"):
        return None
    name = basename[len("predictions_"):-len(".csv")]
    seed_match = re.search(r"_seed\d+", name)
    if seed_match:
        return name[:seed_match.start()]
    return name


def _parse_method_and_seed(filename: str) -> Tuple[Optional[str], Optional[int]]:
    basename = os.path.basename(filename)
    if not basename.startswith("predictions_") or not basename.endswith(".csv"):
        return None, None
    name = basename[len("predictions_"):-len(".csv")]
    seed_match = re.search(r"_seed(\d+)", name)
    if seed_match:
        method = name[:seed_match.start()]
        return method, int(seed_match.group(1))
    return name, None


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


def _compute_gain_percentage(
    method_seed_db_metrics: Dict[str, Dict[int, Dict[str, Dict[str, float]]]],
    metric: str
) -> Dict[str, float]:
    """Compute percentage of DBs where method beats Solo, averaged across seeds."""
    solo_data = method_seed_db_metrics.get("solo", {})
    gains = {}
    
    for method, seed_db_map in method_seed_db_metrics.items():
        if method == "solo":
            continue
        
        pct_per_seed = []
        for seed, db_map in seed_db_map.items():
            solo_db_map = solo_data.get(seed, {})
            if not solo_db_map:
                continue
            
            n_better = 0
            n_total = 0
            for db_id, metrics in db_map.items():
                if db_id not in solo_db_map:
                    continue
                val = metrics.get(metric)
                solo_val = solo_db_map.get(db_id, {}).get(metric)
                if val is None or solo_val is None:
                    continue
                n_total += 1
                if val > solo_val:
                    n_better += 1
            
            if n_total > 0:
                pct_per_seed.append(100.0 * n_better / n_total)
        
        if pct_per_seed:
            gains[method] = float(np.mean(pct_per_seed))
            gains[f"{method}_std"] = float(np.std(pct_per_seed))
    
    return gains


def _generate_latex_table(
    all_results: Dict[str, Dict[str, float]],
    gain_pct: Dict[str, float],
    output_path: Optional[str] = None
) -> str:
    """Generate LaTeX table with Accuracy, F1, and Gain% columns."""
    
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
    
    available_methods = [m for m in method_order if m in all_results]
    
    # Helper to get gain ranking (higher is better)
    def get_gain_ranking(metric: str) -> Tuple[Optional[str], Optional[str]]:
        """Get best and second-best method for gain percentage."""
        values = []
        for method in available_methods:
            if method == "solo":
                continue
            val = gain_pct.get(f"{metric}_{method}")
            if val is not None:
                values.append((method, val))
        if not values:
            return None, None
        values.sort(key=lambda x: x[1], reverse=True)
        best = values[0][0] if values else None
        second = values[1][0] if len(values) > 1 else None
        return best, second
    
    def format_gain_ranked(method: str, metric: str) -> str:
        """Format gain%: bold if best, underline if second-best."""
        if method == "solo":
            return "N/A"
        val = gain_pct.get(f"{metric}_{method}")
        if val is None:
            return "N/A"
        formatted = _format_pct(val)
        best, second = get_gain_ranking(metric)
        if method == best:
            return f"\\textbf{{{formatted}}}"
        elif method == second:
            return f"\\underline{{{formatted}}}"
        return formatted
    
    # Generate LaTeX with cmidrule and multicolumn
    latex = """\\begin{table}[tb]
    \\centering
    \\caption{FedGNN experiment results (per-DB averaged across seeds). Best in \\textbf{bold}, second-best \\underline{underlined}. Gain(\\%) = percentage of DBs where method outperforms Solo.}
    \\label{tab:fedgnn-results}
    \\resizebox{\\columnwidth}{!}{
    \\setlength{\\tabcolsep}{4pt}
    \\begin{tabular}{lcccc}
        \\toprule
        & \\multicolumn{2}{c}{\\textbf{Accuracy}} & \\multicolumn{2}{c}{\\textbf{F1}} \\\\
        \\cmidrule(lr){2-3} \\cmidrule(lr){4-5}
        \\textbf{Method} & Mean$\\pm$Std & Gain(\\%) & Mean$\\pm$Std & Gain(\\%) \\\\
        \\midrule
"""
    
    for method in available_methods:
        display_name = method_names.get(method, method)
        acc = format_ranked(method, "accuracy")
        f1 = format_ranked(method, "f1")
        acc_gain = format_gain_ranked(method, "accuracy")
        f1_gain = format_gain_ranked(method, "f1")
        
        latex += f"        {display_name} & {acc} & {acc_gain} & {f1} & {f1_gain} \\\\\n"
    
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
        default=DEFAULT_PRED_DIR,
        help=f"Directory containing prediction CSVs (default: {DEFAULT_PRED_DIR})"
    )
    parser.add_argument(
        "--per_db",
        action="store_true",
        help="Print metrics per db_id in addition to overall"
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        default=True,
        help="Generate LaTeX table output (default: True)"
    )
    parser.add_argument(
        "--no-latex",
        action="store_false",
        dest="latex",
        help="Disable LaTeX table output"
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
        print(f"Error: No prediction files found in {args.pred_dir}")
        return

    # Group files by method/seed and collect per-DB metrics
    method_seed_db_metrics: Dict[str, Dict[int, Dict[str, Dict[str, float]]]] = defaultdict(lambda: defaultdict(dict))
    method_seed_metrics: Dict[str, Dict[int, Dict[str, Optional[float]]]] = defaultdict(dict)
    
    for path in files:
        if not os.path.exists(path):
            print(f"Warning: Missing file {path}")
            continue
        df = pd.read_csv(path)
        if not {"y_true", "y_pred"}.issubset(df.columns):
            print(f"Warning: Missing y_true/y_pred in {path}")
            continue

        method, seed = _parse_method_and_seed(path)
        
        if not args.latex:
            acc, f1, auc = _compute_metrics(df)
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
        if method and seed is not None:
            per_db = _compute_per_db_metrics(df)
            method_seed_db_metrics[method][seed] = per_db
            per_db_avg = _compute_per_db_average_metrics(df)
            method_seed_metrics[method][seed] = per_db_avg

    if args.latex:
        # Aggregate metrics (mean ± std) across seeds
        all_results: Dict[str, Dict[str, float]] = {}
        for method, seed_map in method_seed_metrics.items():
            all_results[method] = {}
            for metric_name in ["accuracy", "f1", "auc"]:
                values = [
                    metrics.get(metric_name)
                    for metrics in seed_map.values()
                    if metrics.get(metric_name) is not None
                ]
                if values:
                    all_results[method][metric_name] = float(np.mean(values))
                    all_results[method][f"{metric_name}_std"] = float(np.std(values))

        # Compute gain percentages
        gain_pct: Dict[str, float] = {}
        for metric in ["accuracy", "f1"]:
            gains = _compute_gain_percentage(method_seed_db_metrics, metric)
            for method, val in gains.items():
                if "_std" not in method:
                    gain_pct[f"{metric}_{method}"] = val
        
        # Print summary table
        print()
        print("=" * 120)
        print("FedGNN Experiment Results (Per-DB Averaged, Aggregated Across Seeds)")
        print("=" * 120)
        print(f"{'Method':<30} {'Accuracy':<20} {'Acc Gain(%)':<15} {'F1':<20} {'F1 Gain(%)':<15}")
        print("-" * 120)
        
        method_order = ["solo", "fedavg", "fedgnn_none", "fedgnn_node_only", "fedgnn_edge_only", "fedgnn_both"]
        for method in method_order:
            if method not in all_results:
                continue
            results = all_results[method]
            acc_str = _format_metric_std(results.get("accuracy"), results.get("accuracy_std"))
            f1_str = _format_metric_std(results.get("f1"), results.get("f1_std"))
            
            if method == "solo":
                acc_gain = "N/A"
                f1_gain = "N/A"
            else:
                acc_gain_val = gain_pct.get(f"accuracy_{method}")
                f1_gain_val = gain_pct.get(f"f1_{method}")
                acc_gain = f"{acc_gain_val:.1f}%" if acc_gain_val is not None else "N/A"
                f1_gain = f"{f1_gain_val:.1f}%" if f1_gain_val is not None else "N/A"
            
            print(f"{method:<30} {acc_str:<20} {acc_gain:<15} {f1_str:<20} {f1_gain:<15}")
        
        print("=" * 120)
        
        # Generate LaTeX
        output_path = args.output
        if output_path is None and args.pred_dir:
            output_path = os.path.join(args.pred_dir, "results_table.tex")
        
        latex = _generate_latex_table(all_results, gain_pct, output_path)
        print("\nGenerated LaTeX:")
        print("-" * 40)
        print(latex)


if __name__ == "__main__":
    main()
