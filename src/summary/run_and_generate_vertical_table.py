#!/usr/bin/env python3
"""
Vertical Federated Learning Table Generation Script

This script generates specialized LaTeX tables for vertical FL results with:
1. Solo (Primary Client) results - first row  
2. SplitNN method - middle section
3. Combined features baseline - final row

Features:
- Automatically detects and runs missing experiments (no manual intervention needed)
- Multi-seed experiment support with automatic aggregation (mean ± std)
- Option to include/exclude standard deviation in LaTeX tables
- Hyperparameter-based file naming for easy organization

Usage:
    python run_and_generate_vertical_table.py  # Auto-runs missing experiments
    python run_and_generate_vertical_table.py --no-solo --show-std --num-seeds 3
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import glob

# Import and reuse components from horizontal table generator
sys.path.append(str(Path(__file__).parent))
from run_and_generate_horizontal_table import HorizontalFLTableGenerator

class VerticalFLTableGenerator(HorizontalFLTableGenerator):
    """Generator for specialized vertical FL LaTeX tables, extending horizontal generator."""
    
    def find_result_files(self, include_solo: bool = True) -> Dict[str, List[str]]:
        """Find result files for vertical FL experiments."""
        print(f"\n--- Finding Result Files for Vertical FL ---")
        
        result_files = {}
        
        # Find SplitNN result files
        pattern = "splitnn_*_seed*.json"
        pattern_path = str(self.results_dir / "vertical" / pattern)
        files = list(glob.glob(pattern_path))
        if files:
            result_files["splitnn"] = files
            print(f"  Found {len(files)} files for SplitNN")
        
        # SecureBoost uses XGBoost performance data (lossless by design)
        # Load XGBoost solo performance from primary client files
        secureboost_solo_files = []
        for seed in range(5):  # Default 5 seeds
            filename = f"primary_client_xgb_48804_seed{seed}.json"
            filepath = self.results_dir / "primary_client" / filename
            if filepath.exists():
                secureboost_solo_files.append(str(filepath))
        
        # Load XGBoost combined performance from centralized files  
        secureboost_combined_files = []
        for seed in range(5):  # Default 5 seeds
            filename = f"centralized_xgb_48804_00381_seed{seed}.json"
            filepath = self.results_dir / "centralized" / filename
            if filepath.exists():
                secureboost_combined_files.append(str(filepath))
        
        if secureboost_solo_files or secureboost_combined_files:
            result_files["secureboost"] = {
                "solo": secureboost_solo_files,
                "combined": secureboost_combined_files
            }
            print(f"  Found {len(secureboost_solo_files)} solo and {len(secureboost_combined_files)} combined files for SecureBoost")
        else:
            print(f"  No XGBoost files found for SecureBoost")
        
        # Find primary client files if requested (for Solo results)
        if include_solo:
            pattern = "primary_client_nn_*.json"  # Updated to match new naming convention
            pattern_path = str(self.results_dir / "primary_client" / pattern)
            files = list(glob.glob(pattern_path))
            if files:
                result_files["primary"] = files
                print(f"  Found {len(files)} files for primary client")
            else:
                print(f"  No files found for primary client")
                # If no primary client files found, we need to run them
                print(f"  Will need to run primary client experiments")
        
        return result_files
    
    def run_splitnn_experiments(self, num_seeds=5):
        """Run SplitNN experiments for multiple seeds."""
        print(f"\n--- Running SplitNN Experiments for {num_seeds} seeds ---")
        import subprocess
        
        # Get the script directory
        script_dir = self.base_dir / "src" / "demo"
        splitnn_script = script_dir / "train_splitnn.py"
        
        if not splitnn_script.exists():
            print(f"❌ SplitNN script not found: {splitnn_script}")
            return False
        
        success_count = 0
        
        # Run experiments for each seed
        for seed in range(num_seeds):
            print(f"\n🔄 Running SplitNN experiment for seed {seed}...")
            
            # Run the SplitNN script with current seed
            cmd = f"python {splitnn_script} --seed {seed}"
            
            try:
                print(f"Running: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.base_dir)
                
                if result.returncode == 0:
                    print(f"✅ SplitNN experiment for seed {seed} completed successfully")
                    success_count += 1
                else:
                    print(f"❌ SplitNN experiment for seed {seed} failed:")
                    print(f"stdout: {result.stdout}")
                    print(f"stderr: {result.stderr}")
                    
            except Exception as e:
                print(f"❌ Error running SplitNN experiment for seed {seed}: {e}")
        
        print(f"\n📊 Completed {success_count}/{num_seeds} seed experiments")
        return success_count > 0  # Return True if at least one seed succeeded
    
    def check_secureboost_data(self, num_seeds=5):
        """Check if SecureBoost XGBoost data is available."""
        print(f"\n--- Checking SecureBoost XGBoost Data Availability ---")
        
        solo_available = 0
        combined_available = 0
        
        # Check primary client (solo) files
        for seed in range(num_seeds):
            filename = f"primary_client_xgb_48804_seed{seed}.json"
            filepath = self.results_dir / "primary_client" / filename
            if filepath.exists():
                solo_available += 1
        
        # Check centralized (combined) files
        for seed in range(num_seeds):
            filename = f"centralized_xgb_48804_00381_seed{seed}.json"
            filepath = self.results_dir / "centralized" / filename
            if filepath.exists():
                combined_available += 1
        
        print(f"  SecureBoost solo files available: {solo_available}/{num_seeds}")
        print(f"  SecureBoost combined files available: {combined_available}/{num_seeds}")
        
        if solo_available == 0 and combined_available == 0:
            print(f"  ❌ No XGBoost data available for SecureBoost")
            print(f"  Note: SecureBoost requires XGBoost experiments to be run first")
            print(f"  Run: python src/summary/run_and_generate_ml_model_tables.py --scenario vfl --model xgboost")
            return False
        elif solo_available < num_seeds or combined_available < num_seeds:
            print(f"  ⚠️  Incomplete XGBoost data for SecureBoost")
            print(f"  Note: Run full XGBoost experiments for better results")
            return True
        else:
            print(f"  ✅ Complete XGBoost data available for SecureBoost")
            return True
    
    def run_primary_client_experiments(self, num_seeds=5):
        """Run primary client experiments for multiple seeds."""
        print(f"\n--- Running Primary Client Experiments for {num_seeds} seeds ---")
        import subprocess
        
        # Get the script directory
        script_dir = self.base_dir / "src" / "demo"
        primary_script = script_dir / "run_primary_client.py"
        
        if not primary_script.exists():
            print(f"❌ Primary client script not found: {primary_script}")
            return False
        
        success_count = 0
        
        # Run experiments for each seed
        for seed in range(num_seeds):
            print(f"\n🔄 Running primary client experiment for seed {seed}...")
            
            # Run the primary client script with Neural Network and current seed
            cmd = f"python {primary_script} -m nn --seed {seed}"
            
            try:
                print(f"Running: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.base_dir)
                
                if result.returncode == 0:
                    print(f"✅ Primary client experiment for seed {seed} completed successfully")
                    success_count += 1
                else:
                    print(f"❌ Primary client experiment for seed {seed} failed:")
                    print(f"stdout: {result.stdout}")
                    print(f"stderr: {result.stderr}")
                    
            except Exception as e:
                print(f"❌ Error running primary client experiment for seed {seed}: {e}")
        
        print(f"\n📊 Completed {success_count}/{num_seeds} seed experiments")
        return success_count > 0  # Return True if at least one seed succeeded
    
    def aggregate_results(self, result_files: Dict[str, List[str]]) -> Dict:
        """Aggregate results from multiple runs with mean and std."""
        print(f"\n--- Aggregating Results ---")
        
        # Calculate total files processed (handling SecureBoost's special structure)
        total_files = 0
        for algorithm, files in result_files.items():
            if algorithm == "secureboost":
                total_files += len(files.get('solo', [])) + len(files.get('combined', []))
            else:
                total_files += len(files)
        
        aggregated = {
            "experiment_type": "vertical",
            "num_files_processed": total_files,
            "algorithms": {}
        }
        
        for algorithm, files in result_files.items():
            if algorithm == "secureboost":
                # SecureBoost has a special file structure (dict with solo/combined)
                print(f"  Processing {algorithm}: {len(files.get('solo', []))} solo + {len(files.get('combined', []))} combined files")
                aggregated["algorithms"][algorithm] = self._aggregate_secureboost_results(files)
            else:
                # Regular algorithms (primary, splitnn)
                print(f"  Processing {algorithm}: {len(files)} files")
                
                # Load all results for this algorithm
                all_results = []
                for file_path in files:
                    try:
                        with open(file_path, 'r') as f:
                            result = json.load(f)
                            all_results.append(result)
                    except Exception as e:
                        print(f"    Warning: Could not load {file_path}: {e}")
                
                if all_results:
                    if algorithm == "primary":
                        aggregated["algorithms"][algorithm] = self._aggregate_primary_results(all_results)
                    elif algorithm == "splitnn":
                        aggregated["algorithms"][algorithm] = self._aggregate_splitnn_results(all_results)
        
        return aggregated
    
    def _aggregate_primary_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate primary client results."""
        aggregated = {"models": {}}
        
        # Get all models from first result
        if all_results:
            first_result = all_results[0]
            models = list(first_result.get("results", {}).keys())
            
            for model in models:
                # Collect metrics across all runs
                metrics_across_runs = []
                for result in all_results:
                    if model in result.get("results", {}):
                        metrics = result["results"][model]
                        metrics_across_runs.append(metrics)  # [acc, prec, recall, f1]
                
                if metrics_across_runs:
                    # Calculate mean and std for each metric
                    metrics_array = np.array(metrics_across_runs)
                    mean_metrics = np.mean(metrics_array, axis=0)
                    std_metrics = np.std(metrics_array, axis=0)
                    
                    aggregated["models"][model] = {
                        "accuracy": {"mean": float(mean_metrics[0]), "std": float(std_metrics[0])},
                        "precision": {"mean": float(mean_metrics[1]), "std": float(std_metrics[1])},
                        "recall": {"mean": float(mean_metrics[2]), "std": float(std_metrics[2])},
                        "f1": {"mean": float(mean_metrics[3]), "std": float(std_metrics[3])}
                    }
        
        return aggregated
    
    def _aggregate_splitnn_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate SplitNN results."""
        aggregated = {"splitnn": {}, "centralized": {}}
        
        # Collect SplitNN and centralized metrics
        splitnn_metrics = []
        centralized_metrics = []
        
        for result in all_results:
            if "splitnn_results" in result:
                splitnn = result["splitnn_results"]
                splitnn_metrics.append([
                    splitnn.get("accuracy", 0),
                    splitnn.get("precision", 0),
                    splitnn.get("recall", 0),
                    splitnn.get("f1", 0)
                ])
            
            if "centralized_results" in result:
                centralized = result["centralized_results"]
                centralized_metrics.append([
                    centralized.get("accuracy", 0),
                    centralized.get("precision", 0),
                    centralized.get("recall", 0),
                    centralized.get("f1", 0)
                ])
        
        # Calculate aggregated metrics
        if splitnn_metrics:
            splitnn_array = np.array(splitnn_metrics)
            splitnn_mean = np.mean(splitnn_array, axis=0)
            splitnn_std = np.std(splitnn_array, axis=0)
            
            aggregated["splitnn"] = {
                "accuracy": {"mean": float(splitnn_mean[0]), "std": float(splitnn_std[0])},
                "precision": {"mean": float(splitnn_mean[1]), "std": float(splitnn_std[1])},
                "recall": {"mean": float(splitnn_mean[2]), "std": float(splitnn_std[2])},
                "f1": {"mean": float(splitnn_mean[3]), "std": float(splitnn_std[3])}
            }
        
        if centralized_metrics:
            centralized_array = np.array(centralized_metrics)
            centralized_mean = np.mean(centralized_array, axis=0)
            centralized_std = np.std(centralized_array, axis=0)
            
            aggregated["centralized"] = {
                "accuracy": {"mean": float(centralized_mean[0]), "std": float(centralized_std[0])},
                "precision": {"mean": float(centralized_mean[1]), "std": float(centralized_std[1])},
                "recall": {"mean": float(centralized_mean[2]), "std": float(centralized_std[2])},
                "f1": {"mean": float(centralized_mean[3]), "std": float(centralized_std[3])}
            }
        
        return aggregated
    
    def _aggregate_secureboost_results(self, secureboost_files: Dict) -> Dict:
        """Aggregate SecureBoost results from XGBoost performance data."""
        aggregated = {"secureboost_solo": {}, "secureboost_combined": {}}
        
        # Process solo files (primary client files for VFL)
        solo_files = secureboost_files.get("solo", [])
        if solo_files:
            solo_f1_scores = []
            
            for file_path in solo_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # For primary client files, extract XGBoost results
                    if "results" in data:
                        results = data.get("results", {})
                        if "XGBoost" in results:
                            # results["XGBoost"] = [accuracy, precision, recall, f1]
                            f1_score = results["XGBoost"][3]  # F1 is at index 3
                            solo_f1_scores.append(f1_score)
                        
                except Exception as e:
                    print(f"    Warning: Could not load SecureBoost solo file {file_path}: {e}")
            
            # Calculate solo metrics
            if solo_f1_scores:
                solo_mean = np.mean(solo_f1_scores)
                solo_std = np.std(solo_f1_scores)
                aggregated["secureboost_solo"] = {
                    "accuracy": {"mean": solo_mean, "std": solo_std},
                    "precision": {"mean": solo_mean, "std": solo_std},
                    "recall": {"mean": solo_mean, "std": solo_std},
                    "f1": {"mean": solo_mean, "std": solo_std}
                }
                print(f"    Loaded SecureBoost Solo: F1={solo_mean:.4f}±{solo_std:.4f} ({len(solo_f1_scores)} files)")
        
        # Process combined files (centralized training files for VFL)
        combined_files = secureboost_files.get("combined", [])
        if combined_files:
            combined_f1_scores = []
            
            for file_path in combined_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    # For centralized files, check experiment type and extract XGBoost results
                    if "results" in data and data.get("experiment_type") == "centralized":
                        results = data.get("results", {})
                        if "XGBoost" in results:
                            # results["XGBoost"] = [accuracy, precision, recall, f1]
                            f1_score = results["XGBoost"][3]  # F1 is at index 3
                            combined_f1_scores.append(f1_score)
                        
                except Exception as e:
                    print(f"    Warning: Could not load SecureBoost combined file {file_path}: {e}")
            
            # Calculate combined metrics
            if combined_f1_scores:
                combined_mean = np.mean(combined_f1_scores)
                combined_std = np.std(combined_f1_scores)
                aggregated["secureboost_combined"] = {
                    "accuracy": {"mean": combined_mean, "std": combined_std},
                    "precision": {"mean": combined_mean, "std": combined_std},
                    "recall": {"mean": combined_mean, "std": combined_std},
                    "f1": {"mean": combined_mean, "std": combined_std}
                }
                print(f"    Loaded SecureBoost Combined: F1={combined_mean:.4f}±{combined_std:.4f} ({len(combined_f1_scores)} files)")
        
        return aggregated
    
    def generate_vertical_latex_table(self, results: Dict, show_std: bool = False) -> str:
        """Generate specialized vertical FL LaTeX table."""
        print(f"\n--- Generating Vertical FL LaTeX Table (show_std={show_std}) ---")
        
        latex = []
        latex.append("\\begin{table}[ht]")
        latex.append("  \\centering")
        latex.append("  \\caption{Vertical Federated Learning Performance Comparison}")
        latex.append("  \\label{tab:vertical_fl_comparison}")
        latex.append("  \\begin{tabular}{lcccc}")
        latex.append("  \\toprule")
        latex.append("  \\textbf{Method} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} \\\\")
        latex.append("  \\midrule")
        
        def format_metric(metric_dict):
            """Format a metric with optional standard deviation."""
            if show_std and "std" in metric_dict:
                mean = metric_dict["mean"]
                std = metric_dict["std"]
                return f"{mean:.4f} ± {std:.4f}"
            else:
                return f"{metric_dict['mean']:.4f}"
        
        algorithms = results.get("algorithms", {})
        
        # 1. Solo (Primary Client) results - first row
        if "primary" in algorithms:
            models_data = algorithms["primary"].get("models", {})
            
            # Use Neural Network if available, otherwise best model
            if "Neural_Network" in models_data:
                metrics = models_data["Neural_Network"]
                model_name = "Solo (DB 48804)"
            elif models_data:
                # Find best model
                best_model = max(models_data.keys(), 
                               key=lambda x: models_data[x]["f1"]["mean"])
                metrics = models_data[best_model]
                model_name = f"Solo (DB 48804, {best_model.replace('_', ' ')})"
            else:
                metrics = None
                model_name = None
            
            if metrics:
                acc = format_metric(metrics['accuracy'])
                prec = format_metric(metrics['precision'])
                rec = format_metric(metrics['recall'])
                f1 = format_metric(metrics['f1'])
                latex.append(f"  {model_name} & {acc} & {prec} & {rec} & {f1} \\\\")
                latex.append("  \\midrule")
        
        # 2. Federated Learning methods - middle section
        if "splitnn" in algorithms:
            splitnn_data = algorithms["splitnn"]
            if "splitnn" in splitnn_data:
                metrics = splitnn_data["splitnn"]
                acc = format_metric(metrics['accuracy'])
                prec = format_metric(metrics['precision'])
                rec = format_metric(metrics['recall'])
                f1 = format_metric(metrics['f1'])
                
                latex.append(f"  SplitNN (DB 48804 + DB 00381) & {acc} & {prec} & {rec} & {f1} \\\\")
        
        # Add midrule after federated learning methods
        if "splitnn" in algorithms:
            latex.append("  \\midrule")
        
        # 3. Combined features baseline - final row
        centralized_metrics = None
        if "splitnn" in algorithms:
            centralized_metrics = algorithms["splitnn"].get("centralized", {})
        
        if centralized_metrics and "accuracy" in centralized_metrics:
            acc = format_metric(centralized_metrics['accuracy'])
            prec = format_metric(centralized_metrics['precision'])
            rec = format_metric(centralized_metrics['recall'])
            f1 = format_metric(centralized_metrics['f1'])
            
            # Bold centralized as it's typically the best
            latex.append(f"  Combined Features & \\textbf{{{acc}}} & \\textbf{{{prec}}} & \\textbf{{{rec}}} & \\textbf{{{f1}}} \\\\")
        
        latex.append("  \\bottomrule")
        latex.append("  \\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def generate_secureboost_vertical_latex_table(self, results: Dict, show_std: bool = False) -> str:
        """Generate separate LaTeX table for SecureBoost vertical FL results."""
        print(f"\n--- Generating SecureBoost Vertical LaTeX Table (show_std={show_std}) ---")
        
        latex = []
        latex.append("\\begin{table}[ht]")
        latex.append("  \\centering")
        latex.append("  \\caption{SecureBoost Performance on Vertical Federated Learning}")
        latex.append("  \\label{tab:secureboost_vertical_fl}")
        latex.append("  \\begin{tabular}{lcccc}")
        latex.append("  \\toprule")
        latex.append("  \\textbf{Method} & \\textbf{Accuracy} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1-Score} \\\\")
        latex.append("  \\midrule")
        
        def format_metric(metric_dict):
            """Format a metric with optional standard deviation."""
            if show_std and "std" in metric_dict:
                mean = metric_dict["mean"]
                std = metric_dict["std"]
                return f"{mean:.4f} ± {std:.4f}"
            else:
                return f"{metric_dict['mean']:.4f}"
        
        algorithms = results.get("algorithms", {})
        
        # Collect all method data for global best/second best calculation
        all_methods = []
        
        # Add SecureBoost results if available
        if "secureboost" in algorithms:
            secureboost_data = algorithms["secureboost"]
            
            # Add solo results
            if "secureboost_solo" in secureboost_data:
                all_methods.append(("SecureBoost (Solo)", secureboost_data["secureboost_solo"], "solo"))
            
            # Add combined results
            if "secureboost_combined" in secureboost_data:
                all_methods.append(("SecureBoost (Combined)", secureboost_data["secureboost_combined"], "combined"))
        
        # Calculate global best and second best for each metric
        metrics_list = ['accuracy', 'precision', 'recall', 'f1']
        global_rankings = {}
        
        for metric in metrics_list:
            values = []
            for method_name, method_metrics, method_type in all_methods:
                if metric in method_metrics:
                    values.append((method_metrics[metric]['mean'], method_name))
            
            # Sort by value (descending for better performance)
            values.sort(key=lambda x: x[0], reverse=True)
            
            global_rankings[metric] = {
                'best': values[0][1] if len(values) > 0 else None,
                'second_best': values[1][1] if len(values) > 1 else None
            }
        
        def format_metric_with_style(metric_dict, method_name, metric_name):
            """Format metric with global best/second best styling."""
            base_value = format_metric(metric_dict)
            best = global_rankings[metric_name]['best']
            second_best = global_rankings[metric_name]['second_best']
            
            if method_name == best:
                return f"\\textbf{{{base_value}}}"
            elif method_name == second_best:
                return f"\\underline{{{base_value}}}"
            else:
                return base_value
        
        # Generate table rows
        if "secureboost" in algorithms:
            secureboost_data = algorithms["secureboost"]
            
            # Solo row
            if "secureboost_solo" in secureboost_data:
                method_name = "SecureBoost (Solo)"
                metrics = secureboost_data["secureboost_solo"]
                acc = format_metric_with_style(metrics['accuracy'], method_name, 'accuracy')
                prec = format_metric_with_style(metrics['precision'], method_name, 'precision')
                rec = format_metric_with_style(metrics['recall'], method_name, 'recall')
                f1 = format_metric_with_style(metrics['f1'], method_name, 'f1')
                latex.append(f"  {method_name} & {acc} & {prec} & {rec} & {f1} \\\\")
            
            # Combined row
            if "secureboost_combined" in secureboost_data:
                method_name = "SecureBoost (Combined)"
                metrics = secureboost_data["secureboost_combined"]
                acc = format_metric_with_style(metrics['accuracy'], method_name, 'accuracy')
                prec = format_metric_with_style(metrics['precision'], method_name, 'precision')
                rec = format_metric_with_style(metrics['recall'], method_name, 'recall')
                f1 = format_metric_with_style(metrics['f1'], method_name, 'f1')
                latex.append(f"  {method_name} & {acc} & {prec} & {rec} & {f1} \\\\")
        else:
            latex.append("  SecureBoost (Solo) & N/A & N/A & N/A & N/A \\\\")
            latex.append("  SecureBoost (Combined) & N/A & N/A & N/A & N/A \\\\")
        
        latex.append("  \\bottomrule")
        latex.append("  \\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def save_results_and_table(self, aggregated_results: Dict, latex_table: str, show_std: bool = False, num_seeds: int = 5, include_solo: bool = True):
        """Save aggregated results and LaTeX table to files."""
        
        # Create hyperparameter-based filename components
        std_str = "std" if show_std else "mean"
        solo_str = "with_solo" if include_solo else "no_solo"
        
        # Hyperparameter-based filename: vertical_fl_splitnn_5seeds_mean_with_solo (FedTree has separate table)
        base_filename = f"vertical_fl_splitnn_{num_seeds}seeds_{std_str}_{solo_str}"
        
        # Save aggregated results
        results_file = self.output_dir / f"{base_filename}_aggregated.json"
        with open(results_file, 'w') as f:
            json.dump(aggregated_results, f, indent=2)
        
        # Save LaTeX table
        latex_file = self.output_dir / f"{base_filename}_table.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        print(f"✅ Saved aggregated results: {results_file}")
        print(f"✅ Saved LaTeX table: {latex_file}")
        
        return str(results_file), str(latex_file)


def main():
    parser = argparse.ArgumentParser(
        description="Generate specialized vertical FL LaTeX tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate table with SplitNN and solo results (auto-runs missing experiments)
  python run_and_generate_vertical_table.py
  
  # Generate table with standard deviations included
  python run_and_generate_vertical_table.py --show-std
  
  # Generate table without solo results
  python run_and_generate_vertical_table.py --no-solo
  
  # Use custom number of seeds for missing experiments
  python run_and_generate_vertical_table.py --num-seeds 3 --show-std
        """
    )
    
    parser.add_argument("--algorithms", type=str, default="splitnn,secureboost",
                       help="Comma-separated FL algorithms (default: splitnn,secureboost)")
    parser.add_argument("--no-solo", action="store_true",
                       help="Exclude solo (primary client) results from table")
    parser.add_argument("--show-std", action="store_true",
                       help="Include standard deviation in LaTeX table (default: mean only)")
    parser.add_argument("--num-seeds", type=int, default=5,
                       help="Number of seeds for experiments (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Parse algorithms
    algorithms = [alg.strip() for alg in args.algorithms.split(",")]
    include_solo = not args.no_solo
    
    # Initialize generator
    generator = VerticalFLTableGenerator(args.output)
    
    print(f"\n{'='*100}")
    print(f"GENERATING VERTICAL FEDERATED LEARNING TABLE")
    print(f"{'='*100}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Include solo results: {include_solo}")
    print(f"Show standard deviation: {args.show_std}")
    print(f"Number of seeds: {args.num_seeds}")
    
    # Find result files
    result_files = generator.find_result_files(include_solo)
    
    # Check for missing experiments and insufficient seeds
    # Handle SplitNN
    if "splitnn" in algorithms:
        if "splitnn" not in result_files:
            print(f"\n🔄 SplitNN results missing, automatically running experiments...")
            success = generator.run_splitnn_experiments(num_seeds=args.num_seeds)
            if success:
                # Re-scan for SplitNN files
                splitnn_pattern = str(generator.results_dir / "vertical" / "splitnn_*.json")
                splitnn_files = list(glob.glob(splitnn_pattern))
                if splitnn_files:
                    result_files["splitnn"] = splitnn_files
                    print(f"✅ Found {len(splitnn_files)} SplitNN files after running experiments")
            else:
                print(f"⚠️  SplitNN experiments failed, proceeding without them")
        else:
            # Check if we have enough seed files for SplitNN
            current_files = result_files["splitnn"]
            if len(current_files) < args.num_seeds:
                print(f"⚠️  SplitNN: Found {len(current_files)} files but need {args.num_seeds} seeds")
                print(f"🔄 Running additional SplitNN seed experiments...")
                success = generator.run_splitnn_experiments(num_seeds=args.num_seeds)
                if success:
                    splitnn_pattern = str(generator.results_dir / "vertical" / "splitnn_*.json")
                    splitnn_files = list(glob.glob(splitnn_pattern))
                    if splitnn_files:
                        result_files["splitnn"] = splitnn_files
                        print(f"✅ Now found {len(splitnn_files)} SplitNN files after additional seeds")
    
    # Handle SecureBoost
    if "secureboost" in algorithms:
        if "secureboost" not in result_files:
            print(f"\n🔄 SecureBoost XGBoost data missing, checking availability...")
            success = generator.check_secureboost_data(num_seeds=args.num_seeds)
            if success:
                # Re-scan for SecureBoost data
                result_files = generator.find_result_files(include_solo)
                if "secureboost" in result_files:
                    print(f"✅ Found SecureBoost XGBoost data")
                else:
                    print(f"⚠️  SecureBoost data still not available, proceeding without it")
            else:
                print(f"⚠️  SecureBoost XGBoost data not available, proceeding without it")
        else:
            print(f"✅ SecureBoost XGBoost data already available")
    
    # Automatically run missing primary client experiments
    if include_solo and "primary" not in result_files:
        print(f"\n🔄 Primary client results missing, automatically running experiments...")
        success = generator.run_primary_client_experiments(num_seeds=args.num_seeds)
        if success:
            # Re-scan for primary client files
            primary_pattern = str(generator.results_dir / "primary_client" / "primary_client_nn_*.json")
            primary_files = list(glob.glob(primary_pattern))
            if primary_files:
                result_files["primary"] = primary_files
                print(f"✅ Found {len(primary_files)} primary client files after running experiments")
        else:
            print(f"⚠️  Primary client experiments failed, proceeding without them")
    
    if not result_files:
        print(f"❌ No result files found after attempting to run missing experiments")
        return
    
    # Aggregate results
    aggregated = generator.aggregate_results(result_files)
    
    # Separate SecureBoost from other algorithms
    secureboost_only = set(algorithms) == {"secureboost"}
    other_algorithms = [alg for alg in algorithms if alg != "secureboost"]
    has_secureboost = "secureboost" in algorithms
    
    # Generate main LaTeX table (excluding SecureBoost) only if there are other algorithms
    if other_algorithms and not secureboost_only:
        print(f"\n📊 Generating main vertical table with algorithms: {other_algorithms}")
        latex_table = generator.generate_vertical_latex_table(aggregated, show_std=args.show_std)
        
        # Save main table results
        results_file, latex_file = generator.save_results_and_table(aggregated, latex_table, show_std=args.show_std, num_seeds=args.num_seeds, include_solo=include_solo)
        
        print(f"\n--- Main LaTeX Table Preview ---")
        print(latex_table)
        print(f"--- End Main Table Preview ---")
    
    # Generate separate SecureBoost table if SecureBoost is included
    if has_secureboost:
        print(f"\n📊 Generating separate SecureBoost vertical table")
        secureboost_latex_table = generator.generate_secureboost_vertical_latex_table(aggregated, show_std=args.show_std)
        
        # Save SecureBoost table separately
        std_str = "std" if args.show_std else "mean"
        secureboost_filename = f"secureboost_vertical_fl_{args.num_seeds}seeds_{std_str}_table.tex"
        secureboost_latex_file = generator.output_dir / secureboost_filename
        with open(secureboost_latex_file, 'w') as f:
            f.write(secureboost_latex_table)
        print(f"✅ Saved SecureBoost LaTeX table: {secureboost_latex_file}")
        
        # Always show SecureBoost table preview when SecureBoost is included
        print(f"\n--- SecureBoost LaTeX Table Preview ---")
        print(secureboost_latex_table)
        print(f"--- End SecureBoost Preview ---")
    
    print(f"\n{'='*100}")
    print("🎉 VERTICAL FL TABLE GENERATION COMPLETED!")
    print(f"📁 Results saved in: {generator.output_dir}")
    if args.show_std:
        print(f"📊 LaTeX table includes mean ± std values")
    else:
        print(f"📊 LaTeX table shows mean values only")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()