#!/usr/bin/env python3
"""
Horizontal Federated Learning Table Generation Script

This script generates specialized LaTeX tables for horizontal FL results with:
1. Individual client results (Solo) - models trained individually, tested on COMBINED test set (A+B)
2. Federated learning methods - middle section
3. Centralized Neural Network baseline - final row

Key Features:
- Solo rows show performance on the overall test set (merged from all clients)
- Automatically detects and runs missing experiments (no manual intervention needed)
- Multi-seed experiment support with mean ± std aggregation
- Hyperparameter-based file naming for easy organization

Usage:
    python run_and_generate_horizontal_table.py  # Auto-runs missing experiments
    python run_and_generate_horizontal_table.py --algorithms fedavg,fedprox --show-std
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

class HorizontalFLTableGenerator:
    """Generator for specialized horizontal FL LaTeX tables."""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent.parent
        self.results_dir = self.base_dir / "results"
        self.output_dir = self.base_dir / "results" / "tables"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def find_result_files(self, algorithms: List[str] = None, include_individual: bool = True) -> Dict[str, List[str]]:
        """Find result files for specified algorithms."""
        print(f"\n--- Finding Result Files for Horizontal FL ---")
        
        if algorithms is None:
            algorithms = ["fedavg", "fedprox", "scaffold", "fedov", "fedtree"]
        
        result_files = {}
        
        # Find federated learning algorithm files
        for algorithm in algorithms:
            pattern = f"{algorithm.lower()}_*_seed*.json"
            pattern_path = str(self.results_dir / "horizontal" / pattern)
            files = list(glob.glob(pattern_path))
            if files:
                result_files[algorithm] = files
                print(f"  Found {len(files)} files for {algorithm}")
            else:
                print(f"  No files found for {algorithm}")
        
        # Find individual client files if requested
        if include_individual:
            pattern = "individual_clients_nn_*.json"  # Updated to match new naming convention
            pattern_path = str(self.results_dir / "horizontal" / pattern)
            files = list(glob.glob(pattern_path))
            if files:
                result_files["individual"] = files
                print(f"  Found {len(files)} files for individual clients")
            else:
                print(f"  No files found for individual clients")
                # If no individual client files found, we need to run them
                print(f"  Will need to run individual client experiments")
        
        return result_files
    
    def run_individual_clients(self, num_seeds=5):
        """Run individual client experiments for multiple seeds."""
        print(f"\n--- Running Individual Client Experiments for {num_seeds} seeds ---")
        import subprocess
        import sys
        
        # Get the script directory
        script_dir = self.base_dir / "src" / "demo"
        individual_script = script_dir / "run_individual_clients.py"
        
        if not individual_script.exists():
            print(f"❌ Individual client script not found: {individual_script}")
            return False
        
        success_count = 0
        
        # Run experiments for each seed
        for seed in range(num_seeds):
            print(f"\n🔄 Running individual clients experiment for seed {seed}...")
            
            # Run the individual client script with Neural Network and current seed
            cmd = f"python {individual_script} -m nn --seed {seed}"
            
            try:
                print(f"Running: {cmd}")
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.base_dir)
                
                if result.returncode == 0:
                    print(f"✅ Individual client experiments for seed {seed} completed successfully")
                    success_count += 1
                else:
                    print(f"❌ Individual client experiments for seed {seed} failed:")
                    print(f"stdout: {result.stdout}")
                    print(f"stderr: {result.stderr}")
                    
            except Exception as e:
                print(f"❌ Error running individual client experiments for seed {seed}: {e}")
        
        print(f"\n📊 Completed {success_count}/{num_seeds} seed experiments")
        return success_count > 0  # Return True if at least one seed succeeded
    
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
    
    def run_federated_algorithms(self, missing_algorithms: List[str], num_seeds: int = 5):
        """Run missing federated learning algorithms for multiple seeds."""
        print(f"\n--- Running Missing Federated Algorithms: {missing_algorithms} ---")
        import subprocess
        
        # Get the script directory
        script_dir = self.base_dir / "src" / "demo"
        
        # Algorithm script mapping
        algorithm_scripts = {
            "fedavg": "train_fedavg.py",
            "fedprox": "train_fedprox.py", 
            "scaffold": "train_scaffold.py",
            "fedov": "train_fedov.py",
            "fedtree": "train_fedtree.py"
        }
        
        overall_success = False
        
        for algorithm in missing_algorithms:
            if algorithm not in algorithm_scripts:
                print(f"⚠️  Unknown algorithm: {algorithm}")
                continue
                
            script_name = algorithm_scripts[algorithm]
            script_path = script_dir / script_name
            
            if not script_path.exists():
                print(f"❌ Script not found: {script_path}")
                continue
            
            print(f"\n🔄 Running {algorithm} experiments for {num_seeds} seeds...")
            success_count = 0
            
            # Run experiments for each seed
            for seed in range(num_seeds):
                print(f"\n🔄 Running {algorithm} experiment for seed {seed}...")
                
                # Run the algorithm script with current seed
                cmd = f"python {script_path} --seed {seed}"
                
                try:
                    print(f"Running: {cmd}")
                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.base_dir)
                    
                    if result.returncode == 0:
                        print(f"✅ {algorithm} experiment for seed {seed} completed successfully")
                        success_count += 1
                    else:
                        print(f"❌ {algorithm} experiment for seed {seed} failed:")
                        print(f"stdout: {result.stdout}")
                        print(f"stderr: {result.stderr}")
                        
                except Exception as e:
                    print(f"❌ Error running {algorithm} experiment for seed {seed}: {e}")
            
            if success_count > 0:
                overall_success = True
                print(f"\n📊 {algorithm}: Completed {success_count}/{num_seeds} seed experiments")
            else:
                print(f"\n❌ {algorithm}: All experiments failed")
        
        return overall_success
    
    def run_centralized_baseline(self):
        """Run centralized baseline if needed."""
        print(f"\n--- Running Centralized Baseline ---")
        import subprocess
        
        # Get the script directory  
        script_dir = self.base_dir / "src" / "demo"
        centralized_script = script_dir / "centralized_training.py"
        
        if not centralized_script.exists():
            print(f"❌ Centralized training script not found: {centralized_script}")
            return False
        
        cmd = f"python {centralized_script}"
        
        try:
            print(f"Running: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                print("✅ Centralized baseline completed successfully")
                return True
            else:
                print(f"❌ Centralized baseline failed:")
                print(f"stdout: {result.stdout}")
                print(f"stderr: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running centralized baseline: {e}")
            return False
    
    def aggregate_results(self, result_files: Dict[str, List[str]]) -> Dict:
        """Aggregate results from multiple runs with mean and std."""
        print(f"\n--- Aggregating Results ---")
        
        aggregated = {
            "experiment_type": "horizontal",
            "num_files_processed": sum(len(files) for files in result_files.values()),
            "algorithms": {}
        }
        
        for algorithm, files in result_files.items():
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
                if algorithm == "individual":
                    aggregated["algorithms"][algorithm] = self._aggregate_individual_results(all_results)
                else:
                    aggregated["algorithms"][algorithm] = self._aggregate_federated_results(all_results, algorithm)
        
        return aggregated
    
    def _aggregate_individual_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate individual client results using combined test set evaluation."""
        aggregated = {"clients": {}}
        
        # Get all client names from first result
        if all_results:
            first_result = all_results[0]
            
            # Use combined_results if available (models tested on merged test set A+B)
            # This ensures Solo rows show performance on overall test set, not individual client test sets
            if "combined_results" in first_result and first_result["combined_results"]:
                print("  Using combined_results for Solo rows (models tested on merged test set A+B)")
                clients = list(first_result.get("combined_results", {}).keys())
                results_key = "combined_results"
            else:
                print("  Using individual_results for Solo rows (fallback - models tested on individual client test sets)")
                clients = list(first_result.get("individual_results", {}).keys())
                results_key = "individual_results"
            
            for client in clients:
                aggregated["clients"][client] = {}
                
                # Get all models for this client - focus on Neural Network
                models = list(first_result[results_key][client].keys())
                
                for model in models:
                    # Collect metrics across all runs
                    metrics_across_runs = []
                    for result in all_results:
                        if client in result.get(results_key, {}) and model in result[results_key][client]:
                            metrics = result[results_key][client][model]
                            metrics_across_runs.append(metrics)  # [acc, prec, recall, f1]
                    
                    if metrics_across_runs:
                        # Calculate mean and std for each metric
                        metrics_array = np.array(metrics_across_runs)
                        mean_metrics = np.mean(metrics_array, axis=0)
                        std_metrics = np.std(metrics_array, axis=0)
                        
                        aggregated["clients"][client][model] = {
                            "accuracy": {"mean": float(mean_metrics[0]), "std": float(std_metrics[0])},
                            "precision": {"mean": float(mean_metrics[1]), "std": float(std_metrics[1])},
                            "recall": {"mean": float(mean_metrics[2]), "std": float(std_metrics[2])},
                            "f1": {"mean": float(mean_metrics[3]), "std": float(std_metrics[3])}
                        }
        
        return aggregated
    
    def _aggregate_federated_results(self, all_results: List[Dict], algorithm: str) -> Dict:
        """Aggregate federated learning algorithm results."""
        aggregated = {"weighted_metrics": {}, "centralized_baseline": {}}
        
        # Collect weighted metrics across runs
        metrics_across_runs = []
        centralized_metrics_runs = []
        
        for result in all_results:
            if "weighted_metrics" in result:
                metrics = result["weighted_metrics"]
                metrics_across_runs.append([
                    metrics.get("accuracy", 0),
                    metrics.get("precision", 0),
                    metrics.get("recall", 0),
                    metrics.get("f1", 0)
                ])
            
            # Also collect centralized baseline if available
            if "centralized_baseline" in result:
                baseline = result["centralized_baseline"]
                centralized_metrics_runs.append([
                    baseline.get("accuracy", 0),
                    baseline.get("precision", 0),
                    baseline.get("recall", 0),
                    baseline.get("f1", 0)
                ])
        
        # Aggregate federated metrics
        if metrics_across_runs:
            metrics_array = np.array(metrics_across_runs)
            mean_metrics = np.mean(metrics_array, axis=0)
            std_metrics = np.std(metrics_array, axis=0)
            
            aggregated["weighted_metrics"] = {
                "accuracy": {"mean": float(mean_metrics[0]), "std": float(std_metrics[0])},
                "precision": {"mean": float(mean_metrics[1]), "std": float(std_metrics[1])},
                "recall": {"mean": float(mean_metrics[2]), "std": float(std_metrics[2])},
                "f1": {"mean": float(mean_metrics[3]), "std": float(std_metrics[3])}
            }
        
        # Aggregate centralized baseline
        if centralized_metrics_runs:
            centralized_array = np.array(centralized_metrics_runs)
            centralized_mean = np.mean(centralized_array, axis=0)
            centralized_std = np.std(centralized_array, axis=0)
            
            aggregated["centralized_baseline"] = {
                "accuracy": {"mean": float(centralized_mean[0]), "std": float(centralized_std[0])},
                "precision": {"mean": float(centralized_mean[1]), "std": float(centralized_std[1])},
                "recall": {"mean": float(centralized_mean[2]), "std": float(centralized_std[2])},
                "f1": {"mean": float(centralized_mean[3]), "std": float(centralized_std[3])}
            }
        
        return aggregated
    
    def load_xgboost_results(self) -> Dict:
        """Load XGBoost results from raw result files (similar to ML model table generator)."""
        print(f"\n--- Loading XGBoost Results ---")
        
        xgboost_results = {}
        
        # Load individual client results (Solo)
        solo_files = []
        for seed in range(5):  # Default 5 seeds
            filename = f"individual_clients_xgb_02799_79665_seed{seed}.json"
            filepath = self.results_dir / "horizontal" / filename
            if filepath.exists():
                solo_files.append(str(filepath))
        
        # Load centralized results (Combined)
        combined_files = []
        for seed in range(5):  # Default 5 seeds
            filename = f"centralized_xgb_02799_79665_seed{seed}.json"
            filepath = self.results_dir / "centralized" / filename
            if filepath.exists():
                combined_files.append(str(filepath))
        
        print(f"  Found {len(solo_files)} solo files and {len(combined_files)} combined files")
        
        # Extract solo metrics
        if solo_files:
            solo1_f1_scores = []
            solo2_f1_scores = []
            
            for file_path in solo_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    individual_results = data.get("individual_results", {})
                    
                    # Extract solo1 (client_02799) and solo2 (client_79665) F1 scores
                    if "client_02799" in individual_results and "XGBoost" in individual_results["client_02799"]:
                        f1_score = individual_results["client_02799"]["XGBoost"][3]  # F1 is at index 3
                        solo1_f1_scores.append(f1_score)
                    
                    if "client_79665" in individual_results and "XGBoost" in individual_results["client_79665"]:
                        f1_score = individual_results["client_79665"]["XGBoost"][3]  # F1 is at index 3
                        solo2_f1_scores.append(f1_score)
                        
                except Exception as e:
                    print(f"    Warning: Could not load solo file {file_path}: {e}")
            
            # Calculate solo metrics
            if solo1_f1_scores:
                solo1_mean = np.mean(solo1_f1_scores)
                solo1_std = np.std(solo1_f1_scores)
                xgboost_results["solo1"] = {
                    "accuracy": {"mean": solo1_mean, "std": solo1_std},
                    "precision": {"mean": solo1_mean, "std": solo1_std},
                    "recall": {"mean": solo1_mean, "std": solo1_std},
                    "f1": {"mean": solo1_mean, "std": solo1_std}
                }
            
            if solo2_f1_scores:
                solo2_mean = np.mean(solo2_f1_scores)
                solo2_std = np.std(solo2_f1_scores)
                xgboost_results["solo2"] = {
                    "accuracy": {"mean": solo2_mean, "std": solo2_std},
                    "precision": {"mean": solo2_mean, "std": solo2_std},
                    "recall": {"mean": solo2_mean, "std": solo2_std},
                    "f1": {"mean": solo2_mean, "std": solo2_std}
                }
        
        # Extract combined metrics
        if combined_files:
            combined_f1_scores = []
            
            for file_path in combined_files:
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    
                    results = data.get("results", {})
                    
                    # Extract combined F1 score
                    if "XGBoost" in results:
                        f1_score = results["XGBoost"][3]  # F1 is at index 3
                        combined_f1_scores.append(f1_score)
                        
                except Exception as e:
                    print(f"    Warning: Could not load combined file {file_path}: {e}")
            
            # Calculate combined metrics
            if combined_f1_scores:
                combined_mean = np.mean(combined_f1_scores)
                combined_std = np.std(combined_f1_scores)
                xgboost_results["combined"] = {
                    "accuracy": {"mean": combined_mean, "std": combined_std},
                    "precision": {"mean": combined_mean, "std": combined_std},
                    "recall": {"mean": combined_mean, "std": combined_std},
                    "f1": {"mean": combined_mean, "std": combined_std}
                }
        
        # Print loaded results
        if "solo1" in xgboost_results:
            print(f"  Loaded XGBoost Solo1: F1={xgboost_results['solo1']['f1']['mean']:.4f}±{xgboost_results['solo1']['f1']['std']:.4f}")
        if "solo2" in xgboost_results:
            print(f"  Loaded XGBoost Solo2: F1={xgboost_results['solo2']['f1']['mean']:.4f}±{xgboost_results['solo2']['f1']['std']:.4f}")
        if "combined" in xgboost_results:
            print(f"  Loaded XGBoost Combined: F1={xgboost_results['combined']['f1']['mean']:.4f}±{xgboost_results['combined']['f1']['std']:.4f}")
        
        return xgboost_results
    
    def generate_horizontal_latex_table(self, results: Dict, show_std: bool = False) -> str:
        """Generate specialized horizontal FL LaTeX table."""
        print(f"\n--- Generating Horizontal FL LaTeX Table (show_std={show_std}) ---")
        
        latex = []
        latex.append("\\begin{table}[ht]")
        latex.append("  \\centering")
        latex.append("  \\caption{Horizontal Federated Learning Performance Comparison (Solo models tested on combined test set)}")
        latex.append("  \\label{tab:horizontal_fl_comparison}")
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
        
        # 1. Individual client results (Solo) - evaluated on combined test set
        if "individual" in algorithms:
            clients = algorithms["individual"].get("clients", {})
            
            for client_name, client_data in clients.items():
                # Use Neural Network results if available, otherwise best model
                if "Neural_Network" in client_data:
                    metrics = client_data["Neural_Network"]
                    db_id = client_name.split('_')[-1]
                    all_methods.append((f"Solo (DB {db_id})", metrics, "solo"))
                else:
                    # Find best model for this client
                    best_model = max(client_data.keys(), 
                                   key=lambda x: client_data[x]["f1"]["mean"])
                    metrics = client_data[best_model]
                    db_id = client_name.split('_')[-1]
                    all_methods.append((f"Solo (DB {db_id})", metrics, "solo"))
        
        # 2. Federated learning methods - middle section (excluding FedTree)
        algorithm_display = {
            "fedavg": "FedAvg",
            "fedprox": "FedProx", 
            "scaffold": "SCAFFOLD",
            "fedov": "FedOV"
        }
        
        fl_algorithms = ["fedavg", "fedprox", "scaffold", "fedov"]
        for alg_key in fl_algorithms:
            if alg_key in algorithms:
                alg_name = algorithm_display.get(alg_key, alg_key)
                weighted = algorithms[alg_key].get("weighted_metrics", {})
                if weighted:
                    all_methods.append((alg_name, weighted, "federated"))
        
        # 3. Centralized Neural Network baseline
        centralized_metrics = None
        for alg_key in algorithms:
            if alg_key != "individual":
                baseline = algorithms[alg_key].get("centralized_baseline", {})
                if baseline and "accuracy" in baseline:
                    centralized_metrics = baseline
                    break
        
        if centralized_metrics:
            all_methods.append(("Combined (NN)", centralized_metrics, "centralized"))
        
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
        # Solo rows
        if "individual" in algorithms:
            clients = algorithms["individual"].get("clients", {})
            
            for client_name, client_data in clients.items():
                # Use Neural Network results if available, otherwise best model
                if "Neural_Network" in client_data:
                    metrics = client_data["Neural_Network"]
                    db_id = client_name.split('_')[-1]
                    method_name = f"Solo (DB {db_id})"
                else:
                    # Find best model for this client
                    best_model = max(client_data.keys(), 
                                   key=lambda x: client_data[x]["f1"]["mean"])
                    metrics = client_data[best_model]
                    db_id = client_name.split('_')[-1]
                    method_name = f"Solo (DB {db_id})"
                
                acc = format_metric_with_style(metrics['accuracy'], method_name, 'accuracy')
                prec = format_metric_with_style(metrics['precision'], method_name, 'precision')
                rec = format_metric_with_style(metrics['recall'], method_name, 'recall')
                f1 = format_metric_with_style(metrics['f1'], method_name, 'f1')
                latex.append(f"  {method_name} & {acc} & {prec} & {rec} & {f1} \\\\")
            
            latex.append("  \\midrule")
        
        # Federated learning rows
        fl_rows_added = False
        for alg_key in fl_algorithms:
            if alg_key in algorithms:
                alg_name = algorithm_display.get(alg_key, alg_key)
                weighted = algorithms[alg_key].get("weighted_metrics", {})
                if weighted:
                    acc = format_metric_with_style(weighted['accuracy'], alg_name, 'accuracy')
                    prec = format_metric_with_style(weighted['precision'], alg_name, 'precision')
                    rec = format_metric_with_style(weighted['recall'], alg_name, 'recall')
                    f1 = format_metric_with_style(weighted['f1'], alg_name, 'f1')
                    
                    latex.append(f"  {alg_name} & {acc} & {prec} & {rec} & {f1} \\\\")
                    fl_rows_added = True
        
        if fl_rows_added:
            latex.append("  \\midrule")
        
        # Centralized baseline row
        if centralized_metrics:
            method_name = "Combined (NN)"
            acc = format_metric_with_style(centralized_metrics['accuracy'], method_name, 'accuracy')
            prec = format_metric_with_style(centralized_metrics['precision'], method_name, 'precision')
            rec = format_metric_with_style(centralized_metrics['recall'], method_name, 'recall')
            f1 = format_metric_with_style(centralized_metrics['f1'], method_name, 'f1')
            
            latex.append(f"  {method_name} & {acc} & {prec} & {rec} & {f1} \\\\")
        
        latex.append("  \\bottomrule")
        latex.append("  \\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def generate_fedtree_latex_table(self, results: Dict, show_std: bool = False) -> str:
        """Generate separate LaTeX table for FedTree results with XGBoost comparisons."""
        print(f"\n--- Generating FedTree LaTeX Table (show_std={show_std}) ---")
        
        latex = []
        latex.append("\\begin{table}[ht]")
        latex.append("  \\centering")
        latex.append("  \\caption{FedTree Performance on Horizontal Federated Learning}")
        latex.append("  \\label{tab:fedtree_horizontal_fl}")
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
        
        # Load XGBoost results
        xgboost_results = self.load_xgboost_results()
        
        # Collect all method data for global best/second best calculation
        all_methods = []
        
        # Add XGBoost Solo results
        if "solo1" in xgboost_results:
            all_methods.append(("Solo (XGBoost DB1)", xgboost_results["solo1"], "xgboost_solo"))
        if "solo2" in xgboost_results:
            all_methods.append(("Solo (XGBoost DB2)", xgboost_results["solo2"], "xgboost_solo"))
        
        # Add FedTree results
        if "fedtree" in algorithms:
            weighted = algorithms["fedtree"].get("weighted_metrics", {})
            if weighted:
                all_methods.append(("FedTree", weighted, "fedtree"))
        
        # Add XGBoost Combined results
        if "combined" in xgboost_results:
            all_methods.append(("Combined (XGBoost)", xgboost_results["combined"], "xgboost_combined"))
        
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
        # XGBoost Solo rows
        if "solo1" in xgboost_results:
            method_name = "Solo (XGBoost DB1)"
            metrics = xgboost_results["solo1"]
            acc = format_metric_with_style(metrics['accuracy'], method_name, 'accuracy')
            prec = format_metric_with_style(metrics['precision'], method_name, 'precision')
            rec = format_metric_with_style(metrics['recall'], method_name, 'recall')
            f1 = format_metric_with_style(metrics['f1'], method_name, 'f1')
            latex.append(f"  {method_name} & {acc} & {prec} & {rec} & {f1} \\\\")
        
        if "solo2" in xgboost_results:
            method_name = "Solo (XGBoost DB2)"
            metrics = xgboost_results["solo2"]
            acc = format_metric_with_style(metrics['accuracy'], method_name, 'accuracy')
            prec = format_metric_with_style(metrics['precision'], method_name, 'precision')
            rec = format_metric_with_style(metrics['recall'], method_name, 'recall')
            f1 = format_metric_with_style(metrics['f1'], method_name, 'f1')
            latex.append(f"  {method_name} & {acc} & {prec} & {rec} & {f1} \\\\")
        
        latex.append("  \\midrule")
        
        # FedTree results
        if "fedtree" in algorithms:
            weighted = algorithms["fedtree"].get("weighted_metrics", {})
            if weighted:
                method_name = "FedTree"
                acc = format_metric_with_style(weighted['accuracy'], method_name, 'accuracy')
                prec = format_metric_with_style(weighted['precision'], method_name, 'precision')
                rec = format_metric_with_style(weighted['recall'], method_name, 'recall')
                f1 = format_metric_with_style(weighted['f1'], method_name, 'f1')
                latex.append(f"  {method_name} & {acc} & {prec} & {rec} & {f1} \\\\")
            else:
                latex.append("  FedTree & N/A & N/A & N/A & N/A \\\\")
        else:
            latex.append("  FedTree & N/A & N/A & N/A & N/A \\\\")
        
        latex.append("  \\midrule")
        
        # XGBoost Combined results
        if "combined" in xgboost_results:
            method_name = "Combined (XGBoost)"
            metrics = xgboost_results["combined"]
            acc = format_metric_with_style(metrics['accuracy'], method_name, 'accuracy')
            prec = format_metric_with_style(metrics['precision'], method_name, 'precision')
            rec = format_metric_with_style(metrics['recall'], method_name, 'recall')
            f1 = format_metric_with_style(metrics['f1'], method_name, 'f1')
            latex.append(f"  {method_name} & {acc} & {prec} & {rec} & {f1} \\\\")
        
        latex.append("  \\bottomrule")
        latex.append("  \\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def save_results_and_table(self, aggregated_results: Dict, latex_table: str, show_std: bool = False, algorithms: List[str] = None, num_seeds: int = 5, include_individual: bool = True):
        """Save aggregated results and LaTeX table to files."""
        
        # Create hyperparameter-based filename components
        if algorithms is None:
            algorithms = ["fedavg", "fedprox", "scaffold", "fedov"]
        
        alg_str = "_".join(sorted(algorithms))
        std_str = "std" if show_std else "mean"
        individual_str = "with_individual" if include_individual else "no_individual"
        
        # Hyperparameter-based filename: horizontal_fl_fedavg_fedprox_scaffold_fedov_5seeds_mean_with_individual
        base_filename = f"horizontal_fl_{alg_str}_{num_seeds}seeds_{std_str}_{individual_str}"
        
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
        description="Generate specialized horizontal FL LaTeX tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate table with all algorithms (auto-runs missing experiments)
  python run_and_generate_horizontal_table.py
  
  # Generate table with standard deviations included
  python run_and_generate_horizontal_table.py --show-std
  
  # Generate table with specific algorithms only
  python run_and_generate_horizontal_table.py --algorithms fedavg,fedprox
  
  # Generate table without individual client results
  python run_and_generate_horizontal_table.py --no-individual
  
  # Use custom number of seeds for missing experiments
  python run_and_generate_horizontal_table.py --num-seeds 3 --show-std
        """
    )
    
    parser.add_argument("--algorithms", type=str, default="fedavg,fedprox,scaffold,fedov,fedtree",
                       help="Comma-separated FL algorithms (default: fedavg,fedprox,scaffold,fedov,fedtree)")
    parser.add_argument("--no-individual", action="store_true",
                       help="Exclude individual client results from table")
    parser.add_argument("--show-std", action="store_true",
                       help="Include standard deviation in LaTeX table (default: mean only)")
    parser.add_argument("--num-seeds", type=int, default=5,
                       help="Number of seeds for experiments (default: 5)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Parse algorithms
    algorithms = [alg.strip() for alg in args.algorithms.split(",")]
    include_individual = not args.no_individual
    
    # Initialize generator
    generator = HorizontalFLTableGenerator(args.output)
    
    print(f"\n{'='*100}")
    print(f"GENERATING HORIZONTAL FEDERATED LEARNING TABLE")
    print(f"{'='*100}")
    print(f"Algorithms: {', '.join(algorithms)}")
    print(f"Include individual clients: {include_individual}")
    print(f"Show standard deviation: {args.show_std}")
    print(f"Number of seeds: {args.num_seeds}")
    
    # Find result files
    result_files = generator.find_result_files(algorithms, include_individual)
    
    # Check for missing algorithms and insufficient seeds
    missing_algorithms = []
    insufficient_seed_algorithms = []
    
    for alg in algorithms:
        if alg not in result_files:
            missing_algorithms.append(alg)
        else:
            # Check if we have enough seed files
            current_files = result_files[alg]
            if len(current_files) < args.num_seeds:
                insufficient_seed_algorithms.append(alg)
                print(f"⚠️  {alg}: Found {len(current_files)} files but need {args.num_seeds} seeds")
    
    # Run missing algorithms
    if missing_algorithms:
        print(f"\n🔄 Missing algorithms detected: {missing_algorithms}")
        print(f"Automatically running missing experiments...")
        success = generator.run_federated_algorithms(missing_algorithms, num_seeds=args.num_seeds)
        if success:
            # Re-scan for algorithm files
            for alg in missing_algorithms:
                pattern = f"{alg.lower()}_*_seed*.json"
                pattern_path = str(generator.results_dir / "horizontal" / pattern)
                alg_files = list(glob.glob(pattern_path))
                if alg_files:
                    result_files[alg] = alg_files
                    print(f"✅ Found {len(alg_files)} files for {alg} after running experiments")
        else:
            print(f"⚠️  Some federated algorithms failed, proceeding with available results")
    
    # Run additional seeds for insufficient algorithms
    if insufficient_seed_algorithms:
        print(f"\n🔄 Insufficient seeds detected for: {insufficient_seed_algorithms}")
        print(f"Running additional seed experiments to reach {args.num_seeds} seeds...")
        success = generator.run_federated_algorithms(insufficient_seed_algorithms, num_seeds=args.num_seeds)
        if success:
            # Re-scan for algorithm files
            for alg in insufficient_seed_algorithms:
                pattern = f"{alg.lower()}_*_seed*.json"
                pattern_path = str(generator.results_dir / "horizontal" / pattern)
                alg_files = list(glob.glob(pattern_path))
                if alg_files:
                    result_files[alg] = alg_files
                    print(f"✅ Now found {len(alg_files)} files for {alg} after running additional seeds")
        else:
            print(f"⚠️  Some additional seed experiments failed, proceeding with available results")
    
    # Automatically run missing individual client experiments
    if include_individual and "individual" not in result_files:
        print(f"\n🔄 Individual client results missing, automatically running experiments...")
        success = generator.run_individual_clients(num_seeds=args.num_seeds)
        if success:
            # Re-scan for individual files
            individual_pattern = str(generator.results_dir / "horizontal" / "individual_clients_*.json")
            individual_files = list(glob.glob(individual_pattern))
            if individual_files:
                result_files["individual"] = individual_files
                print(f"✅ Found {len(individual_files)} individual client files after running experiments")
        else:
            print(f"⚠️  Individual client experiments failed, proceeding without them")
    
    if not result_files:
        print(f"❌ No result files found after attempting to run missing experiments")
        return
    
    # Aggregate results
    aggregated = generator.aggregate_results(result_files)
    
    # Separate FedTree from other algorithms
    fedtree_only = set(algorithms) == {"fedtree"}
    other_algorithms = [alg for alg in algorithms if alg != "fedtree"]
    has_fedtree = "fedtree" in algorithms
    
    # Generate main LaTeX table (excluding FedTree) only if there are other algorithms
    if other_algorithms and not fedtree_only:
        print(f"\n📊 Generating main table with algorithms: {other_algorithms}")
        latex_table = generator.generate_horizontal_latex_table(aggregated, show_std=args.show_std)
        
        # Save main table results
        results_file, latex_file = generator.save_results_and_table(aggregated, latex_table, show_std=args.show_std, algorithms=other_algorithms, num_seeds=args.num_seeds, include_individual=include_individual)
        
        print(f"\n--- Main LaTeX Table Preview ---")
        print(latex_table)
        print(f"--- End Main Table Preview ---")
    
    # Generate separate FedTree table if FedTree is included
    if has_fedtree:
        print(f"\n📊 Generating separate FedTree table")
        fedtree_latex_table = generator.generate_fedtree_latex_table(aggregated, show_std=args.show_std)
        
        # Save FedTree table separately
        std_str = "std" if args.show_std else "mean"
        fedtree_filename = f"fedtree_horizontal_fl_{args.num_seeds}seeds_{std_str}_table.tex"
        fedtree_latex_file = generator.output_dir / fedtree_filename
        with open(fedtree_latex_file, 'w') as f:
            f.write(fedtree_latex_table)
        print(f"✅ Saved FedTree LaTeX table: {fedtree_latex_file}")
        
        # Always show FedTree table preview when FedTree is included
        print(f"\n--- FedTree LaTeX Table Preview ---")
        print(fedtree_latex_table)
        print(f"--- End FedTree Preview ---")
    
    print(f"\n{'='*100}")
    print("🎉 HORIZONTAL FL TABLE GENERATION COMPLETED!")
    print(f"📁 Results saved in: {generator.output_dir}")
    if args.show_std:
        print(f"📊 LaTeX table includes mean ± std values")
    else:
        print(f"📊 LaTeX table shows mean values only")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()