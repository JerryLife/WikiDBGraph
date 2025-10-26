#!/usr/bin/env python3
"""
LaTeX Table Generation Script for Federated Learning Results

This script generates publication-ready LaTeX tables from experimental results.
It reads JSON result files and creates formatted tables for different experiment types.

Usage:
    python generate_latex_tables.py --exp horizontal --results results/horizontal/
    python generate_latex_tables.py --exp vertical --results results/vertical/
    python generate_latex_tables.py --exp both --algorithms fedavg,fedprox,scaffold
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

class LaTeXTableGenerator:
    """Generator for LaTeX tables from federated learning results."""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent.parent
        self.results_dir = self.base_dir / "results"
        self.output_dir = self.base_dir / "results" / "tables"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def find_result_files(self, experiment_type: str, algorithms: List[str] = None) -> Dict[str, List[str]]:
        """Find result files for specified experiment type and algorithms."""
        print(f"\n--- Finding Result Files for {experiment_type.upper()} ---")
        
        if experiment_type == "horizontal":
            return self._find_horizontal_files(algorithms)
        elif experiment_type == "vertical":
            return self._find_vertical_files()
        elif experiment_type == "primary":
            return self._find_primary_files()
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    def _find_horizontal_files(self, algorithms: List[str] = None) -> Dict[str, List[str]]:
        """Find horizontal FL result files."""
        if algorithms is None:
            algorithms = ["fedavg", "fedprox", "scaffold", "fedov", "individual"]
        
        result_files = {}
        
        for algorithm in algorithms:
            # Pattern maps for both demo and autorun naming conventions
            pattern_map = {
                "fedavg": ["fedavg_*_seed*.json", "*_fedavg_results.json"],
                "fedprox": ["fedprox_*_seed*.json", "*_fedprox_results.json"], 
                "scaffold": ["scaffold_*_seed*.json", "*_scaffold_results.json"],
                "fedov": ["fedov_*_seed*.json", "*_fedov_results.json"],
                "individual": ["individual_clients_nn_*.json", "*_solo_results.json"]  # Support both naming conventions
            }
            
            if algorithm in pattern_map:
                all_files = []
                # Try all patterns for this algorithm
                for pattern_suffix in pattern_map[algorithm]:
                    pattern = str(self.results_dir / "horizontal" / pattern_suffix)
                    files = list(glob.glob(pattern))
                    all_files.extend(files)
                
                if all_files:
                    result_files[algorithm] = all_files
                    print(f"  Found {len(all_files)} files for {algorithm}")
                else:
                    print(f"  No files found for {algorithm}")
        
        return result_files
    
    def _find_vertical_files(self) -> Dict[str, List[str]]:
        """Find vertical FL result files."""
        pattern = str(self.results_dir / "vertical" / "splitnn_*.json")
        files = list(glob.glob(pattern))
        
        result_files = {}
        if files:
            result_files["splitnn"] = files
            print(f"  Found {len(files)} files for splitnn")
        else:
            print(f"  No files found for splitnn")
        
        return result_files
    
    def _find_primary_files(self) -> Dict[str, List[str]]:
        """Find primary client result files."""
        pattern = str(self.results_dir / "primary_client" / "primary_client_nn_*.json")  # Updated to match new naming convention
        files = list(glob.glob(pattern))
        
        result_files = {}
        if files:
            result_files["primary"] = files
            print(f"  Found {len(files)} files for primary client")
        else:
            print(f"  No files found for primary client")
        
        return result_files
    
    def aggregate_results(self, result_files: Dict[str, List[str]], experiment_type: str) -> Dict:
        """Aggregate results from multiple runs with mean and std."""
        print(f"\n--- Aggregating Results ---")
        
        aggregated = {
            "experiment_type": experiment_type,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
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
                aggregated["algorithms"][algorithm] = self._aggregate_algorithm_results(
                    all_results, algorithm, experiment_type
                )
        
        return aggregated
    
    def _aggregate_algorithm_results(self, all_results: List[Dict], algorithm: str, experiment_type: str) -> Dict:
        """Aggregate results for a specific algorithm."""
        
        if experiment_type == "horizontal":
            if algorithm == "individual":
                return self._aggregate_individual_results(all_results)
            else:
                return self._aggregate_federated_results(all_results, algorithm)
        elif experiment_type == "vertical":
            return self._aggregate_vertical_results(all_results)
        elif experiment_type == "primary":
            return self._aggregate_primary_results(all_results)
        
        return {}
    
    def _aggregate_individual_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate individual client results."""
        aggregated = {"clients": {}}
        
        # Get all client names from first result
        if all_results:
            first_result = all_results[0]
            clients = list(first_result.get("individual_results", {}).keys())
            
            for client in clients:
                aggregated["clients"][client] = {}
                
                # Get all models for this client
                models = list(first_result["individual_results"][client].keys())
                
                for model in models:
                    # Collect metrics across all runs
                    metrics_across_runs = []
                    for result in all_results:
                        if client in result.get("individual_results", {}) and model in result["individual_results"][client]:
                            metrics = result["individual_results"][client][model]
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
        aggregated = {"weighted_metrics": {}}
        
        # Collect weighted metrics across runs
        metrics_across_runs = []
        for result in all_results:
            if "weighted_metrics" in result:
                metrics = result["weighted_metrics"]
                metrics_across_runs.append([
                    metrics.get("accuracy", 0),
                    metrics.get("precision", 0),
                    metrics.get("recall", 0),
                    metrics.get("f1", 0)
                ])
        
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
        
        return aggregated
    
    def _aggregate_vertical_results(self, all_results: List[Dict]) -> Dict:
        """Aggregate vertical FL results."""
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
    
    def generate_latex_table(self, aggregated_results: Dict, experiment_type: str) -> str:
        """Generate LaTeX table from aggregated results."""
        print(f"\n--- Generating LaTeX Table for {experiment_type.upper()} ---")
        
        if experiment_type == "horizontal":
            return self._generate_horizontal_latex_table(aggregated_results)
        elif experiment_type == "vertical":
            return self._generate_vertical_latex_table(aggregated_results)
        elif experiment_type == "primary":
            return self._generate_primary_latex_table(aggregated_results)
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    def _generate_horizontal_latex_table(self, results: Dict) -> str:
        """Generate LaTeX table for horizontal FL results."""
        latex = []
        latex.append("\\begin{table}[ht]")
        latex.append("  \\centering")
        latex.append("  \\caption{Federated Learning Algorithms Performance Comparison}")
        latex.append("  \\label{tab:horizontal_fl_comparison}")
        latex.append("  \\begin{tabular}{lcccc}")
        latex.append("  \\toprule")
        latex.append("  \\textbf{Algorithm} & \\textbf{Acc.} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} \\\\")
        latex.append("  \\midrule")
        
        algorithms = results.get("algorithms", {})
        algorithm_display = {
            "fedavg": "FedAvg",
            "fedprox": "FedProx", 
            "scaffold": "SCAFFOLD",
            "fedov": "FedOV",
            "individual": "Individual Clients"
        }
        
        # Sort algorithms for consistent ordering
        best_f1 = 0
        best_algorithm = None
        
        for alg_key in ["fedavg", "fedprox", "scaffold", "fedov", "individual"]:
            if alg_key in algorithms:
                alg_data = algorithms[alg_key]
                alg_name = algorithm_display.get(alg_key, alg_key)
                
                if alg_key == "individual":
                    # For individual clients, show best performing client
                    clients = alg_data.get("clients", {})
                    if clients:
                        # Find client with best average F1 across models
                        best_client_f1 = 0
                        best_client_metrics = None
                        
                        for client_name, client_data in clients.items():
                            client_f1_scores = []
                            for model_data in client_data.values():
                                client_f1_scores.append(model_data["f1"]["mean"])
                            if client_f1_scores:
                                avg_f1 = np.mean(client_f1_scores)
                                if avg_f1 > best_client_f1:
                                    best_client_f1 = avg_f1
                                    # Use best model for this client
                                    best_model = max(client_data.keys(), 
                                                   key=lambda x: client_data[x]["f1"]["mean"])
                                    best_client_metrics = client_data[best_model]
                        
                        if best_client_metrics:
                            metrics = best_client_metrics
                            acc = f"{metrics['accuracy']['mean']:.4f}"
                            prec = f"{metrics['precision']['mean']:.4f}"
                            rec = f"{metrics['recall']['mean']:.4f}"
                            f1 = f"{metrics['f1']['mean']:.4f}"
                            
                            if metrics['f1']['mean'] > best_f1:
                                best_f1 = metrics['f1']['mean']
                                best_algorithm = alg_key
                else:
                    # For federated algorithms, use weighted metrics
                    weighted = alg_data.get("weighted_metrics", {})
                    if weighted:
                        acc = f"{weighted['accuracy']['mean']:.4f}"
                        prec = f"{weighted['precision']['mean']:.4f}"
                        rec = f"{weighted['recall']['mean']:.4f}"
                        f1 = f"{weighted['f1']['mean']:.4f}"
                        
                        if weighted['f1']['mean'] > best_f1:
                            best_f1 = weighted['f1']['mean']
                            best_algorithm = alg_key
                    else:
                        continue
                
                # Bold the best performing algorithm
                if alg_key == best_algorithm:
                    line = f"  {alg_name} & \\textbf{{{acc}}} & \\textbf{{{prec}}} & \\textbf{{{rec}}} & \\textbf{{{f1}}} \\\\"
                else:
                    line = f"  {alg_name} & {acc} & {prec} & {rec} & {f1} \\\\"
                
                latex.append(line)
        
        latex.append("  \\bottomrule")
        latex.append("  \\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def _generate_vertical_latex_table(self, results: Dict) -> str:
        """Generate LaTeX table for vertical FL results."""
        latex = []
        latex.append("\\begin{table}[ht]")
        latex.append("  \\centering")
        latex.append("  \\caption{SplitNN performance in instance-overlapped scenario}")
        latex.append("  \\label{tab:vertical_fl_results}")
        latex.append("  \\begin{tabular}{lcccc}")
        latex.append("  \\toprule")
        latex.append("  \\textbf{Data Configuration} & \\textbf{Acc.} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} \\\\")
        latex.append("  \\midrule")
        
        algorithms = results.get("algorithms", {})
        
        # SplitNN results
        if "splitnn" in algorithms:
            splitnn_data = algorithms["splitnn"]
            if "splitnn" in splitnn_data:
                metrics = splitnn_data["splitnn"]
                acc = f"{metrics['accuracy']['mean']:.4f}"
                prec = f"{metrics['precision']['mean']:.4f}"
                rec = f"{metrics['recall']['mean']:.4f}"
                f1 = f"{metrics['f1']['mean']:.4f}"
                
                latex.append(f"  DB 48804 + DB 00381 (SplitNN) & {acc} & {prec} & {rec} & {f1} \\\\")
            
            # Centralized baseline
            if "centralized" in splitnn_data:
                metrics = splitnn_data["centralized"]
                acc = f"{metrics['accuracy']['mean']:.4f}"
                prec = f"{metrics['precision']['mean']:.4f}"
                rec = f"{metrics['recall']['mean']:.4f}"
                f1 = f"{metrics['f1']['mean']:.4f}"
                
                latex.append(f"  Combined features & \\textbf{{{acc}}} & \\textbf{{{prec}}} & \\textbf{{{rec}}} & \\textbf{{{f1}}} \\\\")
        
        latex.append("  \\bottomrule")
        latex.append("  \\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def _generate_primary_latex_table(self, results: Dict) -> str:
        """Generate LaTeX table for primary client results."""
        latex = []
        latex.append("\\begin{table}[ht]")
        latex.append("  \\centering")
        latex.append("  \\caption{Primary client (DB 48804) performance}")
        latex.append("  \\label{tab:primary_client_results}")
        latex.append("  \\begin{tabular}{lcccc}")
        latex.append("  \\toprule")
        latex.append("  \\textbf{Model} & \\textbf{Acc.} & \\textbf{Prec.} & \\textbf{Rec.} & \\textbf{F1} \\\\")
        latex.append("  \\midrule")
        
        algorithms = results.get("algorithms", {})
        
        if "primary" in algorithms:
            models_data = algorithms["primary"].get("models", {})
            
            models_display = {
                "Neural_Network": "Neural Network",
                "XGBoost": "XGBoost",
                "Random_Forest": "Random Forest",
                "Logistic_Regression": "Logistic Regression"
            }
            
            for model_key, model_name in models_display.items():
                if model_key in models_data:
                    metrics = models_data[model_key]
                    
                    acc = f"{metrics['accuracy']['mean']:.4f}"
                    prec = f"{metrics['precision']['mean']:.4f}"
                    rec = f"{metrics['recall']['mean']:.4f}"
                    f1 = f"{metrics['f1']['mean']:.4f}"
                    
                    latex.append(f"  {model_name} & {acc} & {prec} & {rec} & {f1} \\\\")
        
        latex.append("  \\bottomrule")
        latex.append("  \\end{tabular}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def save_results_and_table(self, aggregated_results: Dict, latex_table: str, experiment_type: str):
        """Save aggregated results and LaTeX table to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save aggregated results
        results_file = self.output_dir / f"{experiment_type}_aggregated_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(aggregated_results, f, indent=2)
        
        # Save LaTeX table
        latex_file = self.output_dir / f"{experiment_type}_table_{timestamp}.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        print(f"✅ Saved aggregated results: {results_file}")
        print(f"✅ Saved LaTeX table: {latex_file}")
        
        return str(results_file), str(latex_file)

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from federated learning results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate horizontal FL table with all algorithms
  python generate_latex_tables.py --exp horizontal
  
  # Generate vertical FL table
  python generate_latex_tables.py --exp vertical
  
  # Generate tables for specific algorithms
  python generate_latex_tables.py --exp horizontal --algorithms fedavg,fedprox,scaffold
  
  # Generate all tables
  python generate_latex_tables.py --exp both
        """
    )
    
    parser.add_argument("--exp", choices=["horizontal", "vertical", "primary", "both"], 
                       default="horizontal", help="Experiment type to process")
    parser.add_argument("--algorithms", type=str, default=None,
                       help="Comma-separated algorithms (fedavg,fedprox,scaffold,fedov,individual)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Parse algorithms
    algorithms = None
    if args.algorithms:
        algorithms = [alg.strip() for alg in args.algorithms.split(",")]
    
    # Initialize generator
    generator = LaTeXTableGenerator(args.output)
    
    experiment_types = ["horizontal", "vertical", "primary"] if args.exp == "both" else [args.exp]
    
    for exp_type in experiment_types:
        print(f"\n{'='*100}")
        print(f"PROCESSING {exp_type.upper()} EXPERIMENT RESULTS")
        print(f"{'='*100}")
        
        # Find result files
        result_files = generator.find_result_files(exp_type, algorithms)
        
        if not result_files:
            print(f"❌ No result files found for {exp_type}")
            continue
        
        # Aggregate results
        aggregated = generator.aggregate_results(result_files, exp_type)
        
        # Generate LaTeX table
        latex_table = generator.generate_latex_table(aggregated, exp_type)
        
        # Save results
        results_file, latex_file = generator.save_results_and_table(aggregated, latex_table, exp_type)
        
        print(f"\n--- LaTeX Table Preview ({exp_type}) ---")
        print(latex_table)
        print(f"--- End Preview ---")
    
    print(f"\n{'='*100}")
    print("🎉 LATEX TABLE GENERATION COMPLETED!")
    print(f"📁 Results saved in: {generator.output_dir}")
    print(f"{'='*100}")

if __name__ == "__main__":
    main()