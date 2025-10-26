#!/usr/bin/env python3
"""
ML Model Performance Table Generation Script

This script generates formal LaTeX tables for machine learning model performance
comparison, focusing on XGBoost, Random Forest, Logistic Regression, and SVM models.
It reports F1 scores with mean ± std from multiple runs across different scenarios:

- HFL (Horizontal Federated Learning): Solo 1, Solo 2, Combined
- VFL (Vertical Federated Learning): Solo, Combined

Usage:
    python generate_ml_model_tables.py --scenario hfl --model xgboost --runs 5
    python generate_ml_model_tables.py --scenario vfl --model random_forest --runs 5
    python generate_ml_model_tables.py --scenario both --model all --runs 5
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class MLModelTableGenerator:
    """Generator for LaTeX tables from machine learning model results."""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent.parent.parent
        self.results_dir = self.base_dir / "results"
        self.output_dir = self.base_dir / "results" / "tables"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Model mappings
        self.model_display_names = {
            "neural_network": "Neural Network",
            "xgboost": "XGBoost",
            "random_forest": "Random Forest", 
            "logistic_regression": "Logistic Regression",
            "svm": "SVM"
        }
        
        # Database name mappings
        self.db_names = {
            "02799": "DB 02799",
            "79665": "DB 79665", 
            "48804": "DB 48804",
            "00381": "DB 00381"
        }
        
        # Model abbreviation mappings
        self.model_map = {
            "xgboost": "xgb",
            "random_forest": "rf", 
            "logistic_regression": "lr",
            "neural_network": "nn"
        }
    
    def find_model_result_files(self, scenario: str, model: str, num_runs: int = 5) -> List[str]:
        """Find result files for specified scenario and model."""
        print(f"\n--- Finding Result Files for {scenario.upper()} {model.upper()} ---")
        
        result_files = []
        
        if scenario == "hfl":
            # Horizontal FL: Look for individual_clients files with model names (for all models including SVM)
            model_map = {
                "xgboost": "xgb",
                "random_forest": "rf", 
                "logistic_regression": "lr",
                "neural_network": "nn",
                "svm": "svm"
            }
            model_abbrev = model_map.get(model, model)
            pattern_base = f"individual_clients_{model_abbrev}_02799_79665_seed"
            for seed in range(num_runs):
                filename = f"{pattern_base}{seed}.json"
                filepath = self.results_dir / "horizontal" / filename
                if filepath.exists():
                    result_files.append(str(filepath))
            
            # Also look for centralized training files for combined results
            centralized_model_abbrev = self.model_map.get(model, model)
            centralized_pattern_base = f"centralized_{centralized_model_abbrev}_02799_79665_seed"
            for seed in range(num_runs):
                filename = f"{centralized_pattern_base}{seed}.json"
                filepath = self.results_dir / "centralized" / filename
                if filepath.exists():
                    result_files.append(str(filepath))
                    
        elif scenario == "vfl":
            # Vertical FL: Look for both primary_client files (solo) and splitnn files (combined) (for all models including SVM)
            model_map = {
                "xgboost": "xgb",
                "random_forest": "rf", 
                "logistic_regression": "lr",
                "neural_network": "nn",
                "svm": "svm"
            }
            model_abbrev = model_map.get(model, model)
            
            # Add primary client files for solo results
            primary_pattern_base = f"primary_client_{model_abbrev}_48804_seed"
            for seed in range(num_runs):
                filename = f"{primary_pattern_base}{seed}.json"
                filepath = self.results_dir / "primary_client" / filename
                if filepath.exists():
                    result_files.append(str(filepath))
            
            # Add centralized training files for combined results
            model_abbrev = self.model_map.get(model, model)
            centralized_pattern_base = f"centralized_{model_abbrev}_48804_00381_seed"
            for seed in range(num_runs):
                filename = f"{centralized_pattern_base}{seed}.json"
                filepath = self.results_dir / "centralized" / filename
                if filepath.exists():
                    result_files.append(str(filepath))
            
            # For neural networks, also add SplitNN files as backup
            if model == "neural_network":
                splitnn_pattern_base = f"splitnn_48804_00381_seed"
                for seed in range(num_runs):
                    filename = f"{splitnn_pattern_base}{seed}.json"
                    filepath = self.results_dir / "vertical" / filename
                    if filepath.exists():
                        result_files.append(str(filepath))
        
        print(f"  Found {len(result_files)} files for {scenario} {model}")
        return result_files
    
    def extract_model_metrics(self, result_files: List[str], model: str, scenario: str) -> Dict:
        """Extract metrics for specified model from result files."""
        print(f"\n--- Extracting {model.upper()} Metrics ---")
        
        # Map model names to expected keys in JSON files
        model_key_map = {
            "neural_network": "Neural_Network",
            "xgboost": "XGBoost",
            "random_forest": "Random_Forest", 
            "logistic_regression": "Logistic_Regression",
            "svm": "SVM"
        }
        
        model_key = model_key_map.get(model, model)
        metrics_data = {"solo": [], "combined": []}
        
        # Regular processing for all models
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                if scenario == "hfl":
                    # Check if this is a centralized training result file
                    if "results" in data and data.get("experiment_type") == "centralized":
                        # This is a centralized training result file
                        results = data.get("results", {})
                        if model_key in results:
                            # results[model_key] = [accuracy, precision, recall, f1]
                            f1_score = results[model_key][3]  # F1 is at index 3
                            metrics_data["combined"].append([f1_score])
                    else:
                        # Regular individual clients results
                        individual_results = data.get("individual_results", {})
                        combined_results = data.get("combined_results", {})
                        
                        # Solo results for each client
                        solo_metrics = []
                        for client_name, client_data in individual_results.items():
                            if model_key in client_data:
                                # client_data[model_key] = [accuracy, precision, recall, f1]
                                f1_score = client_data[model_key][3]  # F1 is at index 3
                                solo_metrics.append(f1_score)
                        
                        # Combined results (these are the same as individual due to our fix)
                        combined_metrics = []
                        for client_name, client_data in combined_results.items():
                            if model_key in client_data:
                                f1_score = client_data[model_key][3]  # F1 is at index 3
                                combined_metrics.append(f1_score)
                        
                        if solo_metrics:
                            metrics_data["solo"].append(solo_metrics)
                        # Note: We don't use combined_metrics from individual clients anymore
                        # because they're the same as solo metrics due to our test set fix
                        
                elif scenario == "vfl":
                    # Vertical FL: Extract primary client results (solo)
                    if "results" in data:
                        # Primary client results (solo performance)
                        results = data.get("results", {})
                        if model_key in results:
                            # results[model_key] = [accuracy, precision, recall, f1]
                            f1_score = results[model_key][3]  # F1 is at index 3
                            metrics_data["solo"].append([f1_score])
                    
                    # For VFL, check for centralized training results (combined)
                    if "results" in data and data.get("experiment_type") == "centralized":
                        # This is a centralized training result file
                        results = data.get("results", {})
                        if model_key in results:
                            # results[model_key] = [accuracy, precision, recall, f1]
                            f1_score = results[model_key][3]  # F1 is at index 3
                            metrics_data["combined"].append([f1_score])
                    
                    # Fallback: For neural networks, also check SplitNN centralized baseline
                    elif "centralized_results" in data and model == "neural_network":
                        # Centralized baseline represents the "combined" performance for neural networks only
                        centralized = data["centralized_results"]
                        combined_f1 = centralized.get("f1", 0)
                        metrics_data["combined"].append([combined_f1])
                    
            except Exception as e:
                print(f"    Warning: Could not load {file_path}: {e}")
        
        return metrics_data
    
    def calculate_aggregated_metrics(self, metrics_data: Dict) -> Dict:
        """Calculate mean and std for metrics across runs."""
        aggregated = {}
        
        for scenario_type, runs_data in metrics_data.items():
            if not runs_data:
                continue
                
            # runs_data is a list of runs, each run contains metrics for clients
            # For HFL: each run has [client1_f1, client2_f1]
            # For VFL: each run has [f1]
            
            if scenario_type == "solo":
                # For HFL, we have multiple clients per run
                if runs_data and len(runs_data[0]) > 1:
                    # Multiple clients (HFL scenario)
                    client1_scores = [run[0] for run in runs_data if len(run) > 0]
                    client2_scores = [run[1] for run in runs_data if len(run) > 1]
                    
                    aggregated["solo1"] = {
                        "mean": np.mean(client1_scores) if client1_scores else 0,
                        "std": np.std(client1_scores) if client1_scores else 0
                    }
                    aggregated["solo2"] = {
                        "mean": np.mean(client2_scores) if client2_scores else 0,
                        "std": np.std(client2_scores) if client2_scores else 0
                    }
                else:
                    # Single client (VFL scenario)
                    solo_scores = [run[0] for run in runs_data if len(run) > 0]
                    aggregated["solo"] = {
                        "mean": np.mean(solo_scores) if solo_scores else 0,
                        "std": np.std(solo_scores) if solo_scores else 0
                    }
            
            elif scenario_type == "combined":
                # Combined results
                if runs_data:
                    if len(runs_data[0]) > 1:
                        # Multiple clients - take average of combined performance
                        combined_scores = []
                        for run in runs_data:
                            if len(run) > 0:
                                avg_combined = np.mean(run)
                                combined_scores.append(avg_combined)
                        
                        aggregated["combined"] = {
                            "mean": np.mean(combined_scores) if combined_scores else 0,
                            "std": np.std(combined_scores) if combined_scores else 0
                        }
                    else:
                        # Single combined result
                        combined_scores = [run[0] for run in runs_data if len(run) > 0]
                        aggregated["combined"] = {
                            "mean": np.mean(combined_scores) if combined_scores else 0,
                            "std": np.std(combined_scores) if combined_scores else 0
                        }
        
        return aggregated
    
    def generate_hfl_latex_table(self, all_model_results: Dict, show_std: bool = True) -> str:
        """Generate LaTeX table for HFL scenario showing all models."""
        
        latex = []
        latex.append("\\begin{table}[ht]")
        latex.append("  \\centering")
        latex.append("  \\caption{ML models performance in feature-overlapped scenario}")
        latex.append("  \\label{tab:hfl_ml_models_comparison}")
        latex.append("  % \\small")
        latex.append("  \\begin{tabular}{lccc}")
        latex.append("  \\toprule")
        latex.append("  \\textbf{Model} & \\textbf{DB 02799} & \\textbf{DB 79665} & \\textbf{Combined} \\\\")
        latex.append("  \\midrule")
        
        # Order models for consistent display
        model_order = ["neural_network", "xgboost", "random_forest", "logistic_regression", "svm"]
        
        for model in model_order:
            if model in all_model_results:
                model_data = all_model_results[model]
                model_name = self.model_display_names.get(model, model.title())
                
                # Get F1 scores for each configuration
                db_02799_f1 = ""
                db_79665_f1 = ""
                combined_f1 = ""
                
                if "solo1" in model_data:
                    metrics = model_data["solo1"]
                    if show_std:
                        db_02799_f1 = f"{metrics['mean']:.4f} ± {metrics['std']:.4f}"
                    else:
                        db_02799_f1 = f"{metrics['mean']:.4f}"
                
                if "solo2" in model_data:
                    metrics = model_data["solo2"]
                    if show_std:
                        db_79665_f1 = f"{metrics['mean']:.4f} ± {metrics['std']:.4f}"
                    else:
                        db_79665_f1 = f"{metrics['mean']:.4f}"
                
                if "combined" in model_data:
                    metrics = model_data["combined"]
                    if show_std:
                        combined_f1 = f"\\textbf{{{metrics['mean']:.4f} ± {metrics['std']:.4f}}}"
                    else:
                        combined_f1 = f"\\textbf{{{metrics['mean']:.4f}}}"
                
                latex.append(f"  {model_name} & {db_02799_f1} & {db_79665_f1} & {combined_f1} \\\\")
        
        latex.append("  \\bottomrule")
        latex.append("  \\end{tabular}")
        latex.append("  \\end{table}")
        
        return "\n".join(latex)
    
    def generate_vfl_latex_table(self, all_model_results: Dict, show_std: bool = True) -> str:
        """Generate LaTeX table for VFL scenario showing all models."""
        
        latex = []
        latex.append("\\begin{table}[ht]")
        latex.append("  \\centering")
        latex.append("  \\caption{ML models performance in instance-overlapped scenario}")
        latex.append("  \\label{tab:vfl_ml_models_comparison}")
        latex.append("  % \\small")
        latex.append("  \\begin{tabular}{lcc}")
        latex.append("  \\toprule")
        latex.append("  \\textbf{Model} & \\textbf{DB 48804} & \\textbf{Combined} \\\\")
        latex.append("  \\midrule")
        
        # Order models for consistent display
        model_order = ["neural_network", "xgboost", "random_forest", "logistic_regression", "svm"]
        
        for model in model_order:
            if model in all_model_results:
                model_data = all_model_results[model]
                model_name = self.model_display_names.get(model, model.title())
                
                # Get F1 scores for each configuration
                solo_f1 = ""
                combined_f1 = ""
                
                if "solo" in model_data:
                    metrics = model_data["solo"]
                    if show_std:
                        solo_f1 = f"{metrics['mean']:.4f} ± {metrics['std']:.4f}"
                    else:
                        solo_f1 = f"{metrics['mean']:.4f}"
                
                if "combined" in model_data:
                    metrics = model_data["combined"]
                    if show_std:
                        combined_f1 = f"\\textbf{{{metrics['mean']:.4f} ± {metrics['std']:.4f}}}"
                    else:
                        combined_f1 = f"\\textbf{{{metrics['mean']:.4f}}}"
                else:
                    # No combined results available for this model in VFL scenario
                    combined_f1 = "N/A"
                
                latex.append(f"  {model_name} & {solo_f1} & {combined_f1} \\\\")
        
        latex.append("  \\bottomrule")
        latex.append("  \\end{tabular}")
        latex.append("  \\end{table}")
        
        return "\n".join(latex)
    
    def save_results(self, latex_table: str, aggregated_data: Dict, model: str, scenario: str, 
                    num_runs: int = 5, show_std: bool = True):
        """Save LaTeX table and aggregated results to files using hyperparameter-based naming."""
        
        # Create filename based on hyperparameters instead of timestamp
        std_suffix = "_with_std" if show_std else "_mean_only"
        base_name = f"{scenario}_{model}_{num_runs}runs{std_suffix}"
        
        # Save aggregated results
        results_file = self.output_dir / f"{base_name}_aggregated.json"
        with open(results_file, 'w') as f:
            json.dump(aggregated_data, f, indent=2)
        
        # Save LaTeX table
        latex_file = self.output_dir / f"{base_name}_table.tex"
        with open(latex_file, 'w') as f:
            f.write(latex_table)
        
        print(f"✅ Saved aggregated results: {results_file}")
        print(f"✅ Saved LaTeX table: {latex_file}")
        
        return str(results_file), str(latex_file)
    
    def run_missing_experiments(self, scenario: str, model: str, num_runs: int = 5, skip_runs: bool = False):
        """Run missing experiments for the specified model and scenario."""
        print(f"\n--- Checking for Missing Experiments ---")
        
        missing_runs = []
        
        if scenario == "hfl":
            # Check for missing individual_clients runs with model names
            model_map = {"xgboost": "xgb", "random_forest": "rf", "logistic_regression": "lr", "neural_network": "nn", "svm": "svm"}
            model_abbrev = model_map.get(model, model)
            for seed in range(num_runs):
                filename = f"individual_clients_{model_abbrev}_02799_79665_seed{seed}.json"
                filepath = self.results_dir / "horizontal" / filename
                if not filepath.exists():
                    missing_runs.append(seed)
                    
        elif scenario == "vfl":
            # Check for missing primary_client runs with model names
            model_map = {"xgboost": "xgb", "random_forest": "rf", "logistic_regression": "lr", "neural_network": "nn", "svm": "svm"}
            model_abbrev = model_map.get(model, model)
            for seed in range(num_runs):
                filename = f"primary_client_{model_abbrev}_48804_seed{seed}.json"
                filepath = self.results_dir / "primary_client" / filename
                if not filepath.exists():
                    missing_runs.append(seed)
        
        if missing_runs:
            print(f"  Missing runs for {scenario} {model}: {missing_runs}")
            
            if skip_runs:
                print(f"  Skipping automatic execution due to --skip-runs flag.")
                print(f"  You need to run the following commands manually:")
                
                if scenario == "hfl":
                    for seed in missing_runs:
                        model_args = self._get_model_args(model)
                        print(f"  python src/demo/run_individual_clients.py -m {model_args} --seed {seed}")
                elif scenario == "vfl":
                    for seed in missing_runs:
                        model_args = self._get_model_args(model)
                        print(f"  python src/demo/run_primary_client.py -m {model_args} --seed {seed}")
                        
                return False
            else:
                print(f"  Automatically running missing experiments...")
                return self._execute_missing_experiments(scenario, model, missing_runs)
        else:
            print(f"  All {num_runs} runs found for {scenario} {model}")
            return True
    
    def _execute_missing_experiments(self, scenario: str, model: str, missing_runs: list) -> bool:
        """Execute missing experiments automatically."""
        import subprocess
        import sys
        
        success_count = 0
        model_args = self._get_model_args(model)
        
        for seed in missing_runs:
            if scenario == "hfl":
                script_path = self.base_dir / "src" / "demo" / "run_individual_clients.py"
                cmd = f"python {script_path} -m {model_args} --seed {seed}"
            elif scenario == "vfl":
                script_path = self.base_dir / "src" / "demo" / "run_primary_client.py"
                cmd = f"python {script_path} -m {model_args} --seed {seed}"
            else:
                continue
                
            print(f"\n🔄 Running experiment for seed {seed}...")
            print(f"Command: {cmd}")
            
            try:
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.base_dir)
                
                if result.returncode == 0:
                    print(f"✅ Experiment for seed {seed} completed successfully")
                    success_count += 1
                else:
                    print(f"❌ Experiment for seed {seed} failed")
                    print(f"Error: {result.stderr[:500]}...")  # Show first 500 chars
                    
            except Exception as e:
                print(f"❌ Failed to run experiment for seed {seed}: {e}")
        
        if success_count == len(missing_runs):
            print(f"✅ All {success_count} missing experiments completed successfully")
            return True
        else:
            print(f"⚠️  {success_count}/{len(missing_runs)} experiments completed successfully")
            return success_count > 0
    
    def _get_model_args(self, model: str) -> str:
        """Get command line arguments for model selection."""
        if model == "all":
            return "nn,xgb,rf,lr,svm"
        
        model_arg_map = {
            "xgboost": "xgb",
            "random_forest": "rf", 
            "logistic_regression": "lr",
            "neural_network": "nn",
            "svm": "svm"
        }
        
        return model_arg_map.get(model, model)
    
    def run_centralized_training(self, scenario: str, model: str, seed: int = 0):
        """Run centralized training by calling the centralized training script."""
        print(f"\n--- Running Centralized Training for {scenario.upper()} {model.upper()} (seed {seed}) ---")
        
        import subprocess
        
        # Get the centralized training script path
        script_dir = self.base_dir / "src" / "demo"
        centralized_script = script_dir / "run_centralized_training.py"
        
        if not centralized_script.exists():
            print(f"❌ Centralized training script not found: {centralized_script}")
            return False
        
        # Run the centralized training script
        cmd = f"python {centralized_script} --scenario {scenario} --model {model} --seed {seed}"
        
        try:
            print(f"Running: {cmd}")
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=self.base_dir)
            
            if result.returncode == 0:
                print(f"✅ Centralized training for {scenario} {model} seed {seed} completed successfully")
                return True
            else:
                print(f"❌ Centralized training for {scenario} {model} seed {seed} failed")
                print(f"Error: {result.stderr[:500]}...")  # Show first 500 chars
                return False
                
        except Exception as e:
            print(f"❌ Failed to run centralized training for {scenario} {model} seed {seed}: {e}")
            return False
    
    def check_and_run_centralized_training(self, scenario: str, model: str, num_runs: int = 5, skip_runs: bool = False):
        """Check for missing centralized training results and run if needed."""
        print(f"\n--- Checking Centralized Training for {scenario.upper()} {model.upper()} ---")
        
        missing_runs = []
        centralized_dir = self.results_dir / "centralized"
        
        # Check for missing centralized training results
        model_abbrev = self.model_map.get(model, model)
        if scenario == "hfl":
            db_string = "02799_79665"
        else:  # vfl
            db_string = "48804_00381"
            
        for seed in range(num_runs):
            filename = f"centralized_{model_abbrev}_{db_string}_seed{seed}.json"
            filepath = centralized_dir / filename
            if not filepath.exists():
                missing_runs.append(seed)
        
        if missing_runs:
            print(f"  Missing centralized training runs for {scenario} {model}: {missing_runs}")
            
            if skip_runs:
                print(f"  Skipping automatic centralized training due to --skip-runs flag.")
                return False
            else:
                print(f"  Automatically running missing centralized training...")
                success_count = 0
                
                for seed in missing_runs:
                    if self.run_centralized_training(scenario, model, seed):
                        success_count += 1
                
                if success_count == len(missing_runs):
                    print(f"✅ All {success_count} missing centralized training runs completed successfully")
                    return True
                else:
                    print(f"⚠️  {success_count}/{len(missing_runs)} centralized training runs completed successfully")
                    return success_count > 0
        else:
            print(f"  All {num_runs} centralized training runs found for {scenario} {model}")
            return True

def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for ML model performance comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate XGBoost HFL table with 5 runs (auto-runs missing experiments)
  python run_and_generate_ml_model_tables.py --scenario hfl --model xgboost --runs 5
  
  # Generate Random Forest VFL table with standard deviation  
  python run_and_generate_ml_model_tables.py --scenario vfl --model random_forest --runs 5 --show-std
  
  # Generate all models for both scenarios
  python run_and_generate_ml_model_tables.py --scenario both --model all --runs 5
  
  # Skip auto-running missing experiments (just check and report)
  python run_and_generate_ml_model_tables.py --scenario hfl --model xgboost --skip-runs
        """
    )
    
    parser.add_argument("--scenario", choices=["hfl", "vfl", "both"], 
                       default="hfl", help="Scenario type (default: hfl)")
    parser.add_argument("--model", choices=["neural_network", "xgboost", "random_forest", "logistic_regression", "svm", "all"],
                       default="all", help="Model type (default: all)")
    parser.add_argument("--runs", type=int, default=5,
                       help="Number of runs to aggregate (default: 5)")
    parser.add_argument("--show-std", action="store_true",
                       help="Show standard deviation in results")
    parser.add_argument("--skip-runs", action="store_true",
                       help="Skip running missing experiments (default: auto-run missing experiments)")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = MLModelTableGenerator()
    
    # Determine scenarios and models to process
    scenarios = ["hfl", "vfl"] if args.scenario == "both" else [args.scenario]
    models = ["neural_network", "xgboost", "random_forest", "logistic_regression", "svm"] if args.model == "all" else [args.model]
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"PROCESSING {scenario.upper()} RESULTS FOR ALL MODELS")
        print(f"{'='*80}")
        
        # Collect results for all models in this scenario
        all_model_results = {}
        
        for model in models:
            print(f"\n--- Processing {model.upper()} ---")
            
            # Check for missing experiments and auto-run by default
            all_experiments_available = generator.run_missing_experiments(scenario, model, args.runs, skip_runs=args.skip_runs)
            
            if not all_experiments_available:
                if args.skip_runs:
                    print(f"❌ Missing experiments found for {model}. Run experiments manually or remove --skip-runs flag.")
                    continue
                else:
                    print(f"❌ Some experiments failed to run automatically for {model}.")
                    continue
            
            # Check for missing centralized training and auto-run by default
            centralized_available = generator.check_and_run_centralized_training(scenario, model, args.runs, skip_runs=args.skip_runs)
            
            if not centralized_available:
                if args.skip_runs:
                    print(f"❌ Missing centralized training found for {model}. Run centralized training manually or remove --skip-runs flag.")
                    # Continue anyway - centralized training is optional for display
                else:
                    print(f"❌ Some centralized training failed to run automatically for {model}.")
                    # Continue anyway - centralized training is optional for display
            
            # Find result files
            result_files = generator.find_model_result_files(scenario, model, args.runs)
            
            if not result_files:
                print(f"❌ No result files found for {scenario} {model}")
                continue
            
            # Extract metrics
            metrics_data = generator.extract_model_metrics(result_files, model, scenario)
            
            # Calculate aggregated metrics
            aggregated_metrics = generator.calculate_aggregated_metrics(metrics_data)
            
            if not aggregated_metrics:
                print(f"❌ No metrics found for {scenario} {model}")
                continue
            
            # Store results for this model
            all_model_results[model] = aggregated_metrics
            print(f"✅ Successfully processed {model}")
        
        if not all_model_results:
            print(f"❌ No model results found for {scenario}")
            continue
        
        # Generate LaTeX table for all models
        if scenario == "hfl":
            latex_table = generator.generate_hfl_latex_table(all_model_results, args.show_std)
        elif scenario == "vfl":
            latex_table = generator.generate_vfl_latex_table(all_model_results, args.show_std)
        
        # Save results for both scenarios (combined table)
        results_file, latex_file = generator.save_results(
            latex_table, 
            {
                "scenario": scenario,
                "models": list(all_model_results.keys()),
                "runs": args.runs,
                "show_std": args.show_std,
                "metrics": all_model_results
            },
            "all_models", 
            scenario,
            num_runs=args.runs,
            show_std=args.show_std
        )
        
        print(f"\n--- LaTeX Table Preview ({scenario} all models) ---")
        print(latex_table)
        print(f"--- End Preview ---")
    
    print(f"\n{'='*80}")
    print("🎉 ML MODEL TABLE GENERATION COMPLETED!")
    print(f"📁 Results saved in: {generator.output_dir}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()