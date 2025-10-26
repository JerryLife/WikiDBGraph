#!/usr/bin/env python3
"""
LaTeX Booktab Table Generator for Federated Learning Performance Summary

This script analyzes the results from automated federated learning experiments,
and generates a LaTeX booktab table showing improvement statistics:
1. Total sample size
2. Successfully find a HFL task - Num(x%)
3. Algorithm improvements over Solo baseline for each FL method
   - Supports: FedAvg, FedProx, SCAFFOLD, FedOV, Combined
   - By default, auto-detects and includes all available algorithms

All values (except total sample size) are formatted as Number(Percent%).
"""

import json
import os
import pandas as pd
import argparse
import warnings
warnings.filterwarnings('ignore')


def load_results_from_csv(csv_path):
    """
    Load results from the detailed_pair_comparison.csv file.
    This file already contains aggregated results per pair.
    
    Args:
        csv_path (str): Path to the detailed_pair_comparison.csv file
        
    Returns:
        pd.DataFrame: DataFrame with results for all methods per pair
    """
    print(f"Loading results from CSV: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load the CSV file
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} pairs from CSV file")
    
    return df


def load_total_pairs_from_sampler(sampler_file_path="out/autorun/sampled_pairs.json"):
    """
    Load the total number of pairs from the pair sampler output file.
    
    Args:
        sampler_file_path (str): Path to the sampled_pairs.json file
        
    Returns:
        int: Total number of pairs from pair_sampler, or None if file not found
    """
    if not os.path.exists(sampler_file_path):
        print(f"Warning: Sampler file not found: {sampler_file_path}")
        return None
    
    try:
        with open(sampler_file_path, 'r', encoding='utf-8') as f:
            sampler_data = json.load(f)
        
        total_pairs = sampler_data.get('sampling_params', {}).get('sample_size')
        if total_pairs is not None:
            print(f"Loaded total pairs from sampler: {total_pairs}")
            return total_pairs
        else:
            print("Warning: Could not find sample_size in sampler file")
            return None
            
    except Exception as e:
        print(f"Warning: Error loading sampler file: {e}")
        return None


def determine_task_type(df):
    """
    Determine task type (classification or regression) from CSV data.
    Based on available metrics in the CSV file.
    
    Args:
        df (pd.DataFrame): DataFrame with results
        
    Returns:
        str: 'classification' or 'regression' 
    """
    # Check if classification metrics exist
    if 'solo_accuracy' in df.columns:
        return 'classification'
    elif 'solo_mse' in df.columns or 'solo_r2' in df.columns:
        return 'regression'
    else:
        return 'classification'  # Default fallback


def detect_available_algorithms(df, primary_metric):
    """
    Detect which FL algorithms are available in the CSV data.
    
    Args:
        df (pd.DataFrame): DataFrame with results
        primary_metric (str): Primary metric name (e.g., 'accuracy', 'r2')
        
    Returns:
        list: List of available algorithm names
    """
    available_algorithms = []
    
    # Check for each algorithm
    algorithms_to_check = ['fedavg', 'fedprox', 'scaffold', 'fedov', 'combined']
    
    for algo in algorithms_to_check:
        metric_col = f'{algo}_{primary_metric}'
        if metric_col in df.columns:
            # Check if there's any non-null data
            if df[metric_col].notna().any():
                available_algorithms.append(algo)
    
    return available_algorithms


def analyze_improvements(df, total_pairs_from_sampler=None, algorithms=None):
    """
    Analyze improvement statistics for LaTeX table generation using CSV data.
    
    Args:
        df (pd.DataFrame): DataFrame with results from CSV file
        total_pairs_from_sampler (int): Total pairs from pair_sampler (overrides CSV count)
        algorithms (list): List of algorithms to analyze (default: auto-detect all available)
        
    Returns:
        dict: Statistics for table generation
    """
    print("\nAnalyzing improvement statistics...")
    
    # Determine task type and metrics
    task_type = determine_task_type(df)
    
    if task_type == 'classification':
        primary_metric = 'accuracy'
    elif task_type == 'regression':
        primary_metric = 'r2'  # Higher is better for R²
    else:
        primary_metric = 'accuracy'  # Default fallback
    
    print(f"Using primary metric: {primary_metric}")
    
    # Auto-detect available algorithms if not specified
    if algorithms is None:
        algorithms = detect_available_algorithms(df, primary_metric)
        print(f"Auto-detected algorithms: {algorithms}")
    else:
        print(f"Using specified algorithms: {algorithms}")
    
    # Get column names for the metrics
    solo_metric_col = f'solo_{primary_metric}'
    
    # Build list of required columns for filtering
    required_cols = [solo_metric_col]
    algorithm_metric_cols = {}
    
    for algo in algorithms:
        algo_col = f'{algo}_{primary_metric}'
        if algo_col in df.columns:
            required_cols.append(algo_col)
            algorithm_metric_cols[algo] = algo_col
    
    # Filter out rows with missing data for solo and at least one algorithm
    valid_rows = df.dropna(subset=[solo_metric_col], how='any')
    
    # Statistics to calculate
    total_sample_size = total_pairs_from_sampler if total_pairs_from_sampler is not None else len(df)  # Use sampler total if available
    successful_hfl_tasks = len(valid_rows)  # Pairs with complete data for comparison
    
    # Dictionary to store improvement counts for each algorithm
    improvements_per_algorithm = {algo: 0 for algo in algorithms}
    valid_comparisons_per_algorithm = {algo: 0 for algo in algorithms}
    
    print(f"Total pairs: {total_sample_size}")
    print(f"Pairs with complete solo data: {successful_hfl_tasks}")
    print(f"Analyzing algorithms: {algorithms}")
    
    for _, row in valid_rows.iterrows():
        solo_performance = row[solo_metric_col]
        
        # Skip if solo value is NaN
        if pd.isna(solo_performance):
            continue
        
        # Check improvements for each algorithm
        for algo in algorithms:
            if algo not in algorithm_metric_cols:
                continue
                
            algo_col = algorithm_metric_cols[algo]
            algo_performance = row[algo_col]
            
            # Skip if algorithm value is NaN
            if pd.isna(algo_performance):
                continue
            
            valid_comparisons_per_algorithm[algo] += 1
            
            # Check improvements
            # For regression metrics like MSE/RMSE/MAE, lower is better
            # For R² and classification metrics, higher is better
            if primary_metric in ['mse', 'rmse', 'mae']:
                # Lower is better - improvement means method < solo
                algo_improved = algo_performance < solo_performance
            else:
                # Higher is better - improvement means method > solo
                algo_improved = algo_performance > solo_performance
            
            # Count improvements
            if algo_improved:
                improvements_per_algorithm[algo] += 1
    
    # Print summary for each algorithm
    print("\nImprovement summary:")
    for algo in algorithms:
        valid_count = valid_comparisons_per_algorithm[algo]
        improved_count = improvements_per_algorithm[algo]
        if valid_count > 0:
            pct = (improved_count / valid_count) * 100
            print(f"  {algo}: {improved_count}/{valid_count} ({pct:.1f}%) pairs improved over solo")
    
    return {
        'total_sample_size': total_sample_size,
        'successful_hfl_tasks': successful_hfl_tasks,
        'algorithms': algorithms,
        'improvements_per_algorithm': improvements_per_algorithm,
        'valid_comparisons_per_algorithm': valid_comparisons_per_algorithm,
        'primary_metric': primary_metric,
        'task_type': task_type
    }


def generate_latex_table(stats, output_path=None):
    """
    Generate LaTeX booktab table from statistics (transposed format with algorithms as rows).
    
    Args:
        stats (dict): Statistics from analyze_improvements
        output_path (str): Optional path to save the table
    """
    print("\nGenerating LaTeX booktab table (transposed)...")
    
    # Extract statistics
    total = stats['total_sample_size']
    successful = stats['successful_hfl_tasks']
    algorithms = stats['algorithms']
    improvements_per_algorithm = stats['improvements_per_algorithm']
    valid_comparisons_per_algorithm = stats['valid_comparisons_per_algorithm']
    
    # Build algorithm display names
    algo_display_names = {
        'fedavg': 'FedAvg',
        'fedprox': 'FedProx',
        'scaffold': 'SCAFFOLD',
        'fedov': 'FedOV',
        'combined': 'Combined'
    }
    
    # Create table rows (transposed format)
    # Header row: Metric | Improved | Total | Ratio (%)
    header_row = '\\textbf{Metric} & \\textbf{Improved} & \\textbf{Total} & \\textbf{Ratio (\\%)} \\\\'
    
    # Build data rows
    table_rows = []
    
    # Row 1: Total Pairs
    table_rows.append(f'\\#Pairs & - & {total} & - \\\\')
    
    # Row 2: Valid HFL Tasks
    successful_pct = (successful / total * 100) if total > 0 else 0.0
    table_rows.append(f'Valid HFL Tasks & {successful} & {total} & {successful_pct:.1f} \\\\')
    
    # Rows 3+: Each algorithm improvement
    for algo in algorithms:
        improve_count = improvements_per_algorithm[algo]
        valid_count = valid_comparisons_per_algorithm[algo]
        display_name = algo_display_names.get(algo, algo.capitalize())
        
        if valid_count > 0:
            improvement_pct = (improve_count / valid_count * 100)
            table_rows.append(f'{display_name} $>$ Solo & {improve_count} & {valid_count} & {improvement_pct:.1f} \\\\')
        else:
            table_rows.append(f'{display_name} $>$ Solo & 0 & 0 & 0.0 \\\\')
    
    # Join all rows
    data_rows = '\n'.join(table_rows)
    
    # Generate LaTeX table (4 columns: Metric, Improved, Total, Ratio)
    latex_table = f"""\\begin{{table}}[htbp]
\\centering
\\small
\\caption{{Summary of automated data mining results}}
\\label{{tab:fl_improvement_summary}}
\\begin{{tabular}}{{lccc}}
\\toprule
{header_row}
\\midrule
{data_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}"""

    print("\nGenerated LaTeX Table:")
    print("=" * 80)
    print(latex_table)
    print("=" * 80)
    
    # Save to file if path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        print(f"Table saved to: {output_path}")
    
    # Print detailed statistics
    print("\nDetailed Statistics:")
    print(f"Task Type: {stats['task_type']}")
    print(f"Primary Metric: {stats['primary_metric']}")
    print(f"Total Sample Size: {total}")
    print(f"Successfully Find HFL Task: {successful}/{total} ({successful/total*100:.1f}%)")
    
    for algo in algorithms:
        improve_count = improvements_per_algorithm[algo]
        valid_count = valid_comparisons_per_algorithm[algo]
        display_name = algo_display_names.get(algo, algo.capitalize())
        if valid_count > 0:
            print(f"{display_name} > Solo: {improve_count}/{valid_count} ({improve_count/valid_count*100:.1f}%)")
        else:
            print(f"{display_name} > Solo: No valid comparisons")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='LaTeX Booktab Table Generator for FL Performance Summary')
    parser.add_argument('--csv-path', default='fig/detailed_pair_comparison.csv',
                       help='Path to the detailed_pair_comparison.csv file')
    parser.add_argument('--sampler-path', default='out/autorun/sampled_pairs.json',
                       help='Path to the sampled_pairs.json file')
    parser.add_argument('--output-path', default='tables/fl_improvement_summary.tex',
                       help='Path to save the LaTeX table')
    parser.add_argument('--algorithms', nargs='+', default=None,
                       choices=['fedavg', 'fedprox', 'scaffold', 'fedov', 'combined'],
                       help='Algorithms to include in table (default: auto-detect all available)')
    args = parser.parse_args()
    
    print("LaTeX Booktab Table Generator for FL Performance Summary")
    print("=" * 60)
    print("\nSupported FL Algorithms:")
    print("  - FedAvg: Federated Averaging")
    print("  - FedProx: Federated Proximal")
    print("  - SCAFFOLD: Stochastic Controlled Averaging")
    print("  - FedOV: Federated One-shot Voting")
    print("  - Combined: Combined baseline")
    print("\nUsage examples:")
    print("  # Auto-detect all available algorithms (default)")
    print("  python src/summary/print_auto_horizontal.py")
    print("\n  # Include only specific algorithms")
    print("  python src/summary/print_auto_horizontal.py --algorithms fedavg fedprox combined")
    print("=" * 60)
    
    # Check if CSV file exists
    if not os.path.exists(args.csv_path):
        print(f"ERROR: CSV file not found: {args.csv_path}")
        print("Please ensure the detailed_pair_comparison.csv has been generated.")
        return
    
    # Load results from CSV
    print(f"\nStep 1: Loading results from {args.csv_path}")
    df = load_results_from_csv(args.csv_path)
    
    # Check if we have any data
    if len(df) == 0:
        print("ERROR: No valid results found in the CSV file.")
        return
    
    print(f"\nStep 2: Found {len(df)} pairs in CSV file")
    
    # Load total pairs from sampler
    print("\nStep 3: Loading total pairs from sampler")
    total_pairs_from_sampler = load_total_pairs_from_sampler(args.sampler_path)
    
    # Analyze improvements
    print("\nStep 4: Analyzing improvement statistics")
    if args.algorithms:
        print(f"Using specified algorithms: {args.algorithms}")
    else:
        print("Auto-detecting available algorithms from CSV data")
    
    stats = analyze_improvements(df, total_pairs_from_sampler, algorithms=args.algorithms)
    
    # Generate LaTeX table
    print("\nStep 5: Generating LaTeX booktab table")
    generate_latex_table(stats, args.output_path)
    
    print(f"\n{'='*60}")
    print("TABLE GENERATION COMPLETE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()