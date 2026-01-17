#!/usr/bin/env python3
"""
Raw vs Semantic FL Performance Comparison

This script compares federated learning performance between:
- Raw approach (run_automated_fl_validation.sh): String-based column alignment
- Semantic approach (run_semantic_auto_fl_validation.sh): BGE embedding-based column alignment

It generates bar plots comparing performance metrics for common pairs
across both approaches for algorithms that both pipelines ran.
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-ready plots
plt.style.use('default')
sns.set_palette("husl")

# Configure matplotlib for better fonts and larger text
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
})


def load_results_from_dir(results_dir, algorithms):
    """
    Load FL results from a results directory.
    
    Args:
        results_dir (str): Path to the results directory
        algorithms (list): List of algorithm names to load (e.g., ['fedprox', 'scaffold', 'fedov'])
        
    Returns:
        dict: {algorithm: DataFrame with pair_id, similarity, accuracy, precision, recall, f1}
    """
    print(f"Loading results from: {results_dir}")
    
    data = {}
    for algo in algorithms:
        data[algo] = []
        files = glob.glob(os.path.join(results_dir, f"*_{algo}_results.json"))
        print(f"  Found {len(files)} {algo} result files")
        
        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                
                pair_id = result.get('pair_id', '')
                similarity = result.get('similarity', 0.0)
                task_type = result.get('task_type', 'unknown')
                
                if 'results' in result:
                    results = result['results']
                    
                    # Handle different result formats
                    if 'weighted_metrics' in results:
                        metrics = results['weighted_metrics']
                    else:
                        metrics = results
                    
                    row = {
                        'pair_id': pair_id,
                        'similarity': similarity,
                        'task_type': task_type,
                        'accuracy': metrics.get('accuracy', np.nan),
                        'precision': metrics.get('precision', np.nan),
                        'recall': metrics.get('recall', np.nan),
                        'f1': metrics.get('f1', np.nan)
                    }
                    data[algo].append(row)
                    
            except Exception as e:
                print(f"    Error processing {file_path}: {e}")
                continue
        
        data[algo] = pd.DataFrame(data[algo])
        print(f"  {algo.upper()}: {len(data[algo])} valid results")
    
    return data


def find_common_pairs(raw_data, semantic_data, algorithms):
    """
    Find pairs that exist in both raw and semantic results for all algorithms.
    
    Args:
        raw_data (dict): Raw approach results
        semantic_data (dict): Semantic approach results
        algorithms (list): List of algorithms to check
        
    Returns:
        set: Common pair IDs
    """
    common_pairs = None
    
    for algo in algorithms:
        if len(raw_data[algo]) > 0:
            raw_pairs = set(raw_data[algo]['pair_id'])
            if common_pairs is None:
                common_pairs = raw_pairs
            else:
                common_pairs = common_pairs.intersection(raw_pairs)
        
        if len(semantic_data[algo]) > 0:
            semantic_pairs = set(semantic_data[algo]['pair_id'])
            if common_pairs is None:
                common_pairs = semantic_pairs
            else:
                common_pairs = common_pairs.intersection(semantic_pairs)
    
    return common_pairs if common_pairs else set()


def create_comparison_plots(raw_data, semantic_data, algorithms, common_pairs, output_dir):
    """
    Create bar plots comparing raw vs semantic performance for each algorithm.
    
    Args:
        raw_data (dict): Raw approach results
        semantic_data (dict): Semantic approach results
        algorithms (list): List of algorithms to compare
        common_pairs (set): Common pair IDs to use
        output_dir (str): Directory to save plots
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score'
    }
    
    # Color palette for raw vs semantic
    raw_color = '#1f77b4'      # Blue
    semantic_color = '#ff7f0e' # Orange
    
    os.makedirs(output_dir, exist_ok=True)
    
    for metric in metrics:
        print(f"\nCreating comparison plot for {metric}...")
        
        # Filter data to common pairs only
        plot_data = []
        
        for algo in algorithms:
            # Raw approach
            raw_df = raw_data[algo]
            raw_filtered = raw_df[raw_df['pair_id'].isin(common_pairs)]
            raw_values = raw_filtered[metric].dropna().values
            
            # Semantic approach
            sem_df = semantic_data[algo]
            sem_filtered = sem_df[sem_df['pair_id'].isin(common_pairs)]
            sem_values = sem_filtered[metric].dropna().values
            
            if len(raw_values) > 0:
                plot_data.append({
                    'Algorithm': algo.upper(),
                    'Approach': 'Raw',
                    'values': raw_values,
                    'mean': raw_values.mean(),
                    'std': raw_values.std()
                })
            
            if len(sem_values) > 0:
                plot_data.append({
                    'Algorithm': algo.upper(),
                    'Approach': 'Semantic',
                    'values': sem_values,
                    'mean': sem_values.mean(),
                    'std': sem_values.std()
                })
        
        if not plot_data:
            print(f"  No data for {metric}, skipping")
            continue
        
        # Create the grouped bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Group positions
        n_algorithms = len(algorithms)
        bar_width = 0.35
        x_base = np.arange(n_algorithms)
        
        raw_means = []
        raw_stds = []
        sem_means = []
        sem_stds = []
        raw_values_all = []
        sem_values_all = []
        
        for algo in algorithms:
            # Find raw data for this algorithm
            raw_entry = next((d for d in plot_data if d['Algorithm'] == algo.upper() and d['Approach'] == 'Raw'), None)
            sem_entry = next((d for d in plot_data if d['Algorithm'] == algo.upper() and d['Approach'] == 'Semantic'), None)
            
            if raw_entry:
                raw_means.append(raw_entry['mean'])
                raw_stds.append(raw_entry['std'])
                raw_values_all.append(raw_entry['values'])
            else:
                raw_means.append(0)
                raw_stds.append(0)
                raw_values_all.append(np.array([]))
            
            if sem_entry:
                sem_means.append(sem_entry['mean'])
                sem_stds.append(sem_entry['std'])
                sem_values_all.append(sem_entry['values'])
            else:
                sem_means.append(0)
                sem_stds.append(0)
                sem_values_all.append(np.array([]))
        
        # Plot bars
        bars_raw = ax.bar(x_base - bar_width/2, raw_means, bar_width, 
                         yerr=raw_stds, label='Raw (String Match)',
                         color=raw_color, capsize=5, alpha=0.8)
        bars_sem = ax.bar(x_base + bar_width/2, sem_means, bar_width,
                         yerr=sem_stds, label='Semantic (BGE Embedding)',
                         color=semantic_color, capsize=5, alpha=0.8)
        
        # Add scatter points for individual pairs
        for i, (raw_vals, sem_vals) in enumerate(zip(raw_values_all, sem_values_all)):
            if len(raw_vals) > 0:
                jitter = np.random.normal(0, 0.03, len(raw_vals))
                ax.scatter(np.full(len(raw_vals), x_base[i] - bar_width/2) + jitter, 
                          raw_vals, color=raw_color, alpha=0.15, s=10, 
                          edgecolors='black', linewidth=0.2)
            if len(sem_vals) > 0:
                jitter = np.random.normal(0, 0.03, len(sem_vals))
                ax.scatter(np.full(len(sem_vals), x_base[i] + bar_width/2) + jitter,
                          sem_vals, color=semantic_color, alpha=0.15, s=10,
                          edgecolors='black', linewidth=0.2)
        
        # Add value labels on bars
        for i, (raw_m, raw_s, sem_m, sem_s) in enumerate(zip(raw_means, raw_stds, sem_means, sem_stds)):
            if raw_m > 0:
                ax.text(x_base[i] - bar_width/2, raw_m + raw_s + 0.02, 
                       f'{raw_m:.3f}', ha='center', va='bottom', fontsize=9,
                       fontweight='bold')
            if sem_m > 0:
                ax.text(x_base[i] + bar_width/2, sem_m + sem_s + 0.02,
                       f'{sem_m:.3f}', ha='center', va='bottom', fontsize=9,
                       fontweight='bold')
        
        # Customize plot
        ax.set_xlabel('FL Algorithm', fontweight='bold', fontsize=14)
        ax.set_ylabel(metric_labels[metric], fontweight='bold', fontsize=14)
        ax.set_title(f'{metric_labels[metric]}: Raw vs Semantic Approach (n={len(common_pairs)} pairs)',
                    fontweight='bold', fontsize=16, pad=20)
        ax.set_xticks(x_base)
        ax.set_xticklabels([algo.upper() for algo in algorithms], fontweight='bold')
        ax.legend(loc='lower right', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_axisbelow(True)
        
        # Set y-axis limits
        all_values = np.concatenate([v for v in raw_values_all + sem_values_all if len(v) > 0])
        if len(all_values) > 0:
            y_min = max(0, min(all_values) - 0.1)
            y_max = min(1.0, max(all_values) + 0.15)
            ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        
        # Save plots
        filepath = os.path.join(output_dir, f'raw_vs_semantic_{metric}.png')
        plt.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {filepath}")
        
        pdf_path = os.path.join(output_dir, f'raw_vs_semantic_{metric}.pdf')
        plt.savefig(pdf_path, dpi=600, bbox_inches='tight', facecolor='white')
        
        plt.close()


def create_delta_plot(raw_data, semantic_data, algorithms, common_pairs, output_dir):
    """
    Create a delta plot showing (Semantic - Raw) difference for each algorithm.
    """
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1-Score'
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for metric in metrics:
        print(f"\nCreating delta plot for {metric}...")
        
        delta_data = {}
        
        for algo in algorithms:
            raw_df = raw_data[algo]
            sem_df = semantic_data[algo]
            
            # Calculate deltas for common pairs
            deltas = []
            for pair_id in common_pairs:
                raw_row = raw_df[raw_df['pair_id'] == pair_id]
                sem_row = sem_df[sem_df['pair_id'] == pair_id]
                
                if len(raw_row) > 0 and len(sem_row) > 0:
                    raw_val = raw_row.iloc[0][metric]
                    sem_val = sem_row.iloc[0][metric]
                    
                    if not (pd.isna(raw_val) or pd.isna(sem_val)):
                        deltas.append(sem_val - raw_val)
            
            if deltas:
                delta_data[algo] = {
                    'values': np.array(deltas),
                    'mean': np.mean(deltas),
                    'std': np.std(deltas)
                }
        
        if not delta_data:
            print(f"  No delta data for {metric}, skipping")
            continue
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Color palette
        colors = ['#66C2A5', '#FC8D62', '#8DA0CB', '#E78AC3', '#A6D854']
        
        x_positions = np.arange(len(algorithms)) * 0.8
        
        # Zero baseline
        ax.axhline(y=0, color='black', linestyle=':', linewidth=2, alpha=0.7)
        
        for i, algo in enumerate(algorithms):
            if algo in delta_data:
                values = delta_data[algo]['values']
                mean_val = delta_data[algo]['mean']
                std_val = delta_data[algo]['std']
                color = colors[i % len(colors)]
                
                # Scatter points
                if len(values) > 0:
                    jitter = np.random.normal(0, 0.05, len(values))
                    ax.scatter(np.full(len(values), x_positions[i]) + jitter, values,
                              color=color, alpha=0.15, s=20,
                              edgecolors='black', linewidth=0.3)
                
                # Mean line
                line_width = 0.15
                ax.plot([x_positions[i] - line_width, x_positions[i] + line_width],
                       [mean_val, mean_val], color=color, linewidth=4, alpha=0.9)
                
                # Std span
                if std_val > 0:
                    ax.fill_between([x_positions[i] - line_width, x_positions[i] + line_width],
                                   [mean_val - std_val, mean_val - std_val],
                                   [mean_val + std_val, mean_val + std_val],
                                   color=color, alpha=0.15)
                
                # Annotation
                improved = np.sum(values > 0)
                total = len(values)
                ax.text(x_positions[i], mean_val + std_val + 0.02,
                       f'{mean_val:+.3f}\n({improved}/{total})',
                       ha='center', va='bottom', fontweight='bold', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Customize
        ax.set_ylabel(f'Δ {metric_labels[metric]} (Semantic - Raw)', fontweight='bold', fontsize=14)
        ax.set_title(f'{metric_labels[metric]} Difference: Semantic vs Raw (n={len(common_pairs)} pairs)',
                    fontweight='bold', fontsize=16, pad=20)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([algo.upper() for algo in algorithms], fontweight='bold')
        ax.set_xlim(-0.5, max(x_positions) + 0.5)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save
        filepath = os.path.join(output_dir, f'semantic_vs_raw_delta_{metric}.png')
        plt.savefig(filepath, dpi=600, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {filepath}")
        
        pdf_path = os.path.join(output_dir, f'semantic_vs_raw_delta_{metric}.pdf')
        plt.savefig(pdf_path, dpi=600, bbox_inches='tight', facecolor='white')
        
        plt.close()


def print_summary_statistics(raw_data, semantic_data, algorithms, common_pairs):
    """Print summary statistics comparing raw vs semantic approaches."""
    print("\n" + "="*80)
    print("RAW vs SEMANTIC APPROACH COMPARISON SUMMARY")
    print("="*80)
    print(f"Common pairs analyzed: {len(common_pairs)}")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    for metric in metrics:
        print(f"\n{metric.upper()}:")
        print("-" * 60)
        print(f"{'Algorithm':<12} {'Raw Mean':<12} {'Semantic Mean':<15} {'Delta':<10} {'Improved':<10}")
        print("-" * 60)
        
        for algo in algorithms:
            raw_df = raw_data[algo]
            sem_df = semantic_data[algo]
            
            raw_filtered = raw_df[raw_df['pair_id'].isin(common_pairs)]
            sem_filtered = sem_df[sem_df['pair_id'].isin(common_pairs)]
            
            raw_mean = raw_filtered[metric].dropna().mean()
            sem_mean = sem_filtered[metric].dropna().mean()
            delta = sem_mean - raw_mean if not (pd.isna(raw_mean) or pd.isna(sem_mean)) else 0
            
            # Calculate improvement count
            improved = 0
            total = 0
            for pair_id in common_pairs:
                raw_row = raw_df[raw_df['pair_id'] == pair_id]
                sem_row = sem_df[sem_df['pair_id'] == pair_id]
                if len(raw_row) > 0 and len(sem_row) > 0:
                    rv = raw_row.iloc[0][metric]
                    sv = sem_row.iloc[0][metric]
                    if not (pd.isna(rv) or pd.isna(sv)):
                        total += 1
                        if sv > rv:
                            improved += 1
            
            improved_pct = f"{improved}/{total}" if total > 0 else "N/A"
            
            print(f"{algo.upper():<12} {raw_mean:.4f}       {sem_mean:.4f}          {delta:+.4f}     {improved_pct}")


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Raw vs Semantic FL Approaches')
    parser.add_argument('--raw-results-dir', default='out/autorun/results',
                       help='Path to raw approach results (default: out/autorun/results)')
    parser.add_argument('--semantic-results-dir', default='out/autorun_semantic/results',
                       help='Path to semantic approach results (default: out/autorun_semantic/results)')
    parser.add_argument('--output-dir', default='fig',
                       help='Directory to save plots (default: fig)')
    args = parser.parse_args()
    
    print("="*60)
    print("Raw vs Semantic FL Performance Comparison")
    print("="*60)
    
    raw_results_dir = args.raw_results_dir
    semantic_results_dir = args.semantic_results_dir
    output_dir = args.output_dir
    
    # The semantic pipeline ran only these algorithms
    algorithms = ['fedprox', 'scaffold', 'fedov']
    
    # Load data
    print("\nLoading raw approach results...")
    raw_data = load_results_from_dir(raw_results_dir, algorithms)
    
    print("\nLoading semantic approach results...")
    semantic_data = load_results_from_dir(semantic_results_dir, algorithms)
    
    # Find common pairs
    common_pairs = find_common_pairs(raw_data, semantic_data, algorithms)
    print(f"\nFound {len(common_pairs)} common pairs across both approaches")
    
    if len(common_pairs) == 0:
        print("ERROR: No common pairs found between raw and semantic results.")
        return
    
    # Print summary statistics
    print_summary_statistics(raw_data, semantic_data, algorithms, common_pairs)
    
    # Create comparison plots
    print("\nCreating comparison plots...")
    create_comparison_plots(raw_data, semantic_data, algorithms, common_pairs, output_dir)
    
    # Create delta plots
    print("\nCreating delta plots...")
    create_delta_plot(raw_data, semantic_data, algorithms, common_pairs, output_dir)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print("="*60)
    print(f"Output saved to: {output_dir}/")
    print("Generated files:")
    print("  - raw_vs_semantic_*.png (comparison bar plots)")
    print("  - raw_vs_semantic_*.pdf (publication-ready)")
    print("  - semantic_vs_raw_delta_*.png (delta distribution plots)")
    print("  - semantic_vs_raw_delta_*.pdf (publication-ready)")


if __name__ == "__main__":
    main()
