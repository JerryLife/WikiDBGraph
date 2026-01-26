#!/usr/bin/env python3
"""
Dataset Size vs Performance Gain Analysis

This script analyzes the relationship between dataset size and collaborative
learning performance gain to address reviewer concerns about regression on
larger datasets.

It creates:
1. A scatter plot with trend line showing Gain vs Dataset Size
2. Quantitative statistics for the rebuttal
"""

import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configure matplotlib for publication-quality plots
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 26,
    'axes.labelsize': 22,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'legend.fontsize': 16,
    'figure.titlesize': 28,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2
})


def load_preprocessing_summary(summary_file):
    """Load preprocessing summary to get dataset sizes."""
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Build a lookup: pair_id -> data_shapes
    size_lookup = {}
    for result in summary.get('results', []):
        if 'error' not in result:
            pair_id = result['pair_id']
            data_shapes = result.get('data_shapes', {})
            # Compute total training rows (sum of both DBs)
            db1_train = data_shapes.get('db1_train', [0, 0])[0]
            db2_train = data_shapes.get('db2_train', [0, 0])[0]
            total_train_rows = db1_train + db2_train
            size_lookup[pair_id] = {
                'db1_train_rows': db1_train,
                'db2_train_rows': db2_train,
                'total_train_rows': total_train_rows,
                'similarity': result.get('similarity', 0.0)
            }
    
    return size_lookup


def load_results(results_dir, cl_algorithm='fedavg'):
    """
    Load Solo and CL results for all pairs.
    
    Args:
        results_dir: Directory containing result JSON files
        cl_algorithm: Which CL algorithm to compare against Solo
        
    Returns:
        DataFrame with pair_id, solo_f1, cl_f1, gain
    """
    results = []
    
    # Get all solo result files
    solo_files = glob.glob(os.path.join(results_dir, "*_solo_results.json"))
    print(f"Found {len(solo_files)} solo result files")
    
    for solo_file in solo_files:
        # Extract pair_id
        basename = os.path.basename(solo_file)
        pair_id = basename.replace("_solo_results.json", "")
        
        # Find corresponding CL result
        cl_file = os.path.join(results_dir, f"{pair_id}_{cl_algorithm}_results.json")
        
        if not os.path.exists(cl_file):
            continue
        
        try:
            # Load solo results
            with open(solo_file, 'r') as f:
                solo_data = json.load(f)
            
            # Load CL results
            with open(cl_file, 'r') as f:
                cl_data = json.load(f)
            
            # Extract Solo F1 score - weighted average of clients
            solo_results = solo_data.get('results', {})
            solo_f1 = None
            
            # Solo results have per-client metrics (client_0, client_1)
            if 'client_0' in solo_results:
                client_f1s = []
                for key in solo_results:
                    if key.startswith('client_'):
                        client_f1 = solo_results[key].get('f1')
                        if client_f1 is not None:
                            client_f1s.append(client_f1)
                if client_f1s:
                    solo_f1 = np.mean(client_f1s)  # Average across clients
            elif 'weighted_metrics' in solo_results:
                solo_f1 = solo_results['weighted_metrics'].get('f1')
            elif 'f1' in solo_results:
                solo_f1 = solo_results.get('f1')
            
            # Extract CL F1 score - direct metrics
            cl_results = cl_data.get('results', {})
            cl_f1 = None
            
            # CL results have metrics directly in results dict
            if 'f1' in cl_results:
                cl_f1 = cl_results.get('f1')
            elif 'weighted_metrics' in cl_results:
                cl_f1 = cl_results['weighted_metrics'].get('f1')
            elif cl_algorithm in cl_results:
                cl_f1 = cl_results[cl_algorithm].get('f1')
            
            if solo_f1 is not None and cl_f1 is not None:
                results.append({
                    'pair_id': pair_id,
                    'solo_f1': solo_f1,
                    'cl_f1': cl_f1,
                    'gain': cl_f1 - solo_f1,
                    'similarity': solo_data.get('similarity', 0.0)
                })
                
        except Exception as e:
            print(f"Error processing {pair_id}: {e}")
            continue
    
    df = pd.DataFrame(results)
    print(f"Loaded {len(df)} valid pairs with Solo and {cl_algorithm.upper()} results")
    return df


def merge_with_sizes(results_df, size_lookup):
    """Merge results with dataset size information."""
    sizes = []
    for _, row in results_df.iterrows():
        pair_id = row['pair_id']
        if pair_id in size_lookup:
            sizes.append(size_lookup[pair_id])
        else:
            sizes.append({
                'db1_train_rows': None,
                'db2_train_rows': None,
                'total_train_rows': None,
                'similarity': row.get('similarity', 0.0)
            })
    
    size_df = pd.DataFrame(sizes)
    merged = pd.concat([results_df.reset_index(drop=True), size_df], axis=1)
    
    # Remove rows without size info
    merged = merged.dropna(subset=['total_train_rows'])
    print(f"After merging with sizes: {len(merged)} pairs")
    
    return merged


def compute_statistics(df):
    """Compute key statistics for the rebuttal."""
    print("\n" + "="*70)
    print("QUANTITATIVE ANALYSIS FOR REVIEWER REBUTTAL")
    print("="*70)
    
    total_tasks = len(df)
    print(f"\nTotal tasks analyzed: {total_tasks}")
    
    # Dataset size statistics
    print(f"\n--- Dataset Size Statistics ---")
    print(f"  Min training rows: {df['total_train_rows'].min():.0f}")
    print(f"  Max training rows: {df['total_train_rows'].max():.0f}")
    print(f"  Mean training rows: {df['total_train_rows'].mean():.1f}")
    print(f"  Median training rows: {df['total_train_rows'].median():.0f}")
    
    # Overall gain statistics
    print(f"\n--- Performance Gain Statistics ---")
    print(f"  Mean gain (CL - Solo): {df['gain'].mean():.4f}")
    print(f"  Median gain: {df['gain'].median():.4f}")
    print(f"  Std of gain: {df['gain'].std():.4f}")
    
    # Positive transfer rate
    positive_transfer = (df['gain'] > 0).sum()
    neutral_transfer = (df['gain'] == 0).sum()
    negative_transfer = (df['gain'] < 0).sum()
    significant_negative = (df['gain'] < -0.01).sum()
    
    print(f"\n--- Transfer Analysis ---")
    print(f"  Positive transfer (Gain > 0): {positive_transfer} ({100*positive_transfer/total_tasks:.1f}%)")
    print(f"  Neutral transfer (Gain = 0): {neutral_transfer} ({100*neutral_transfer/total_tasks:.1f}%)")
    print(f"  Negative transfer (Gain < 0): {negative_transfer} ({100*negative_transfer/total_tasks:.1f}%)")
    print(f"  Significant negative (Gain < -0.01): {significant_negative} ({100*significant_negative/total_tasks:.1f}%)")
    
    # Size-based analysis
    print(f"\n--- Size-Based Analysis ---")
    
    # Define size bins
    small = df[df['total_train_rows'] < 500]
    medium = df[(df['total_train_rows'] >= 500) & (df['total_train_rows'] < 2000)]
    large = df[df['total_train_rows'] >= 2000]
    
    # Top quartile
    size_75th = df['total_train_rows'].quantile(0.75)
    top_quartile = df[df['total_train_rows'] >= size_75th]
    
    print(f"\n  Small datasets (< 500 rows): {len(small)} tasks")
    if len(small) > 0:
        print(f"    Mean gain: {small['gain'].mean():.4f}")
        print(f"    Positive transfer rate: {100*(small['gain'] > 0).mean():.1f}%")
    
    print(f"\n  Medium datasets (500-2000 rows): {len(medium)} tasks")
    if len(medium) > 0:
        print(f"    Mean gain: {medium['gain'].mean():.4f}")
        print(f"    Positive transfer rate: {100*(medium['gain'] > 0).mean():.1f}%")
    
    print(f"\n  Large datasets (>= 2000 rows): {len(large)} tasks")
    if len(large) > 0:
        print(f"    Mean gain: {large['gain'].mean():.4f}")
        print(f"    Positive transfer rate: {100*(large['gain'] > 0).mean():.1f}%")
    
    print(f"\n  Top quartile (>= {size_75th:.0f} rows): {len(top_quartile)} tasks")
    if len(top_quartile) > 0:
        top_positive = (top_quartile['gain'] >= 0).sum()
        print(f"    Mean gain: {top_quartile['gain'].mean():.4f}")
        print(f"    Neutral or positive: {top_positive} ({100*top_positive/len(top_quartile):.1f}%)")
    
    # Neutral or positive rate (gain >= 0)
    neutral_or_positive = (df['gain'] >= 0).sum()
    neutral_or_positive_rate = 100 * neutral_or_positive / total_tasks
    print(f"  Neutral or positive (Gain >= 0): {neutral_or_positive} ({neutral_or_positive_rate:.1f}%)")
    
    # Return key stats for annotation
    return {
        'total_tasks': total_tasks,
        'mean_gain': df['gain'].mean(),
        'positive_rate': 100 * positive_transfer / total_tasks,
        'neutral_or_positive_rate': neutral_or_positive_rate,
        'significant_negative_rate': 100 * significant_negative / total_tasks,
        'size_75th': size_75th,
        'top_quartile_positive_rate': 100 * (top_quartile['gain'] >= 0).mean() if len(top_quartile) > 0 else 0,
        'small_gain': small['gain'].mean() if len(small) > 0 else 0,
        'large_gain': large['gain'].mean() if len(large) > 0 else 0
    }


def create_scatter_plot(df, output_dir, stats, cl_algorithm='fedavg', semantic_df=None, semantic_stats=None):
    """Create the publication-quality scatter plot with trend line."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Colors
    raw_color = '#3274A1'       # Professional blue for String Match
    semantic_color = '#E1812C'  # Professional orange for DeepJoin
    
    # --- Plot Raw (String Match) results ---
    x_raw = df['total_train_rows'].values
    y_raw = df['gain'].values
    
    scatter_raw = ax.scatter(x_raw, y_raw, c=raw_color, alpha=0.05, s=35, 
                             edgecolors='darkblue', linewidth=0.3, zorder=3,
                             label=f'String Match ({stats["neutral_or_positive_rate"]:.1f}% neutral/positive)')
    
    # Add trend line for raw with variance shadow
    log_bins = np.logspace(np.log10(x_raw.min()), np.log10(x_raw.max()), 20)
    bin_centers_raw = []
    bin_means_raw = []
    bin_stds_raw = []
    
    for i in range(len(log_bins) - 1):
        mask = (x_raw >= log_bins[i]) & (x_raw < log_bins[i+1])
        if mask.sum() >= 5:
            bin_centers_raw.append(np.sqrt(log_bins[i] * log_bins[i+1]))
            bin_means_raw.append(y_raw[mask].mean())
            bin_stds_raw.append(y_raw[mask].std())
    
    if len(bin_centers_raw) > 2:
        bin_centers_raw = np.array(bin_centers_raw)
        bin_means_raw = np.array(bin_means_raw)
        bin_stds_raw = np.array(bin_stds_raw)
        
        # Variance shadow
        ax.fill_between(bin_centers_raw, bin_means_raw - bin_stds_raw, bin_means_raw + bin_stds_raw,
                        color=raw_color, alpha=0.2, zorder=1)
        # Trend line
        ax.plot(bin_centers_raw, bin_means_raw, color=raw_color, linewidth=4, 
                linestyle='-', zorder=5, alpha=1.0)
    
    # --- Plot Semantic (DeepJoin) results if provided ---
    if semantic_df is not None and len(semantic_df) > 0:
        x_sem = semantic_df['total_train_rows'].values
        y_sem = semantic_df['gain'].values
        
        scatter_sem = ax.scatter(x_sem, y_sem, c=semantic_color, alpha=0.05, s=35, 
                                 edgecolors='darkorange', linewidth=0.3, zorder=3,
                                 label=f'DeepJoin Embedding ({semantic_stats["neutral_or_positive_rate"]:.1f}% neutral/positive)')
        
        # Add trend line for semantic with variance shadow
        bin_centers_sem = []
        bin_means_sem = []
        bin_stds_sem = []
        
        for i in range(len(log_bins) - 1):
            mask = (x_sem >= log_bins[i]) & (x_sem < log_bins[i+1])
            if mask.sum() >= 5:
                bin_centers_sem.append(np.sqrt(log_bins[i] * log_bins[i+1]))
                bin_means_sem.append(y_sem[mask].mean())
                bin_stds_sem.append(y_sem[mask].std())
        
        if len(bin_centers_sem) > 2:
            bin_centers_sem = np.array(bin_centers_sem)
            bin_means_sem = np.array(bin_means_sem)
            bin_stds_sem = np.array(bin_stds_sem)
            
            # Variance shadow
            ax.fill_between(bin_centers_sem, bin_means_sem - bin_stds_sem, bin_means_sem + bin_stds_sem,
                            color=semantic_color, alpha=0.2, zorder=1)
            # Trend line
            ax.plot(bin_centers_sem, bin_means_sem, color=semantic_color, linewidth=4, 
                    linestyle='-', zorder=5, alpha=1.0)
    
    # Add horizontal reference line at y=0 (break-even)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2.5, 
               label='Break-even (No Regression)', zorder=2)
    
    # Linear scale for x-axis (no log)
    ax.set_xscale('log')
    
    # Labels and title
    ax.set_xlabel('Dataset Size (Total Training Rows)', fontweight='bold')
    ax.set_ylabel(f'Performance Gain ({cl_algorithm.upper()} F1 − Solo F1)', fontweight='bold')
    ax.set_title('Collaborative Learning Gain vs Dataset Size', fontweight='bold', pad=15)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='black', fontsize=14)
    
    # Set y-axis limits with some padding
    all_y = list(y_raw)
    if semantic_df is not None:
        all_y.extend(list(semantic_df['gain'].values))
    y_abs_max = max(abs(min(all_y)), abs(max(all_y)))
    ax.set_ylim(-y_abs_max * 1.1, y_abs_max * 1.1)
    
    # Background
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    # Save (always use cl_algorithm only, no _comparison suffix)
    os.makedirs(output_dir, exist_ok=True)
    
    png_path = os.path.join(output_dir, f'dataset_size_vs_gain_{cl_algorithm}.png')
    plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {png_path}")
    
    pdf_path = os.path.join(output_dir, f'dataset_size_vs_gain_{cl_algorithm}.pdf')
    plt.savefig(pdf_path, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"Saved: {pdf_path}")
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Dataset Size vs Performance Gain Analysis')
    parser.add_argument('--results-dir', default='out/autorun/results',
                       help='Directory containing result JSON files')
    parser.add_argument('--semantic-results-dir', default='out/autorun_semantic/results',
                       help='Directory containing semantic result JSON files')
    parser.add_argument('--preprocessing-summary', default='data/auto/preprocessing_summary.json',
                       help='Path to preprocessing summary file')
    parser.add_argument('--output-dir', default='fig',
                       help='Output directory for plots')
    parser.add_argument('--cl-algorithm', default='fedavg',
                       choices=['fedavg', 'fedprox', 'scaffold', 'fedov'],
                       help='CL algorithm to compare against Solo')
    parser.add_argument('--include-semantic', action='store_true', default=True,
                       help='Include semantic results in comparison (default: True)')
    parser.add_argument('--no-semantic', action='store_false', dest='include_semantic',
                       help='Exclude semantic results')
    args = parser.parse_args()
    
    print("="*60)
    print("DATASET SIZE vs PERFORMANCE GAIN ANALYSIS")
    print("="*60)
    print(f"Results directory: {args.results_dir}")
    print(f"Preprocessing summary: {args.preprocessing_summary}")
    print(f"CL algorithm: {args.cl_algorithm.upper()}")
    if args.include_semantic:
        print(f"Semantic results: {args.semantic_results_dir}")
    print("="*60)
    
    # Load preprocessing summary for dataset sizes
    print("\nLoading preprocessing summary...")
    size_lookup = load_preprocessing_summary(args.preprocessing_summary)
    print(f"Found size info for {len(size_lookup)} pairs")
    
    # Load raw results
    print("\nLoading raw (String Match) results...")
    raw_results_df = load_results(args.results_dir, args.cl_algorithm)
    
    if len(raw_results_df) == 0:
        print("ERROR: No raw results found!")
        return 1
    
    # Merge with sizes
    print("\nMerging raw results with dataset sizes...")
    raw_merged_df = merge_with_sizes(raw_results_df, size_lookup)
    
    if len(raw_merged_df) == 0:
        print("ERROR: No raw data after merging!")
        return 1
    
    # Load semantic results if requested
    semantic_merged_df = None
    semantic_stats = None
    
    if args.include_semantic:
        print("\nLoading semantic (DeepJoin) results...")
        # For semantic, we use the SAME solo baseline from raw, but CL from semantic dir
        # So we need a custom load that takes solo from raw dir and CL from semantic dir
        semantic_results = []
        
        # Get solo files from raw directory
        solo_files = glob.glob(os.path.join(args.results_dir, "*_solo_results.json"))
        
        for solo_file in solo_files:
            basename = os.path.basename(solo_file)
            pair_id = basename.replace("_solo_results.json", "")
            
            # Find corresponding CL result in SEMANTIC directory
            cl_file = os.path.join(args.semantic_results_dir, f"{pair_id}_{args.cl_algorithm}_results.json")
            
            if not os.path.exists(cl_file):
                continue
            
            try:
                # Load solo results from RAW directory
                with open(solo_file, 'r') as f:
                    solo_data = json.load(f)
                
                # Load CL results from SEMANTIC directory
                with open(cl_file, 'r') as f:
                    cl_data = json.load(f)
                
                # Extract Solo F1 score
                solo_results = solo_data.get('results', {})
                solo_f1 = None
                
                if 'client_0' in solo_results:
                    client_f1s = []
                    for key in solo_results:
                        if key.startswith('client_'):
                            client_f1 = solo_results[key].get('f1')
                            if client_f1 is not None:
                                client_f1s.append(client_f1)
                    if client_f1s:
                        solo_f1 = np.mean(client_f1s)
                elif 'weighted_metrics' in solo_results:
                    solo_f1 = solo_results['weighted_metrics'].get('f1')
                elif 'f1' in solo_results:
                    solo_f1 = solo_results.get('f1')
                
                # Extract CL F1 score from semantic
                cl_results = cl_data.get('results', {})
                cl_f1 = None
                
                if 'f1' in cl_results:
                    cl_f1 = cl_results.get('f1')
                elif 'weighted_metrics' in cl_results:
                    cl_f1 = cl_results['weighted_metrics'].get('f1')
                elif args.cl_algorithm in cl_results:
                    cl_f1 = cl_results[args.cl_algorithm].get('f1')
                
                if solo_f1 is not None and cl_f1 is not None:
                    semantic_results.append({
                        'pair_id': pair_id,
                        'solo_f1': solo_f1,
                        'cl_f1': cl_f1,
                        'gain': cl_f1 - solo_f1,
                        'similarity': solo_data.get('similarity', 0.0)
                    })
                    
            except Exception as e:
                continue
        
        print(f"Found {len(semantic_results)} valid pairs with Solo and semantic {args.cl_algorithm.upper()} results")
        
        if len(semantic_results) > 0:
            semantic_results_df = pd.DataFrame(semantic_results)
            
            # Merge with sizes
            print("\nMerging semantic results with dataset sizes...")
            semantic_merged_df = merge_with_sizes(semantic_results_df, size_lookup)
            
            # Filter to common pairs
            common_pairs = set(raw_merged_df['pair_id']).intersection(set(semantic_merged_df['pair_id']))
            print(f"\nFiltering to {len(common_pairs)} common pairs...")
            
            raw_merged_df = raw_merged_df[raw_merged_df['pair_id'].isin(common_pairs)]
            semantic_merged_df = semantic_merged_df[semantic_merged_df['pair_id'].isin(common_pairs)]
            
            print(f"Raw results after filtering: {len(raw_merged_df)} pairs")
            print(f"Semantic results after filtering: {len(semantic_merged_df)} pairs")
            
            # Compute statistics for semantic
            print("\n--- SEMANTIC (DeepJoin Embedding) STATISTICS ---")
            semantic_stats = compute_statistics(semantic_merged_df)
        else:
            print("WARNING: No semantic results found, proceeding with raw only")
    
    # Compute statistics for raw
    print("\n--- RAW (String Match) STATISTICS ---")
    raw_stats = compute_statistics(raw_merged_df)
    
    # Create plot
    print("\nCreating scatter plot...")
    create_scatter_plot(raw_merged_df, args.output_dir, raw_stats, args.cl_algorithm, 
                        semantic_merged_df, semantic_stats)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())
