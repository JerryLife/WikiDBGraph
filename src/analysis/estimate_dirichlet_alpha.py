#!/usr/bin/env python3
"""
Dirichlet Alpha Estimation for Federated Learning Database Pairs

This script estimates the Dirichlet concentration parameter (alpha) for each
database pair used in the automated FL validation pipeline using the Method
of Moments estimator with non-parametric bootstrapping to assess uncertainty.

The Dirichlet alpha quantifies heterogeneity:
- Low alpha (e.g., < 1): Strong heterogeneity (non-IID)
- High alpha (e.g., > 10): Near homogeneous (close to IID)

Usage:
    python src/analysis/estimate_dirichlet_alpha.py [options]

References:
    - Non-parametric Bootstrapping for Variance Estimation
    - Method of Moments for Dirichlet parameter estimation
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress numpy warnings for inf values during bootstrap statistics
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value.*')


def estimate_alpha_mom(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """
    Estimate Dirichlet alpha using Method of Moments.
    
    Given two databases with their label distributions, estimate the concentration
    parameter alpha that best explains the observed variance in class proportions.
    
    Args:
        labels_a: Array of label values (encoded as integers) for database A
        labels_b: Array of label values (encoded as integers) for database B
        
    Returns:
        Estimated alpha value (clipped to [0, 100])
    """
    # Get all unique classes across both databases
    all_classes = np.unique(np.concatenate([labels_a, labels_b]))
    n_classes = len(all_classes)
    
    if n_classes <= 1:
        return 100.0  # Single class = perfect IID (undefined alpha)
    
    # Compute class proportions for each database
    n_a = len(labels_a)
    n_b = len(labels_b)
    
    # Count frequencies (fixed vector size for all classes)
    counts_a = np.array([np.sum(labels_a == c) for c in all_classes])
    counts_b = np.array([np.sum(labels_b == c) for c in all_classes])
    
    # Stack into (2, n_classes) matrix
    counts = np.array([counts_a, counts_b], dtype=float)
    
    # Compute proportions per client (rows sum to 1)
    client_totals = counts.sum(axis=1, keepdims=True)
    # Avoid division by zero
    proportions = counts / (client_totals + 1e-12)
    
    # Global mean proportion q (across clients)
    q = proportions.mean(axis=0)  # shape: (n_classes,)
    
    # Variance of proportions across clients
    var = proportions.var(axis=0, ddof=1)  # shape: (n_classes,)
    
    # Method of Moments estimate
    # alpha = (sum(q_k * (1 - q_k)) / sum(var_k)) - 1
    total_var = var.sum()
    
    if total_var < 1e-12:
        return np.inf  # Perfect agreement = undefined alpha (mark as invalid)
    
    numerator = (q * (1 - q)).sum()
    alpha = (numerator / total_var) - 1
    
    # Clip negative to 0, but allow high values (will be filtered as invalid if >= 100)
    alpha = max(0.0, alpha)
    
    return float(alpha)


def bootstrap_alpha(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    n_iterations: int = 1000,
    random_seed: Optional[int] = None
) -> np.ndarray:
    """
    Perform non-parametric bootstrapping to estimate the distribution of alpha.
    
    Resamples rows with replacement from each database to generate pseudo-datasets,
    then estimates alpha for each iteration.
    
    Args:
        labels_a: Array of label values for database A
        labels_b: Array of label values for database B
        n_iterations: Number of bootstrap samples
        random_seed: Random seed for reproducibility
        
    Returns:
        Array of alpha estimates from bootstrap samples
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    n_a = len(labels_a)
    n_b = len(labels_b)
    
    alphas = np.zeros(n_iterations)
    
    for i in range(n_iterations):
        # Resample with replacement
        sample_a = np.random.choice(labels_a, size=n_a, replace=True)
        sample_b = np.random.choice(labels_b, size=n_b, replace=True)
        
        # Estimate alpha for this bootstrap sample
        alphas[i] = estimate_alpha_mom(sample_a, sample_b)
    
    return alphas


def load_pair_labels(pair_dir: Path, config: Dict[str, Any]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load label arrays from a preprocessed database pair.
    
    Args:
        pair_dir: Path to the pair directory (e.g., data/auto/29470_49877)
        config: Configuration dict with pair metadata
        
    Returns:
        Tuple of (labels_a, labels_b) or None if loading fails
    """
    try:
        # Extract db_ids from pair_id (to handle zero-padded IDs like 02574)
        pair_id = config['pair_id']
        db_id1_str, db_id2_str = pair_id.split('_')
        label_col = config['label_column']
        
        # Load train + test for each database (combine for full distribution)
        train1 = pd.read_csv(pair_dir / f"{db_id1_str}_train.csv")
        test1 = pd.read_csv(pair_dir / f"{db_id1_str}_test.csv")
        train2 = pd.read_csv(pair_dir / f"{db_id2_str}_train.csv")
        test2 = pd.read_csv(pair_dir / f"{db_id2_str}_test.csv")
        
        # Combine train and test for full label distribution
        labels_a = np.concatenate([train1[label_col].values, test1[label_col].values])
        labels_b = np.concatenate([train2[label_col].values, test2[label_col].values])
        
        return labels_a, labels_b
        
    except Exception as e:
        logger.warning(f"Failed to load pair {pair_dir.name}: {e}")
        return None


def process_single_pair(
    pair_info: Dict[str, Any],
    data_dir: Path,
    n_bootstrap: int,
    seed: int
) -> Optional[Dict[str, Any]]:
    """
    Process a single pair: load data, estimate alpha with bootstrapping.
    
    Args:
        pair_info: Pair metadata from preprocessing summary
        data_dir: Base data directory
        n_bootstrap: Number of bootstrap iterations
        seed: Random seed
        
    Returns:
        Dict with alpha estimates and statistics, or None if failed
    """
    pair_id = pair_info['pair_id']
    pair_dir = data_dir / pair_id
    
    if not pair_dir.exists():
        return None
    
    # Load config
    config_path = pair_dir / "config.json"
    if not config_path.exists():
        return None
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Skip non-classification tasks
    if config.get('task_type') != 'classification':
        return None
    
    # Load labels
    labels = load_pair_labels(pair_dir, config)
    if labels is None:
        return None
    
    labels_a, labels_b = labels
    
    # Estimate point alpha
    point_alpha = estimate_alpha_mom(labels_a, labels_b)
    
    # Bootstrap for uncertainty
    alpha_samples = bootstrap_alpha(labels_a, labels_b, n_bootstrap, seed)
    
    # Calculate statistics
    mean_alpha = np.mean(alpha_samples)
    median_alpha = np.median(alpha_samples)
    ci_lower = np.percentile(alpha_samples, 2.5)
    ci_upper = np.percentile(alpha_samples, 97.5)
    std_alpha = np.std(alpha_samples)
    
    return {
        'pair_id': pair_id,
        'db_id1': config['db_id1'],
        'db_id2': config['db_id2'],
        'similarity': config.get('similarity', np.nan),
        'n_classes': config.get('label_metadata', {}).get('n_classes', 0),
        'n_samples_db1': len(labels_a),
        'n_samples_db2': len(labels_b),
        'point_alpha': point_alpha,
        'mean_alpha': mean_alpha,
        'median_alpha': median_alpha,
        'std_alpha': std_alpha,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_samples': alpha_samples
    }


def estimate_alphas_parallel(
    preprocessing_summary: Path,
    data_dir: Path,
    n_bootstrap: int = 1000,
    n_workers: int = None,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Estimate Dirichlet alpha for all pairs in parallel.
    
    Args:
        preprocessing_summary: Path to preprocessing_summary.json
        data_dir: Path to data directory with preprocessed pairs
        n_bootstrap: Number of bootstrap iterations per pair
        n_workers: Number of parallel workers (None = CPU count)
        seed: Base random seed
        
    Returns:
        List of result dicts for each successfully processed pair
    """
    # Load preprocessing summary
    with open(preprocessing_summary, 'r') as f:
        summary = json.load(f)
    
    # Filter to successful classification pairs
    pairs = [
        p for p in summary['results']
        if p.get('status') == 'success' and p.get('task_type') == 'classification'
    ]
    
    logger.info(f"Found {len(pairs)} classification pairs to process")
    
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 16)
    
    results = []
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(
                process_single_pair,
                pair,
                data_dir,
                n_bootstrap,
                seed + i
            ): pair['pair_id']
            for i, pair in enumerate(pairs)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Estimating alpha"):
            pair_id = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                logger.warning(f"Error processing {pair_id}: {e}")
    
    return results


def plot_alpha_distribution(
    results: List[Dict[str, Any]],
    output_path: Path,
    title: str = None
):
    """
    Create publication-quality visualization of alpha distribution.
    
    Two panels:
    - (a) Non-IID regime histogram with KDE
    - (b) Cumulative distribution function
    
    Args:
        results: List of result dicts with alpha estimates
        output_path: Path to save the figure
        title: Optional figure title
    """
    from scipy import stats
    
    # Extract data
    alphas = np.array([r['point_alpha'] for r in results])
    n_pairs = len(alphas)
    
    # Configure matplotlib for publication quality
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 24,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 18,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'axes.linewidth': 1.0,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })
    
    # Color palette (colorblind-friendly)
    colors = {
        'hist': '#4C72B0',      # Steel blue
        'kde': '#C44E52',       # Muted red
        'mean': '#DD8452',      # Orange
        'median': '#55A868',    # Green
        'threshold': '#8172B3', # Purple
    }
    
    # Create figure with 2 panels
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # ==================== Panel (a): Non-IID Distribution ====================
    ax1 = axes[0]
    
    # Filter to alpha < 10 (non-IID regime)
    alphas_low = alphas[alphas < 10]
    pct_low = 100 * len(alphas_low) / len(alphas)
    
    if len(alphas_low) > 0:
        bins_zoom = np.linspace(0, 10, 40)
        ax1.hist(
            alphas_low, bins=bins_zoom, density=True,
            color=colors['hist'], alpha=0.7, edgecolor='white', linewidth=0.5
        )
        
        # KDE overlay
        try:
            kde_low = stats.gaussian_kde(alphas_low, bw_method='scott')
            x_kde = np.linspace(0, 10, 200)
            ax1.plot(x_kde, kde_low(x_kde), color=colors['kde'], linewidth=2)
        except:
            pass
    
    # Add threshold lines with percentages in legend (consistent style with CDF)
    pct_extreme = 100 * np.sum(alphas < 0.5) / len(alphas)
    pct_high_het = 100 * np.sum(alphas < 1.0) / len(alphas)
    pct_mod_het = 100 * np.sum(alphas < 5.0) / len(alphas)
    
    ax1.axvline(0.5, color='#E63946', linestyle='--', linewidth=2, 
                label=fr'$\alpha<0.5$: {pct_extreme:.1f}%')
    ax1.axvline(1.0, color=colors['threshold'], linestyle='--', linewidth=2, 
                label=fr'$\alpha<1.0$: {pct_high_het:.1f}%')
    ax1.axvline(5.0, color=colors['mean'], linestyle='--', linewidth=2, 
                label=fr'$\alpha<5.0$: {pct_mod_het:.1f}%')
    
    ax1.set_xlabel(r'Dirichlet $\alpha$', fontsize=16)
    ax1.set_ylabel('Density', fontsize=16)
    ax1.set_title(r'(a) Non-IID Distribution ($\alpha < 10$)', fontsize=16, fontweight='bold', pad=10)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, None)
    ax1.legend(loc='upper right', framealpha=0.95, edgecolor='gray', fontsize=14)
    
    # ==================== Panel (b): Cumulative Distribution ====================
    ax2 = axes[1]
    
    # Sort alphas for CDF (use log scale, filter out zero/negative)
    alphas_positive = alphas[alphas > 0]
    alphas_sorted = np.sort(alphas_positive)
    cdf = np.arange(1, len(alphas_sorted) + 1) / len(alphas_sorted)
    
    ax2.plot(alphas_sorted, cdf * 100, color=colors['hist'], linewidth=2.5)
    ax2.fill_between(alphas_sorted, 0, cdf * 100, alpha=0.2, color=colors['hist'])
    
    # Set log scale for x-axis
    ax2.set_xscale('log')
    
    # Add threshold markers (same colors and line styles as panel a)
    thresholds = [
        (0.5, '#E63946'),
        (1.0, colors['threshold']),
        (5.0, colors['mean']),
    ]
    
    for threshold, color in thresholds:
        pct = 100 * np.sum(alphas <= threshold) / len(alphas)
        ax2.axvline(threshold, color=color, linestyle='--', linewidth=2, alpha=0.8)
        ax2.axhline(pct, color=color, linestyle=':', linewidth=1, alpha=0.4)
        ax2.plot(threshold, pct, 'o', color=color, markersize=7, zorder=5)
    
    # # Interpretation guide (upper right)
    # interp_text = (
    #     r'$\\alpha \\to 0$: Extreme Non-IID' + '\\n'
    #     r'$\\alpha = 1$: High Heterogeneity' + '\\n'
    #     r'$\\alpha > 10$: Near-IID'
    # )
    # ax2.text(0.97, 0.97, interp_text, transform=ax2.transAxes, fontsize=14,
    #          verticalalignment='top', horizontalalignment='right',
    #          bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8',
    #                    edgecolor='gray', alpha=0.95))
    
    ax2.set_xlabel(r'Dirichlet $\alpha$ (log scale)', fontsize=16)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=16)
    ax2.set_title('(b) Cumulative Distribution', fontsize=16, fontweight='bold', pad=10)
    ax2.set_xlim(0.1, min(alphas_sorted.max(), 100))  # Stop at max alpha or 100
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5, which='both')
    
    # Tight layout
    plt.tight_layout()
    
    # Save to fig/ directory
    fig_dir = Path('fig')
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = fig_dir / 'dirichlet_alpha_distribution.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(fig_path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    
    # Also save to original output_path for JSON reference
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    logger.info(f"Saved figures to {fig_path} (PNG, PDF) and {output_path}")
    plt.close()



def print_summary_statistics(results: List[Dict[str, Any]]):
    """Print summary statistics of the alpha distribution."""
    alphas = np.array([r['point_alpha'] for r in results])
    
    print("\n" + "=" * 60)
    print("DIRICHLET ALPHA ESTIMATION SUMMARY")
    print("=" * 60)
    print(f"Total pairs analyzed: {len(results)}")
    print()
    print("Point Estimate Statistics:")
    print(f"  Mean α:   {np.mean(alphas):.3f}")
    print(f"  Median α: {np.median(alphas):.3f}")
    print(f"  Std α:    {np.std(alphas):.3f}")
    print(f"  Min α:    {np.min(alphas):.3f}")
    print(f"  Max α:    {np.max(alphas):.3f}")
    print()
    print("Percentiles:")
    for p in [5, 25, 50, 75, 95]:
        print(f"  P{p}: {np.percentile(alphas, p):.3f}")
    print()
    print("Heterogeneity Distribution:")
    print(f"  α < 0.5 (Extreme non-IID):  {100 * np.sum(alphas < 0.5) / len(alphas):.1f}%")
    print(f"  α < 1.0 (Highly heterog.):  {100 * np.sum(alphas < 1.0) / len(alphas):.1f}%")
    print(f"  α < 5.0 (Moderately het.):  {100 * np.sum(alphas < 5.0) / len(alphas):.1f}%")
    print(f"  α < 10.0 (Some heterog.):   {100 * np.sum(alphas < 10.0) / len(alphas):.1f}%")
    print(f"  α >= 10.0 (Near-IID):       {100 * np.sum(alphas >= 10.0) / len(alphas):.1f}%")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Estimate Dirichlet alpha for FL database pairs using bootstrapping'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/auto',
        help='Directory containing preprocessed pair data'
    )
    parser.add_argument(
        '--preprocessing-summary', type=str, default='data/auto/preprocessing_summary.json',
        help='Path to preprocessing summary JSON'
    )
    parser.add_argument(
        '--output-dir', type=str, default='out/analysis',
        help='Directory to save outputs'
    )
    parser.add_argument(
        '--n-bootstrap', type=int, default=1000,
        help='Number of bootstrap iterations per pair'
    )
    parser.add_argument(
        '--n-workers', type=int, default=None,
        help='Number of parallel workers (default: CPU count)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for bootstrapping'
    )
    
    args = parser.parse_args()
    
    # Paths
    data_dir = Path(args.data_dir)
    preprocessing_summary = Path(args.preprocessing_summary)
    output_dir = Path(args.output_dir)
    
    # Validate inputs
    if not preprocessing_summary.exists():
        logger.error(f"Preprocessing summary not found: {preprocessing_summary}")
        sys.exit(1)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run estimation
    logger.info("Starting Dirichlet alpha estimation...")
    all_results = estimate_alphas_parallel(
        preprocessing_summary=preprocessing_summary,
        data_dir=data_dir,
        n_bootstrap=args.n_bootstrap,
        n_workers=args.n_workers,
        seed=args.seed
    )
    
    if not all_results:
        logger.error("No pairs were successfully processed!")
        sys.exit(1)
    
    logger.info(f"Successfully processed {len(all_results)} pairs")
    
    # Filter out invalid pairs (alpha >= 100 or inf means near-IID, mark as invalid)
    valid_results = [r for r in all_results if r['point_alpha'] < 100 and np.isfinite(r['point_alpha'])]
    invalid_count = len(all_results) - len(valid_results)
    
    logger.info(f"Valid non-IID pairs: {len(valid_results)} (excluded {invalid_count} near-IID pairs with α >= 100)")
    
    if not valid_results:
        logger.error("No valid non-IID pairs found! All pairs have α >= 100 (near-IID).")
        sys.exit(1)
    
    # Print summary for valid pairs only
    print_summary_statistics(valid_results)
    print(f"\nNote: {invalid_count} pairs excluded (α >= 100, considered near-IID/invalid)")
    
    # Save results (without bootstrap samples for JSON serialization)
    results_for_json = [
        {k: v for k, v in r.items() if k != 'bootstrap_samples'}
        for r in valid_results
    ]
    
    results_path = output_dir / 'dirichlet_alpha_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'n_bootstrap': args.n_bootstrap,
                'seed': args.seed,
                'data_dir': str(data_dir),
            },
            'summary': {
                'total_pairs_processed': len(all_results),
                'valid_pairs': len(valid_results),
                'invalid_pairs_excluded': invalid_count,
                'mean_alpha': float(np.mean([r['point_alpha'] for r in valid_results])),
                'median_alpha': float(np.median([r['point_alpha'] for r in valid_results])),
            },
            'results': results_for_json
        }, f, indent=2)
    logger.info(f"Saved results to {results_path}")
    
    # Save full results with bootstrap samples as numpy
    bootstrap_path = output_dir / 'bootstrap_samples.npz'
    np.savez_compressed(
        bootstrap_path,
        pair_ids=np.array([r['pair_id'] for r in valid_results]),
        point_alphas=np.array([r['point_alpha'] for r in valid_results]),
        bootstrap_samples=np.array([r['bootstrap_samples'] for r in valid_results])
    )
    logger.info(f"Saved bootstrap samples to {bootstrap_path}")
    
    # Generate plots
    plot_path = output_dir / 'alpha_distribution.png'
    plot_alpha_distribution(valid_results, plot_path)
    
    logger.info("Done!")


if __name__ == '__main__':
    main()
