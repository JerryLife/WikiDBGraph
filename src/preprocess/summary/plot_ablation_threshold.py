"""
Threshold ablation analysis for graph density.

Analyzes how similarity threshold affects graph properties:
- Number of edges
- Average degree
- Node coverage (% of nodes with at least one edge)

Efficiently processes a single predictions file at multiple thresholds.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm

from .config import COLORS, MARKERS, FONTSIZE, LINEWIDTH, MARKERSIZE, STYLE, GRID, DPI


def analyze_threshold_effects(
    predictions_path: str,
    thresholds: List[float],
    output_dir: str,
    figsize: Tuple[float, float] = (10, 6)
) -> Dict[str, List[float]]:
    """
    Analyze how different thresholds affect graph properties.
    
    Args:
        predictions_path: Path to predictions .pt file (new tensor format)
        thresholds: List of threshold values to evaluate
        output_dir: Directory to save plots
        figsize: Figure size for plots
        
    Returns:
        Dictionary with metrics for each threshold
    """
    print(f"Loading predictions from {predictions_path}")
    data = torch.load(predictions_path, weights_only=False)
    
    # Handle both old and new format
    if isinstance(data, list):
        # Old format: list of [src, tgt, sim, label, edge]
        print("Detected old format (list of records)")
        scores = torch.tensor([r[2] for r in data], dtype=torch.float32)
        sources = torch.tensor([i for i, _ in enumerate(data)], dtype=torch.int32)
        targets = sources.clone()  # Placeholder
        num_nodes = len(set(r[0] for r in data) | set(r[1] for r in data))
        
        # Extract actual source/target indices
        all_ids = sorted(set(r[0] for r in data) | set(r[1] for r in data))
        id_to_idx = {uid: i for i, uid in enumerate(all_ids)}
        sources = torch.tensor([id_to_idx[r[0]] for r in data], dtype=torch.int32)
        targets = torch.tensor([id_to_idx[r[1]] for r in data], dtype=torch.int32)
    else:
        # New format: dict with tensors
        print("Detected new format (tensor dict)")
        sources = data["sources"]
        targets = data["targets"]
        scores = data["scores"]
        num_nodes = data.get("num_nodes", len(data.get("sorted_db_ids", [])))
    
    print(f"Total pairs: {len(scores):,}")
    print(f"Total nodes: {num_nodes:,}")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")
    
    # Sort thresholds for efficient processing
    thresholds = sorted(thresholds)
    
    # Results storage
    results = {
        "thresholds": thresholds,
        "num_edges": [],
        "avg_degree": [],
        "node_coverage": [],
    }
    
    print(f"\nAnalyzing {len(thresholds)} thresholds...")
    print("=" * 70)
    print(f"{'Threshold':<12} {'Edges':<15} {'Avg Degree':<15} {'Node Coverage':<15}")
    print("-" * 70)
    
    for threshold in tqdm(thresholds, desc="Processing thresholds"):
        # Filter edges by threshold
        mask = scores >= threshold
        num_edges = mask.sum().item()
        
        if num_edges > 0:
            # Get nodes that have at least one edge
            edge_sources = sources[mask]
            edge_targets = targets[mask]
            
            # Count unique nodes with edges
            nodes_with_edges = torch.unique(torch.cat([edge_sources, edge_targets]))
            node_coverage = len(nodes_with_edges) / num_nodes * 100
            
            # Average degree = 2 * edges / nodes (undirected graph)
            avg_degree = 2 * num_edges / num_nodes
        else:
            node_coverage = 0.0
            avg_degree = 0.0
        
        results["num_edges"].append(num_edges)
        results["avg_degree"].append(avg_degree)
        results["node_coverage"].append(node_coverage)
        
        print(f"{threshold:<12.4f} {num_edges:<15,} {avg_degree:<15.2f} {node_coverage:<15.2f}%")
    
    print("=" * 70)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Number of Edges
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(thresholds, results["num_edges"], 
            marker=MARKERS['primary'], markersize=MARKERSIZE['main'], linewidth=LINEWIDTH['main'], color=COLORS['ablation_1'],
            markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Similarity Threshold', fontsize=FONTSIZE['axis_label'], fontweight='bold')
    ax.set_ylabel('Number of Edges', fontsize=FONTSIZE['axis_label'], fontweight='bold')
    ax.set_title('Effect of Threshold on Edge Count', fontsize=FONTSIZE['title'], fontweight='bold', pad=15)
    ax.tick_params(axis='both', labelsize=FONTSIZE['tick_label'])
    ax.grid(True, alpha=GRID['alpha'], linestyle=GRID['linestyle'])
    
    # Format y-axis with thousands separator
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    plt.tight_layout()
    edge_plot_path = os.path.join(output_dir, "threshold_vs_edges")
    plt.savefig(f"{edge_plot_path}.png", dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{edge_plot_path}.pdf", dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved edge count plot to {edge_plot_path}.png")
    
    # Plot 2: Average Degree
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(thresholds, results["avg_degree"],
            marker=MARKERS['secondary'], markersize=MARKERSIZE['main'], linewidth=LINEWIDTH['main'], color=COLORS['ablation_2'],
            markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Similarity Threshold', fontsize=FONTSIZE['axis_label'], fontweight='bold')
    ax.set_ylabel('Average Degree', fontsize=FONTSIZE['axis_label'], fontweight='bold')
    ax.set_title('Effect of Threshold on Average Node Degree', fontsize=FONTSIZE['title'], fontweight='bold', pad=15)
    ax.tick_params(axis='both', labelsize=FONTSIZE['tick_label'])
    ax.grid(True, alpha=GRID['alpha'], linestyle=GRID['linestyle'])
    
    plt.tight_layout()
    degree_plot_path = os.path.join(output_dir, "threshold_vs_degree")
    plt.savefig(f"{degree_plot_path}.png", dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{degree_plot_path}.pdf", dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved average degree plot to {degree_plot_path}.png")
    
    # Plot 3: Node Coverage
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(thresholds, results["node_coverage"],
            marker=MARKERS['tertiary'], markersize=MARKERSIZE['main'], linewidth=LINEWIDTH['main'], color=COLORS['ablation_3'],
            markerfacecolor='white', markeredgewidth=2)
    
    ax.set_xlabel('Similarity Threshold', fontsize=FONTSIZE['axis_label'], fontweight='bold')
    ax.set_ylabel('Node Coverage (%)', fontsize=FONTSIZE['axis_label'], fontweight='bold')
    ax.set_title('Effect of Threshold on Node Coverage', fontsize=FONTSIZE['title'], fontweight='bold', pad=15)
    ax.set_ylim([0, 105])
    ax.tick_params(axis='both', labelsize=FONTSIZE['tick_label'])
    ax.grid(True, alpha=GRID['alpha'], linestyle=GRID['linestyle'])
    
    plt.tight_layout()
    coverage_plot_path = os.path.join(output_dir, "threshold_vs_coverage")
    plt.savefig(f"{coverage_plot_path}.png", dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.savefig(f"{coverage_plot_path}.pdf", dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Saved node coverage plot to {coverage_plot_path}.png")
    
    # Save results as text
    results_path = os.path.join(output_dir, "threshold_ablation_results.txt")
    with open(results_path, "w") as f:
        f.write("Threshold Ablation Analysis Results\n")
        f.write("=" * 70 + "\n")
        f.write(f"Predictions file: {predictions_path}\n")
        f.write(f"Total nodes: {num_nodes:,}\n")
        f.write(f"Total pairs analyzed: {len(scores):,}\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"{'Threshold':<12} {'Edges':<15} {'Avg Degree':<15} {'Node Coverage':<15}\n")
        f.write("-" * 70 + "\n")
        for i, t in enumerate(thresholds):
            f.write(f"{t:<12.4f} {results['num_edges'][i]:<15,} "
                   f"{results['avg_degree'][i]:<15.2f} {results['node_coverage'][i]:<15.2f}%\n")
    print(f"✅ Saved results to {results_path}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze threshold effects on graph density")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions .pt file")
    parser.add_argument("--output-dir", type=str, default="fig/ablation_threshold",
                        help="Output directory for plots")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                        help="Threshold values to analyze")
    parser.add_argument("--figsize", type=float, nargs=2, default=[10, 6],
                        help="Figure size (width height)")
    
    args = parser.parse_args()
    
    analyze_threshold_effects(
        predictions_path=args.predictions,
        thresholds=args.thresholds,
        output_dir=args.output_dir,
        figsize=tuple(args.figsize)
    )
