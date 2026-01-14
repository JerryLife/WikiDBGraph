#!/usr/bin/env python3
"""
Visualize a FedGNN graph example with professional styling.

Usage:
    python src/analysis/visualize_graph.py -f data/analysis/graph_example.txt
    python src/analysis/visualize_graph.py --db_ids 00155,00176 --output graph.png
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Using basic visualization.")


def parse_config_file(config_path: str) -> Dict:
    """Parse the graph_example.txt config file."""
    config = {
        'databases': [],
        'label_column': None,
        'positive_token': None,
        'description': None
    }
    
    current_section = None
    with open(config_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
            elif current_section == 'databases':
                config['databases'].append(line)
            elif current_section == 'label':
                if '=' in line:
                    key, val = line.split('=', 1)
                    key, val = key.strip(), val.strip()
                    if key == 'column':
                        config['label_column'] = val
                    elif key == 'positive_token':
                        config['positive_token'] = val
                    elif key == 'description':
                        config['description'] = val
    
    return config


def load_subgraph_data(db_ids: List[str], cache_dir: Optional[str] = None) -> Dict:
    """Load subgraph structure for the given database IDs."""
    try:
        from analysis.WikiDBSubgraph import WikiDBSubgraph
        subgraph = WikiDBSubgraph(cache_dir=cache_dir)
        return subgraph.load_or_construct(db_ids)
    except Exception as e:
        print(f"Warning: Could not load subgraph data: {e}")
        # Return minimal structure
        return {
            'n_nodes': len(db_ids),
            'n_edges': 0,
            'node_id_map': {db_id: i for i, db_id in enumerate(db_ids)},
            'edges_src': [],
            'edges_dst': []
        }


def get_db_display_name(db_id: str) -> str:
    """Get a human-readable name for a database."""
    schema_dir = Path("data/schema")
    for f in schema_dir.iterdir():
        if f.name.startswith(db_id):
            # Extract name from filename
            name = f.stem.replace(db_id + "_", "").replace("_", " ")
            return name[:25] + "..." if len(name) > 25 else name
    return f"DB{db_id}"


def visualize_graph(
    db_ids: List[str],
    subgraph_data: Dict,
    output_path: str = "graph_visualization.png",
    label_info: str = None,
    figsize: tuple = (12, 8)
):
    """Create a professional visualization of the database graph."""
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    n_nodes = len(db_ids)
    node_id_map = subgraph_data.get('node_id_map', {})
    edges_src = subgraph_data.get('edges_src', [])
    edges_dst = subgraph_data.get('edges_dst', [])
    edge_props = subgraph_data.get('edge_props', {})
    edge_sims = edge_props.get('similarity', None)
    
    # Create position layout (circular)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    radius = 3
    positions = {
        db_id: (radius * np.cos(angles[i]), radius * np.sin(angles[i]))
        for i, db_id in enumerate(db_ids)
    }
    
    # Color scheme
    node_colors = plt.cm.Set3(np.linspace(0, 1, n_nodes))
    edge_color = '#888888'
    
    # Draw edges
    idx_to_db = {v: k for k, v in node_id_map.items()}
    labeled_pairs = set()
    for edge_idx, (s, d) in enumerate(zip(edges_src, edges_dst)):
        src_db = idx_to_db.get(s)
        dst_db = idx_to_db.get(d)
        if src_db in positions and dst_db in positions:
            x1, y1 = positions[src_db]
            x2, y2 = positions[dst_db]
            ax.annotate(
                '', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(
                    arrowstyle='-',
                    color=edge_color,
                    lw=1.5,
                    alpha=0.6
                )
            )
            if edge_sims is not None:
                pair_key = tuple(sorted([src_db, dst_db]))
                if pair_key not in labeled_pairs:
                    if edge_idx < len(edge_sims):
                        try:
                            sim_val = float(edge_sims[edge_idx])
                        except (TypeError, ValueError):
                            sim_val = None
                        sim_label = "NA" if sim_val is None else f"{sim_val:.2f}"
                        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                        ax.text(
                            mid_x, mid_y, sim_label,
                            fontsize=8, color='#444444',
                            ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7)
                        )
                    labeled_pairs.add(pair_key)
    
    # Draw nodes
    for i, db_id in enumerate(db_ids):
        x, y = positions[db_id]
        circle = plt.Circle(
            (x, y), 0.5,
            color=node_colors[i],
            ec='black',
            linewidth=2,
            zorder=10
        )
        ax.add_patch(circle)
        
        # Add label
        display_name = get_db_display_name(db_id)
        ax.annotate(
            f"DB{db_id}\n{display_name}",
            xy=(x, y - 0.8),
            ha='center', va='top',
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )
    
    # Set axis properties
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Title
    title = f"FedGNN Database Graph ({n_nodes} nodes, {len(edges_src)} edges)"
    if label_info:
        title += f"\nLabel: {label_info}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    node_patches = [
        mpatches.Patch(color=node_colors[i], label=f"DB{db_ids[i]}")
        for i in range(n_nodes)
    ]
    ax.legend(
        handles=node_patches,
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved visualization to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize FedGNN database graph")
    parser.add_argument(
        "-f", "--file",
        type=str,
        default=None,
        help="Path to graph config file (e.g., data/analysis/graph_example.txt)"
    )
    parser.add_argument(
        "--db_ids",
        type=str,
        default=None,
        help="Comma-separated database IDs (alternative to config file)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="data/analysis/graph_visualization.png",
        help="Output path for the visualization"
    )
    parser.add_argument(
        "--subgraph_cache_dir",
        type=str,
        default="results/subgraph_cache",
        help="Cache directory for constructed subgraphs"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.file:
        config = parse_config_file(args.file)
        db_ids = config['databases']
        label_info = config.get('label_column')
        if config.get('positive_token'):
            label_info = f"{label_info} (positive: {config['positive_token']})"
    elif args.db_ids:
        db_ids = [db_id.strip().zfill(5) for db_id in args.db_ids.split(',')]
        label_info = None
    else:
        print("Error: Please provide either -f (config file) or --db_ids")
        return
    
    if not db_ids:
        print("Error: No database IDs found")
        return
    
    print(f"Visualizing graph with {len(db_ids)} databases: {db_ids}")
    
    # Load subgraph structure
    subgraph_data = load_subgraph_data(db_ids, cache_dir=args.subgraph_cache_dir)
    
    # Create visualization
    visualize_graph(
        db_ids,
        subgraph_data,
        output_path=args.output,
        label_info=label_info
    )


if __name__ == "__main__":
    main()
