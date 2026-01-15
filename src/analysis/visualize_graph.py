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

try:
    from adjustText import adjust_text
    HAS_ADJUSTTEXT = True
except ImportError:
    HAS_ADJUSTTEXT = False
    print("Warning: adjustText not installed. Text overlap prevention disabled.")


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
            return name[:40] + "..." if len(name) > 40 else name
    return f"Database {db_id}"


def visualize_graph(
    db_ids: List[str],
    subgraph_data: Dict,
    output_path: str = "graph_visualization.png",
    label_info: str = None,
    figsize: tuple = (12, 8)
):
    """Create a professional visualization of the database graph."""
    
    # Use a professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_facecolor('#f8f9fa')
    
    n_nodes = len(db_ids)
    node_id_map = subgraph_data.get('node_id_map', {})
    edges_src = subgraph_data.get('edges_src', [])
    edges_dst = subgraph_data.get('edges_dst', [])
    edge_props = subgraph_data.get('edge_props', {})
    edge_sims = edge_props.get('similarity', None)
    
    # Create position layout (elliptical - flatter shape)
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False) - np.pi / 2
    radius_x = 5.5  # Wider horizontal spread
    radius_y = 3.0  # Shorter vertical spread
    positions = {
        db_id: (radius_x * np.cos(angles[i]), radius_y * np.sin(angles[i]))
        for i, db_id in enumerate(db_ids)
    }
    
    # Professional color palette
    colors = [
        '#4C72B0', '#DD8452', '#55A868', '#C44E52',
        '#8172B3', '#937860', '#DA8BC3', '#8C8C8C',
        '#CCB974', '#64B5CD', '#4C9A2A', '#E45756'
    ]
    node_colors = [colors[i % len(colors)] for i in range(n_nodes)]
    
    # Draw edges with varying thickness based on similarity
    idx_to_db = {v: k for k, v in node_id_map.items()}
    labeled_pairs = set()
    
    for edge_idx, (s, d) in enumerate(zip(edges_src, edges_dst)):
        src_db = idx_to_db.get(s)
        dst_db = idx_to_db.get(d)
        if src_db in positions and dst_db in positions:
            x1, y1 = positions[src_db]
            x2, y2 = positions[dst_db]
            
            # Get similarity for edge width
            sim_val = None
            if edge_sims is not None and edge_idx < len(edge_sims):
                try:
                    sim_val = float(edge_sims[edge_idx])
                except (TypeError, ValueError):
                    pass
            
            edge_width = 1.0 + (sim_val * 3 if sim_val else 0)
            edge_alpha = 0.3 + (sim_val * 0.4 if sim_val else 0.2)
            
            ax.plot(
                [x1, x2], [y1, y2],
                color='#555555',
                linewidth=edge_width,
                alpha=edge_alpha,
                zorder=1
            )
            
            # Track edges to avoid duplicates
            pair_key = tuple(sorted([src_db, dst_db]))
            labeled_pairs.add(pair_key)
    
    # Draw nodes as circles with IDs
    node_size = 0.6
    for i, db_id in enumerate(db_ids):
        x, y = positions[db_id]
        circle = plt.Circle(
            (x, y), node_size,
            color=node_colors[i],
            ec='white',
            linewidth=3,
            zorder=10
        )
        ax.add_patch(circle)
        
        # Add ID inside node
        ax.text(
            x, y, db_id,
            fontsize=14, fontweight='bold',
            color='white',
            ha='center', va='center',
            zorder=11
        )
    
    # Set axis properties - adjusted for elliptical layout
    ax.set_xlim(-7.0, 7.0)
    ax.set_ylim(-4.5, 4.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add legend below the graph with full database names
    legend_handles = []
    for i, db_id in enumerate(db_ids):
        display_name = get_db_display_name(db_id)
        patch = mpatches.Patch(
            color=node_colors[i],
            label=f"{db_id}: {display_name}"
        )
        legend_handles.append(patch)
    
    legend = ax.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,  # Two columns for compact layout
        fontsize=11,
        frameon=True,
        fancybox=True,
        handlelength=1.5,
        handleheight=1.0,
        columnspacing=1.0,
    )
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('#cccccc')
    
    plt.tight_layout()
    
    # Save to both PNG and PDF
    base_path = os.path.splitext(output_path)[0]
    
    # PNG (high resolution)
    png_path = f"{base_path}.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    print(f"Saved PNG visualization to: {png_path}")
    
    # PDF (vector format)
    pdf_path = f"{base_path}.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white', pad_inches=0.05)
    print(f"Saved PDF visualization to: {pdf_path}")
    
    plt.close()
    
    # Print LaTeX include script
    print("\n" + "="*60)
    print("LaTeX code to include this figure:")
    print("="*60)
    
    # Generate a more meaningful caption based on metadata if possible
    caption = f"Network graph of {len(db_ids)} databases. Nodes represent local databases, and edges denote schema similarity."
    if label_info:
        caption += f" The graph highlights connections relevant to the task: {label_info}."
        
    latex_code = f'''\\begin{{figure}}[htbp]
    \\centering
    \\includegraphics[width=1.0\\linewidth]{{{pdf_path}}}
    \\caption{{{caption}}}
    \\label{{fig:database_network_graph}}
\\end{{figure}}'''
    print(latex_code)
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Visualize FedGNN database graph")
    parser.add_argument(
        "-f", "--file",
        type=str,
        default="results/graph_example.txt",
        help="Path to graph config file (default: results/graph_example.txt)"
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
        default="fig/graph_DB_example.png",
        help="Output path for the visualization (will generate both .png and .pdf)"
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
