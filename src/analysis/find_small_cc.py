"""
Find and visualize several small connected components from the database graph.

Usage example:
  python src/analysis/find_small_cc.py -t 0.94 --min-size 3 --max-size 20 \
         --max-components 3 --output-dir data/graph/cc_vis

The script loads `data/graph/graph_raw_{threshold}.dgl`, scans nodes until it
collects up to `max_components` connected components whose size is within the
given range, and for each component it writes a publication-ready PNG (with
edge-weight annotations) plus a JSON summary capturing node names and edge
weights.
"""

import os
import json
import argparse
from collections import deque
from typing import Dict, List, Tuple

import torch
import dgl

# Use non-interactive backend for headless environments
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

# Check for pydot availability
PYDOT_AVAILABLE = False
try:
    from networkx.drawing.nx_pydot import graphviz_layout
    import pydot
    PYDOT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: pydot/graphviz not available ({e}). Will use spring layout.")
    print("For better layouts, install with: pip install pydot")
    graphviz_layout = None  # Set to None so we know it's not available

# Configure seaborn for beautiful plots
sns.set_theme(style="white", context="paper")
sns.set_palette("husl")


def load_graph(threshold: str = "0.94") -> dgl.DGLGraph:
    graph_path = f"data/graph/graph_raw_{threshold}.dgl"
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    graphs, _ = dgl.load_graphs(graph_path)
    g = graphs[0]
    print(f"Loaded graph with {g.num_nodes()} nodes and {g.num_edges()} edges")
    return g


def split_camel_case(text: str, max_chars_per_line: int = 20) -> str:
    """Split camelCase text into multiple lines for better readability."""
    import re
    
    # Insert space before uppercase letters that follow lowercase letters
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Insert space before uppercase letters followed by lowercase when preceded by uppercase
    spaced = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', spaced)
    # Replace underscores with spaces
    spaced = spaced.replace('_', ' ')
    
    # Split into words
    words = spaced.split()
    
    # Group words into lines with max_chars_per_line characters
    lines = []
    current_line = []
    current_length = 0
    
    for word in words:
        word_length = len(word)
        if current_length + word_length + len(current_line) <= max_chars_per_line:
            current_line.append(word)
            current_length += word_length
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
            current_length = word_length
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)


def get_database_name(db_id: int, schema_dir: str = "data/schema") -> str:
    """Map numeric ID to database name via file prefix in data/schema."""
    db_id_str = f"{db_id:05d}"
    try:
        for filename in os.listdir(schema_dir):
            if filename.startswith(db_id_str + "_") and filename.endswith(".json"):
                return filename[len(db_id_str) + 1 : -5]
    except FileNotFoundError:
        pass
    return f"ID_{db_id:05d}"


def load_database_sizes(data_volume_path: str = "data/graph/data_volume.csv") -> Dict[int, float]:
    """Load database sizes from data volume CSV file."""
    import pandas as pd
    
    if not os.path.exists(data_volume_path):
        print(f"Warning: Data volume file not found at {data_volume_path}")
        return {}
    
    try:
        df = pd.read_csv(data_volume_path)
        # Assuming the CSV has 'db_id' and 'size_mb' or 'size_bytes' columns
        size_dict = {}
        
        if 'db_id' in df.columns:
            if 'size_mb' in df.columns:
                size_dict = dict(zip(df['db_id'], df['size_mb']))
            elif 'size_bytes' in df.columns:
                # Convert bytes to MB
                size_dict = dict(zip(df['db_id'], df['size_bytes'] / (1024 * 1024)))
            elif 'file_size' in df.columns:
                # Assume file_size is in bytes
                size_dict = dict(zip(df['db_id'], df['file_size'] / (1024 * 1024)))
            else:
                # Try to find any numeric column that might represent size
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
                if len(numeric_cols) > 1:  # db_id is also numeric, so we need at least 2
                    size_col = [col for col in numeric_cols if col != 'db_id'][0]
                    size_dict = dict(zip(df['db_id'], df[size_col]))
        
        print(f"Loaded sizes for {len(size_dict)} databases")
        return size_dict
    except Exception as e:
        print(f"Warning: Failed to load database sizes: {e}")
        return {}


def bfs_component(g: dgl.DGLGraph, start: int, visited_global: torch.Tensor) -> List[int]:
    """Return the full connected component discovered via BFS from `start`."""
    q = deque([start])
    component: List[int] = []
    visited_global[start] = True

    while q:
        u = q.popleft()
        component.append(u)
        nbrs = g.successors(u).numpy()
        for v in nbrs:
            if not visited_global[v]:
                visited_global[v] = True
                q.append(v)
    return component


def find_small_components(
    g: dgl.DGLGraph,
    max_size: int,
    seed: int = 42,
    min_size: int = 1,
    limit: int = 3,
) -> List[List[int]]:
    """Find up to `limit` components with size constrained to [min_size, max_size]."""
    if min_size > max_size:
        raise ValueError("min_size must not exceed max_size")

    num_nodes = g.num_nodes()
    order = torch.randperm(num_nodes, generator=torch.Generator().manual_seed(seed)).tolist()
    visited = torch.zeros(num_nodes, dtype=torch.bool)

    components: List[List[int]] = []
    for u in order:
        if visited[u]:
            continue
        comp = bfs_component(g, u, visited)
        size = len(comp)
        if min_size <= size <= max_size:
            components.append(sorted(comp))
            print(f"Found component #{len(components)} (size={size}): {components[-1]}")
            if len(components) >= limit:
                break

    if not components:
        raise RuntimeError(f"No component found with size in [{min_size}, {max_size}].")

    return components


def build_component_graph(
    g: dgl.DGLGraph,
    nodes: List[int],
    schema_dir: str,
    db_sizes: Dict[int, float] = None,
) -> Tuple[nx.Graph, Dict[int, str], Dict[Tuple[int, int], str], List[Dict[str, float]], int, Dict[int, float]]:
    """Create a NetworkX graph plus metadata for visualization and export."""
    nodes_sorted = sorted(nodes)
    sub = dgl.node_subgraph(g, nodes_sorted)

    orig_ids = sub.ndata[dgl.NID].numpy().tolist()
    node_labels: Dict[int, str] = {}
    node_sizes: Dict[int, float] = {}
    max_label_length = 0
    
    for oid in orig_ids:
        name = get_database_name(int(oid), schema_dir)
        # Split camelCase name into multiple lines
        split_name = split_camel_case(name, max_chars_per_line=20)
        node_labels[int(oid)] = f"{int(oid):05d}\n{split_name}"
        # Track max length for width calculation (longest line)
        max_label_length = max(max_label_length, max(len(line) for line in split_name.split('\n')))
        
        # Get database size
        if db_sizes and int(oid) in db_sizes:
            node_sizes[int(oid)] = db_sizes[int(oid)]
        else:
            node_sizes[int(oid)] = 1.0  # Default size

    weight_tensor = sub.edata["weight"] if "weight" in sub.edata else None
    weights = weight_tensor.numpy().tolist() if weight_tensor is not None else [None] * sub.num_edges()

    edge_accumulator: Dict[Tuple[int, int], Dict[str, float]] = {}
    src, dst = sub.edges()
    src = src.numpy().tolist()
    dst = dst.numpy().tolist()

    for s, d, w in zip(src, dst, weights):
        a = int(orig_ids[s])
        b = int(orig_ids[d])
        if a == b:
            continue
        key = tuple(sorted((a, b)))
        stats = edge_accumulator.setdefault(key, {"weight_sum": 0.0, "weight_count": 0})
        if w is not None:
            stats["weight_sum"] += float(w)
            stats["weight_count"] += 1

    nxg = nx.Graph()
    for oid in orig_ids:
        nxg.add_node(int(oid))

    edge_labels: Dict[Tuple[int, int], str] = {}
    edge_export: List[Dict[str, float]] = []
    for (a, b), stats in edge_accumulator.items():
        weight = None
        if stats["weight_count"] > 0:
            weight = stats["weight_sum"] / stats["weight_count"]
        nxg.add_edge(a, b, weight=weight)
        if weight is not None:
            edge_labels[(a, b)] = f"{weight:.3f}"
        edge_export.append({"source": a, "target": b, "weight": weight})

    return nxg, node_labels, edge_labels, edge_export, max_label_length, node_sizes


def visualize_component(
    nxg: nx.Graph,
    node_labels: Dict[int, str],
    edge_labels: Dict[Tuple[int, int], str],
    out_png: str,
    layout_seed: int,
    max_label_length: int = 25,
    node_sizes: Dict[int, float] = None,
) -> None:
    """Render the NetworkX graph with publication-friendly styling using seaborn."""
    if nxg.number_of_nodes() == 0:
        raise ValueError("Component graph has no nodes")

    n_nodes = nxg.number_of_nodes()
    
    # Process node sizes for visualization
    if node_sizes is None:
        node_sizes = {node: 1.0 for node in nxg.nodes()}
    
    # Normalize sizes for better visualization (between 0.3 and 1.5 for padding scale)
    size_values = list(node_sizes.values())
    min_size = min(size_values) if size_values else 1.0
    max_size = max(size_values) if size_values else 1.0
    
    if max_size > min_size:
        normalized_sizes = {
            node: 0.3 + 1.2 * (node_sizes.get(node, 1.0) - min_size) / (max_size - min_size)
            for node in nxg.nodes()
        }
    else:
        normalized_sizes = {node: 0.8 for node in nxg.nodes()}
    
    # Calculate figure size dynamically based on number of nodes AND label length
    # Add extra width for longer labels to prevent going out of bounds
    base_width = max(10.0, n_nodes * 2.5)
    label_width_factor = max(1.0, max_label_length / 20.0)  # Scale up for longer labels
    width = base_width * label_width_factor
    
    base_height = max(8.0, n_nodes * 2.0)
    height = base_height * label_width_factor
    
    _, ax = plt.subplots(figsize=(width, height), facecolor='white')
    ax.set_facecolor("#f8f9fa")
    
    # Prefer Graphviz (pydot) layouts for cleaner placement; fall back to spring layout if needed
    pos = None
    k_param = 0.5 / (n_nodes ** 0.5) if n_nodes > 2 else 1.0
    if PYDOT_AVAILABLE and graphviz_layout is not None:
        try:
            # Use neato for undirected graphs with nice spacing
            pos = graphviz_layout(nxg, prog="neato")
            print(f"Using graphviz/pydot layout (neato)")
        except Exception as e:
            print(f"Graphviz layout failed ({e}), falling back to spring layout")
            pos = None
    
    # Fallback to spring layout if graphviz is not available or failed
    if pos is None:
        for u, v in nxg.edges():
            weight = nxg[u][v].get('weight', 0.5)
            nxg[u][v]['layout_weight'] = weight if weight > 0 else 0.5
        pos = nx.spring_layout(nxg, seed=layout_seed, k=k_param, iterations=100, weight='layout_weight')
        print(f"Using spring layout")

    # Draw edges with gradient based on weights
    # INVERTED: High similarity (high weight) = thin/short edge, low similarity = thick/long edge
    edge_weights = [nxg[u][v].get('weight', 0.5) for u, v in nxg.edges()]
    if edge_weights:
        # Normalize weights for edge styling
        max_weight = max(edge_weights) if edge_weights else 1.0
        min_weight = min(edge_weights) if edge_weights else 0.0
        weight_range = max_weight - min_weight if max_weight > min_weight else 1.0
        
        # Create color map for edges: red = high similarity, blue = low similarity
        edge_cmap = sns.color_palette("coolwarm", as_cmap=True)  # Normal colormap (red=high, blue=low)
        edge_colors = [(w - min_weight) / weight_range for w in edge_weights]
        
        # CORRECT: High similarity (high weight) gets THICK edge, low similarity gets THIN edge
        # Range from 3.0 (low similarity) to 6.0 (high similarity)
        edge_widths = [3.0 + 3.0 * ((w - min_weight) / weight_range) for w in edge_weights]
        
        nx.draw_networkx_edges(
            nxg, pos, 
            width=edge_widths,
            edge_color=edge_colors,
            edge_cmap=edge_cmap,
            alpha=0.8,  # Slightly more opaque for bolder appearance
            edge_vmin=0,
            edge_vmax=1,
            arrows=False
        )
    else:
        nx.draw_networkx_edges(
            nxg, pos, 
            width=5.0,  # Bolder default
            edge_color="#607d8b", 
            alpha=0.7,
            arrows=False
        )

    # No circular nodes drawn - text boxes will serve as nodes
    
    # Draw text boxes as nodes with sizes representing actual database size
    # Adjust font size based on label length to prevent overflow
    font_size = 15
    
    # Use grey color for the box background
    box_color = "#e0e0e0"  # Light grey
    
    # Draw each label individually to control box size based on database size
    for node in nxg.nodes():
        # Scale padding based on normalized database size
        size_scale = normalized_sizes.get(node, 0.8)
        padding = 0.3 + 0.4 * size_scale  # Range from 0.3 to 0.7
        
        # Draw individual label with size-based padding
        nx.draw_networkx_labels(
            nxg,
            pos,
            labels={node: node_labels[node]},
            font_size=font_size,
            font_weight="bold",
            font_family="sans-serif",
            verticalalignment="center",
            horizontalalignment="center",
            bbox={
                "boxstyle": f"round,pad={padding}",  # Variable padding based on size
                "facecolor": box_color,
                "edgecolor": "#2c3e50",
                "linewidth": 2.0,  # Thicker border for prominence
                "alpha": 0.95,
            },
            clip_on=True,  # Clip labels to axes boundaries
        )

    # Draw edge labels with enhanced styling and larger font
    if edge_labels:
        nx.draw_networkx_edge_labels(
            nxg,
            pos,
            edge_labels=edge_labels,
            font_size=13,  # Increased from 9 to 13 for better visibility
            font_weight="bold",  # Changed to bold for better visibility
            font_family="sans-serif",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "#fff9e6",
                "edgecolor": "#95a5a6",
                "linewidth": 1.2,
                "alpha": 0.9,
            },
            label_pos=0.5,
        )

    # Remove axis but set limits to ensure labels don't go out of bounds
    ax.set_axis_off()
    
    # Expand axis limits to accommodate labels with padding
    x_values = [pos[node][0] for node in nxg.nodes()]
    y_values = [pos[node][1] for node in nxg.nodes()]
    
    x_margin = 0.15 * (max(x_values) - min(x_values)) if len(x_values) > 1 else 0.2
    y_margin = 0.15 * (max(y_values) - min(y_values)) if len(y_values) > 1 else 0.2
    
    # Add extra margin for long labels
    extra_margin = max(0, (max_label_length - 20) / 40.0)
    x_margin += extra_margin
    y_margin += extra_margin
    
    ax.set_xlim(min(x_values) - x_margin, max(x_values) + x_margin)
    ax.set_ylim(min(y_values) - y_margin, max(y_values) + y_margin)
    
    # Add subtle border
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.tight_layout()  # Increased padding
    plt.savefig(out_png, dpi=320, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    print(f"Saved visualization to {out_png}")


def save_summary(
    nodes: List[int],
    edges: List[Dict[str, float]],
    out_json: str,
    schema_dir: str,
) -> None:
    records: List[Dict[str, object]] = []
    for oid in sorted(nodes):
        records.append(
            {
                "id": int(oid),
                "id_str": f"{oid:05d}",
                "name": get_database_name(int(oid), schema_dir),
            }
        )

    payload = {
        "size": len(nodes),
        "nodes": records,
        "edges": edges,
    }

    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved summary to {out_json}")


def main():
    parser = argparse.ArgumentParser(description="Find and visualize small connected components")
    parser.add_argument("-t", "--threshold", type=str, default="0.94", help="Graph threshold tag (default: 0.94)")
    parser.add_argument("--min-size", type=int, default=3, help="Minimum component size (default: 3)")
    parser.add_argument("--max-size", type=int, default=20, help="Maximum component size (default: 20)")
    parser.add_argument("--max-components", type=int, default=3, help="Number of components to output (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for node traversal")
    parser.add_argument("--schema-dir", type=str, default="data/schema", help="Directory for schema metadata")
    parser.add_argument("--output-dir", type=str, default="data/graph/cc_vis", help="Destination directory for artifacts")
    parser.add_argument("--data-volume", type=str, default="data/graph/data_volume.csv", help="Path to data volume CSV file")
    args = parser.parse_args()
    
    # Load database sizes
    db_sizes = load_database_sizes(args.data_volume)
    
    g = load_graph(args.threshold)
    components = find_small_components(
        g,
        max_size=args.max_size,
        seed=args.seed,
        min_size=args.min_size,
        limit=args.max_components,
    )
    
    for idx, comp_nodes in enumerate(components, start=1):
        size = len(comp_nodes)
        first_id = f"{min(comp_nodes):05d}"
        tag = f"t{args.threshold}_c{idx:02d}_n{size}"
        out_png = os.path.join(args.output_dir, f"small_cc_{tag}_{first_id}.png")
        out_json = os.path.join(args.output_dir, f"small_cc_{tag}_{first_id}.json")

        nxg, node_labels, edge_labels, edge_records, max_label_len, node_sizes = build_component_graph(
            g, comp_nodes, args.schema_dir, db_sizes
        )
        visualize_component(
            nxg, node_labels, edge_labels, out_png, 
            layout_seed=args.seed + idx, max_label_length=max_label_len, node_sizes=node_sizes
        )
        save_summary(comp_nodes, edge_records, out_json, args.schema_dir)


if __name__ == "__main__":
    main()
