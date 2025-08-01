import dgl
import os
import pandas as pd
import argparse
import cudf
import cugraph
import matplotlib.pyplot as plt
import collections
import numpy as np

def analyze_graph(dgl_path, output_base_path, use_cache=False, threshold='0.94'):
    """
    Combined function to analyze both connected components and communities in a graph.
    Args:
        dgl_path: Path to the DGL graph file
        output_base_path: Base path for output files (without extension)
        use_cache: Whether to use cached community assignments
        threshold: Threshold for community detection
    """
    print(f"Loading DGL graph from {dgl_path}...")
    g = dgl.load_graphs(dgl_path)[0][0]
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()

    # Prepare graph for analysis
    src, dst = g.edges()
    src = src.numpy()
    dst = dst.numpy()
    
    # Create DataFrame with original node IDs
    df = cudf.DataFrame({
        'src': src,
        'dst': dst,
        'src_id': src,  # Keep original IDs
        'dst_id': dst   # Keep original IDs
    })
    
    G = cugraph.Graph()
    G.from_cudf_edgelist(df, source='src', destination='dst', renumber=True)

    # 1. Connected Components Analysis
    print("Running connected components analysis...")
    components_df = cugraph.connected_components(G)
    labels = components_df['labels'].to_pandas().values

    # Unrenumber the vertex IDs back to original
    if hasattr(G, 'renumber_map'):
        components_df = G.unrenumber(components_df, 'vertex')
    
    size_counter = collections.Counter(labels)
    sorted_cc_sizes = sorted(size_counter.values(), reverse=True)
    largest_cc_size = sorted_cc_sizes[0]

    # 2. Community Detection
    if use_cache and os.path.exists("data/graph/community_assignment.csv"):
        print("Loading community assignments from cache...")
        partition = pd.read_csv("data/graph/community_assignment.csv")
        community_sizes = partition['partition'].value_counts().sort_values(ascending=False)
        sorted_community_sizes = community_sizes.values.tolist()
        modularity_score = 0
        print(f"Found {partition['partition'].nunique()} communities from cache")
    else:
        print("Running Louvain community detection...")
        partition_df, modularity_score = cugraph.louvain(G)
        
        # Unrenumber the vertex IDs back to original
        if hasattr(G, 'renumber_map'):
            partition_df = G.unrenumber(partition_df, 'vertex')
        
        partition = partition_df.to_pandas()
        print(f"Found {partition['partition'].nunique()} communities")
        print(f"Modularity score: {modularity_score:.4f}")

        # Save community assignments
        os.makedirs("data/graph", exist_ok=True)
        partition.to_csv("data/graph/community_assignment.csv", index=False)
        print("Saved community assignments to data/graph/community_assignment.csv")

        community_sizes = partition['partition'].value_counts().sort_values(ascending=False)
        sorted_community_sizes = community_sizes.values.tolist()

    # Generate reports
    with open(f"{output_base_path}_report.txt", "w") as f:
        f.write(f"Graph Path: {dgl_path}\n")
        f.write(f"Total Nodes: {num_nodes}\n")
        f.write(f"Total Edges: {num_edges}\n\n")

        # Connected Components section
        f.write("=== Connected Components Analysis ===\n")
        f.write(f"Number of Connected Components: {len(size_counter)}\n")
        f.write("\nTop 10 Component Sizes:\n")
        for idx, size in enumerate(sorted_cc_sizes[:10]):
            f.write(f"  Component {idx + 1}: {size} nodes\n")
        if len(sorted_cc_sizes) > 10:
            f.write("  ...\n")
        f.write(f"\nLargest Connected Component: {largest_cc_size} nodes\n\n")

        # Community Detection section
        f.write("=== Community Detection Analysis ===\n")
        f.write(f"Number of Communities: {len(sorted_community_sizes)}\n")
        f.write(f"Modularity Score: {modularity_score:.4f}\n\n")
        f.write("Top 10 Community Sizes:\n")
        for idx, size in enumerate(sorted_community_sizes[:10]):
            f.write(f"  Community {idx + 1}: {size} nodes\n")
        if len(sorted_community_sizes) > 10:
            f.write("  ...\n")

    print(f"[Done] Analysis report saved to {output_base_path}_report.txt")

    # Generate plots
    # 1. Connected Components plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sorted_cc_sizes) + 1), sorted_cc_sizes, marker='o', linestyle='-')
    plt.yscale("log")
    plt.xlabel("Component Rank (Largest → Smallest)")
    plt.ylabel("Component Size (Number of Nodes)")
    plt.title("Connected Component Size Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_components.png")
    plt.close()

    # 2. Community size distribution plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sorted_community_sizes) + 1), sorted_community_sizes, marker='o', linestyle='-')
    plt.yscale("log")
    plt.xlabel("Community Rank (Largest → Smallest)")
    plt.ylabel("Number of Nodes")
    plt.title("Community Size Distribution (Louvain)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_base_path}_communities.png")
    plt.close()

    print(f"[Done] Plots saved to {output_base_path}_components.png and {output_base_path}_communities.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Graph analysis script')
    parser.add_argument('-t', '--threshold', type=str, default='0.94', help='Threshold for community detection')
    args = parser.parse_args()

    dgl_file = f"out/graph/graph_raw_{args.threshold}.dgl"
    output_base = f"out/graph/analysis_{args.threshold}"
    
    analyze_graph(dgl_file, output_base, use_cache=True, threshold=args.threshold) 