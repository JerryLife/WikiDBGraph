import dgl
import os
import pandas as pd
import argparse
import cudf
import cugraph
import matplotlib.pyplot as plt
import collections

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
    
    # Unrenumber the vertex IDs back to original
    if G.renumbered:
        renumber_map_df = G.renumber_map.to_dataframe()
        components_df = components_df.reset_index(drop=True)
        components_df = components_df.merge(
            renumber_map_df,
            left_on='vertex',
            right_on=G.renumber_map.internal_col_name,
            how='left'
        )
    
    labels = components_df['labels'].to_pandas().values
    
    size_counter = collections.Counter(labels)
    sorted_cc_sizes = sorted(size_counter.values(), reverse=True)
    largest_cc_size = sorted_cc_sizes[0]

    # 2. Community Detection
    community_assignment_path = f"data/graph/community_assignment_{threshold}.csv"
    if use_cache and os.path.exists(community_assignment_path):
        print(f"Loading community assignments from cache: {community_assignment_path}")
        partition = pd.read_csv(community_assignment_path)
        community_sizes = partition['partition'].value_counts().sort_values(ascending=False)
        sorted_community_sizes = community_sizes.values.tolist()
        modularity_score = 0
        print(f"Found {partition['partition'].nunique()} communities from cache")
    else:
        print("Running Louvain community detection...")
        partition_df, modularity_score = cugraph.louvain(G)
        
        # Unrenumber the vertex IDs back to original
        if G.renumbered:
            renumber_map_df = G.renumber_map.to_dataframe()
            partition_df = partition_df.reset_index(drop=True)
            merged_df = partition_df.merge(
                renumber_map_df,
                left_on='vertex',
                right_on=G.renumber_map.internal_col_name,
                how='left'
            )
            merged_df[G.renumber_map.external_col_name] = merged_df[G.renumber_map.external_col_name].fillna(-1)
            partition = cudf.DataFrame({
                'node_id': merged_df[G.renumber_map.external_col_name].astype('int64'),
                'partition': merged_df['partition'].astype('int64')
            }).to_pandas()
        else:
            partition = cudf.DataFrame({
                'node_id': partition_df['vertex'],
                'partition': partition_df['partition']
            }).to_pandas()
        
        print(f"Found {partition['partition'].nunique()} communities")
        print(f"Modularity score: {modularity_score:.4f}")

        # Save community assignments
        os.makedirs("data/graph", exist_ok=True)
        partition.to_csv(community_assignment_path, index=False)
        print(f"Saved community assignments to {community_assignment_path}")

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

    dgl_file = f"data/graph/graph_raw_{args.threshold}.dgl"
    output_base = f"data/graph/analysis_{args.threshold}"
    
    analyze_graph(dgl_file, output_base, use_cache=True, threshold=args.threshold) 