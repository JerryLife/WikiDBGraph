import dgl
import os
import pandas as pd
import argparse
import cudf
import cugraph
import matplotlib.pyplot as plt
import collections
import numpy as np
import cupy as cp
import networkx as nx


def analyze_components_and_plot(dgl_path, output_txt_path, output_plot_path):
    print(f"Loading DGL graph from {dgl_path}...")
    g = dgl.load_graphs(dgl_path)[0][0]
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()

    src, dst = g.edges()
    src = src.numpy()
    dst = dst.numpy()

    df = cudf.DataFrame({'src': src, 'dst': dst})
    G = cugraph.Graph()
    G.from_cudf_edgelist(df, source='src', destination='dst', renumber=True)

    print("Running connected components...")
    components_df = cugraph.connected_components(G)
    labels = components_df['labels'].to_pandas().values

    size_counter = collections.Counter(labels)
    sorted_sizes = sorted(size_counter.values(), reverse=True)

    largest_cc_size = sorted_sizes[0]

    with open(output_txt_path, "w") as f:
        f.write(f"Graph Path: {dgl_path}\n")
        f.write(f"Total Nodes: {num_nodes}\n")
        f.write(f"Total Edges: {num_edges}\n")
        f.write(f"Connected Components: {len(size_counter)}\n\n")

        f.write("Top 10 Component Sizes:\n")
        for idx, size in enumerate(sorted_sizes[:10]):
            f.write(f"  Component {idx + 1}: {size} nodes\n")
        if len(sorted_sizes) > 10:
            f.write("  ...\n")
        f.write("\nLargest Connected Component:\n")
        f.write(f"  Nodes: {largest_cc_size}\n")

    print(f"[Done] Info saved to {output_txt_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sorted_sizes) + 1), sorted_sizes, marker='o', linestyle='-')
    plt.yscale("log")
    plt.xlabel("Component Rank (Largest → Smallest)")
    plt.ylabel("Component Size (Number of Nodes)")
    plt.title("Connected Component Size Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"[Plot saved] {output_plot_path}")
    plt.close()


def detect_communities_louvain(dgl_path, output_txt_path, output_plot_path, output_graph_plot_path, use_cache=False, threshold='0.94', top_n_communities_to_visualize=10):
    """
    Detects communities using Louvain and generates visualizations.
    Args:
        dgl_path (str): Path to the saved DGL graph.
        output_txt_path (str): Path to save the text summary.
        output_plot_path (str): Path to save the histogram of community sizes.
        output_graph_plot_path (str): Path to save the graph layout plot of top N communities.
        use_cache (bool): Whether to use cached community assignments.
        threshold (str): Threshold info (used for logging, not directly in Louvain here).
        top_n_communities_to_visualize (int): Number of largest communities to visualize in the graph plot.
    """
    community_assignments_csv_path = "data/graph/community_assignments_louvain.csv"
    os.makedirs("data/graph", exist_ok=True)

    if use_cache and os.path.exists(community_assignments_csv_path):
        print(f"Loading community assignments from cache: {community_assignments_csv_path}")
        partition_df_pd = pd.read_csv(community_assignments_csv_path) # Expects 'node_id', 'community_id'
        modularity_score = 0  # Modularity not typically stored with just assignments
        print(f"Found {partition_df_pd['community_id'].nunique()} communities from cache for {len(partition_df_pd)} nodes.")

        print(f"Loading DGL graph from {dgl_path} for stats...")
        g_list = dgl.load_graphs(dgl_path)
        g = g_list[0][0]
        num_nodes = g.num_nodes()
        num_edges = g.num_edges()

        node_to_community = pd.Series(partition_df_pd.community_id.values, index=partition_df_pd.node_id.values)
        all_dgl_node_ids = np.arange(num_nodes)
        community_labels_for_plot = node_to_community.reindex(all_dgl_node_ids, fill_value=-1).values

    else:
        print(f"Loading DGL graph from {dgl_path} for community detection...")
        g_list = dgl.load_graphs(dgl_path)
        g = g_list[0][0]
        num_nodes = g.num_nodes()
        num_edges = g.num_edges()

        if num_nodes == 0 or num_edges == 0:
            print("Graph is empty. Skipping community detection and visualization.")
            with open(output_txt_path, "w") as f:
                f.write(f"Graph Path: {dgl_path}\nTotal Nodes: {num_nodes}\nTotal Edges: {num_edges}\nGraph is empty.\n")
            for plot_path in [output_plot_path, output_graph_plot_path]:
                plt.figure(figsize=(8,8))
                plt.text(0.5, 0.5, "Graph is empty", ha='center', va='center', fontsize=16)
                plt.savefig(plot_path, dpi=100)
                plt.close()
            return

        print("Preparing graph for cuGraph...")
        src, dst = g.edges()
        src_np = src.cpu().numpy()
        dst_np = dst.cpu().numpy()

        df_for_cugraph = cudf.DataFrame()
        df_for_cugraph['src'] = src_np
        df_for_cugraph['dst'] = dst_np
        
        G_cugraph = cugraph.Graph(directed=False) # Louvain typically works on undirected graphs
        # When renumber=True, cuGraph builds an internal mapping from 'src'/'dst' values to contiguous integers.
        G_cugraph.from_cudf_edgelist(df_for_cugraph, source='src', destination='dst', renumber=True)

        print("Running Louvain community detection with cuGraph...")
        louvain_parts_df, modularity_score = cugraph.louvain(G_cugraph) # louvain_parts_df has 'vertex' and 'partition'
        
        print(f"Found {louvain_parts_df['partition'].nunique()} communities")
        print(f"Modularity score: {modularity_score:.4f}")

        # --- Modified Unrenumbering ---
        print("Unrenumbering community assignments...")
        if G_cugraph.renumbered:
            # Get the renumbering map from cuGraph
            # This DataFrame typically has 'internal_vertex_id' and 'external_vertex_id'
            renumber_map_df = G_cugraph.renumber_map.to_dataframe()
            
            # Merge louvain_parts_df (which has internal 'vertex' IDs) with the renumbering map
            # Ensure column names for merging are correct.
            # louvain_parts_df has 'vertex' (internal IDs from cuGraph's perspective)
            # renumber_map_df has 'internal_vertex_id' and 'external_vertex_id' (original DGL node IDs)
            
            # Ensure louvain_parts_df has a clean index before merging
            louvain_parts_df = louvain_parts_df.reset_index(drop=True)
            
            merged_df = louvain_parts_df.merge(
                renumber_map_df,
                left_on='vertex', # Internal ID column from Louvain output
                right_on=G_cugraph.renumber_map.internal_col_name, # Internal ID column in the map
                how='left' # Keep all Louvain assignments
            )
            
            # The 'external_vertex_id' column now contains the original DGL node IDs
            # Handle cases where a vertex might not be in the renumber map (should not happen if map is complete)
            merged_df[G_cugraph.renumber_map.external_col_name] = merged_df[G_cugraph.renumber_map.external_col_name].fillna(-1)

            partition_df_pd = cudf.DataFrame({
                'node_id': merged_df[G_cugraph.renumber_map.external_col_name].astype('int64'), # Original DGL Node IDs
                'community_id': merged_df['partition'].astype('int64')      # Community IDs
            }).to_pandas()
        else:
            # If graph was not renumbered, 'vertex' column in louvain_parts_df already contains original IDs
            print("Graph was not renumbered by cuGraph. Using 'vertex' as original node IDs.")
            partition_df_pd = cudf.DataFrame({
                'node_id': louvain_parts_df['vertex'],
                'community_id': louvain_parts_df['partition']
            }).to_pandas()
        # --- End of Modified Unrenumbering ---

        partition_df_pd.sort_values(by='node_id').to_csv(community_assignments_csv_path, index=False)
        print(f"Saved community assignments to {community_assignments_csv_path}")

        node_to_community = pd.Series(partition_df_pd.community_id.values, index=partition_df_pd.node_id.values)
        all_dgl_node_ids = np.arange(num_nodes)
        community_labels_for_plot = node_to_community.reindex(all_dgl_node_ids, fill_value=-1).values

    # --- Summary Text File ---
    community_series_for_stats = pd.Series(community_labels_for_plot)
    community_sizes = community_series_for_stats[community_series_for_stats != -1].value_counts().sort_values(ascending=False)
    
    sorted_sizes = community_sizes.values.tolist()
    num_detected_communities = len(sorted_sizes)

    with open(output_txt_path, "w") as f:
        f.write(f"Graph Path: {dgl_path}\n")
        f.write(f"Threshold for graph construction (info): {threshold}\n")
        f.write(f"Total Nodes: {num_nodes}\n")
        f.write(f"Total Edges (in DGL graph): {num_edges}\n")
        f.write(f"Detected Communities: {num_detected_communities}\n")
        f.write(f"Modularity Score: {modularity_score:.4f} {'(Not available from cache)' if use_cache and modularity_score == 0 else ''}\n\n")
        f.write(f"Top 10 Community Sizes (excluding unassigned if any):\n")
        for idx, size in enumerate(sorted_sizes[:10]):
            f.write(f"  Community Rank {idx + 1}: {size} nodes\n")
        if len(sorted_sizes) > 10:
            f.write("  ...\n")
    print(f"[Done] Community info saved to {output_txt_path}")

    # --- Histogram of Community Sizes ---
    if sorted_sizes:
        plt.figure(figsize=(6, 5))
        # Ensure min_size is at least 1 for log scale if all sizes are > 0
        positive_sizes = [s for s in sorted_sizes if s > 0]
        if not positive_sizes:
            print("No communities with size > 0. Skipping histogram.")
        else:
            min_size = min(positive_sizes) 
            max_size = max(positive_sizes)
            
            # Create logarithmically spaced bins
            bins = np.logspace(np.log10(min_size), np.log10(max_size), 20)
            
            plt.hist(positive_sizes, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
            plt.xscale('log')
            plt.yscale('log') 
            
            plt.xlabel("Community Size", fontsize=16)
            plt.ylabel("Frequency", fontsize=16)
            plt.title(f"Distribution of {num_detected_communities} Community Sizes", fontsize=18)
            plt.tight_layout()
            plt.savefig(output_plot_path, dpi=600)
            print(f"[Plot saved] Community size distribution plot saved to {output_plot_path}")
            plt.close()
    else:
        print("No communities detected or all communities are empty, skipping histogram plot.")

    # --- Visualize the largest N communities ---
    print(f"Preparing to visualize the largest {top_n_communities_to_visualize} communities...")
    if num_detected_communities == 0:
        print(f"No communities detected. Skipping visualization of top {top_n_communities_to_visualize} communities.")
        plt.figure(figsize=(10, 10))
        plt.text(0.5, 0.5, f"No communities detected.", ha='center', va='center')
        plt.savefig(output_graph_plot_path, dpi=300)
        plt.close()
        return

    top_n_ids_actual = community_sizes.index[:top_n_communities_to_visualize].tolist()
    print(f"Actual top community IDs selected for visualization: {top_n_ids_actual}")

    nodes_in_top_n_mask = np.isin(community_labels_for_plot, top_n_ids_actual)
    nodes_to_visualize_original_ids = np.arange(num_nodes)[nodes_in_top_n_mask]

    if len(nodes_to_visualize_original_ids) == 0:
        print(f"No nodes found belonging to the top {top_n_communities_to_visualize} communities. Skipping graph plot.")
        plt.figure(figsize=(10, 10))
        plt.text(0.5, 0.5, f"No nodes in top {top_n_communities_to_visualize} communities.", ha='center', va='center')
        plt.savefig(output_graph_plot_path, dpi=300)
        plt.close()
        return

    print(f"Creating subgraph for {len(nodes_to_visualize_original_ids)} nodes from top {len(top_n_ids_actual)} communities...")
    # Create subgraph from the original DGL graph 'g'
    sub_g_dgl = dgl.node_subgraph(g.to('cpu'), nodes_to_visualize_original_ids, store_ids=True)
    print(f"Subgraph created with {sub_g_dgl.num_nodes()} nodes and {sub_g_dgl.num_edges()} edges.")

    if sub_g_dgl.num_nodes() == 0:
        print("Subgraph is empty. Skipping visualization.")
        plt.figure(figsize=(10, 10))
        plt.text(0.5, 0.5, "Subgraph for top communities is empty.", ha='center', va='center')
        plt.savefig(output_graph_plot_path, dpi=300)
        plt.close()
        return

    nx_sub_g = dgl.to_networkx(sub_g_dgl)
    print(f"Converted DGL subgraph to NetworkX graph.")

    # Get community labels for the nodes in the subgraph
    # The nodes in nx_sub_g are indexed 0 to sub_g_dgl.num_nodes()-1.
    # Their original DGL IDs are in sub_g_dgl.ndata[dgl.NID]
    original_ids_in_subgraph = sub_g_dgl.ndata[dgl.NID].cpu().numpy()
    subgraph_node_community_labels = community_labels_for_plot[original_ids_in_subgraph]
    
    num_unique_communities_in_subgraph = len(np.unique(subgraph_node_community_labels))

    print("Calculating node positions for subgraph (this may take a while)...")
    try:
        if nx_sub_g.number_of_nodes() > 500:
             pos = nx.spring_layout(nx_sub_g, k=0.1, iterations=20, seed=42)
        elif nx_sub_g.number_of_nodes() > 0 : # Ensure graph is not empty for layout
             pos = nx.spring_layout(nx_sub_g, k=0.15, iterations=50, seed=42)
        else: # Should be caught by earlier check, but as a safeguard
            pos = {} 
    except Exception as e:
        print(f"Could not compute spring_layout for subgraph, falling back to random_layout: {e}")
        pos = nx.random_layout(nx_sub_g, seed=42) if nx_sub_g.number_of_nodes() > 0 else {}

    plt.figure(figsize=(15, 15)) # Increased figure size for potentially complex subgraphs
    
    print("Drawing subgraph nodes...")
    node_size_val = 50
    if nx_sub_g.number_of_nodes() > 0:
        node_size_val = max(10, min(100, 5000 // nx_sub_g.number_of_nodes()))


    nx.draw_networkx_nodes(nx_sub_g, pos, 
                           node_color=subgraph_node_community_labels,
                           cmap=plt.cm.get_cmap('viridis', num_unique_communities_in_subgraph if num_unique_communities_in_subgraph > 0 else 1),
                           node_size=node_size_val, 
                           alpha=0.85,
                           linewidths=0.1,
                           edgecolors='face'
                           )
    
    print("Drawing subgraph edges...")
    edge_alpha_val = 0.2
    edge_width_val = 0.5
    if nx_sub_g.number_of_edges() > 0: # Avoid division by zero
        edge_alpha_val = max(0.01, 0.3 - nx_sub_g.number_of_edges()/max(1,50000)) # Dynamic alpha
        edge_width_val = max(0.02, 0.5 - nx_sub_g.number_of_edges()/max(1,20000))# Dynamic width


    nx.draw_networkx_edges(nx_sub_g, pos, 
                           alpha=edge_alpha_val, 
                           width=edge_width_val 
                           )

    plt.title(f"Top {len(top_n_ids_actual)} Largest Communities (Louvain)", fontsize=22)
    plt.xlabel(f"Subgraph: {nx_sub_g.number_of_nodes()} nodes, {nx_sub_g.number_of_edges()} edges. Modularity (full graph): {modularity_score:.4f} {'(N/A from cache)' if use_cache and modularity_score == 0 else ''}", fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_graph_plot_path, dpi=200, bbox_inches='tight') # Adjusted DPI
    print(f"[Plot saved] Graph visualization of top {len(top_n_ids_actual)} communities saved to {output_graph_plot_path}")
    plt.close()
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Graph analysis script')
    parser.add_argument('-t', '--threshold', type=str, default='0.94', help='Threshold for community detection')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='GPU ID')
    args = parser.parse_args()

    dgl_file = f"data/graph/graph_raw_{args.threshold}.dgl"
    out_txt = f"data/graph/community_assignments_louvain_{args.threshold}.txt"
    out_plot = f"fig/community_size_distribution_louvain_{args.threshold}.png"
    out_graph_plot = f"fig/community_visualization_louvain_{args.threshold}.png"

    cp.cuda.Device(args.gpu).use()

    detect_communities_louvain(dgl_file, out_txt, out_plot, out_graph_plot, use_cache=True, threshold=args.threshold, top_n_communities_to_visualize=1)
