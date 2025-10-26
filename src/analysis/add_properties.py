"""
Graph Attributes:

Node Properties:
* Structural Properties:
    - Number of tables
    - Number of columns
    - Proportion of data types (categorical, numerical etc.)
    - Foreign key density (number of foreign keys / number of columns)
    - Average table connectivity (number of potential joins / number of tables)
* Semantic Properties:
    - Database Embedding (we already have this)
    - Topic (e.g. medical, financial, etc., can be derived by clustering database embeddings)
* Statistical Properties:
    - Data volume (file size)
    - All-join size (number of rows when joining all tables with foreign keys)
    - Average/Median Column Cardinality (the average or median number of distinct values across columns)
    - Average/Median Column Sparsity (number of columns with nulls / total number of columns)
    - Average/Median Column Entropy (measure of randomness in the column)
    
Edge Properties:
* Structural Properties:
    - Jaccard index of set of table names (number of overlapping table names / total number of table names)
    - Jaccard index of set of columns (number of overlapping columns / total number of columns)
    - Jaccard index of set of data types (number of overlapping data types / total number of data types)
    - Similarity of internal graph structure (use graph matching; node: table, edge: foreign key)
* Semantic Properties:
    - Similarity of database embedding (we already have this)
    - Confidence of similarity
* Statistical Properties:
    - Divergence of distribution for shared columns
    - Ratio of overlap for shared columns
    
"""

import os
import sys
import argparse
import pandas as pd

import dgl
import torch
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import collections

BASE_DIR = "data/graph"
PLOT_DIR = "fig"

def load_graph(path: str, device: str = "cpu") -> dgl.DGLGraph:
    """
    Load a graph from a file and print the basic graph information.
    
    Args:
        path: Path to the graph file
        device: Device to load the graph to ('cpu' or 'cuda:N')
    """
    g = dgl.load_graphs(path)[0][0]
    g = g.to(device)
    print(f"Loaded graph with {g.num_nodes()} nodes and {g.num_edges()} edges on {device}")
    return g

def print_basic_info(g: dgl.DGLGraph, use_cache: bool = False):
    """
    Print the basic graph information.
    """
    print(f"Number of nodes: {g.num_nodes()}")
    print(f"Number of bidirectional edges: {g.num_edges() // 2}")  # Divide by 2 since edges are bidirectional
    print(f"Node types: {g.ntypes}")
    print(f"Edge types: {g.etypes}")
    print(f"In-degree min, max, mean, median: {torch.min(g.in_degrees().float()):.2f}, {torch.max(g.in_degrees().float()):.2f}, {torch.mean(g.in_degrees().float()):.2f}, {torch.median(g.in_degrees().float()):.2f}")
    print(f"Out-degree min, max, mean, median: {torch.min(g.out_degrees().float()):.2f}, {torch.max(g.out_degrees().float()):.2f}, {torch.mean(g.out_degrees().float()):.2f}, {torch.median(g.out_degrees().float()):.2f}")
    print(f"Is multigraph: {g.is_multigraph}")
    print(f"Is homogeneous: {g.is_homogeneous}")
    
    # Node properties
    print(f"Node features: {g.ndata.keys()}")
    print(f"Node feature shapes: {g.ndata.values()}")
    
    # Edge properties
    print(f"Edge features: {g.edata.keys()}")
    print(f"Edge feature shapes: {g.edata.values()}")

    # Plot the distribution of node degrees
    plt.figure(figsize=(6, 5))
    in_degrees = g.in_degrees().numpy()
    plt.hist(in_degrees, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    plt.yscale('log')
    plt.xlabel("Node Degree", fontsize=16)
    plt.ylabel("Frequency", fontsize=16)
    plt.title("Node Degree Distribution", fontsize=18)
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"degree_distribution_{args.threshold}.png"), dpi=600)
    print(f"[Plot saved] Degree distribution plot saved")
    plt.close()

    
    connected_components_file = os.path.join(BASE_DIR, f"connected_components_{args.threshold}.csv")
    if use_cache and os.path.exists(connected_components_file):
        print(f"Connected components file already exists: {connected_components_file}")
        # Read the CSV file and convert to list of lists format
        connected_components_df = pd.read_csv(connected_components_file, header=None, names=['node', 'component_id'])
        # Group by component_id and convert to list of lists
        connected_components = [group['node'].tolist() for _, group in connected_components_df.groupby('component_id')]
        connected_components_sizes = [len(component) for component in connected_components]
    else:
        # Convert to NetworkX undirected graph for connected components
        nx_graph = g.to("cpu").to_networkx().to_undirected()
        connected_components = list(nx.connected_components(nx_graph))
        connected_components_sizes = [len(component) for component in connected_components]
    
    n_isolated_nodes = np.count_nonzero(np.array(connected_components_sizes) == 1)
    print(f"Number of connected components: {len(connected_components)}")
    print(f"Size of connected components (min, max, mean, median): {np.min(connected_components_sizes)}, {np.max(connected_components_sizes)}, {np.mean(connected_components_sizes)}, {np.median(connected_components_sizes)}")
    print(f"Number of isolated nodes: {n_isolated_nodes}")

    # Plot the distribution of connected components sizes
    plt.figure(figsize=(6, 5))
    sorted_sizes = sorted(connected_components_sizes, reverse=True)
    bins = np.logspace(np.log10(min(sorted_sizes)), np.log10(max(sorted_sizes)), 20)
    plt.hist(sorted_sizes, bins=bins, alpha=0.7, color='steelblue', edgecolor='black')
    plt.xscale('log')
    plt.yscale("log")
    plt.xlabel("Connected Component Size (Number of Nodes)", fontsize=16)
    plt.ylabel("Frequency (Number of Components)", fontsize=16)
    plt.title("Connected Component Size Distribution", fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"component_size_distribution_{args.threshold}.png"), dpi=600)
    print(f"[Plot saved] Component size distribution plot saved")
    plt.close()
    
    

    if use_cache:
        # save connected components assigned to each node to csv
        with open(connected_components_file, "w") as f:
            for i, component in enumerate(connected_components):
                for node in component:
                    f.write(f"{node}, {i}\n")



# Constants for filenames, assuming cache_file is the directory path
# These filenames are based on the provided context and scripts.



def read_edge_structure_properties(cache_file: str):
    """
    Read the edge structure properties from the cache file (directory).
    File: edge_structural_properties_GED_0.94.csv
    Expected Columns: db_id1, db_id2, jaccard_table_names, jaccard_columns, jaccard_data_types, graph_edit_distance, ...
    """
    file_path = os.path.join(cache_file, EDGE_STRUCTURAL_FILE)
    if not os.path.exists(file_path):
        print(f"Warning: Edge structural properties file not found: {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print(f"Error reading edge structural properties from {file_path}: {e}")
        return pd.DataFrame()

def read_node_structure_properties(cache_file: str):
    """
    Read the node structure properties from the cache file (directory).
    File: node_structural_properties.csv
    Expected Columns: db_id, num_tables, num_columns, foreign_key_density, ...
    """
    file_path = os.path.join(cache_file, NODE_STRUCTURAL_FILE)
    if not os.path.exists(file_path):
        print(f"Warning: Node structural properties file not found: {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        if 'db_id' in df.columns:
            df = df.rename(columns={'db_id': 'node_id'})
        return df
    except Exception as e:
        print(f"Error reading node structural properties from {file_path}: {e}")
        return pd.DataFrame()

def read_node_semantic_properties(cache_file: str):
    """
    Read the node semantic properties from cache files (directory).
    Properties:
        - Cluster assignments from cluster_assignments_dim2_sz100_msNone.csv (db_id, cluster)
        - Community assignments from community_assignment.csv (vertex, partition)
        - Database embeddings from database_embeddings.pt (Tensor)
    Returns:
        Tuple (pd.DataFrame, torch.Tensor or None):
            - DataFrame with columns ['node_id', 'cluster_id', 'community_id'] (outer merged)
            - Tensor of embeddings (num_nodes, embedding_dim) or None if not found.
              The order of embeddings in the tensor needs to be mapped to node_ids by the caller.
    """
    # Cluster assignments
    cluster_file_path = os.path.join(cache_file, CLUSTER_ASSIGNMENTS_FILE)
    df_clusters = pd.DataFrame()
    if os.path.exists(cluster_file_path):
        try:
            df_clusters_raw = pd.read_csv(cluster_file_path)
            if 'db_id' in df_clusters_raw.columns and 'cluster' in df_clusters_raw.columns:
                df_clusters = df_clusters_raw.rename(columns={'db_id': 'node_id', 'cluster': 'cluster_id'})[['node_id', 'cluster_id']]
            else:
                print(f"Warning: Expected columns 'db_id', 'cluster' not found in {cluster_file_path}. Got {df_clusters_raw.columns.tolist()}")
        except Exception as e:
            print(f"Error reading cluster assignments from {cluster_file_path}: {e}")
    else:
        print(f"Warning: Cluster assignments file not found: {cluster_file_path}")

    # Community assignments
    community_file_path = os.path.join(cache_file, COMMUNITY_ASSIGNMENTS_FILE)
    df_communities = pd.DataFrame()
    if os.path.exists(community_file_path):
        try:
            df_communities_raw = pd.read_csv(community_file_path)
            if 'vertex' in df_communities_raw.columns and 'partition' in df_communities_raw.columns:
                df_communities = df_communities_raw.rename(columns={'vertex': 'node_id', 'partition': 'community_id'})[['node_id', 'community_id']]
            else:
                print(f"Warning: Expected columns 'vertex', 'partition' not found in {community_file_path}. Got {df_communities_raw.columns.tolist()}")
        except Exception as e:
            print(f"Error reading community assignments from {community_file_path}: {e}")
    else:
        print(f"Warning: Community assignments file not found: {community_file_path}")

    # Merge cluster and community assignments
    merged_df = pd.DataFrame()
    if not df_clusters.empty and not df_communities.empty:
        merged_df = pd.merge(df_clusters, df_communities, on='node_id', how='outer')
    elif not df_clusters.empty:
        merged_df = df_clusters
    elif not df_communities.empty:
        merged_df = df_communities
    
    # Database embeddings
    embeddings_tensor = None
    embeddings_file_path = os.path.join(cache_file, DB_EMBEDDINGS_FILE)
    if os.path.exists(embeddings_file_path):
        try:
            embeddings_tensor = torch.load(embeddings_file_path)
        except Exception as e:
            print(f"Error loading embeddings from {embeddings_file_path}: {e}")
    else:
        print(f"Warning: Database embeddings file not found: {embeddings_file_path}")
        
    return merged_df, embeddings_tensor

def read_edge_semantic_properties(cache_file_similarity: str, cache_file_confidence: str):
    """
    Read edge semantic properties from cache files (directory).
    Properties:
        - Embedding Similarity from filtered_edges_threshold_0.94.csv (src, tgt, similarity)
        - Similarity Confidence from filtered_edges_0.94_with_confidence.csv (src, tgt, confidence)
    Returns:
        pd.DataFrame with columns ['db_id1', 'db_id2', 'similarity', 'confidence'] (outer merged)
    """
    # Embedding Similarity
    embed_sim_file_path = os.path.join(cache_file_similarity, EDGE_EMBED_SIM_FILE)
    df_embed_sim = pd.DataFrame()
    if os.path.exists(embed_sim_file_path):
        try:
            df_embed_sim_raw = pd.read_csv(embed_sim_file_path)
            if {'src', 'tgt', 'similarity'}.issubset(df_embed_sim_raw.columns):
                # Rename columns to match expected format
                df_embed_sim = df_embed_sim_raw.rename(columns={'src': 'db_id1', 'tgt': 'db_id2'})[['db_id1', 'db_id2', 'similarity']]
            else:
                print(f"Warning: Expected columns 'src', 'tgt', 'similarity' not in {embed_sim_file_path}. Got {df_embed_sim_raw.columns.tolist()}")
        except Exception as e:
            print(f"Error reading edge embedding similarity from {embed_sim_file_path}: {e}")
    else:
        print(f"Warning: Edge embedding similarity file not found: {embed_sim_file_path}")

    # Similarity Confidence
    sim_conf_file_path = os.path.join(cache_file_confidence, EDGE_SIM_CONF_FILE)
    df_sim_conf = pd.DataFrame()
    if os.path.exists(sim_conf_file_path):
        try:
            df_sim_conf_raw = pd.read_csv(sim_conf_file_path)
            if {'src', 'tgt', 'confidence'}.issubset(df_sim_conf_raw.columns):
                # Rename columns to match expected format
                df_sim_conf = df_sim_conf_raw.rename(columns={'src': 'db_id1', 'tgt': 'db_id2'})[['db_id1', 'db_id2', 'confidence']]
            else:
                print(f"Warning: Expected columns 'src', 'tgt', 'confidence' not in {sim_conf_file_path}. Got {df_sim_conf_raw.columns.tolist()}")
        except Exception as e:
            print(f"Error reading similarity confidence from {sim_conf_file_path}: {e}")
    else:
        print(f"Warning: Similarity confidence file not found: {sim_conf_file_path}")

    # Merge properties
    merged_df = pd.DataFrame()
    if not df_embed_sim.empty and not df_sim_conf.empty:
        merged_df = pd.merge(df_embed_sim, df_sim_conf, on=['db_id1', 'db_id2'], how='outer')
    elif not df_embed_sim.empty:
        merged_df = df_embed_sim
    elif not df_sim_conf.empty:
        merged_df = df_sim_conf
        
    return merged_df

def read_node_statistical_properties(cache_file: str):
    """
    Read and compute node statistical properties from cache files (directory).
    Properties: DataVol, AllJoinSize, AvgCard, AvgSparsity, AvgEntropy.
    These are read from respective CSV files and aggregated per node (db_id).
    Returns:
        pd.DataFrame with 'node_id' and columns for each statistical property (left merged).
    """
    dfs_to_merge = []
    
    datavol_path = os.path.join(cache_file, DATA_VOLUME_FILE)
    if os.path.exists(datavol_path):
        try:
            df_datavol = pd.read_csv(datavol_path)
            if 'db_id' in df_datavol.columns and 'volume_bytes' in df_datavol.columns:
                 df_datavol = df_datavol.rename(columns={'db_id': 'node_id', 'volume_bytes': 'DataVol'})[['node_id', 'DataVol']]
                 dfs_to_merge.append(df_datavol)
            else: print(f"Warning: Expected columns for DataVol not found in {datavol_path}. Got {df_datavol.columns.tolist()}")
        except Exception as e: print(f"Error reading data volume from {datavol_path}: {e}")
    else: print(f"Warning: Data volume file not found: {datavol_path}")

    joinsize_path = os.path.join(cache_file, ALL_JOIN_SIZE_FILE)
    if os.path.exists(joinsize_path):
        try:
            df_joinsize = pd.read_csv(joinsize_path)
            if 'db_id' in df_joinsize.columns and 'all_join_size' in df_joinsize.columns:
                df_joinsize = df_joinsize.rename(columns={'db_id': 'node_id', 'all_join_size': 'AllJoinSize'})[['node_id', 'AllJoinSize']]
                dfs_to_merge.append(df_joinsize)
            else: print(f"Warning: Expected columns for AllJoinSize not found in {joinsize_path}. Got {df_joinsize.columns.tolist()}")
        except Exception as e: print(f"Error reading all join size from {joinsize_path}: {e}")
    else: print(f"Warning: All join size file not found: {joinsize_path}")

    card_path = os.path.join(cache_file, COLUMN_CARDINALITY_FILE)
    if os.path.exists(card_path):
        try:
            df_card = pd.read_csv(card_path)
            if 'db_id' in df_card.columns and 'n_distinct' in df_card.columns:
                avg_card = df_card.groupby('db_id')['n_distinct'].mean().reset_index()
                avg_card = avg_card.rename(columns={'db_id': 'node_id', 'n_distinct': 'AvgCard'})
                dfs_to_merge.append(avg_card)
            else: print(f"Warning: Expected columns for AvgCard not found in {card_path}. Got {df_card.columns.tolist()}")
        except Exception as e: print(f"Error processing column cardinality from {card_path}: {e}")
    else: print(f"Warning: Column cardinality file not found: {card_path}")

    sparsity_path = os.path.join(cache_file, COLUMN_SPARSITY_FILE)
    if os.path.exists(sparsity_path):
        try:
            df_sparsity = pd.read_csv(sparsity_path)
            if 'db_id' in df_sparsity.columns and 'sparsity' in df_sparsity.columns:
                avg_sparsity = df_sparsity.groupby('db_id')['sparsity'].mean().reset_index()
                avg_sparsity = avg_sparsity.rename(columns={'db_id': 'node_id', 'sparsity': 'AvgSparsity'})
                dfs_to_merge.append(avg_sparsity)
            else: print(f"Warning: Expected columns for AvgSparsity not found in {sparsity_path}. Got {df_sparsity.columns.tolist()}")
        except Exception as e: print(f"Error processing column sparsity from {sparsity_path}: {e}")
    else: print(f"Warning: Column sparsity file not found: {sparsity_path}")

    entropy_path = os.path.join(cache_file, COLUMN_ENTROPY_FILE)
    if os.path.exists(entropy_path):
        try:
            df_entropy = pd.read_csv(entropy_path, encoding="utf-8-sig")
            if 'db_id' in df_entropy.columns and 'entropy' in df_entropy.columns:
                df_entropy['entropy'] = pd.to_numeric(df_entropy['entropy'], errors='coerce')
                avg_entropy = df_entropy.dropna(subset=['entropy']).groupby('db_id')['entropy'].mean().reset_index()
                avg_entropy = avg_entropy.rename(columns={'db_id': 'node_id', 'entropy': 'AvgEntropy'})
                dfs_to_merge.append(avg_entropy)
            else: print(f"Warning: Expected columns for AvgEntropy not found in {entropy_path}. Got {df_entropy.columns.tolist()}")
        except Exception as e: print(f"Error processing column entropy from {entropy_path}: {e}")
    else: print(f"Warning: Column entropy file not found: {entropy_path}")

    if not dfs_to_merge:
        return pd.DataFrame()

    all_node_ids = pd.Series(dtype='object') # Ensure dtype compatibility or handle specific types
    for df_prop in dfs_to_merge:
        if 'node_id' in df_prop.columns:
            # Ensure node_id is of a consistent type for concatenation, e.g., string
            all_node_ids = pd.concat([all_node_ids, df_prop['node_id'].astype(str)]).drop_duplicates().reset_index(drop=True)
    
    if all_node_ids.empty:
        return pd.DataFrame()

    merged_df = pd.DataFrame({'node_id': all_node_ids})

    for df_prop in dfs_to_merge:
        if 'node_id' in df_prop.columns and not df_prop.empty:
            # Ensure node_id is of the same type for merging
            df_prop_copy = df_prop.copy()
            df_prop_copy['node_id'] = df_prop_copy['node_id'].astype(str)
            merged_df = pd.merge(merged_df, df_prop_copy, on='node_id', how='left')
    return merged_df

def read_edge_statistical_properties(cache_file: str):
    """
    Read edge statistical properties from the cache file (directory).
    File: distdiv_results.csv
    Expected Properties: distdiv, overlap_ratio
    Returns:
        pd.DataFrame with ['db_id1', 'db_id2', 'distdiv', 'overlap_ratio']
    """
    file_path = os.path.join(cache_file, EDGE_DISTDIV_FILE)
    if not os.path.exists(file_path):
        print(f"Warning: Edge statistical properties (distdiv) file not found: {file_path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(file_path)
        # Check if we have src/tgt columns instead of db_id1/db_id2
        if {'src', 'tgt', 'distdiv', 'overlap_ratio'}.issubset(df.columns):
            df = df.rename(columns={'src': 'db_id1', 'tgt': 'db_id2'})
            return df[['db_id1', 'db_id2', 'distdiv', 'overlap_ratio']]
        elif {'db_id1', 'db_id2', 'distdiv', 'overlap_ratio'}.issubset(df.columns):
            return df[['db_id1', 'db_id2', 'distdiv', 'overlap_ratio']]
        else:
            print(f"Warning: Expected columns not found in {file_path}. Available: {df.columns.tolist()}")
            # Return available columns if key identifiers are present
            if ('src' in df.columns and 'tgt' in df.columns) or ('db_id1' in df.columns and 'db_id2' in df.columns):
                # Select existing expected columns plus keys
                cols_to_return = ['distdiv', 'overlap_ratio']
                if 'src' in df.columns:
                    df = df.rename(columns={'src': 'db_id1', 'tgt': 'db_id2'})
                cols_to_return = ['db_id1', 'db_id2'] + [col for col in cols_to_return if col in df.columns]
                return df[list(dict.fromkeys(cols_to_return))] # Unique columns, preserving order
            return pd.DataFrame()
    except Exception as e:
        print(f"Error reading edge statistical properties from {file_path}: {e}")
        return pd.DataFrame()

def combine_into_graph(g: dgl.DGLGraph, node_properties: dict, edge_properties: dict) -> dgl.DGLGraph:
    """
    Combine various properties into the graph.
    
    Args:
        g: DGL graph to add properties to
        node_properties: Dictionary of node property DataFrames with 'node_id' as key column
        edge_properties: Dictionary of edge property DataFrames with 'db_id1'/'db_id2' as key columns
    
    Returns:
        DGL graph with properties added
    """
    device = g.device
    print(f"Adding properties to graph with {g.num_nodes()} nodes and {g.num_edges()} edges on {device}")
    
    # Add node properties
    # Create node ID mapping from node indices
    node_ids = torch.arange(g.num_nodes(), device=device)
    g.ndata['nid'] = node_ids
    node_id_map = {i: i for i in range(g.num_nodes())}
    
    for prop_name, prop_df in node_properties.items():
        if prop_df.empty:
            print(f"Skipping empty node property: {prop_name}")
            continue
            
        print(f"Adding node property: {prop_name} with shape {prop_df.shape}")
        
        # Get property columns (excluding node_id)
        prop_cols = [col for col in prop_df.columns if col != 'node_id']
        
        for col in prop_cols:
            # Initialize tensor with NaNs
            prop_tensor = torch.full((g.num_nodes(),), float('nan'), dtype=torch.float32, device=device)
            
            # Map values from DataFrame to tensor
            for _, row in prop_df.iterrows():
                if pd.notna(row[col]):
                    node_idx = int(row['node_id'])
                    if node_idx in node_id_map:
                        try:
                            # Special handling for data_type_proportions column
                            if col == 'data_type_proportions':
                                # Convert string representation of dict to actual dict
                                if isinstance(row[col], str):
                                    try:
                                        data_types = eval(row[col])
                                        # Extract specific type proportions
                                        if 'string' in data_types:
                                            prop_tensor[node_idx] = float(data_types['string'])
                                    except:
                                        print(f"Warning: Could not parse data_type_proportions value: {row[col]}")
                                else:
                                    # If it's already a dict
                                    if 'string' in row[col]:
                                        prop_tensor[node_idx] = float(row[col]['string'])
                            else:
                                prop_tensor[node_idx] = float(row[col])
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Could not convert {col} value '{row[col]}' to float: {e}")
            
            # Add property to graph
            prop_name_col = f"{prop_name}_{col}" if len(prop_cols) > 1 else prop_name
            g.ndata[prop_name_col] = prop_tensor
            print(f"  - Added {prop_name_col} to graph.ndata")
    
    # Add edge properties
    for prop_name, prop_df in edge_properties.items():
        if prop_df.empty:
            print(f"Skipping empty edge property: {prop_name}")
            continue
            
        print(f"Adding edge property: {prop_name} with shape {prop_df.shape}")
        
        # Get property columns (excluding db_id1 and db_id2)
        prop_cols = [col for col in prop_df.columns if col not in ['db_id1', 'db_id2']]
        
        for col in prop_cols:
            # Initialize tensor with NaNs
            prop_tensor = torch.full((g.num_edges(),), float('nan'), dtype=torch.float32, device=device)
            
            # Create a mapping of node id pairs to edge indices
            src_nodes = g.edges()[0].cpu().numpy()
            dst_nodes = g.edges()[1].cpu().numpy()
            
            # Create a mapping from (src_id, dst_id) to edge index
            edge_map = {}
            for i, (src, dst) in enumerate(zip(src_nodes, dst_nodes)):
                edge_map[(int(src), int(dst))] = i
            
            # Map values from DataFrame to tensor
            for _, row in prop_df.iterrows():
                src_id = int(row['db_id1'])
                dst_id = int(row['db_id2'])
                
                if (src_id, dst_id) in edge_map:
                    edge_idx = edge_map[(src_id, dst_id)]
                    try:
                        prop_tensor[edge_idx] = float(row[col])
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Could not convert {col} value '{row[col]}' to float: {e}")
            
            # Add property to graph
            prop_name_col = f"{prop_name}_{col}" if len(prop_cols) > 1 else prop_name
            g.edata[prop_name_col] = prop_tensor
            print(f"  - Added {prop_name_col} to graph.edata")
    
    return g

if __name__ == "__main__":
        
    # Node properties
    NODE_STRUCTURAL_FILE = "node_structural_properties.csv"
    CLUSTER_ASSIGNMENTS_FILE = "cluster_assignments_dim2_sz100_msNone.csv"
    COMMUNITY_ASSIGNMENTS_FILE = "community_assignment.csv"
    DB_EMBEDDINGS_FILE = "database_embeddings.pt"

    DATA_VOLUME_FILE = "data_volume.csv"
    ALL_JOIN_SIZE_FILE = "all_join_size_results.csv"
    COLUMN_CARDINALITY_FILE = "column_cardinality.csv"
    COLUMN_SPARSITY_FILE = "column_sparsity.csv"
    COLUMN_ENTROPY_FILE = "column_entropy.csv"

    # Edge properties
    EDGE_STRUCTURAL_FILE = "edge_structural_properties_GED_0.94.csv"
    EDGE_EMBED_SIM_FILE = "filtered_edges_threshold_0.94.csv"
    EDGE_SIM_CONF_FILE = "filtered_edges_0.94_with_confidence.csv"
    EDGE_DISTDIV_FILE = "distdiv_results.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", type=str, default='0.94')
    parser.add_argument("-o", "--output_file", type=str, default=None)
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Device to use (cpu or cuda:N)")
    args = parser.parse_args()
    
    # Set output file if not provided
    if args.output_file is None:
        args.output_file = os.path.join(BASE_DIR, f"graph_with_properties_{args.threshold}.dgl")
    
    # Load the graph
    g = load_graph(os.path.join(BASE_DIR, f"graph_raw_{args.threshold}.dgl"), device=args.device)
    # print_basic_info(g, use_cache=False)
    
    # Collect node properties
    print("\nCollecting node properties...")
    node_struct_props = read_node_structure_properties("data/graph")
    print(f"Node structural properties: {len(node_struct_props)} rows")
    
    node_semantic_props, node_embeddings = read_node_semantic_properties("data/graph")
    print(f"Node semantic properties: {len(node_semantic_props)} rows")
    
    node_stat_props = read_node_statistical_properties("data/data/graph/ziyangw")
    print(f"Node statistical properties: {len(node_stat_props)} rows")
    
    # Collect edge properties
    print("\nCollecting edge properties...")
    edge_struct_props = read_edge_structure_properties("data/graph")
    print(f"Edge structural properties: {len(edge_struct_props)} rows")
    
    edge_semantic_props = read_edge_semantic_properties("data/graph", "data/data/graph/ziyangw")
    print(f"Edge semantic properties: {len(edge_semantic_props)} rows")
    
    edge_stat_props = read_edge_statistical_properties("data/data/graph/ziyangw")
    print(f"Edge statistical properties: {len(edge_stat_props)} rows")
    
    # Prepare property dictionaries
    node_properties = {
        'structural': node_struct_props,
        'semantic': node_semantic_props,
        'statistical': node_stat_props
    }
    
    edge_properties = {
        'structural': edge_struct_props,
        'semantic': edge_semantic_props,
        'statistical': edge_stat_props
    }
    
    # Add node embeddings directly if available
    if node_embeddings is not None:
        print(f"Adding node embeddings of shape {node_embeddings.shape}")
        g.ndata['embedding'] = node_embeddings.to(args.device)
    
    # Combine properties into graph
    g = combine_into_graph(g, node_properties, edge_properties)
    
    # Save graph with properties
    print(f"\nSaving graph with properties to {args.output_file}")
    dgl.save_graphs(args.output_file, [g])
    print(f"Graph saved successfully with {g.num_nodes()} nodes and {g.num_edges()} edges")
    print(f"Node features: {list(g.ndata.keys())}")
    print(f"Edge features: {list(g.edata.keys())}")




