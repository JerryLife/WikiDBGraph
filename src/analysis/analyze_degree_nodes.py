"""
Analyze DGL graph to find highest degree node and lowest degree neighbors.

This script loads a DGL graph with a given threshold, finds the node with the 
highest degree, then among its neighbors finds the ones with the lowest degrees.
Finally, it prints their database IDs, names, and shared columns.
"""

import os
import json
import argparse
import dgl
import torch
from typing import Tuple, List, Set


def load_graph(threshold: str = "0.94") -> dgl.DGLGraph:
    """
    Load the DGL graph for the given threshold.
    
    Args:
        threshold: Similarity threshold used to build the graph
        
    Returns:
        DGL graph
    """
    graph_path = f"data/graph/graph_raw_{threshold}.dgl"
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    
    print(f"Loading graph from {graph_path}...")
    graphs, _ = dgl.load_graphs(graph_path)
    g = graphs[0]
    print(f"Loaded graph with {g.num_nodes()} nodes and {g.num_edges()} edges")
    return g


def get_database_name(db_id: int, schema_dir: str = "data/schema") -> str:
    """
    Get the database name for the given database ID.
    
    Args:
        db_id: Database ID (integer)
        schema_dir: Directory containing schema files
        
    Returns:
        Database name as a string
    """
    db_id_str = f"{db_id:05d}"  # Format as 5-digit string with leading zeros
    
    # Find the schema file for this database ID
    for filename in os.listdir(schema_dir):
        if filename.startswith(db_id_str + "_"):
            # Extract name from filename (remove .json extension and db_id prefix)
            name = filename[6:-5]  # Skip "00000_" and ".json"
            return name
    
    return f"Unknown (ID: {db_id})"


def load_schema(db_id: int, unzip_dir: str = "data/unzip") -> dict:
    """
    Load the schema for a given database.
    
    Args:
        db_id: Database ID (integer)
        unzip_dir: Directory containing unzipped database folders
        
    Returns:
        Schema dictionary
    """
    db_id_str = f"{db_id:05d}"  # Format as 5-digit string
    
    # Find the database folder
    for folder in os.listdir(unzip_dir):
        if folder.startswith(db_id_str + " "):
            schema_path = os.path.join(unzip_dir, folder, "schema.json")
            if os.path.exists(schema_path):
                with open(schema_path, "r", encoding="utf-8") as f:
                    return json.load(f)
    
    return {}


def get_all_columns(schema: dict) -> Set[str]:
    """
    Extract all column names from a schema.
    
    Args:
        schema: Schema dictionary
        
    Returns:
        Set of all column names (normalized to lowercase)
    """
    columns = set()
    
    for table in schema.get("tables", []):
        for column in table.get("columns", []):
            col_name = column.get("column_name", "")
            # Normalize: convert to lowercase and remove non-alphanumeric
            normalized = ''.join(c for c in col_name.lower() if c.isalnum())
            if normalized:
                columns.add(normalized)
    
    return columns


def find_shared_columns(db_id1: int, db_id2: int, unzip_dir: str = "data/unzip") -> List[str]:
    """
    Find shared columns between two databases.
    
    Args:
        db_id1: First database ID
        db_id2: Second database ID
        unzip_dir: Directory containing database folders
        
    Returns:
        List of shared column names (normalized)
    """
    schema1 = load_schema(db_id1, unzip_dir)
    schema2 = load_schema(db_id2, unzip_dir)
    
    columns1 = get_all_columns(schema1)
    columns2 = get_all_columns(schema2)
    
    shared = columns1 & columns2
    return sorted(list(shared))


def find_highest_and_lowest_degree_neighbors(g: dgl.DGLGraph, n_lowest: int = 2) -> Tuple[int, List[int]]:
    """
    Find the highest degree node and its n lowest degree neighbors.
    
    Args:
        g: DGL graph
        n_lowest: Number of lowest degree neighbors to return
        
    Returns:
        Tuple of (highest_degree_node_id, list of lowest_degree_neighbor_ids)
    """
    # Get degrees for all nodes
    # Since the graph is undirected (edges are bidirectional), use in_degrees
    degrees = g.in_degrees()
    
    # Find the node with the highest degree
    highest_degree_node = torch.argmax(degrees).item()
    highest_degree = degrees[highest_degree_node].item()
    
    print(f"\nHighest degree node: {highest_degree_node} with degree {highest_degree}")
    
    # Get neighbors of the highest degree node
    # Since edges are bidirectional, we can use either predecessors or successors
    neighbors = g.successors(highest_degree_node).numpy()
    
    if len(neighbors) == 0:
        raise ValueError(f"Node {highest_degree_node} has no neighbors!")
    
    print(f"Number of neighbors: {len(neighbors)}")
    
    # Find the n neighbors with the lowest degrees
    neighbor_degrees = degrees[neighbors]
    
    # Limit n_lowest to the actual number of neighbors
    n_lowest = min(n_lowest, len(neighbors))
    
    # Get indices of the n lowest degrees
    lowest_indices = torch.argsort(neighbor_degrees)[:n_lowest]
    lowest_degree_neighbors = neighbors[lowest_indices.numpy()]
    
    print(f"\nFound {n_lowest} lowest degree neighbor(s):")
    for i, neighbor_id in enumerate(lowest_degree_neighbors, 1):
        degree = degrees[neighbor_id].item()
        print(f"  {i}. Node {neighbor_id} with degree {degree}")
    
    return highest_degree_node, lowest_degree_neighbors.tolist()


def main(threshold: str = "0.94", schema_dir: str = "data/schema", unzip_dir: str = "data/unzip", n_lowest: int = 2):
    """
    Main function to analyze the graph.
    
    Args:
        threshold: Similarity threshold for the graph
        schema_dir: Directory containing schema files
        unzip_dir: Directory containing database folders
        n_lowest: Number of lowest degree neighbors to analyze
    """
    # Load the graph
    g = load_graph(threshold)
    
    # Find the highest degree node and its lowest degree neighbors
    highest_node, lowest_neighbors = find_highest_and_lowest_degree_neighbors(g, n_lowest)
    
    # Get database name for highest node
    highest_name = get_database_name(highest_node, schema_dir)
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print("\n1. Highest Degree Node:")
    print(f"   Database ID: {highest_node:05d}")
    print(f"   Database Name: {highest_name}")
    print(f"   Degree: {g.in_degrees()[highest_node].item()}")
    
    print(f"\n2. Lowest Degree Neighbor(s) ({len(lowest_neighbors)} total):")
    for i, lowest_neighbor in enumerate(lowest_neighbors, 1):
        lowest_name = get_database_name(lowest_neighbor, schema_dir)
        
        # Find shared columns
        print(f"\n   {i}. Neighbor Node:")
        print(f"      Database ID: {lowest_neighbor:05d}")
        print(f"      Database Name: {lowest_name}")
        print(f"      Degree: {g.in_degrees()[lowest_neighbor].item()}")
        
        print("\n      Finding shared columns with highest degree node...")
        shared_columns = find_shared_columns(highest_node, lowest_neighbor, unzip_dir)
        
        print(f"      Shared Columns ({len(shared_columns)} total):")
        if shared_columns:
            # Print first 20 shared columns
            for j, col in enumerate(shared_columns[:20], 1):
                print(f"         {j}. {col}")
            if len(shared_columns) > 20:
                print(f"         ... and {len(shared_columns) - 20} more")
        else:
            print("         No shared columns found")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze DGL graph to find highest degree node and lowest degree neighbors"
    )
    parser.add_argument(
        "-t", "--threshold",
        type=str,
        default="0.94",
        help="Similarity threshold for the graph (default: 0.94)"
    )
    parser.add_argument(
        "-n", "--n-lowest",
        type=int,
        default=2,
        help="Number of lowest degree neighbors to analyze (default: 2)"
    )
    parser.add_argument(
        "--schema-dir",
        type=str,
        default="data/schema",
        help="Directory containing schema files (default: data/schema)"
    )
    parser.add_argument(
        "--unzip-dir",
        type=str,
        default="data/unzip",
        help="Directory containing database folders (default: data/unzip)"
    )
    
    args = parser.parse_args()
    
    main(args.threshold, args.schema_dir, args.unzip_dir, args.n_lowest)

