"""
Build raw graph from filtered edges and embeddings.

This script loads filtered edges and node embeddings to construct a DGL graph
for further analysis.
"""

import os
import sys
import torch
import dgl
import numpy as np
from tqdm import tqdm
import argparse


def build_dgl_graph_from_pt(pt_file_path, device=None, num_nodes=100000):
    """
    Build a DGL graph from a PyTorch tensor file containing edge information.
    
    Args:
        pt_file_path (str): Path to the PyTorch tensor file
        device (str, optional): Device to build the graph on ('cpu', 'cuda', etc.)
        num_nodes (int, optional): Total number of nodes to ensure in the graph (default: 100000)
        
    Returns:
        dgl.DGLGraph: Constructed graph
    """
    tensor = torch.load(pt_file_path, map_location="cpu")
    print("Loaded tensor with shape", tensor.shape)

    valid_edges = tensor[tensor[:, 4] == 1]
    print("Valid edges:", valid_edges.shape)

    src = valid_edges[:, 0].long()
    dst = valid_edges[:, 1].long()
    weights = valid_edges[:, 2].to(device) if device else valid_edges[:, 2]

    if device:
        src = src.to(device)
        dst = dst.to(device)
        weights = weights.to(device)
    
    # Create graph with explicit number of nodes to ensure all nodes are included
    g = dgl.graph((src, dst), num_nodes=num_nodes, device=device)
    g.edata["weight"] = weights

    if 'label' in tensor.columns:
        labels = torch.tensor(tensor['label'].values, dtype=torch.float32)
        if device:
            labels = labels.to(device)
        g.edata["gt_edge"] = labels
        print(f"Added ground truth labels to edges")

    add_reverse = dgl.AddReverse(copy_edata=True)
    g = add_reverse(g)

    print(f"DGLGraph built with {g.num_nodes()} nodes and {g.num_edges()} edges on {g.device}.")

    return g


def build_dgl_graph_from_csv(csv_file_path, device=None, num_nodes=100000):
    """
    Build a DGL graph from a CSV file containing filtered edges.
    
    Args:
        csv_file_path (str): Path to the CSV file with filtered edges
        device (str, optional): Device to build the graph on ('cpu', 'cuda', etc.)
        num_nodes (int, optional): Total number of nodes to ensure in the graph (default: 100000)
        
    Returns:
        dgl.DGLGraph: Constructed graph
    """
    import pandas as pd
    
    df = pd.read_csv(csv_file_path)
    print(f"Loaded {len(df)} edges from CSV file")
    
    src = torch.tensor(df['src'].values, dtype=torch.int64)
    dst = torch.tensor(df['tgt'].values, dtype=torch.int64)
    weights = torch.tensor(df['similarity'].values, dtype=torch.float32)
    
    if device:
        src = src.to(device)
        dst = dst.to(device)
        weights = weights.to(device)
    
    # Create graph with explicit number of nodes to ensure all nodes are included
    g = dgl.graph((src, dst), num_nodes=num_nodes, device=device)
    g.edata["weight"] = weights

    # Add ground truth label if available
    if 'label' in df.columns:
        labels = torch.tensor(df['label'].values, dtype=torch.float32)
        if device:
            labels = labels.to(device)
        g.edata["gt_edge"] = labels
        print(f"Added ground truth labels to edges")
    
    # Add reverse edges with the same weights
    add_reverse = dgl.AddReverse(copy_edata=True)
    g = add_reverse(g)
    
    print(f"DGLGraph built with {g.num_nodes()} nodes and {g.num_edges()} edges on {g.device}.")
    
    return g


def add_node_embeddings(g, embeddings_file, device=None):
    """
    Add node embeddings to the graph.
    
    Args:
        g (dgl.DGLGraph): The graph to add embeddings to
        embeddings_file (str): Path to the embeddings file
        device (str, optional): Device to place embeddings on
        
    Returns:
        dgl.DGLGraph: Graph with node embeddings
    """
    embeddings = torch.load(embeddings_file, map_location="cpu", weights_only=False)
    print(f"Loaded embeddings with shape {embeddings.shape}")
    
    # Make sure we have embeddings for all nodes
    if embeddings.shape[0] < g.num_nodes():
        print(f"Warning: Embeddings only available for {embeddings.shape[0]} nodes, but graph has {g.num_nodes()} nodes")
        # Pad with zeros if needed
        padding = torch.zeros((g.num_nodes() - embeddings.shape[0], embeddings.shape[1]), dtype=embeddings.dtype)
        embeddings = torch.cat([embeddings, padding], dim=0)
    
    if device:
        embeddings = embeddings.to(device)
    
    g.ndata["embedding"] = embeddings
    print(f"Added node embeddings with dimension {embeddings.shape[1]}")
    
    return g


def save_dgl_graph(g, path):
    """
    Save a DGL graph to a file.
    
    Args:
        g (dgl.DGLGraph): The graph to save
        path (str): Path to save the graph
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Move to CPU before saving if on GPU
    if g.device != torch.device('cpu'):
        g = g.to('cpu')
    dgl.save_graphs(path, [g])
    print(f"DGLGraph saved to {path}")


def load_dgl_graph(path, device=None):
    """
    Load a DGL graph from a file.
    
    Args:
        path (str): Path to the saved graph
        device (str, optional): Device to load the graph to
        
    Returns:
        dgl.DGLGraph: Loaded graph
    """
    graphs, _ = dgl.load_graphs(path)
    g = graphs[0]
    if device:
        g = g.to(device)
    print(f"Loaded DGLGraph with {g.num_nodes()} nodes and {g.num_edges()} edges from {path} on {g.device}.")
    return g


def build_and_save_graph(edges_file, embeddings_file=None, output_path=None, device=None):
    """
    Build a graph from edges and embeddings and save it.
    
    Args:
        edges_file (str): Path to the edges file (.pt or .csv)
        embeddings_file (str, optional): Path to the embeddings file
        output_path (str, optional): Path to save the graph
        device (str, optional): Device to build the graph on ('cpu', 'cuda', etc.)
        
    Returns:
        dgl.DGLGraph: The constructed graph
    """
    # Determine file type and build graph
    if edges_file.endswith('.pt'):
        g = build_dgl_graph_from_pt(edges_file, device)
    elif edges_file.endswith('.csv'):
        g = build_dgl_graph_from_csv(edges_file, device)
    else:
        raise ValueError(f"Unsupported file format for edges: {edges_file}")
    
    # Add embeddings if provided
    if embeddings_file:
        g = add_node_embeddings(g, embeddings_file, device)
    
    # Save graph if output path is provided
    if output_path:
        save_dgl_graph(g, output_path)
    
    return g


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", type=str, default='0.94')
    args = parser.parse_args()
    # Example usage
    edges_file = f"data/graph/filtered_edges_threshold_{args.threshold}.csv"
    embeddings_file = "data/graph/database_embeddings.pt"
    output_path = f"data/graph/graph_raw_{args.threshold}.dgl"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    g = build_and_save_graph(edges_file, embeddings_file, output_path, device)
