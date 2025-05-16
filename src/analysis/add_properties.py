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

import dgl
import torch

BASE_DIR = "data/graph"

def load_graph(path: str, device: str = "cpu") -> dgl.DGLGraph:
    """
    Load a graph from a file and print the basic graph information.
    """
    g = dgl.load_graphs(path)[0][0]
    g = g.to(device)
    print(f"Loaded graph with {g.num_nodes()} nodes and {g.num_edges()} edges")
    return g

def print_basic_info(g: dgl.DGLGraph):
    """
    Print the basic graph information.
    """
    print(f"Number of nodes: {g.num_nodes()}")
    print(f"Number of edges: {g.num_edges()}")
    print(f"Node types: {g.ntypes}")
    print(f"Edge types: {g.etypes}")
    print(f"In-degree min, max, mean: {torch.min(g.in_degrees().float()):.2f}, {torch.max(g.in_degrees().float()):.2f}, {torch.mean(g.in_degrees().float()):.2f}")
    print(f"Out-degree min, max, mean: {torch.min(g.out_degrees().float()):.2f}, {torch.max(g.out_degrees().float()):.2f}, {torch.mean(g.out_degrees().float()):.2f}")
    print(f"Is multigraph: {g.is_multigraph}")
    print(f"Is homogeneous: {g.is_homogeneous}")
    
    # Node properties
    print(f"Node features: {g.ndata.keys()}")
    print(f"Node feature shapes: {g.ndata.values()}")
    
    # Edge properties
    print(f"Edge features: {g.edata.keys()}")
    print(f"Edge feature shapes: {g.edata.values()}")
    




if __name__ == "__main__":
    g = load_graph(os.path.join(BASE_DIR, "graph_raw.dgl"), device="cuda:4")
    print_basic_info(g)




