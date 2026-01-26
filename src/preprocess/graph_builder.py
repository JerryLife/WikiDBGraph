"""
DGL graph construction from filtered edges and embeddings.

Builds graph structures for downstream analysis and GNN training.
"""

import os
import torch
import dgl
import pandas as pd
from typing import Optional


class GraphBuilder:
    """
    Build DGL graphs from filtered edges and node embeddings.
    
    Creates graph structures with node embeddings and edge weights
    for downstream analysis and GNN-based methods.
    
    Args:
        num_nodes: Total number of nodes in the graph (default: 100000)
        add_reverse: Whether to add reverse edges (default: True)
    
    Example:
        >>> builder = GraphBuilder(num_nodes=100000)
        >>> g = builder.build(
        ...     edges_file="data/graph/filtered_edges.csv",
        ...     embeddings_file="data/graph/all_embeddings.pt",
        ...     output_path="data/graph/graph.dgl"
        ... )
    """
    
    def __init__(
        self,
        num_nodes: int = 100000,
        add_reverse: bool = True
    ):
        """Initialize the graph builder."""
        self.num_nodes = num_nodes
        self.add_reverse = add_reverse
    
    def build_from_csv(
        self,
        edges_file: str,
        device: Optional[str] = None
    ) -> dgl.DGLGraph:
        """
        Build DGL graph from CSV edge file.
        
        Args:
            edges_file: Path to CSV with columns: src, tgt, similarity
            device: Device to build graph on (default: CPU)
        
        Returns:
            DGL graph with edge weights
        """
        df = pd.read_csv(edges_file)
        print(f"Loaded {len(df):,} edges from {edges_file}")
        
        src = torch.tensor(df['src'].values, dtype=torch.int64)
        dst = torch.tensor(df['tgt'].values, dtype=torch.int64)
        weights = torch.tensor(df['similarity'].values, dtype=torch.float32)
        
        if device:
            src = src.to(device)
            dst = dst.to(device)
            weights = weights.to(device)
        
        g = dgl.graph((src, dst), num_nodes=self.num_nodes, device=device)
        g.edata["weight"] = weights
        
        # Add ground truth labels if available
        if 'label' in df.columns:
            labels = torch.tensor(df['label'].values, dtype=torch.float32)
            if device:
                labels = labels.to(device)
            g.edata["gt_edge"] = labels
            print(f"Added ground truth labels to edges")
        
        # Add reverse edges
        if self.add_reverse:
            add_reverse_transform = dgl.AddReverse(copy_edata=True)
            g = add_reverse_transform(g)
        
        print(f"Built DGLGraph: {g.num_nodes():,} nodes, {g.num_edges():,} edges")
        return g
    
    def build_from_pt(
        self,
        pt_file: str,
        device: Optional[str] = None
    ) -> dgl.DGLGraph:
        """
        Build DGL graph from PyTorch tensor file.
        
        Expected format: tensor with columns [src, tgt, similarity, label, edge]
        
        Args:
            pt_file: Path to .pt file
            device: Device to build graph on
        
        Returns:
            DGL graph with edge weights
        """
        tensor = torch.load(pt_file, map_location="cpu")
        if isinstance(tensor, list):
            tensor = torch.tensor(tensor)
        print(f"Loaded tensor with shape {tensor.shape}")
        
        # Filter to valid edges
        valid_edges = tensor[tensor[:, 4] == 1]
        print(f"Valid edges: {len(valid_edges):,}")
        
        src = valid_edges[:, 0].long()
        dst = valid_edges[:, 1].long()
        weights = valid_edges[:, 2].float()
        
        if device:
            src = src.to(device)
            dst = dst.to(device)
            weights = weights.to(device)
        
        g = dgl.graph((src, dst), num_nodes=self.num_nodes, device=device)
        g.edata["weight"] = weights
        
        # Add labels if available
        if tensor.shape[1] > 3:
            labels = valid_edges[:, 3].float()
            if device:
                labels = labels.to(device)
            g.edata["gt_edge"] = labels
        
        if self.add_reverse:
            add_reverse_transform = dgl.AddReverse(copy_edata=True)
            g = add_reverse_transform(g)
        
        print(f"Built DGLGraph: {g.num_nodes():,} nodes, {g.num_edges():,} edges")
        return g
    
    def add_node_embeddings(
        self,
        g: dgl.DGLGraph,
        embeddings_file: str,
        device: Optional[str] = None
    ) -> dgl.DGLGraph:
        """
        Add node embeddings to an existing graph.
        
        Args:
            g: DGL graph to add embeddings to
            embeddings_file: Path to embeddings .pt file
            device: Device to place embeddings on
        
        Returns:
            Graph with node embeddings
        """
        embeddings_data = torch.load(embeddings_file, map_location="cpu", weights_only=False)
        
        if isinstance(embeddings_data, dict):
            embeddings = embeddings_data["embeddings"]
        else:
            embeddings = embeddings_data
        
        print(f"Loaded embeddings with shape {embeddings.shape}")
        
        # Pad embeddings if needed
        if embeddings.shape[0] < g.num_nodes():
            print(f"Warning: Padding embeddings from {embeddings.shape[0]} to {g.num_nodes()} nodes")
            padding = torch.zeros(
                (g.num_nodes() - embeddings.shape[0], embeddings.shape[1]),
                dtype=embeddings.dtype
            )
            embeddings = torch.cat([embeddings, padding], dim=0)
        
        if device:
            embeddings = embeddings.to(device)
        
        g.ndata["embedding"] = embeddings
        print(f"Added node embeddings with dimension {embeddings.shape[1]}")
        
        return g
    
    def build(
        self,
        edges_file: str,
        embeddings_file: Optional[str] = None,
        output_path: Optional[str] = None,
        device: Optional[str] = None
    ) -> dgl.DGLGraph:
        """
        Build a complete graph from edges and optionally embeddings.
        
        Args:
            edges_file: Path to edge file (.csv or .pt)
            embeddings_file: Optional path to embeddings file
            output_path: Optional path to save the graph
            device: Device to build on (default: CUDA if available)
        
        Returns:
            Complete DGL graph
        """
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Build base graph
        if edges_file.endswith('.pt'):
            g = self.build_from_pt(edges_file, device)
        else:
            g = self.build_from_csv(edges_file, device)
        
        # Add embeddings if provided
        if embeddings_file:
            g = self.add_node_embeddings(g, embeddings_file, device)
        
        # Save if output path provided
        if output_path:
            self.save(g, output_path)
        
        return g
    
    def save(self, g: dgl.DGLGraph, output_path: str) -> None:
        """
        Save a DGL graph to file.
        
        Args:
            g: DGL graph to save
            output_path: Path to save the graph
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        # Move to CPU before saving
        if g.device != torch.device('cpu'):
            g = g.to('cpu')
        
        dgl.save_graphs(output_path, [g])
        print(f"✅ Saved DGLGraph to {output_path}")
    
    @staticmethod
    def load(path: str, device: Optional[str] = None) -> dgl.DGLGraph:
        """
        Load a DGL graph from file.
        
        Args:
            path: Path to saved graph
            device: Device to load graph to
        
        Returns:
            Loaded DGL graph
        """
        graphs, _ = dgl.load_graphs(path)
        g = graphs[0]
        if device:
            g = g.to(device)
        print(f"Loaded DGLGraph: {g.num_nodes():,} nodes, {g.num_edges():,} edges from {path}")
        return g
    
    @classmethod
    def from_config(cls, config: "PreprocessConfig") -> "GraphBuilder":
        """Create a GraphBuilder from a PreprocessConfig."""
        from .config import PreprocessConfig
        return cls(num_nodes=config.num_nodes)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build DGL graph from edges and embeddings")
    parser.add_argument("--edges", type=str, required=True, help="Path to edges file")
    parser.add_argument("--embeddings", type=str, default=None, help="Path to embeddings file")
    parser.add_argument("--output", type=str, required=True, help="Output path for graph")
    parser.add_argument("--num-nodes", type=int, default=100000, help="Number of nodes")
    parser.add_argument("--no-reverse", action="store_true", help="Don't add reverse edges")
    
    args = parser.parse_args()
    
    builder = GraphBuilder(num_nodes=args.num_nodes, add_reverse=not args.no_reverse)
    builder.build(
        edges_file=args.edges,
        embeddings_file=args.embeddings,
        output_path=args.output
    )
