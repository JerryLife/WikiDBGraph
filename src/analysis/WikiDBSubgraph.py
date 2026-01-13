"""
WikiDBSubgraph: Dynamic subgraph construction with node/edge properties.

This module constructs a DGL subgraph for a given set of database IDs,
loading all node and edge properties from cached files.
"""

import os
import sys
import hashlib
import pickle
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
import pandas as pd

try:
    import dgl
except ImportError:
    dgl = None
    print("Warning: DGL not installed. Some features will be unavailable.")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WikiDBSubgraph:
    """
    Dynamic subgraph construction for WikiDB databases.
    
    Constructs a DGL graph for specified database IDs with all available
    node and edge properties from the graph analysis cache.
    """
    
    # Default cache directories
    GRAPH_DIR = "data/graph"
    CACHE_DIR = "data/graph/subgraph_cache"
    
    # Node property files
    NODE_STRUCTURAL_FILE = "node_structural_properties.csv"
    CLUSTER_ASSIGNMENTS_FILE = "cluster_assignments_dim2_sz100_msNone.csv"
    
    # Statistical property files (in data/data/graph/ziyangw)
    DATA_VOLUME_FILE = "data_volume.csv"
    COLUMN_CARDINALITY_FILE = "column_cardinality.csv"
    COLUMN_SPARSITY_FILE = "column_sparsity.csv"
    COLUMN_ENTROPY_FILE = "column_entropy.csv"
    
    # Edge property files
    EDGE_STRUCTURAL_FILE = "edge_structural_properties_GED_0.94.csv"
    EDGE_SIMILARITY_FILE = "filtered_edges_threshold_0.94.csv"
    
    def __init__(
        self, 
        graph_dir: str = None,
        cache_dir: str = None,
        stat_dir: str = None,
        device: str = "cpu"
    ):
        """
        Initialize the subgraph constructor.
        
        Args:
            graph_dir: Directory containing graph property files
            cache_dir: Directory to cache constructed subgraphs
            stat_dir: Directory containing statistical property files
            device: Device for graph tensors ('cpu' or 'cuda')
        """
        self.graph_dir = graph_dir or self.GRAPH_DIR
        self.cache_dir = cache_dir or self.CACHE_DIR
        self.stat_dir = stat_dir or os.path.join("data", "data", "graph", "ziyangw")
        self.device = device
        
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_key(self, db_ids: List[str]) -> str:
        """Generate a cache key for the given database IDs."""
        sorted_ids = sorted(db_ids)
        key_str = "_".join(sorted_ids)
        return hashlib.md5(key_str.encode()).hexdigest()[:12]
    
    def _get_cache_path(self, db_ids: List[str]) -> str:
        """Get the cache file path for the given database IDs."""
        cache_key = self._get_cache_key(db_ids)
        return os.path.join(self.cache_dir, f"subgraph_{cache_key}.pkl")
    
    def load_node_structural_properties(self) -> pd.DataFrame:
        """Load node structural properties."""
        path = os.path.join(self.graph_dir, self.NODE_STRUCTURAL_FILE)
        print(f"  [1/5] Loading node structural properties...")
        if not os.path.exists(path):
            print(f"    Warning: Not found: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        if 'db_id' in df.columns:
            df['db_id'] = df['db_id'].astype(str).str.zfill(5)
        print(f"    Loaded {len(df)} rows")
        return df
    
    def load_node_semantic_properties(self) -> pd.DataFrame:
        """Load node semantic properties (cluster assignments)."""
        path = os.path.join(self.graph_dir, self.CLUSTER_ASSIGNMENTS_FILE)
        print(f"  [2/5] Loading node semantic properties (clusters)...")
        if not os.path.exists(path):
            print(f"    Warning: Not found: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        if 'db_id' in df.columns:
            df['db_id'] = df['db_id'].astype(str).str.zfill(5)
        print(f"    Loaded {len(df)} rows")
        return df
    
    def load_node_statistical_properties(self) -> pd.DataFrame:
        """Load and aggregate node statistical properties."""
        print(f"  [3/5] Loading node statistical properties...")
        dfs = []
        
        # Data volume
        vol_path = os.path.join(self.graph_dir, self.DATA_VOLUME_FILE)
        print(f"    Loading data volume...")
        if os.path.exists(vol_path):
            df_vol = pd.read_csv(vol_path)
            if 'db_id' in df_vol.columns:
                df_vol['db_id'] = df_vol['db_id'].astype(str).str.zfill(5)
                dfs.append(df_vol[['db_id', 'volume_bytes']].rename(
                    columns={'volume_bytes': 'data_volume'}
                ))
                print(f"      Loaded {len(df_vol)} rows")
        
        # Cardinality (average per db)
        card_path = os.path.join(self.graph_dir, self.COLUMN_CARDINALITY_FILE)
        print(f"    Loading column cardinality...")
        if os.path.exists(card_path):
            df_card = pd.read_csv(card_path)
            print(f"      Loaded {len(df_card)} rows, aggregating...")
            if 'db_id' in df_card.columns:
                df_card['db_id'] = df_card['db_id'].astype(str).str.zfill(5)
                avg_card = df_card.groupby('db_id')['n_distinct'].mean().reset_index()
                avg_card = avg_card.rename(columns={'n_distinct': 'avg_cardinality'})
                dfs.append(avg_card)
        
        # Sparsity (average per db)
        sparse_path = os.path.join(self.graph_dir, self.COLUMN_SPARSITY_FILE)
        print(f"    Loading column sparsity...")
        if os.path.exists(sparse_path):
            df_sparse = pd.read_csv(sparse_path)
            print(f"      Loaded {len(df_sparse)} rows, aggregating...")
            if 'db_id' in df_sparse.columns:
                df_sparse['db_id'] = df_sparse['db_id'].astype(str).str.zfill(5)
                avg_sparse = df_sparse.groupby('db_id')['sparsity'].mean().reset_index()
                avg_sparse = avg_sparse.rename(columns={'sparsity': 'avg_sparsity'})
                dfs.append(avg_sparse)
        
        # Entropy (average per db)
        entropy_path = os.path.join(self.graph_dir, self.COLUMN_ENTROPY_FILE)
        print(f"    Loading column entropy...")
        if os.path.exists(entropy_path):
            df_entropy = pd.read_csv(entropy_path, encoding='utf-8-sig')
            print(f"      Loaded {len(df_entropy)} rows, aggregating...")
            if 'db_id' in df_entropy.columns:
                df_entropy['db_id'] = df_entropy['db_id'].astype(str).str.zfill(5)
                df_entropy['entropy'] = pd.to_numeric(df_entropy['entropy'], errors='coerce')
                avg_entropy = df_entropy.dropna(subset=['entropy']).groupby('db_id')['entropy'].mean().reset_index()
                avg_entropy = avg_entropy.rename(columns={'entropy': 'avg_entropy'})
                dfs.append(avg_entropy)
        
        if not dfs:
            return pd.DataFrame()
        
        # Merge all statistics
        result = dfs[0]
        for df in dfs[1:]:
            result = result.merge(df, on='db_id', how='outer')
        
        return result
    
    def load_edge_structural_properties(self) -> pd.DataFrame:
        """Load edge structural properties (Jaccard, GED)."""
        path = os.path.join(self.graph_dir, self.EDGE_STRUCTURAL_FILE)
        print(f"  [4/5] Loading edge structural properties...")
        if not os.path.exists(path):
            print(f"    Warning: Not found: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        # Normalize db_id columns - handle float format like 26218.0
        if 'db_id1' in df.columns:
            df['db_id1'] = pd.to_numeric(df['db_id1'], errors='coerce').fillna(0).astype(int).astype(str).str.zfill(5)
        if 'db_id2' in df.columns:
            df['db_id2'] = pd.to_numeric(df['db_id2'], errors='coerce').fillna(0).astype(int).astype(str).str.zfill(5)
        print(f"    Loaded {len(df)} edges")
        return df
    
    def load_edge_similarity_properties(self) -> pd.DataFrame:
        """Load edge embedding similarity."""
        path = os.path.join(self.graph_dir, self.EDGE_SIMILARITY_FILE)
        print(f"  [5/5] Loading edge similarity properties...")
        if not os.path.exists(path):
            print(f"    Warning: Not found: {path}")
            return pd.DataFrame()
        
        df = pd.read_csv(path)
        # Normalize src/tgt columns - handle float format like 26218.0
        if 'src' in df.columns:
            df['db_id1'] = df['src'].astype(float).astype(int).astype(str).str.zfill(5)
        if 'tgt' in df.columns:
            df['db_id2'] = df['tgt'].astype(float).astype(int).astype(str).str.zfill(5)
        print(f"    Loaded {len(df)} edges")
        return df
    
    def construct_subgraph(
        self, 
        db_ids: List[str],
        include_node_props: bool = True,
        include_edge_props: bool = True,
        fully_connected: bool = True
    ) -> Dict:
        """
        Construct a subgraph for the given database IDs.
        
        Args:
            db_ids: List of database IDs to include
            include_node_props: Whether to load node properties
            include_edge_props: Whether to load edge properties
            fully_connected: If True, create edges between all nodes when
                           target DBs aren't found in edge file
            
        Returns:
            Dictionary with graph structure and properties
        """
        # Normalize db_ids to 5-digit format
        db_ids = [str(db_id).zfill(5) for db_id in db_ids]
        n_nodes = len(db_ids)
        
        # Create node ID mapping
        node_id_map = {db_id: idx for idx, db_id in enumerate(db_ids)}
        
        # Load edge properties to find edges
        edge_sim_df = self.load_edge_similarity_properties()
        edge_struct_df = self.load_edge_structural_properties()
        
        # Find edges within subgraph
        edges_src = []
        edges_dst = []
        edge_features = {}
        
        found_edges = False
        if not edge_sim_df.empty:
            # Filter to edges within subgraph
            mask = (edge_sim_df['db_id1'].isin(db_ids)) & (edge_sim_df['db_id2'].isin(db_ids))
            sub_edges = edge_sim_df[mask]
            
            if len(sub_edges) > 0:
                found_edges = True
                for _, row in sub_edges.iterrows():
                    src_idx = node_id_map[row['db_id1']]
                    dst_idx = node_id_map[row['db_id2']]
                    
                    # Add bidirectional edges
                    edges_src.extend([src_idx, dst_idx])
                    edges_dst.extend([dst_idx, src_idx])
        
        # If no edges found and fully_connected is True, create fully connected graph
        if not found_edges and fully_connected:
            print(f"    Warning: Target DBs not in edge file. Creating fully connected graph.")
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    edges_src.extend([i, j])
                    edges_dst.extend([j, i])
        
        # Load node properties
        node_props = {}
        if include_node_props:
            # Structural
            struct_df = self.load_node_structural_properties()
            if not struct_df.empty:
                struct_df = struct_df[struct_df['db_id'].isin(db_ids)]
                for col in ['num_tables', 'num_columns', 'foreign_key_density']:
                    if col in struct_df.columns:
                        values = np.zeros(n_nodes)
                        for _, row in struct_df.iterrows():
                            if row['db_id'] in node_id_map:
                                idx = node_id_map[row['db_id']]
                                values[idx] = float(row[col]) if pd.notna(row[col]) else 0
                        node_props[col] = values
            
            # Statistical
            stat_df = self.load_node_statistical_properties()
            if not stat_df.empty:
                stat_df = stat_df[stat_df['db_id'].isin(db_ids)]
                for col in ['data_volume', 'avg_cardinality', 'avg_sparsity', 'avg_entropy']:
                    if col in stat_df.columns:
                        values = np.zeros(n_nodes)
                        for _, row in stat_df.iterrows():
                            if row['db_id'] in node_id_map:
                                idx = node_id_map[row['db_id']]
                                values[idx] = float(row[col]) if pd.notna(row[col]) else 0
                        node_props[col] = values
            
            # Semantic (cluster)
            sem_df = self.load_node_semantic_properties()
            if not sem_df.empty:
                sem_df = sem_df[sem_df['db_id'].isin(db_ids)]
                cluster_col = 'cluster' if 'cluster' in sem_df.columns else None
                if cluster_col:
                    values = np.zeros(n_nodes)
                    for _, row in sem_df.iterrows():
                        if row['db_id'] in node_id_map:
                            idx = node_id_map[row['db_id']]
                            values[idx] = float(row[cluster_col]) if pd.notna(row[cluster_col]) else -1
                    node_props['cluster'] = values
        
        # Load edge properties
        edge_props = {}
        if include_edge_props and edges_src:
            # Build edge lookup
            edge_lookup = {}
            for i, (s, d) in enumerate(zip(edges_src, edges_dst)):
                edge_lookup[(s, d)] = i
            
            n_edges = len(edges_src)
            
            # Similarity from edge_sim_df
            if found_edges and not edge_sim_df.empty and 'similarity' in edge_sim_df.columns:
                sim_values = np.zeros(n_edges)
                mask = (edge_sim_df['db_id1'].isin(db_ids)) & (edge_sim_df['db_id2'].isin(db_ids))
                for _, row in edge_sim_df[mask].iterrows():
                    s = node_id_map[row['db_id1']]
                    d = node_id_map[row['db_id2']]
                    val = float(row['similarity']) if pd.notna(row['similarity']) else 0
                    if (s, d) in edge_lookup:
                        sim_values[edge_lookup[(s, d)]] = val
                    if (d, s) in edge_lookup:
                        sim_values[edge_lookup[(d, s)]] = val
                edge_props['similarity'] = sim_values
            else:
                # Default uniform similarity for fully connected graph
                edge_props['similarity'] = np.ones(n_edges)
            
            # Structural from edge_struct_df
            if found_edges and not edge_struct_df.empty:
                for col in ['jaccard_columns', 'jaccard_table_names']:
                    if col in edge_struct_df.columns:
                        values = np.zeros(n_edges)
                        mask = (edge_struct_df['db_id1'].isin(db_ids)) & (edge_struct_df['db_id2'].isin(db_ids))
                        for _, row in edge_struct_df[mask].iterrows():
                            s = node_id_map.get(row['db_id1'])
                            d = node_id_map.get(row['db_id2'])
                            if s is not None and d is not None:
                                val = float(row[col]) if pd.notna(row[col]) else 0
                                if (s, d) in edge_lookup:
                                    values[edge_lookup[(s, d)]] = val
                                if (d, s) in edge_lookup:
                                    values[edge_lookup[(d, s)]] = val
                        edge_props[col] = values
            else:
                # Default uniform weights for fully connected graph
                edge_props['jaccard_columns'] = np.ones(n_edges) * 0.5
                edge_props['jaccard_table_names'] = np.ones(n_edges) * 0.5
        
        result = {
            'db_ids': db_ids,
            'node_id_map': node_id_map,
            'n_nodes': n_nodes,
            'n_edges': len(edges_src),
            'edges_src': edges_src,
            'edges_dst': edges_dst,
            'node_props': node_props,
            'edge_props': edge_props
        }
        
        return result
    
    def build_dgl_graph(self, subgraph_data: Dict):
        """
        Build a DGL graph from subgraph data.
        
        Args:
            subgraph_data: Dictionary from construct_subgraph()
            
        Returns:
            DGL graph with node and edge features
        """
        if dgl is None:
            raise ImportError("DGL is required for build_dgl_graph()")
        
        edges_src = subgraph_data['edges_src']
        edges_dst = subgraph_data['edges_dst']
        n_nodes = subgraph_data['n_nodes']
        
        # Create graph
        if edges_src:
            g = dgl.graph((edges_src, edges_dst), num_nodes=n_nodes)
        else:
            g = dgl.graph(([], []), num_nodes=n_nodes)
        
        # Add node features
        for name, values in subgraph_data['node_props'].items():
            g.ndata[name] = torch.tensor(values, dtype=torch.float32)
        
        # Add edge features
        for name, values in subgraph_data['edge_props'].items():
            g.edata[name] = torch.tensor(values, dtype=torch.float32)
        
        g = g.to(self.device)
        return g
    
    def load_or_construct(
        self, 
        db_ids: List[str],
        use_cache: bool = True,
        include_node_props: bool = True,
        include_edge_props: bool = True,
        fully_connected: bool = True
    ) -> Dict:
        """
        Load subgraph from cache or construct it.
        
        Args:
            db_ids: List of database IDs
            use_cache: Whether to use cached subgraph
            include_node_props: Whether to include node properties
            include_edge_props: Whether to include edge properties
            fully_connected: Create fully connected graph if DBs not in edge file
            
        Returns:
            Dictionary with subgraph data
        """
        cache_path = self._get_cache_path(db_ids)
        
        if use_cache and os.path.exists(cache_path):
            print(f"Loading cached subgraph from: {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        print(f"Constructing subgraph for {len(db_ids)} databases...")
        subgraph_data = self.construct_subgraph(
            db_ids,
            include_node_props=include_node_props,
            include_edge_props=include_edge_props,
            fully_connected=fully_connected
        )
        
        # Cache the result
        if use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(subgraph_data, f)
            print(f"Cached subgraph to: {cache_path}")
        
        return subgraph_data
    
    def get_node_feature_tensor(self, subgraph_data: Dict) -> torch.Tensor:
        """
        Stack all node properties into a single feature tensor.
        
        Returns:
            Tensor of shape (n_nodes, n_features)
        """
        if not subgraph_data['node_props']:
            return torch.zeros(subgraph_data['n_nodes'], 0)
        
        features = []
        for name in sorted(subgraph_data['node_props'].keys()):
            feat = torch.tensor(subgraph_data['node_props'][name], dtype=torch.float32)
            features.append(feat.unsqueeze(1))
        
        return torch.cat(features, dim=1)
    
    def get_edge_feature_tensor(self, subgraph_data: Dict) -> torch.Tensor:
        """
        Stack all edge properties into a single feature tensor.
        
        Returns:
            Tensor of shape (n_edges, n_features)
        """
        n_edges = subgraph_data['n_edges']
        if not subgraph_data['edge_props'] or n_edges == 0:
            return torch.zeros(n_edges, 0)
        
        features = []
        for name in sorted(subgraph_data['edge_props'].keys()):
            feat = torch.tensor(subgraph_data['edge_props'][name], dtype=torch.float32)
            features.append(feat.unsqueeze(1))
        
        return torch.cat(features, dim=1)


def main():
    """Test the WikiDBSubgraph class."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test WikiDBSubgraph construction")
    parser.add_argument(
        "--db_ids",
        type=str,
        default="54379,37176,85770,50469",
        help="Comma-separated database IDs"
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Don't use cached subgraph"
    )
    
    args = parser.parse_args()
    
    db_ids = [db_id.strip() for db_id in args.db_ids.split(",")]
    
    print(f"Constructing subgraph for databases: {db_ids}")
    
    subgraph = WikiDBSubgraph()
    data = subgraph.load_or_construct(db_ids, use_cache=not args.no_cache)
    
    print(f"\n=== Subgraph Summary ===")
    print(f"Nodes: {data['n_nodes']}")
    print(f"Edges: {data['n_edges']}")
    print(f"Node properties: {list(data['node_props'].keys())}")
    print(f"Edge properties: {list(data['edge_props'].keys())}")
    
    # Get feature tensors
    node_feat = subgraph.get_node_feature_tensor(data)
    edge_feat = subgraph.get_edge_feature_tensor(data)
    
    print(f"\nNode feature tensor shape: {node_feat.shape}")
    print(f"Edge feature tensor shape: {edge_feat.shape}")
    
    # Try building DGL graph if available
    if dgl is not None:
        g = subgraph.build_dgl_graph(data)
        print(f"\nDGL Graph: {g}")
        print(f"DGL Node features: {list(g.ndata.keys())}")
        print(f"DGL Edge features: {list(g.edata.keys())}")


if __name__ == "__main__":
    main()
