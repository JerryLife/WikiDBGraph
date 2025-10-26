import networkx as nx
import pandas as pd
import os

def create_overlap_graph(seed):
    """Create graph from pairwise stats for a single seed"""
    # Initialize empty graph
    G = nx.Graph()
    
    # Read pairwise stats for this seed
    df = pd.read_csv(f'out/pairwise_stats_seed{seed}.csv')
    
    # Add edges for overlapping pairs
    for _, row in df.iterrows():
        if row['overlap_features'] > 0:
            # Add edge with overlap count as weight
            G.add_edge(row['db1'], row['db2'], weight=row['overlap_features'])
    
    return G

def save_graph(G, seed):
    """Save graph to GEXF format"""
    os.makedirs('out/graphs', exist_ok=True)
    nx.write_gexf(G, f'out/graphs/overlap_graph_seed{seed}.gexf')

def process_seed(seed):
    """Process a single seed's data into graph"""
    print(f"Processing seed {seed}...")
    G = create_overlap_graph(seed)
    print(f"\nGraph stats for seed {seed}: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, avg_degree={sum(dict(G.degree()).values())/G.number_of_nodes():.2f}")
    save_graph(G, seed)
    return G


if __name__ == "__main__":
    # Process each seed
    for seed in range(10):
        G = process_seed(seed)
