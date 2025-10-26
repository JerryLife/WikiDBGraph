import pandas as pd
import networkx as nx

def save_valid_pairs(seed):
    """Save valid pairs for a given seed."""
    # Load pairwise stats for this seed
    df = pd.read_csv(f'out/pairwise_stats_seed{seed}.csv')

    # Load graph to get connected components
    G = nx.read_gexf(f'out/graphs/overlap_graph_seed{seed}.gexf')
    components = list(nx.connected_components(G))

    # remove self-pairs
    df = df[df['db1'] != df['db2']]
    
    # Only keep pairs within same component
    print(f"Number of components: {len(components)}")
    valid_pairs = []
    for _, row in df.iterrows():
        db1, db2 = row['db1'], row['db2']
        for component in components:
            if db1 in component and db2 in component:
                valid_pairs.append(True)
                break
        else:
            valid_pairs.append(False)
    
    df = df[valid_pairs]
    print(f"Number of valid pairs: {len(df)}")
    
    # save the valid pairs
    df.to_csv(f'out/valid_pairs_stats_seed{seed}.csv', index=False)

if __name__ == "__main__":
    for seed in range(10):
        print(f"\nProcessing seed {seed}")
        save_valid_pairs(seed)
