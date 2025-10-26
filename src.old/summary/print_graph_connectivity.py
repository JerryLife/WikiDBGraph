import networkx as nx
import os
import numpy as np

def get_component_sizes(seed):
    """Get list of sizes of all connected components for a given seed"""
    # Load graph
    G = nx.read_gexf(f'out/graphs/overlap_graph_seed{seed}.gexf')
    
    # Get sizes of all connected components
    component_sizes = [len(c) for c in nx.connected_components(G)]
    return component_sizes

def calc_different_component_prob(seed):
    """Calculate probability that two random nodes are in different components"""
    sizes = get_component_sizes(seed)
    total_nodes = sum(sizes)
    
    # Probability is 1 - sum of probabilities of picking two nodes from same component
    same_component_prob = sum((size/total_nodes) * ((size-1)/(total_nodes-1)) for size in sizes)
    diff_component_prob = 1 - same_component_prob
    
    print(f"Seed {seed}:")
    print(f"Number of components: {len(sizes)}")
    print(f"Component sizes: {sizes}")
    print(f"Probability of different components: {diff_component_prob:.3f}\n")
    
    return diff_component_prob, len(sizes)


if __name__ == "__main__":
    # Calculate probabilities and component counts for all seeds
    results = [calc_different_component_prob(seed) for seed in range(10)]
    probs = [r[0] for r in results]
    comp_counts = [r[1] for r in results]

    print("Metric & Mean & Std & Min & Max \\\\")
    print("\\hline")
    print(f"Probability & {np.mean(probs):.3f} & {np.std(probs):.3f} & {np.min(probs):.3f} & {np.max(probs):.3f} \\\\")
    print(f"Components & {np.mean(comp_counts):.1f} & {np.std(comp_counts):.1f} & {np.min(comp_counts)} & {np.max(comp_counts)} \\\\")



"""
Metric & Mean & Std & Min & Max \\
\hline
Probability & 0.490 & 0.016 & 0.474 & 0.530 \\
Components & 3.1 & 0.3 & 3 & 4 \\
"""