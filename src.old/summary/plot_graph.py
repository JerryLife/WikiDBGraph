import networkx as nx
import matplotlib.pyplot as plt
import os

def plot_graph(seed):
    """Plot overlap graph for a given seed"""
    # Load graph
    G = nx.read_gexf(f'out/graphs/overlap_graph_seed{seed}.gexf')
    
    # Create figure
    plt.figure(figsize=(6, 5))
    
    # Draw graph
    pos = nx.spring_layout(G, k=1, iterations=80)
    
    # Get connected components and assign colors
    components = list(nx.connected_components(G))
    node_colors = []
    edge_colors = []
    
    # Assign colors to nodes based on component
    for node in G.nodes():
        for i, component in enumerate(components):
            if node in component:
                node_colors.append(f'C{i}')
                break
    
    # Assign lighter colors to edges based on connected nodes
    for edge in G.edges():
        for i, component in enumerate(components):
            if edge[0] in component:
                # Create color with alpha transparency
                edge_colors.append(('#48474E', 0.1))  # Tuple of color and alpha
                break
    
    # Draw nodes and edges
    nx.draw(G, pos,
            node_color=node_colors,
            edge_color=[color for color, alpha in edge_colors],
            node_size=10,
            width=0.5,
            with_labels=False)  # Removed edge_alpha parameter
    
    # Save plot
    os.makedirs('fig', exist_ok=True)
    plt.savefig(f'fig/overlap_graph_seed{seed}.png', bbox_inches='tight', dpi=600)
    plt.close()

    print(f"Saved fig/overlap_graph_seed{seed}.png")

if __name__ == "__main__":
    # Plot graph for each seed
    for seed in range(10):
        plot_graph(seed)
