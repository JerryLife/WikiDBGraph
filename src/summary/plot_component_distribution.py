import dgl
import cudf
import cugraph
import matplotlib.pyplot as plt
import collections
import numpy as np

def analyze_components_and_plot(dgl_path, output_txt_path, output_plot_path):
    print(f"Loading DGL graph from {dgl_path}...")
    g = dgl.load_graphs(dgl_path)[0][0]
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()

    src, dst = g.edges()
    src = src.numpy()
    dst = dst.numpy()

    df = cudf.DataFrame({'src': src, 'dst': dst})
    G = cugraph.Graph()
    G.from_cudf_edgelist(df, source='src', destination='dst', renumber=True)

    print("Running connected components...")
    components_df = cugraph.connected_components(G)
    labels = components_df['labels'].to_pandas().values

    size_counter = collections.Counter(labels)
    sorted_sizes = sorted(size_counter.values(), reverse=True)

    largest_cc_size = sorted_sizes[0]

    # 保存文本信息
    with open(output_txt_path, "w") as f:
        f.write(f"Graph Path: {dgl_path}\n")
        f.write(f"Total Nodes: {num_nodes}\n")
        f.write(f"Total Edges: {num_edges}\n")
        f.write(f"Connected Components: {len(size_counter)}\n\n")

        f.write("Top 10 Component Sizes:\n")
        for idx, size in enumerate(sorted_sizes[:10]):
            f.write(f"  Component {idx + 1}: {size} nodes\n")
        if len(sorted_sizes) > 10:
            f.write("  ...\n")
        f.write("\nLargest Connected Component:\n")
        f.write(f"  Nodes: {largest_cc_size}\n")

    print(f"[Done] Info saved to {output_txt_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sorted_sizes) + 1), sorted_sizes, marker='o', linestyle='-')
    plt.yscale("log")
    plt.xlabel("Component Rank (Largest → Smallest)")
    plt.ylabel("Component Size (Number of Nodes)")
    plt.title("Connected Component Size Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"[Plot saved] {output_plot_path}")
    plt.close()


def detect_communities_louvain(dgl_path, output_txt_path, output_plot_path):
    print(f"Loading DGL graph from {dgl_path}...")
    g = dgl.load_graphs(dgl_path)[0][0]
    num_nodes = g.num_nodes()
    num_edges = g.num_edges()

    src, dst = g.edges()
    src = src.numpy()
    dst = dst.numpy()

    df = cudf.DataFrame({'src': src, 'dst': dst})

    G = cugraph.Graph()
    G.from_cudf_edgelist(df, source='src', destination='dst', renumber=True)

    print("Running Louvain community detection...")
    partition_df, modularity_score = cugraph.louvain(G)
    partition = partition_df.to_pandas()

    print(f"Found {partition['partition'].nunique()} communities")
    print(f"Modularity score: {modularity_score:.4f}")

    community_sizes = partition['partition'].value_counts().sort_values(ascending=False)
    sorted_sizes = community_sizes.values.tolist()

    with open(output_txt_path, "w") as f:
        f.write(f"Graph Path: {dgl_path}\n")
        f.write(f"Total Nodes: {num_nodes}\n")
        f.write(f"Total Edges: {num_edges}\n")
        f.write(f"Detected Communities: {len(sorted_sizes)}\n")
        f.write(f"Modularity Score: {modularity_score:.4f}\n\n")

        f.write("Top 10 Community Sizes:\n")
        for idx, size in enumerate(sorted_sizes[:10]):
            f.write(f"  Community {idx + 1}: {size} nodes\n")
        if len(sorted_sizes) > 10:
            f.write("  ...\n")

    print(f"[Done] Community info saved to {output_txt_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(sorted_sizes) + 1), sorted_sizes, marker='o', linestyle='-')
    plt.yscale("log")
    plt.xlabel("Community Rank (Largest → Smallest)")
    plt.ylabel("Number of Nodes")
    plt.title("Community Size Distribution (Louvain)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"[Plot saved] {output_plot_path}")
    plt.close()

if __name__ == "__main__":
    dgl_file = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_exhaustive_split/full_graph.dgl"
    out_txt = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_exhaustive_split/louvain_communities.txt"
    out_plot = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_exhaustive_split/louvain_community_distribution.png"

    detect_communities_louvain(dgl_file, out_txt, out_plot)


# if __name__ == "__main__":
#     dgl_file = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_exhaustive_split/full_graph.dgl"
#     out_txt = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_exhaustive_split/connected_components.txt"
#     out_plot = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_exhaustive_split/component_size_distribution.png"

#     analyze_components_and_plot(dgl_file, out_txt, out_plot)
