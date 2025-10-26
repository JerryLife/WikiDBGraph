import argparse

from src.analysis.NodeSemantic import GraphSemantic


def plot_db_cluster(tsne_embed_path, cluster_assign_path, output_dir="fig", filename="database_clusters.png", n_threads=1):
    """
    Plot database clusters based on semantic embeddings.
    """
    analyzer = GraphSemantic()
    analyzer.load_tsne_from_csv(tsne_embed_path)
    analyzer.load_cluster_from_csv(cluster_assign_path)
    analyzer.visualize_clusters(output_dir=output_dir, filename=filename, n_threads=n_threads)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot database clusters based on semantic embeddings.")
    parser.add_argument("-tsne", "--tsne_embed_path", type=str, default="data/graph/tsne_embeddings_dim2.csv", help="Path to the t-SNE embeddings CSV file")
    parser.add_argument("-cluster", "--cluster_assign_path", type=str, default="data/graph/cluster_assignments_dim2_sz100_msNone.csv", help="Path to the cluster assignments CSV file")
    parser.add_argument("-o", "--output_dir", type=str, default="fig", help="Output directory for the plot")
    parser.add_argument("-f", "--filename", type=str, default="database_clusters.png", help="Output filename for the plot")
    parser.add_argument("-t", "--n_threads", type=int, default=1, help="Number of threads to use for the plot")
    args = parser.parse_args()

    plot_db_cluster(args.tsne_embed_path, args.cluster_assign_path, args.output_dir, args.filename, args.n_threads)







