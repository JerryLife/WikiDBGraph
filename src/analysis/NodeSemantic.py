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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import torch

os.environ["OPENBLAS_NUM_THREADS"] = "32"


class GraphSemantic:
    """
    Class for analyzing semantic properties of database nodes using embeddings.
    Performs dimensionality reduction, clustering, and visualization.
    """
    
    def __init__(self, embeddings = None, db_ids = None, tsne_embeddings = None, clusters = None):
        """
        Initialize with the path to database embeddings.
        
        Args:
            embeddings_path (str): Path to the CSV or pickle file containing database embeddings
        """
        self.embeddings = embeddings
        self.db_ids = db_ids
        self.tsne_embeddings = tsne_embeddings
        self.clusters = clusters
        
    def load_embeddings(self, embeddings_path):
        """Load database embeddings from file."""
        print(f"Loading embeddings from {embeddings_path}")
        
        self.embeddings = torch.load(embeddings_path, weights_only=False).numpy()
        self.db_ids = list(range(self.embeddings.shape[0]))

        print(f"Loaded {len(self.db_ids)} database embeddings with {self.embeddings.shape[1]} dimensions")
        return self
    
    def load_tsne_from_csv(self, tsne_embed_path):
        """Load database embeddings from CSV file."""
        self.tsne_embeddings = pd.read_csv(tsne_embed_path).to_numpy()
        return self

    def load_cluster_from_csv(self, cluster_assign_path):
        """Load cluster assignments from CSV file."""
        id_to_cluster = pd.read_csv(cluster_assign_path)
        self.db_ids = id_to_cluster.iloc[:, 0].tolist()
        self.clusters = id_to_cluster.iloc[:, 1].tolist()
        return self
    
    def reduce_dimensions(self, n_components=2, perplexity=30, random_state=42, n_threads=1, save_path=None):
        """
        Reduce dimensions of embeddings using t-SNE.
        
        Args:
            n_components (int): Number of dimensions to reduce to (must be 2 or 3 for barnes_hut algorithm)
            perplexity (int): t-SNE perplexity parameter
            random_state (int): Random seed for reproducibility
            n_threads (int): Number of threads to use for computation
        """
        if self.embeddings is None:
            self.load_embeddings()
            
        print(f"Reducing dimensions to {n_components} using t-SNE...")
        
        # Standardize the data
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)
        
        # Apply t-SNE
        if n_components >= 4:
            # Use exact algorithm for n_components >= 4
            tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                    random_state=random_state, n_jobs=n_threads, method='exact')
        else:
            # Use barnes_hut for n_components < 4 (faster)
            tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                    random_state=random_state, n_jobs=n_threads)
        
        self.tsne_embeddings = tsne.fit_transform(embeddings_scaled)
        print(f"Reduced dimensions to {self.tsne_embeddings.shape[1]}")

        if save_path:   
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            # Create DataFrame with column names for each dimension
            column_names = [f"dim_{i}" for i in range(self.tsne_embeddings.shape[1])]
            df = pd.DataFrame(self.tsne_embeddings, columns=column_names)
            df.to_csv(save_path, index=False)
            print(f"Saved t-SNE embeddings to {save_path}")
        return self
    
    def cluster_databases(self, cluster_algo='kmeans', n_threads=1, n_clusters=5, min_cluster_size=100, min_samples=None, random_state=42, save_path=None):
        """
        Cluster databases using K-means on the reduced embeddings.
        
        Args:
            n_clusters (int): Number of clusters
            random_state (int): Random seed for reproducibility
            n_threads (int): Number of threads to use for computation
            min_cluster_size (int): Minimum number of samples in a cluster
            min_samples (int): Minimum number of samples in a cluster for HDBSCAN
            save_path (str): Path to save the cluster assignments as CSV (if None, won't save)
        """
        if self.tsne_embeddings is None:
            self.reduce_dimensions()
            
        print(f"Clustering databases with {cluster_algo}...")
        self.db_ids = list(range(self.tsne_embeddings.shape[0]))
        
        # Apply K-means clustering
        if cluster_algo == 'kmeans':
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            self.clusters = kmeans.fit_predict(self.tsne_embeddings)
        elif cluster_algo == 'hdbscan':
            hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean', n_jobs=n_threads)
            self.clusters = hdbscan.fit_predict(self.tsne_embeddings)
        
        # Save cluster assignments to CSV if path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df = pd.DataFrame({'db_id': self.db_ids, 'cluster': self.clusters})
            df.to_csv(save_path, index=False)
            print(f"Saved cluster assignments to {save_path}")
            
        return self
    
    def visualize_clusters(self, output_dir="fig", filename="database_clusters.png", n_threads=1):
        """
        Visualize clusters in 2D space and save the figure.
        
        Args:
            output_dir (str): Directory to save the figure
            filename (str): Filename for the saved figure
        """

        plt.rcParams['font.size'] = 24
        plt.rcParams['axes.titlesize'] = 24
        plt.rcParams['xtick.labelsize'] = 24
        plt.rcParams['ytick.labelsize'] = 24
        
        if self.clusters is None:
            raise ValueError("Cluster assignments are not available. Please run cluster_databases first.")
        if self.tsne_embeddings is None:
            raise ValueError("t-SNE embeddings are not available. Please run reduce_dimensions first.")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # Ensure we have 2D embeddings for visualization
        vis_embeddings = self.tsne_embeddings
        if self.tsne_embeddings.shape[1] > 2:
            print("Reducing dimensions to 2 for visualization...")
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(self.tsne_embeddings)
            tsne = TSNE(n_components=2, random_state=42, n_jobs=n_threads)
            vis_embeddings = tsne.fit_transform(embeddings_scaled)
        
        # Create a 2D visualization
        plt.figure(figsize=(12, 10))
        
        # Get unique clusters and create a color map
        unique_clusters = np.unique(self.clusters)
        n_clusters = len(unique_clusters)
        colors = plt.cm.tab20(np.linspace(0, 1, n_clusters))  # Using tab20 colormap
        
        # Create a mapping from cluster ID to color
        cluster_to_color = {cluster: 'grey' if cluster < 0 else color 
                          for cluster, color in zip(unique_clusters, colors)}
        
        # Plot each cluster
        for cluster_id in unique_clusters:
            mask = self.clusters == cluster_id
            plt.scatter(
                vis_embeddings[mask, 0],
                vis_embeddings[mask, 1],
                c=[cluster_to_color[cluster_id]],  # Use the mapped color
                alpha=0.8,
                s=5,
                edgecolors='w',
                label=f'Cluster {cluster_id}',
                marker='o'
            )
            
            # Add cluster centroids only for non-negative clusters
            if cluster_id >= 0:
                cluster_points = vis_embeddings[mask]
                centroid = cluster_points.mean(axis=0)
                plt.scatter(
                    centroid[0], centroid[1],
                    marker='*', s=200,
                    color=cluster_to_color[cluster_id],  # Use the mapped color
                    edgecolors='black'
                )
                plt.annotate(
                    f'Cluster {cluster_id}',
                    (centroid[0], centroid[1]),
                    fontsize=12,
                    fontweight='bold',
                    ha='center',
                    va='bottom',
                    xytext=(0, 10),
                    textcoords='offset points'
                )
        plt.title('Database Clusters based on Semantic Embeddings', fontsize=24)
        plt.xlabel('t-SNE Dimension 1', fontsize=24)
        plt.ylabel('t-SNE Dimension 2', fontsize=24)
        plt.grid(True, linestyle='--', alpha=0.7)
        # Replace legend labels to show "Unknown" instead of "Cluster -1" for HDBSCAN
        handles, labels = plt.gca().get_legend_handles_labels()
        labels = ['Unknown' if 'Cluster -1' in label else label for label in labels]
        plt.legend(handles, labels, bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=4)  # Increased marker size in legend
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_path}")
        
        return self


def analyze_database_semantics(embeddings_path, output_dir="fig", n_clusters=5, cluster_algo='kmeans', min_cluster_size=10, min_samples=None, n_components=2, n_threads=1):
    """
    Analyze semantic properties of databases using embeddings.
    
    Args:
        embeddings_path (str): Path to database embeddings file
        output_dir (str): Directory to save visualizations
        n_clusters (int): Number of clusters for K-means
        n_components (int): Number of dimensions for t-SNE (2 or 3 recommended)
    """
    analyzer = GraphSemantic()
    embeddings_dir = os.path.dirname(embeddings_path)
    
    # Check if tsne embeddings already exist
    tsne_path = os.path.join(embeddings_dir, f"tsne_embeddings_dim{n_components}.csv")
    if cluster_algo == 'kmeans':
        cluster_path = os.path.join(embeddings_dir, f"cluster_assignments_dim{n_components}_c{n_clusters}.csv")
    elif cluster_algo == 'hdbscan':
        cluster_path = os.path.join(embeddings_dir, f"cluster_assignments_dim{n_components}_sz{min_cluster_size}_ms{min_samples}.csv")

    if os.path.exists(tsne_path):
        # Load existing results
        analyzer.load_tsne_from_csv(tsne_path)
    else:
        # Calculate embeddings and clusters
        analyzer.load_embeddings(embeddings_path)
        analyzer.reduce_dimensions(n_components=n_components, n_threads=n_threads, save_path=tsne_path)
    
    if os.path.exists(cluster_path):
        # Load existing results
        analyzer.load_cluster_from_csv(cluster_path)
    else:
        # Calculate clusters
        analyzer.cluster_databases(cluster_algo=cluster_algo, n_threads=n_threads, n_clusters=n_clusters, min_cluster_size=min_cluster_size, min_samples=min_samples, save_path=cluster_path)
    
    # Visualize results
    analyzer.visualize_clusters(output_dir=output_dir, n_threads=n_threads)
    
    return analyzer


if __name__ == "__main__":
    # Example usage
    GRAPH_DIR = os.path.join("data", "graph")
    embeddings_path = os.path.join(GRAPH_DIR, "database_embeddings.pt")
    
    analyzer = analyze_database_semantics(
        embeddings_path=embeddings_path,
        cluster_algo='hdbscan',
        min_cluster_size=100,
        output_dir="fig",
        n_clusters=8,
        n_components=2,
        n_threads=8
    )
