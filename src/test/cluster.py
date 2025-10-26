import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
from sklearn.decomposition import PCA

# 1. 从.pt文件直接加载embeddings
def load_embeddings_from_pt(path):
    data = torch.load(path, map_location="cpu")
    embeddings = data["embeddings"].numpy()  # (N, D)
    db_id_to_index = data["db_id_to_index"]
    db_ids = list(db_id_to_index.keys())
    return embeddings, db_ids

# 2. 用HDBSCAN做聚类
def hdbscan_clustering(embeddings, min_cluster_size=5, min_samples=5):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'  # 也可以是 'leaf'
    )
    labels = clusterer.fit_predict(embeddings)
    return labels

# 3. 保存聚类结果
def save_cluster_results(ids, labels, output_csv_path):
    df = pd.DataFrame({
        "db_id": ids,
        "cluster_id": labels
    })
    df.to_csv(output_csv_path, index=False)
    print(f"Cluster results saved to {output_csv_path}")

# 4. 可视化
def plot_embeddings_2d(embeddings, labels, save_path=None):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab10', s=10)
    handles, _ = scatter.legend_elements()
    plt.legend(handles, [f"Cluster {i}" for i in np.unique(labels) if i != -1], title="Clusters")

    plt.title("PCA 2D Visualization of HDBSCAN Clusters")
    plt.xlabel("PCA Dim 1")
    plt.ylabel("PCA Dim 2")
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {save_path}")
    plt.close()

# 5. 主函数
if __name__ == "__main__":
    pt_path = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/embeddings/all_embeddings.pt"
    output_csv_path = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/embeddings/db_id_cluster_mapping.csv"

    print("Loading embeddings...")
    embeddings, db_ids = load_embeddings_from_pt(pt_path)

    print(f"Loaded {len(db_ids)} embeddings. Clustering with HDBSCAN...")

    labels = hdbscan_clustering(embeddings, min_cluster_size=5, min_samples=5)

    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"✅ Found {num_clusters} clusters (excluding noise)")

    print("Saving clustering results...")
    save_cluster_results(db_ids, labels, output_csv_path)

    print("Plotting...")
    plot_embeddings_2d(embeddings, labels, save_path="/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/embeddings/db_id_cluster_mapping.png")

    print("Done!")
