import csv
import random
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from get_dbs_similarity import load_database_stats
import os
import pickle
import numpy as np
import pandas as pd

def build_and_visualize_sampled_graph(
    db_stats,
    qid_csv,
    output_image="sampled_database_graph.png",
    sample_size=None,
    seed=42,
    graph_output_path="sampled_database_graph.gexf",
    layout_cache_path="layout_cache.pkl"
):
    """
    Builds and visualizes a sampled graph with optimized performance.
    Saves the graph as GEXF and skips expensive rendering settings.

    :param db_stats: Dictionary of all database names.
    :param qid_csv: Path to the Q-ID CSV file.
    :param output_image: File name for the saved graph image.
    :param sample_size: If set, randomly sample this many databases.
    :param seed: Random seed for reproducibility.
    :param graph_output_path: Path to save the GEXF graph file.
    :param layout_cache_path: Path to optionally cache layout positions.
    """
    if sample_size is not None and sample_size < len(db_stats):
        print(f"🎯 Sampling {sample_size} databases from {len(db_stats)}...")
        random.seed(seed)
        sampled_dbs = set(random.sample(list(db_stats.keys()), sample_size))
    else:
        sampled_dbs = set(db_stats.keys())

    G = nx.Graph()
    print("🔄 Reading Q-ID file and building graph...")
    with open(qid_csv, 'r', encoding="utf-8") as f:
        reader = list(csv.reader(f))
        rows = reader[1:]

        for row in tqdm(rows, desc="➕ Adding edges between sampled databases"):
            databases = row[3].split("; ")
            filtered_dbs = [db for db in databases if db in sampled_dbs]
            if len(filtered_dbs) < 2:
                continue
            for i in range(len(filtered_dbs)):
                for j in range(i + 1, len(filtered_dbs)):
                    G.add_edge(filtered_dbs[i], filtered_dbs[j])

    for db in sampled_dbs:
        G.add_node(db)

    print(f"✅ Final graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # 💾 Save graph to GEXF
    os.makedirs(os.path.dirname(graph_output_path), exist_ok=True)
    nx.write_gexf(G, graph_output_path)
    print(f"💾 Graph saved to {graph_output_path} (GEXF format)")

import networkx as nx
import matplotlib.pyplot as plt
import random

def sample_and_visualize_gexf(
    gexf_path,
    sample_size=500,
    seed=42,
):
    # Load full graph
    print(f"📥 Loading GEXF graph from {gexf_path}")
    G = nx.read_gexf(gexf_path)
    print(f"✅ Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

    # Sample nodes
    if sample_size >= len(G.nodes):
        print(f"⚠️ Sample size >= total nodes. Using full graph.")
        sampled_nodes = list(G.nodes)
    else:
        random.seed(seed)
        sampled_nodes = random.sample(list(G.nodes), sample_size)

    # Induced subgraph
    subgraph = G.subgraph(sampled_nodes).copy()

    nodes_with_edges = [n for n in subgraph.nodes if subgraph.degree(n) > 0]
    subgraph = subgraph.subgraph(nodes_with_edges).copy()

    print(f"🔍 Filtered subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges")

    # Layout and draw
    print("🖼️ Visualizing...")
    pos = nx.spring_layout(subgraph)
    for node in pos:
        pos[node] += np.random.normal(0, 0.06, size=2)

    plt.figure(figsize=(12, 8))
    nx.draw(
        subgraph,
        pos,
        node_color="red",
        node_size=30,
        edge_color="gray",
        alpha=0.5,
        with_labels=False
    )
    output_image = f"sampled_subgraph_{seed}.png"
    plt.savefig(output_image, dpi=150, bbox_inches="tight")
    print(f"📂 Sampled subgraph image saved to {output_image}")

def analyze_duplicate_qids(csv_path):
    df = pd.read_csv(csv_path)

    print(f"📄 Loaded {len(df)} connected components from: {csv_path}")

    num_components = len(df)
    db_counts = df["Number of Databases"]
    
    print(f"Connected Component Stats:")
    print(f"- Total components: {num_components}")
    print(f"- Total databases (sum): {db_counts.sum()}")
    print(f"- Unique databases (approx): {len(set(';'.join(df['Databases']).replace(' ', '').split(';')))}")
    print(f"- Max databases in a component: {db_counts.max()}")
    print(f"- Min databases in a component: {db_counts.min()}")
    print(f"- Average databases per component: {db_counts.mean():.2f}")
    print(f"- Median databases per component: {db_counts.median()}")
    print(f"- Std deviation: {db_counts.std():.2f}")

    largest = df.loc[db_counts.idxmax()]
    print(f"\n🏆 Largest Component:")
    print(f"- Q-ID: {largest['Q-ID']}")
    print(f"- Label: {largest['Label']}")
    print(f"- Databases ({largest['Number of Databases']}):")
    for db in largest["Databases"].split(";"):
        print(f"  • {db.strip()}")

def plot_duplicate_qids_column_distribution(qid_csv_path, column_count_csv):
    qid_df = pd.read_csv(qid_csv_path)
    col_df = pd.read_csv(column_count_csv)

    # 建立列数查找字典，key → db_id (小写)
    col_map = {str(k).lower(): v for k, v in zip(col_df["db_id"], col_df["num_columns"])}

    # 提取所有数据库 ID，统一小写
    db_set = set()
    for db_str in qid_df["Databases"]:
        db_list = [db.strip().split("_")[0].lower() for db in db_str.split(";") if db.strip()]
        db_set.update(db_list)

    print(f"🔍 Total unique databases in QID groups: {len(db_set)}")

    # 收集列数信息（跳过没有的）
    col_counts = [col_map[db] for db in db_set if db in col_map]
    print(f"✅ Found column counts for {len(col_counts)} databases.")

    # 转为 DataFrame 方便处理
    col_df = pd.DataFrame({"num_columns": col_counts})

    # 打印统计信息
    print("\n📊 Column Count Summary:")
    print(col_df["num_columns"].describe())

    # 分布统计
    col_series = col_df["num_columns"].value_counts().sort_index()
    x = col_series.index
    y = col_series.values

    # 绘图（面积图 + log x 轴）
    plt.figure(figsize=(10, 5))
    plt.fill_between(x, y, color="skyblue", alpha=0.6)
    plt.xscale("log")
    plt.gca().xaxis.set_major_formatter(plt.ScalarFormatter())
    plt.gca().xaxis.set_major_locator(plt.LogLocator(base=10.0))
    plt.xlabel("Number of Columns per Database (log scale)")
    plt.ylabel("Count")
    plt.title("📊 Column Count Distribution of Duplicate-QID Databases")
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig("column_distribution_duplicate_qids.png")
    plt.show()
# 使用方式：



if __name__ == "__main__":
    # column_stats_path = "../data/column_stats.txt"
    # schema_folder = "../data/schema"

    # print("📥 Loading database stats...")
    # db_stats = load_database_stats(column_stats_path, schema_folder, 100000)
    # print(f"✅ Loaded {len(db_stats)} databases")

    # qid_csv = "duplicate_qids.csv"
    # build_and_visualize_sampled_graph(
    #     db_stats,
    #     qid_csv,
    #     sample_size=100000,
    #     graph_output_path="out/graph/sample_graph.gexf",
    #     layout_cache_path="out/graph/layout_cache.pkl"
    # )
    # gexf_file = "/hpctmp/e1351271/wkdbs/src/out/graph/sample_graph.gexf"
    # for seed in range(10):
    #     sample_and_visualize_gexf(gexf_file, sample_size=100000, seed=seed)
    plot_duplicate_qids_column_distribution(
        qid_csv_path="/hpctmp/e1351271/wkdbs/data/duplicate_qids.csv",
        column_count_csv="schema_column_counts.csv"
    )