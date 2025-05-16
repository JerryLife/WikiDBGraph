import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score, f1_score
import matplotlib.colors as mcolors
import matplotlib.cm as cm
sns.set(style="whitegrid")

SCHEMA_COLUMN_COUNT_FILE = "data/schema_column_counts.csv"

def load_column_count_lookup():
    column_count_df = pd.read_csv(SCHEMA_COLUMN_COUNT_FILE)
    column_count_df["simple_db_id"] = column_count_df["db_id"].astype(str).str.split("_").str[0]
    return dict(zip(column_count_df["simple_db_id"], column_count_df["num_columns"]))

def load_all_sources(method_dir):
    dfs = []
    for filename in os.listdir(method_dir):
        if filename.endswith(".csv") and filename.startswith("col_"):
            source = filename.replace("col_matcher_", "").replace(".csv", "")
            csv_path = os.path.join(method_dir, filename)
            df = pd.read_csv(csv_path)
            if {"db_1", "db_2", "runtime_seconds", "similarity"}.issubset(df.columns):
                df["source"] = source
                dfs.append(df)
            else:
                print(f"Skipping {csv_path}: missing required columns.")
    if not dfs:
        print(f"No valid source files found in: {method_dir}")
        return None
    return pd.concat(dfs, ignore_index=True)


def process_runtime_and_similarity(method_dir, column_count_lookup, mode="neg"):
    df = load_all_sources(method_dir)
    if df is None:
        return

    df["db1_id"] = df["db_1"].astype(str)
    df["db2_id"] = df["db_2"].astype(str)

    df["total_column_count"] = df["db1_id"].map(column_count_lookup) + df["db2_id"].map(column_count_lookup)

    summary_csv = os.path.join(method_dir, "column_vs_time_summary.csv")
    df[["db1_id", "db2_id", "total_column_count", "runtime_seconds", "similarity", "source"]].to_csv(summary_csv, index=False)
    print(f"Saved summary: {summary_csv}")

    output_dir = os.path.join(method_dir, "summary")
    df["dataset"] = df["source"].apply(
        lambda x: f"Positive (qid {x.split('_')[-1]})" if x.startswith("qid") 
        else f"Random (sample {x.split('_')[-1]})" if x.startswith("sample") 
        else f"Random (neg {x.split('_')[-1]})" if x.startswith("neg") 
        else x
    )
    os.makedirs(output_dir, exist_ok=True)
    plot_runtime_trend(df, output_dir)
    plot_column_distribution(df, output_dir)
    plot_similarity_histogram(df, output_dir)
    plot_similarity_boxplot(df, output_dir)
    save_similarity_statistics(df, output_dir)
    plot_precision_recall_curve(df, output_dir)

    for src in df["source"].unique():
        src_df = df[df["source"] == src]
        lowest_sim_df = src_df.nsmallest(10, "similarity")[["db1_id", "db2_id", "similarity"]]
        lowest_path = os.path.join(output_dir, f"lowest_similarity_pairs_{src}.txt")
        with open(lowest_path, "w") as f:
            for _, row in lowest_sim_df.iterrows():
                f.write(f"{row['db1_id']}, {row['db2_id']}, similarity={row['similarity']:.4f}\n")
        print(f"Saved lowest similarity pairs for [{src}]: {lowest_path}")
# def plot_runtime_trend(df, output_dir):
#     plt.figure(figsize=(10, 6))
#     for src in df["source"].unique():
#         subset = df[df["source"] == src]
#         plt.scatter(subset["total_column_count"], subset["runtime_seconds"], alpha=0.4, label=f"{src} (n={len(subset)})")

#     df["column_bin"] = (df["total_column_count"] // 1000) * 1000
#     avg_df = df.groupby("column_bin")["runtime_seconds"].mean().reset_index()
#     sns.lineplot(x="column_bin", y="runtime_seconds", data=avg_df, color="red", linewidth=2, label="Avg Trend")

#     plt.axvline(df["total_column_count"].mean(), color="green", linestyle="--", label="Mean Columns")
#     plt.axhline(df["runtime_seconds"].mean(), color="blue", linestyle="--", label="Mean Runtime")

#     plt.xlabel("Total number of columns in database pair")
#     plt.ylabel("Runtime (seconds)")
#     plt.title("Runtime vs Total Columns by Source")
#     plt.grid(True, linestyle="--", alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     plt.ylim(0, 50)
#     save_path = os.path.join(output_dir, "runtime_vs_columns_trend.png")
#     plt.savefig(save_path)
#     plt.close()
#     print(f"Saved runtime trend plot: {save_path}")


def plot_precision_recall_curve(df, output_dir):
    df_ap = df[df["dataset"].str.startswith("Positive (qid") | df["dataset"].str.startswith("Random (sample seed42")].copy()
    df_ap["label"] = df_ap["dataset"].apply(lambda s: 1 if s.startswith("Positive (qid") else 0)


    y_true = df_ap["label"].values
    y_scores = df_ap["similarity"].values

    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_idx = np.argmax(f1)
    best_threshold = thresholds[best_idx]
    best_precision = precision[best_idx]
    best_recall = recall[best_idx]
    best_f1 = f1[best_idx]

    plt.figure(figsize=(5, 3))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}", color="darkorange", linewidth=2)
    plt.scatter(best_recall, best_precision, color="red", s=80, label=f"Best F1 = {best_f1:.3f}\n@Sim ≥ {best_threshold:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve (Positive = qid)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Saved PR curve with best threshold: {save_path}")


def plot_runtime_trend(df, output_dir):
    import numpy as np
    from sklearn.linear_model import LinearRegression

    plt.figure(figsize=(10, 6))

    for src in df["source"].unique():
        subset = df[df["source"] == src]
        plt.scatter(subset["total_column_count"], subset["runtime_seconds"], alpha=0.4, label=f"{src} (n={len(subset)})")

    df["column_bin"] = (df["total_column_count"] // 1000) * 1000
    avg_df = df.groupby("column_bin")["runtime_seconds"].mean().reset_index()
    sns.lineplot(x="column_bin", y="runtime_seconds", data=avg_df, color="red", linewidth=2, label="Avg Trend")

    bin_width = 100
    df["fit_bin"] = (df["total_column_count"] // bin_width) * bin_width
    min_df = df.groupby("fit_bin")["runtime_seconds"].min().reset_index()

    min_df = min_df[min_df["runtime_seconds"] < 60]

    X = min_df["fit_bin"].values.reshape(-1, 1)
    y = min_df["runtime_seconds"].values
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)

    plt.plot(min_df["fit_bin"], y_pred, color="black", linestyle="-", linewidth=10, alpha=0.4, label="Min Runtime Fit")


    a, b = reg.coef_[0], reg.intercept_
    formula = f"y = {a:.4f}x + {b:.2f}"
    plt.text(0.05, 0.95, f"Min Fit: {formula}", transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.6))

    plt.axvline(df["total_column_count"].mean(), color="green", linestyle="--", label="Mean Columns")
    plt.axhline(df["runtime_seconds"].mean(), color="blue", linestyle="--", label="Mean Runtime")

    plt.xlabel("Total number of columns in database pair")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime vs Total Columns")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.ylim(0, 50)

    save_path = os.path.join(output_dir, "runtime_vs_columns_trend.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved runtime trend plot: {save_path}")



def plot_column_distribution(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, hue="source", x="total_column_count", bins=30, element="step", stat="count")
    plt.xlabel("Total number of columns in database pairs")
    plt.ylabel("Frequency")
    plt.title("Column Count Distribution")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "column_count_distribution.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Saved column distribution: {save_path}")

def plot_similarity_histogram(df, output_dir):
    unique_datasets = df["dataset"].unique()

    positive_datasets = [name for name in unique_datasets if name.startswith("Positive")]
    other_datasets = [name for name in unique_datasets if not name.startswith("Positive")]

    orange_palette = sns.color_palette("Oranges", n_colors=len(positive_datasets))
    blue_palette = sns.color_palette("Blues", n_colors=len(other_datasets))

    palette = {}
    for i, name in enumerate(positive_datasets):
        palette[name] = orange_palette[i]
    for i, name in enumerate(other_datasets):
        palette[name] = blue_palette[i]

    plt.figure(figsize=(10, 6))
    sns.histplot(
        data=df,
        x="similarity",
        hue="dataset",
        bins=30,
        kde=True,
        alpha=0.6,
        palette=palette
    )
    plt.xlabel("Similarity Score")
    plt.ylabel("Frequency")
    plt.title("Similarity Score Distribution")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(output_dir, "hist_similarity.png")
    plt.savefig(path)
    plt.close()
    print(f"📊 Saved similarity histogram: {path}")


def plot_similarity_boxplot(df, output_dir):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="source", y="similarity")
    plt.title("Boxplot of Similarity Score")
    plt.tight_layout()
    path = os.path.join(output_dir, "boxplot_similarity.png")
    plt.savefig(path)
    plt.close()
    print(f"📦 Saved similarity boxplot: {path}")


import os

def save_similarity_statistics(df, output_dir, similarity_threshold=0.8):
    grouped = df.groupby("source").agg(
        runtime_mean=("runtime_seconds", "mean"),
        runtime_min=("runtime_seconds", "min"),
        runtime_max=("runtime_seconds", "max"),
        runtime_25=("runtime_seconds", lambda x: x.quantile(0.25)),
        runtime_75=("runtime_seconds", lambda x: x.quantile(0.75)),
        similarity_mean=("similarity", "mean"),
        similarity_min=("similarity", "min"),
        similarity_max=("similarity", "max"),
    )
    proportion = (
        df[df["similarity"] > similarity_threshold]
        .groupby("source")
        .size()
        .div(df.groupby("source").size())
        .fillna(0)
    )
    grouped["similarity>%.2f_ratio" % similarity_threshold] = proportion

    grouped.rename(columns={
        "runtime_25": "runtime_25%",
        "runtime_75": "runtime_75%",
    }, inplace=True)

    stat_path = os.path.join(output_dir, "similarity_statistics.csv")
    grouped.to_csv(stat_path)
    print(f"Saved similarity & runtime statistics by source: {stat_path}")


def plot_all_precision_recall_curves(base_dir, method_dirs, output_path="all_pr_curve.png"):
    plt.figure(figsize=(7, 5))
    
    for method_name, subdir in method_dirs.items():
        full_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(full_path):
            print(f"Skipping PR curve: {full_path} not found")
            continue

        df = load_all_sources(full_path)
        if df is None or not {"source", "similarity"}.issubset(df.columns):
            continue

        df_ap = df[df["source"].str.startswith("qid") | df["source"].str.endswith("42")].copy()
        df_ap["label"] = df_ap["source"].apply(lambda s: 1 if s.startswith("qid") else 0)

        y_true = df_ap["label"].values
        y_scores = df_ap["similarity"].values

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)

        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
        best_idx = np.argmax(f1)
        best_f1 = f1[best_idx]
        best_recall = recall[best_idx]
        best_precision = precision[best_idx]
        best_threshold = thresholds[best_idx]

        plt.plot(recall, precision, label=f"{method_name} (AP={ap:.4f}, F1={best_f1:.3f})", linewidth=2)
        plt.scatter(best_recall, best_precision, s=60, color="red")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves (Positive = qid)")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📈 Saved combined PR curve with best F1 markers: {output_path}")



def main():
    BASE_DIR = "/hpctmp/e1351271/wkdbs/out"
    column_count_lookup = load_column_count_lookup()
    # print(column_count_lookup)

    METHOD_DIRS = {
        # "col-level": "col_matcher_cosine",
        # "col-level + idf": "col_matcher_cosine_idf",
        # "col-level (+ db name) + idf": "col_matcher_cosine_db_idf",
        "db-level": "col_matcher_bge-m3_database",
        "ft_db-level": "col_matcher_bge-m3_ft_database",
        # "ft_lr1e-5_db-level": "col_matcher_bge-m3_lr1e-5_ft_database",
        # "db-level-old": "col_matcher_bge-m3_database_old",
        # "db-wkid-level": "col_matcher_bge-m3_database_wkid",
        # "col-level": "col_matcher_bge_m3_idf_column",
        # "cosine_db_refined_idf": "col_matcher_cosine_db_refined_idf",
        # "cross_encoder": "col_matcher_cross_encoder",
        # "cross_encoder_idf": "col_matcher_cross_encoder_idf"
    }

    for method_name, subdir in METHOD_DIRS.items():
        full_path = os.path.join(BASE_DIR, subdir)
        if not os.path.isdir(full_path):
            print(f"Directory not found: {full_path}")
            continue
        print(f"\nProcessing [{method_name}]: {full_path}")
        process_runtime_and_similarity(full_path, column_count_lookup)

    # plot_all_precision_recall_curves(BASE_DIR, METHOD_DIRS, output_path=os.path.join(BASE_DIR, "summary", "all_pr_curves.png"))



if __name__ == "__main__":
    main()
