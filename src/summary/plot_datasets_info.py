import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from joblib import Parallel, delayed
import tqdm

dataset_dir = "out/test_results_exhaustive_split"
dataset_pred_dir = os.path.join(dataset_dir, "test_results_exhaustive_split")
output_plot_dir = "/out/similarity_distributions"
os.makedirs(output_plot_dir, exist_ok=True)

sim_threshold = 0.6713

def read_csv_file(file_path):
    try:
        df = pd.read_csv(file_path, usecols=["similarity", "label"])
        df["label"] = df["label"].astype(str)
        df["similarity"] = pd.to_numeric(df["similarity"], errors="coerce")
        df = df[np.isfinite(df["similarity"])]
        return df
    except Exception as e:
        print(f"⚠️ Skipped {file_path}: {e}")
        return pd.DataFrame(columns=["similarity", "label"])

def process_file(file_path):
    df = read_csv_file(file_path)
    if df.empty:
        return

    file_name = os.path.splitext(os.path.basename(file_path))[0]
    sim_all = df["similarity"].values
    sim_positive = df[df["label"] == "1"]["similarity"].values

    plt.figure(figsize=(10, 6))
    pd.Series(sim_all).plot(kind='kde', label='All pairs', color='blue', alpha=0.7)
    if len(sim_positive) > 0:
        pd.Series(sim_positive).plot(kind='kde', label='Positive', color='orange', alpha=0.7)
    plt.axvline(sim_threshold, color='gray', linestyle='--', label=f'Threshold = {sim_threshold}')
    plt.xlabel("Similarity")
    plt.ylabel("Density (log scale)")
    plt.yscale("log", base=2)
    plt.title(f"Distribution for {file_name}")
    plt.legend()
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_plot_dir, f"{file_name}_kde.png"))
    plt.close()


    stats = {
        "file": file_name,
        "total_count": len(sim_all),
        "positive_count": len(sim_positive),
        "all>threshold": np.sum(sim_all > sim_threshold),
        "positive>threshold": np.sum(sim_positive > sim_threshold),
        "all_ratio_above": np.mean(sim_all > sim_threshold) if len(sim_all) else 0,
        "positive_ratio_above": np.mean(sim_positive > sim_threshold) if len(sim_positive) else 0,
        "mean_all": np.mean(sim_all),
        "std_all": np.std(sim_all),
        "min_all": np.min(sim_all),
        "max_all": np.max(sim_all),
        "mean_positive": np.mean(sim_positive) if len(sim_positive) else np.nan,
        "std_positive": np.std(sim_positive) if len(sim_positive) else np.nan,
        "min_positive": np.min(sim_positive) if len(sim_positive) else np.nan,
        "max_positive": np.max(sim_positive) if len(sim_positive) else np.nan
    }
    stats_path = os.path.join(output_plot_dir, f"{file_name}_stats.csv")
    pd.DataFrame([stats]).to_csv(stats_path, index=False)
    return stats

# 获取所有文件并分析
files = [os.path.join(dataset_pred_dir, f) for f in os.listdir(dataset_pred_dir) if f.endswith(".csv")]
results = Parallel(n_jobs=4)(delayed(process_file)(f) for f in tqdm.tqdm(files))
df_stats = pd.DataFrame([r for r in results if r is not None])
df_stats.to_csv(os.path.join(output_plot_dir, "similarity_stats_summary.csv"), index=False)
