import sys
sys.path.append(".")
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from src.model.WKDataset import WKDataset

INPUT_DIRS = {
    # "QID_pairs": "/hpctmp/e1351271/wkdbs/out/qid_pairs_max_similarity",
    # "Sample_pairs": "/hpctmp/e1351271/wkdbs/out/random_pairs_max_similarity_10k"
    "QID_pairs": "/hpctmp/e1351271/wkdbs/out/qid_pairs_max_similarity_default_pooling",
    "Sample_pairs": "/hpctmp/e1351271/wkdbs/out/random_pairs_max_similarity_10k_default_pooling"
}

SCHEMA_COLUMN_COUNT_FILE = "data/schema_column_counts.csv"
runtime_pattern = re.compile(r"Runtime:\s+([0-9.]+)\s+seconds")

column_count_df = pd.read_csv(SCHEMA_COLUMN_COUNT_FILE)

column_count_df["simple_db_id"] = column_count_df["db_id"].str.split("_").str[0]
column_count_lookup = dict(zip(column_count_df["simple_db_id"], column_count_df["num_columns"]))

records = []

for source, path in INPUT_DIRS.items():
    print(f"📂 Processing {source}...")
    for filename in tqdm(os.listdir(path)):
        if not filename.endswith(".txt"):
            continue
        full_path = os.path.join(path, filename)
        with open(full_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        runtime_match = runtime_pattern.search(lines[-1])
        if runtime_match is None:
            print(f"⚠️ Runtime not found in {filename}, skipping...")
            continue
        runtime = float(runtime_match.group(1))

        db1_id, db2_id = os.path.basename(filename).split("_")[0:2]

        if db1_id not in column_count_lookup or db2_id not in column_count_lookup:
            print(f"❌ Column count missing for {db1_id} or {db2_id}, skipping...")
            continue

        db1_column_count = column_count_lookup[db1_id]
        db2_column_count = column_count_lookup[db2_id]

        records.append({
            "source": source,
            "db1_id": db1_id,
            "db2_id": db2_id,
            "total_column_count": db1_column_count + db2_column_count,
            "runtime_sec": runtime
        })

df = pd.DataFrame(records)
df.to_csv("column_vs_time_summary.csv", index=False)
print(f"✅ Parsed {len(df)} files.")

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

df = pd.read_csv("column_vs_time_summary.csv")

plt.figure(figsize=(10, 6))

for source in df["source"].unique():
    subset = df[df["source"] == source]
    plt.scatter(
        subset["total_column_count"],
        subset["runtime_sec"],
        alpha=0.5,
        label=f"{source} (n={len(subset)})"
    )

df["column_bin"] = (df["total_column_count"] // 1000) * 1000

avg_df = df.groupby("column_bin")["runtime_sec"].mean().reset_index()
avg_df = avg_df.sort_values("column_bin")

sns.lineplot(
    x="column_bin",
    y="runtime_sec",
    data=avg_df,
    color="red",
    linewidth=2.0,
    label="Avg Trend"
)

mean_column_count = df["total_column_count"].mean()
plt.axvline(mean_column_count, color="green", linestyle="--", linewidth=2, label=f"Mean Columns = {mean_column_count:.1f}")

mean_runtime = df["runtime_sec"].mean()
plt.axhline(mean_runtime, color="blue", linestyle="--", linewidth=2, label=f"Mean Runtime = {mean_runtime:.2f}s")

plt.xlabel("Total number of columns in databases pair")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime vs Total Columns")
plt.grid(True, linestyle="--", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.ylim(0, 50)
plt.savefig("runtime_vs_columns_trend_default_pooling.png")
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="total_column_count", bins=30, hue="source", element="step", stat="count", common_norm=False)
plt.xlabel("Total number of columns in database pairs")
plt.ylabel("Frequency")
plt.title("Distribution of Total Column Count by Source")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig("column_count_distribution_by_source_default_pooling.png")
plt.show()