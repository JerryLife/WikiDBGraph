import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# ======================
# 📂 Input CSV files
# ======================
INPUT_CSVS = {
    "qid_pairs": "/hpctmp/e1351271/wkdbs/out/col_matcher_cross_encoder/col_matcher_qid.csv",
    "sample_pairs": "/hpctmp/e1351271/wkdbs/out/col_matcher_cross_encoder/col_matcher_sample_10k.csv"
}

# ======================
# 📥 Load and combine
# ======================
all_dfs = []
for source_name, csv_path in INPUT_CSVS.items():
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df["source"] = source_name
        all_dfs.append(df)
    else:
        print(f"⚠️ File not found: {csv_path}")

df = pd.concat(all_dfs, ignore_index=True)
print(f"\n✅ Total loaded: {len(df)} rows across sources: {list(INPUT_CSVS.keys())}")
df.to_csv("similarity_summary_cross_encoder.csv", index=False)

# ======================
# 📊 Summary statistics
# ======================
agg_funcs = {
    col: ["mean", "max", "min",
          lambda x: x.quantile(0.25),
          lambda x: x.quantile(0.75)]
    for col in ["similarity", "runtime_seconds"]
}

stats = df.groupby("source").agg(agg_funcs)
stats.columns = [
    (col[0] + " " + {
        "<lambda_0>": "25%",
        "<lambda_1>": "75%"
    }.get(col[1], col[1]))
    for col in stats.columns
]
stats = stats.T

source_counts = df["source"].value_counts().to_dict()
source_label_map = {
    source: f"{source} (n={count})"
    for source, count in source_counts.items()
}
df["source_label"] = df["source"].map(source_label_map)

print("\n📈 Summary Statistics by Source:")
print(stats)
stats.to_csv("similarity_statistics_cross_encoder.csv")

# ======================
# 📈 Histogram
# ======================
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="similarity", hue="source_label", bins=30, kde=True, alpha=0.6)
plt.xlabel("Similarity Score")
plt.ylabel("Count")
plt.title("Similarity Score Distribution by Source (CrossEncoder)")
plt.tight_layout()
plt.savefig("hist_similarity_cross_encoder.png")
plt.show()

# ======================
# 📦 Boxplot
# ======================
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="source", y="similarity")
plt.title("Boxplot of Max Similarity by Source (CrossEncoder)")
plt.tight_layout()
plt.savefig("boxplot_similarity_cross_encoder.png")
plt.show()
