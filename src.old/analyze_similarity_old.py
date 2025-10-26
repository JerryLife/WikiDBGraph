import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import seaborn as sns
sns.set(style="whitegrid")

INPUT_DIRS = {
    "qid_pairs": "/hpctmp/e1351271/wkdbs/out/qid_pairs_max_similarity",
    "sample_pairs": "/hpctmp/e1351271/wkdbs/out/random_pairs_max_similarity_10k",
    # "qid_pairs": "/hpctmp/e1351271/wkdbs/out/qid_pairs_max_similarity_default_pooling",
    # "sample_pairs": "/hpctmp/e1351271/wkdbs/out/random_pairs_max_similarity_10k_default_pooling"
}

PATTERN_MAX = re.compile(r"Max similarity:\s+([0-9.]+)", re.IGNORECASE)
PATTERN_TIME = re.compile(r"Runtime:\s+([0-9.]+) seconds")

all_records = []

for source_name, input_dir in INPUT_DIRS.items():
    for filename in tqdm(os.listdir(input_dir), desc=f"Processing {source_name}"):
        if filename.endswith(".txt"):
            filepath = os.path.join(input_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            max_match = PATTERN_MAX.search(content)
            time_match = PATTERN_TIME.search(content)

            if max_match and time_match:
                all_records.append({
                    "source": source_name,
                    "file": filename,
                    "max_similarity": float(max_match.group(1)),
                    "runtime_sec": float(time_match.group(1))
                })

df = pd.DataFrame(all_records)
print(f"\n✅ Total loaded: {len(df)} files across {list(INPUT_DIRS.keys())}")
df.to_csv("similarity_summary_all_sources.csv", index=False)

# ======================
# 📊 Summary statistics
# ======================
agg_funcs = {
    col: ["mean", "max", "min",
          lambda x: x.quantile(0.25), 
          lambda x: x.quantile(0.75)]
    for col in ["max_similarity", "runtime_sec"]
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
print("\n📈 Summary Statistics by Source:")
print(stats)
stats.to_csv("similarity_statistics_all_sources.csv")

source_counts = df["source"].value_counts().to_dict()
source_label_map = {
    source: f"{source} (n={count})"
    for source, count in source_counts.items()
}
df["source_label"] = df["source"].map(source_label_map)

# ======================
# 📈 Histogram comparison
# ======================
plt.figure(figsize=(10, 5))
sns.histplot(data=df, x="max_similarity", hue="source_label", bins=30, kde=True, alpha=0.6)
plt.xlabel("Similarity Score")
plt.ylabel("Count")
plt.title("Similarity Score Distribution by Source")
plt.tight_layout()
# plt.savefig("hist_similarity_all_sources_default_pooling.png")
plt.savefig("hist_similarity_all_sources.png")
plt.show()

# ======================
# 📦 Boxplot comparison
# ======================
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x="source", y="max_similarity")
plt.title("Boxplot of Max Similarity by Source")
plt.tight_layout()
plt.savefig("boxplot_similarity_all_sources.png")
# plt.savefig("boxplot_similarity_all_sources.png")
plt.show()
