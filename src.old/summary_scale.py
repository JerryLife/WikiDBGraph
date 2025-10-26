import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import LogLocator, LogFormatterMathtext
SCHEMA_DIR = "/hpctmp/e1351271/wkdbs/data/schema"
records = []

# for filename in tqdm(os.listdir(SCHEMA_DIR), desc="📦 Scanning Schemas"):
#     if filename.endswith(".json"):
#         path = os.path.join(SCHEMA_DIR, filename)
#         try:
#             with open(path, "r", encoding="utf-8") as f:
#                 schema = json.load(f)

#             total_columns = 0
#             for table in schema.get("tables", []):
#                 total_columns += len(table.get("columns", []))

#             db_id = schema.get("db_id", filename.replace(".json", ""))
#             records.append({"db_id": db_id, "num_columns": total_columns})

#         except Exception as e:
#             print(f"❌ Failed to read {filename}: {e}")

# # 保存为 CSV
# df = pd.DataFrame(records)
# df.to_csv("schema_column_counts.csv", index=False)

df = pd.read_csv("schema_column_counts.csv")
col_counts = df["num_columns"]

print("📊 Column count summary:")
print(col_counts.describe())

count_series = col_counts.value_counts().sort_index()
x = count_series.index
y = count_series.values

plt.figure(figsize=(5, 5))
plt.fill_between(x, y, color="skyblue", alpha=0.6, linewidth=1)
plt.xscale("log")
plt.gca().xaxis.set_major_locator(LogLocator(base=10.0))
plt.gca().xaxis.set_major_formatter(LogFormatterMathtext())
plt.gca().tick_params(axis="x", which="major", labelsize=10)
plt.xlabel("Number of Columns per Database")
plt.ylabel("Count")
plt.title("Column Count Distribution")
plt.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.3)
plt.tight_layout()
plt.savefig("column_count_area_logX_exponent_ticks.png")
plt.show()