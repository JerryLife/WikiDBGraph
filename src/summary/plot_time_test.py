import re
import matplotlib.pyplot as plt
import numpy as np

with open("/hpctmp/e1351271/wkdbs/logs/col_matcher_qid_cosine_test_idf_20250327_035057.out", "r") as f:
    text = f.read()

blocks = text.strip().split('--------------------')
data = []

for i, block in enumerate(blocks):
    if not block.strip():
        continue
    try:
        source_match = re.search(r"Time taken to get source embeddings: ([\d.]+)", block)
        target_match = re.search(r"Time taken to get target embeddings: ([\d.]+)", block)
        idf_match = re.search(r"Time taken to get idf values: ([\d.]+)", block)
        cosine_match = re.search(r"Time taken to compute max cosine similarity: ([\d.]+)", block)
        col_count_match = re.search(r"Total column count: (\d+)", block)

        if not all([source_match, target_match, idf_match, cosine_match, col_count_match]):
            print(f"Skipping block {i} due to missing fields:\n{block}\n")
            continue

        source_time = float(source_match.group(1))
        target_time = float(target_match.group(1))
        idf_time = float(idf_match.group(1))
        cosine_time = float(cosine_match.group(1))
        col_count = int(col_count_match.group(1))

        data.append((col_count, source_time, target_time, idf_time, cosine_time))
    except Exception as e:
        print(f"Failed to parse block {i}: {e}\n{block}\n")

if not data:
    raise ValueError("没有成功解析任何数据块！")

data = np.array(data)
col_counts = data[:, 0]
source_times = data[:, 1]
target_times = data[:, 2]
idf_times = data[:, 3]
cosine_times = data[:, 4]

fit = lambda x, y: np.poly1d(np.polyfit(x, y, 1))

source_fit = fit(col_counts, source_times)
target_fit = fit(col_counts, target_times)
idf_fit = fit(col_counts, idf_times)
cosine_fit = fit(col_counts, cosine_times)

plt.figure(figsize=(7, 4))
plt.scatter(col_counts, source_times, label="Source Embeddings", color="blue")
plt.plot(col_counts, source_fit(col_counts), color="blue", linestyle='--')

plt.scatter(col_counts, target_times, label="Target Embeddings", color="green")
plt.plot(col_counts, target_fit(col_counts), color="green", linestyle='--')

plt.scatter(col_counts, idf_times, label="IDF Values", color="orange")
plt.plot(col_counts, idf_fit(col_counts), color="orange", linestyle='--')

plt.scatter(col_counts, cosine_times, label="Cosine Similarity", color="red")
plt.plot(col_counts, cosine_fit(col_counts), color="red", linestyle='--')

plt.xlabel("Total Column Count")
plt.ylabel("Time (seconds)")
plt.title("Time vs. Total Column Count")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("time_vs_column_count.png", dpi=300)
print("图像已保存为 time_vs_column_count.png")
