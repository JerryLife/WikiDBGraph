import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. 指定 CSV 文件的完整路径
csv_path = '/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_fullneg/predictions.csv'

# 2. 读取 CSV 文件
df = pd.read_csv(csv_path)

# 3. 分别筛选出 label=1 与 label=0 的数据
df_label_1 = df[df['label'] == 1]
df_label_0 = df[df['label'] == 0]

# 4. 绘制相似度分布直方图（以绝对数量为纵轴）
plt.figure(figsize=(8, 6))
plt.hist(df_label_1['similarity'], bins=30, alpha=0.5, label='label=1', color='orange', density=False)
plt.hist(df_label_0['similarity'], bins=30, alpha=0.5, label='label=0', color='blue', density=False)
plt.yscale('log', base=2)

# 5. 设置图例、标题和坐标轴
plt.legend()
plt.xlabel('Similarity')
plt.ylabel('Density')
plt.title('Distribution of Similarity by Label')

# 6. 将图形保存在 CSV 同一目录下
csv_dir = os.path.dirname(csv_path)                 # 获取 CSV 所在文件夹路径
output_filename = 'similarity_distribution_frequency.png'      # 想要保存的文件名
output_path = os.path.join(csv_dir, output_filename) # 拼接保存路径
plt.savefig(output_path, dpi=300)

# 7. 显示图形（可选）
plt.show()

print(f"Figure saved to: {output_path}")


# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_recall_curve, average_precision_score

# def analyze_and_plot_pr_curve(csv_path, output_dir):
#     os.makedirs(output_dir, exist_ok=True)

#     # ========== 1. 读取CSV ==========
#     df = pd.read_csv(csv_path)

#     # 确保 label 为整数（0或1）
#     df['label'] = df['label'].astype(int)

#     y_true = df['label'].values
#     y_scores = df['similarity'].values

#     # ========== 2. 计算 PR 曲线 ==========
#     precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
#     ap = average_precision_score(y_true, y_scores)

#     f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
#     best_idx = np.argmax(f1)
#     best_threshold = thresholds[best_idx]
#     best_precision = precision[best_idx]
#     best_recall = recall[best_idx]
#     best_f1 = f1[best_idx]

#     # ========== 3. 绘制PR曲线 ==========
#     plt.figure(figsize=(5, 3))
#     plt.plot(recall, precision, label=f"AP = {ap:.4f}", color="darkorange", linewidth=2)
#     plt.scatter(best_recall, best_precision, color="red", s=80,
#                 label=f"Best F1 = {best_f1:.3f}\n@Sim ≥ {best_threshold:.4f}")
#     plt.xlabel("Recall")
#     plt.ylabel("Precision")
#     plt.title("Precision-Recall Curve")
#     plt.grid(True, linestyle="--", alpha=0.3)
#     plt.legend()
#     plt.tight_layout()

#     pr_path = os.path.join(output_dir, "precision_recall_curve.png")
#     plt.savefig(pr_path)
#     plt.close()
#     print(f"📈 Saved PR curve: {pr_path}")
#     print(f"✅ Best threshold: {best_threshold:.4f}, F1: {best_f1:.4f}")

#     # ========== 4. 筛选 label=0 且 similarity > best_threshold ==========
#     df_label_0 = df[df['label'] == 0]
#     df_label_0_exceed = df_label_0[df_label_0['similarity'] > best_threshold]

#     count_exceed = len(df_label_0_exceed)
#     count_total = len(df_label_0)
#     ratio = count_exceed / count_total if count_total else 0.0

#     print(f"在所有 label=0 的数据中，similarity > {best_threshold:.4f} 的组合占比: "
#           f"{ratio:.4f} ({count_exceed}/{count_total})")

#     # ========== 5. 补0并保存 ==========
#     df_label_0_exceed['anchor_id'] = df_label_0_exceed['anchor_id'].astype(str).str.zfill(5)
#     df_label_0_exceed['target_id'] = df_label_0_exceed['target_id'].astype(str).str.zfill(5)

#     output_file = os.path.join(output_dir, f'label0_similarity_exceed_{best_threshold:.4f}.csv')
#     df_label_0_exceed.to_csv(output_file, index=False)
#     print(f"📄 已将筛选结果保存到: {output_file}")

# # 示例调用方式
# csv_path = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_fullneg/predictions.csv"
# output_dir = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results_fullneg"

# analyze_and_plot_pr_curve(csv_path, output_dir)
