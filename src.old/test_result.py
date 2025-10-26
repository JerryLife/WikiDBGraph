import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# === Load CSV ===
path = "/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results/predictions.csv"
df = pd.read_csv(path)

# === Threshold prediction ===
threshold = 0.6405
df["pred"] = (df["similarity"] > threshold).astype(int)

# === Extract ground truth and predictions ===
y_true = df["label"]
y_pred = df["pred"]
y_score = df["similarity"]

# === Classification Metrics ===
print("=== Classification Report ===")
print(classification_report(y_true, y_pred, digits=4))

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_score)
ap = average_precision_score(y_true, y_score)

print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print(f"ROC AUC        : {roc_auc:.4f}")
print(f"Average Prec.  : {ap:.4f}")

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)
labels = ["False", "True"]

print("\n=== Confusion Matrix (percentage) ===")
cm_percent = cm / cm.sum() * 100
cm_display = pd.DataFrame(cm_percent, index=[f"Actual {l}" for l in labels], columns=[f"Predicted {l}" for l in labels])
print(cm_display.round(2))

# === PR Curve ===
precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_score)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, label=f"AP = {ap:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results/pr_curve.png")
print("Saved PR curve to /hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/test_results/pr_curve.png")
