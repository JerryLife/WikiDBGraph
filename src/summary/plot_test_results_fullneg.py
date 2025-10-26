import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_curve,
    roc_auc_score
)

def analyze_and_plot_metrics(csv_path, output_dir):
    """
    Analyze similarity predictions using PR and ROC curves.
    Save plots and export label=0 pairs exceeding thresholds from both metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df['label'] = df['label'].astype(int)

    y_true = df['label'].values
    y_scores = df['similarity'].values

    # ====================== Precision-Recall Curve ======================
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    best_pr_idx = np.argmax(f1)
    best_pr_threshold = thresholds_pr[best_pr_idx]
    best_pr = precision[best_pr_idx]
    best_rc = recall[best_pr_idx]
    best_f1 = f1[best_pr_idx]

    plt.figure(figsize=(5, 3))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}", color="darkorange", linewidth=2)
    plt.scatter(best_rc, best_pr, color="red", s=80,
                label=f"Best F1 = {best_f1:.3f}\n@Sim ≥ {best_pr_threshold:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    pr_path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(pr_path)
    plt.close()

    # =========================== ROC Curve =============================
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    j_scores = tpr - fpr
    best_roc_idx = np.argmax(j_scores)
    best_roc_threshold = thresholds_roc[best_roc_idx]
    best_tpr = tpr[best_roc_idx]
    best_fpr = fpr[best_roc_idx]
    best_j = j_scores[best_roc_idx]

    plt.figure(figsize=(5, 3))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="blue", linewidth=2)
    plt.scatter(best_fpr, best_tpr, color="red", s=80,
                label=f"Best J = {best_j:.3f}\n@Sim ≥ {best_roc_threshold:.4f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    auc_path = os.path.join(output_dir, "roc_auc_curve.png")
    plt.savefig(auc_path)
    plt.close()

    print(f"PR curve saved to:  {pr_path}")
    print(f"ROC curve saved to: {auc_path}")

    # ===================== Export PR-based label=0 =====================
    df_label_0 = df[df['label'] == 0]
    df_label_0_exceed_pr = df_label_0[df_label_0['similarity'] > best_pr_threshold]

    count_exceed_pr = len(df_label_0_exceed_pr)
    total_0 = len(df_label_0)
    ratio_pr = count_exceed_pr / total_0 if total_0 else 0.0
    print(f"[PR-F1] Label=0 exceeding {best_pr_threshold:.4f}: {count_exceed_pr}/{total_0} ({ratio_pr:.2%})")

    df_label_0_exceed_pr['anchor_id'] = df_label_0_exceed_pr['anchor_id'].astype(str).str.zfill(5)
    df_label_0_exceed_pr['target_id'] = df_label_0_exceed_pr['target_id'].astype(str).str.zfill(5)

    pr_output_file = os.path.join(output_dir, f'label0_similarity_exceed_PRF1_{best_pr_threshold:.4f}.csv')
    df_label_0_exceed_pr.to_csv(pr_output_file, index=False)
    print(f"Exceeding PR-F1 cases saved to: {pr_output_file}")

    # ===================== Export ROC-based label=0 =====================
    df_label_0_exceed_roc = df_label_0[df_label_0['similarity'] > best_roc_threshold]

    count_exceed_roc = len(df_label_0_exceed_roc)
    ratio_roc = count_exceed_roc / total_0 if total_0 else 0.0
    print(f"[ROC-Youden] Label=0 exceeding {best_roc_threshold:.4f}: {count_exceed_roc}/{total_0} ({ratio_roc:.2%})")

    df_label_0_exceed_roc['anchor_id'] = df_label_0_exceed_roc['anchor_id'].astype(str).str.zfill(5)
    df_label_0_exceed_roc['target_id'] = df_label_0_exceed_roc['target_id'].astype(str).str.zfill(5)

    roc_output_file = os.path.join(output_dir, f'label0_similarity_exceed_ROC_{best_roc_threshold:.4f}.csv')
    df_label_0_exceed_roc.to_csv(roc_output_file, index=False)
    print(f"Exceeding ROC-Youden cases saved to: {roc_output_file}")

def main():
    csv_path = "out/col_matcher_bge-m3_lr1e-5_ft_database/test/test_results_fullneg/predictions.csv"
    output_dir = "fig/test_results_fullneg"
    os.makedirs(output_dir, exist_ok=True)
    analyze_and_plot_metrics(csv_path, output_dir)

if __name__ == "__main__":
    main()
