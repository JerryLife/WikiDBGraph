
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=str, required=True, help="Input scores CSV")
    parser.add_argument("--plot", type=str, default=None, help="Output ROC plot path")
    args = parser.parse_args()
    
    df = pd.read_csv(args.scores)
    
    if len(df) == 0:
        print("No scores found.")
        return
        
    y_true = df["label"].values
    y_scores = df["score"].values
    
    try:
        auc_score = roc_auc_score(y_true, y_scores)
        print(f"ROC AUC: {auc_score:.4f}")
        
        if args.plot:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(args.plot)
            print(f"Saved ROC plot to {args.plot}")
            
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        # Identify issue (e.g. only one class)
        print(f"Labels distribution: {df['label'].value_counts()}")

if __name__ == "__main__":
    main()
