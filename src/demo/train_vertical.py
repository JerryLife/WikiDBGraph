import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

# MODIFIED: Label is now from DB 48804 (which will be party A)


def train_xgboost(df_train, n_classes=2, device='cuda'):
    """
    Train a XGBoost model on the given data.
    Return a model and label encoder for the target variable.
    """
    cols_to_drop = [LABEL]
    if 'Common_Gene_ID' in df_train.columns: # Common_Gene_ID is for alignment, not a feature
        cols_to_drop.append('Common_Gene_ID')
    
    X = df_train.drop(columns=cols_to_drop, errors='ignore')
    y = df_train[LABEL]
    
    X = X.to_numpy()
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    if n_classes == 2:
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=0,
            device=device
        )
    else:
        model = xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=n_classes,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=0,
            device=device
        )
    
    model.fit(X, y)
    return model, le

def eval_xgboost(model, df_test, le, n_classes=2, device='cuda'):
    """
    Evaluate the performance of the XGBoost model on the given data.
    Uses the label encoder 'le' fitted on the training data.
    """
    cols_to_drop = [LABEL]
    if 'Common_Gene_ID' in df_test.columns:
        cols_to_drop.append('Common_Gene_ID')

    X_test = df_test.drop(columns=cols_to_drop, errors='ignore')
    y_test_original = df_test[LABEL].copy()
    
    X_test = X_test.to_numpy()
    y_pred = model.predict(X_test)
    
    y_test_encoded = np.full(len(y_test_original), -1, dtype=int)
    seen_mask_indices = []
    seen_labels_original = []

    for i, label_val in enumerate(y_test_original):
        if label_val in le.classes_:
            seen_mask_indices.append(i)
            seen_labels_original.append(label_val)
    
    if seen_labels_original:
        y_test_encoded[seen_mask_indices] = le.transform(seen_labels_original)

    unseen_count = len(y_test_original) - len(seen_labels_original)
    if unseen_count > 0:
        print(f"Warning: Found {unseen_count} unseen labels in test data for label '{LABEL}'. Encoded as -1.")

    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_encoded, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return accuracy, precision, recall, f1

def train_eval_vertical(df_a_train, df_a_test, df_b_train, df_b_test, name_a=None, name_b=None, n_classes=2, device='cuda'):
    """
    Train XGBoost models for VFL scenario and evaluate performance.
    df_a_train contains the label and features for party A.
    df_b_train contains features for party B (already prefixed).
    Assumes df_a_train and df_b_train are row-aligned.
    """
    
    print(f"\n--- Training and Evaluating Model for Party {name_a} (using only its own features) ---")
    model_a, le_a = train_xgboost(df_a_train, n_classes, device)
    print(f"Evaluating model for Party {name_a} on its test set...")
    metrics_a = eval_xgboost(model_a, df_a_test, le_a, n_classes, device)

    # Prepare data for the combined model (VFL scenario)
    # Features from A (excluding label and ID)
    cols_to_drop_a = [LABEL]
    if 'Common_Gene_ID' in df_a_train.columns:
        cols_to_drop_a.append('Common_Gene_ID')
    X_a_train = df_a_train.drop(columns=cols_to_drop_a, errors='ignore')
    X_a_test = df_a_test.drop(columns=cols_to_drop_a, errors='ignore')

    # Features from B (excluding ID)
    cols_to_drop_b = []
    if 'Common_Gene_ID' in df_b_train.columns:
        cols_to_drop_b.append('Common_Gene_ID')
    X_b_train = df_b_train.drop(columns=cols_to_drop_b, errors='ignore')
    X_b_test = df_b_test.drop(columns=cols_to_drop_b, errors='ignore')
    
    # Ensure X_b_train and X_b_test have the same columns in the same order
    # This can happen if some Common_Gene_IDs in train are not in test or vice-versa,
    # leading to different sets of lpg_ or tc_ columns after merge if some were all NaN and dropped.
    # However, the way aligned CSVs are created (from a single merged df split) should prevent this issue
    # as long as the column selection for saving was `df_x_final_cols`.

    # Concatenate features from A and B. Assumes row-alignment from prior data prep.
    # Columns in X_a_train and X_b_train should already be uniquely prefixed and distinct.
    X_ab_train_combined = pd.concat([X_a_train, X_b_train], axis=1)
    X_ab_test_combined = pd.concat([X_a_test, X_b_test], axis=1)

    # Re-attach the label (which comes from df_a_train) for training the combined model
    y_train_labels = df_a_train[LABEL]
    y_test_labels = df_a_test[LABEL]

    df_ab_train_for_xgb = pd.concat([X_ab_train_combined, y_train_labels], axis=1)
    df_ab_test_for_xgb = pd.concat([X_ab_test_combined, y_test_labels], axis=1)
    
    print(f"\n--- Training and Evaluating Combined Model (VFL: {name_a} + {name_b}) ---")
    # le_a (fitted on df_a_train[LABEL]) is the correct encoder for the combined model's label
    model_ab, le_ab_combined = train_xgboost(df_ab_train_for_xgb, n_classes, device) # le_ab_combined will be same as le_a
    print(f"Evaluating combined model on its combined test set...")
    metrics_ab = eval_xgboost(model_ab, df_ab_test_for_xgb, le_a, n_classes, device)

    return metrics_a, metrics_ab

if __name__ == "__main__":
    LABEL = "lpg_Uni_Prot_Protein_Id"
    base_output_dir = "data/clean"  # Directory used in your data preparation script

    # MODIFIED: Party A is now DB 48804 (LPG1L Orthologs), Party B is DB 00381 (TC Orthologs)
    # Party A (df_a - from DB 48804, prefixed lpg_, contains the LABEL)
    path_a_train = os.path.join(base_output_dir, "48804", "aligned", "lpg_aligned_train.csv")
    path_a_test = os.path.join(base_output_dir, "48804", "aligned", "lpg_aligned_test.csv")

    # Party B (df_b - from DB 00381, prefixed tc_)
    path_b_train = os.path.join(base_output_dir, "00381", "aligned", "tc_aligned_train.csv")
    path_b_test = os.path.join(base_output_dir, "00381", "aligned", "tc_aligned_test.csv")

    print(f"Loading data for party A (48804 - LPG1L) from: {path_a_train}, {path_a_test}")
    print(f"Loading data for party B (00381 - TC) from: {path_b_train}, {path_b_test}")

    if not (os.path.exists(path_a_train) and os.path.exists(path_a_test) and \
            os.path.exists(path_b_train) and os.path.exists(path_b_test)):
        print("Error: One or more required aligned CSV files not found. " \
              "Please ensure the data preparation script (`create_merged_dataset`) has run successfully " \
              "and paths are correct.")
    else:
        df_a_train = pd.read_csv(path_a_train)
        df_a_test = pd.read_csv(path_a_test)
        df_b_train = pd.read_csv(path_b_train)
        df_b_test = pd.read_csv(path_b_test)

        print(f"Shape of df_a_train (48804 - LPG1L): {df_a_train.shape}")
        print(f"Shape of df_b_train (00381 - TC): {df_b_train.shape}")

        if LABEL not in df_a_train.columns:
            print(f"Error: Label column '{LABEL}' not found in df_a_train (from 48804 data). " \
                  f"Available columns: {df_a_train.columns.tolist()}")
        elif df_a_train.empty or df_b_train.empty:
            print("Error: One or both training dataframes are empty after loading.")
        else:
            # Calculate n_classes based on the chosen LABEL column from the training and test sets of Party A
            try:
                # Ensure LABEL column exists and handle potential NaNs before nunique
                label_series_train = df_a_train[LABEL].dropna().astype(str)
                label_series_test = df_a_test[LABEL].dropna().astype(str)
                combined_labels_unique = pd.concat([label_series_train, label_series_test]).unique()
                n_classes = len(combined_labels_unique)
                
                if n_classes == 0:
                    print(f"Error: Label column '{LABEL}' has no non-NaN unique values. Cannot determine n_classes.")
                    n_classes = 2 # Default to avoid crash, but this indicates a data problem
                elif n_classes == 1:
                    print(f"Warning: Label column '{LABEL}' has only 1 unique class: {combined_labels_unique[0]}. " \
                          "This is not suitable for classification. Setting n_classes=2 for binary setup, but check your label.")
                    n_classes = 2 
                # If n_classes = 2, XGBoost binary:logistic is appropriate.
                # If n_classes > 2, XGBoost multi:softmax is appropriate.
                # If n_classes = 1 after LabelEncoding (0), it's problematic.
            except KeyError:
                print(f"Error: Label column '{LABEL}' not found while trying to determine n_classes.")
                n_classes = 2 # Fallback

            print(f"Determined n_classes = {n_classes} for label '{LABEL}'.")

            metrics_party_a, metrics_combined = train_eval_vertical(
                df_a_train, df_a_test,
                df_b_train, df_b_test,
                name_a="48804 (LPG1L)", name_b="00381 (TC)",
                n_classes=n_classes, device="cuda:1" # Change to "cpu" if no GPU
            )

            print("-" * 100)
            print(f"FINAL VFL RESULTS (Prediction Target: '{LABEL}', Num Classes: {n_classes})")
            print("---")
            print(f"Model trained on DB 48804 (LPG1L) features only (tested on 48804 test set):")
            print(f"  Accuracy: {metrics_party_a[0]:.4f} \t Precision: {metrics_party_a[1]:.4f} \t " \
                  f"Recall: {metrics_party_a[2]:.4f} \t F1 Score: {metrics_party_a[3]:.4f}")
            print("---")
            print(f"Model trained on Combined (48804_LPG1L + 00381_TC) features (tested on combined test set):")
            print(f"  Accuracy: {metrics_combined[0]:.4f} \t Precision: {metrics_combined[1]:.4f} \t " \
                  f"Recall: {metrics_combined[2]:.4f} \t F1 Score: {metrics_combined[3]:.4f}")
            print("-" * 100)