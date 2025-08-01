import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import numpy as np





def train_xgboost(df_train, n_classes=2, device='cuda'):
    """
    Train a XGBoost model on the given data and evaluate it on the test set
    Return a model
    """
    # Prepare the data
    # Using gene_classification as the target variable
    X = df_train.drop(LABEL, axis=1)
    y = df_train[LABEL]
    
    # Fill NaN values with 0
    X = X.fillna(0)
    
    if n_classes == 2:
        # Create and train the model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            num_class=len(y.unique()),
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=0,
            device=device
        )
    else:
        # Create and train the model
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
    return model


def eval_xgboost(model, df_test, n_classes=2, device='cuda'):
    """
    Evaluate the performance of the XGBoost model on the given data
    Return the accuracy, precision, recall, and F1 score
    """
    # Prepare the test data
    X_test = df_test.drop(LABEL, axis=1)
    y_test = df_test[LABEL]
    
    # Fill NaN values with 0
    X_test = X_test.fillna(0)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    if n_classes == 2:
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    else:
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return accuracy, precision, recall, f1


def eval_horizontal(df_a_train, df_a_test, df_b_train, df_b_test, name_a=None, name_b=None, n_classes=2, device='cuda'):
    """
    Evaluate the performance of the XGBoost model on the given data
    """

    print(f"Training model on dataset {name_a}...")
    model_a = train_xgboost(df_a_train, n_classes, device)
    
    print(f"Training model on dataset {name_b}...")
    model_b = train_xgboost(df_b_train, n_classes, device)
    
    print(f"Training model on combined datasets {name_a} and {name_b}...")
    df_ab_train = pd.concat([df_a_train, df_b_train])
    model_ab = train_xgboost(df_ab_train, device)
    
    print(f"\nEvaluating model trained on {name_a} with test data from {name_a}:")
    metrics_a_a = eval_xgboost(model_a, df_a_test, device)
    
    print(f"\nEvaluating model trained on {name_b} with test data from {name_b}:")
    metrics_b_b = eval_xgboost(model_b, df_b_test, device)
    
    print(f"\nEvaluating model trained on combined data with test data from {name_a}:")
    metrics_ab_a = eval_xgboost(model_ab, df_a_test, device)
    
    print(f"\nEvaluating model trained on combined data with test data from {name_b}:")
    metrics_ab_b = eval_xgboost(model_ab, df_b_test, device)
    
    # Combine test sets for final evaluation
    df_ab_test = pd.concat([df_a_test, df_b_test])
    print(f"\nEvaluating model trained on combined data with combined test data:")
    metrics_ab_ab = eval_xgboost(model_ab, df_ab_test, device)
    
    return {
        f"{name_a}_model": metrics_a_a,
        f"{name_b}_model": metrics_b_b,
        "combined_model_on_combined_data": metrics_ab_ab,
        f"combined_model_on_{name_a}": metrics_ab_a,
        f"combined_model_on_{name_b}": metrics_ab_b
    }


if __name__ == "__main__":
    df_a_train = pd.read_csv("data/clean/02799/train_data.csv")
    df_a_test = pd.read_csv("data/clean/02799/test_data.csv")
    df_b_train = pd.read_csv("data/clean/79665/train_data.csv")
    df_b_test = pd.read_csv("data/clean/79665/test_data.csv")

    LABEL = 'details_encoded_protein'
    n_classes = len(set(df_a_train[LABEL].unique()) | set(df_b_train[LABEL].unique()) | set(df_a_test[LABEL].unique()) | set(df_b_test[LABEL].unique()))
    results = eval_horizontal(df_a_train, df_a_test, df_b_train, df_b_test, name_a="02799", name_b="79665", n_classes=2)
    
    print("-" * 100)
    print(f"Number of classes: {n_classes}")
    print(f"Label: {LABEL}")
    print(f"Results for 02799 and 79665:")
    for method, result in results.items():
        print(f"{method}: Accuracy: {result[0]:.4f} \t Precision: {result[1]:.4f} \t Recall: {result[2]:.4f} \t F1 Score: {result[3]:.4f}")
    print("-" * 100)


