import pandas as pd
import os
import sys
import numpy as np
import argparse
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Add src to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.FedAvg import FedAvg

"""
Primary Client Training Script

This script trains different machine learning models on only the primary client's data (48804 - LPG1L).
This is the client that contains the labels (lpg_Uni_Prot_Protein_Id) in the vertical federated learning setup.

The script applies similar data preprocessing (standardization and label encoding) as used in 
train_fedavg.py to ensure consistent comparisons.

Usage:
    python run_primary_client.py -m nn,xgb,rf,lr
    python run_primary_client.py -m nn        # Run only neural network
"""


def get_project_root():
    """Get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up from src/demo to project root
    return os.path.dirname(os.path.dirname(current_dir))

def load_primary_client_data(base_output_dir="data/clean"):
    """
    Load primary client data (48804 - LPG1L) that contains the labels.
    Returns data for training and testing.
    """
    # Get absolute paths
    project_root = get_project_root()
    if not os.path.isabs(base_output_dir):
        base_output_dir = os.path.join(project_root, base_output_dir)
    
    # Primary client data (DB 48804 - LPG1L) - contains labels
    path_train = os.path.join(base_output_dir, "48804", "aligned", "lpg_aligned_train.csv")
    path_test = os.path.join(base_output_dir, "48804", "aligned", "lpg_aligned_test.csv")

    print(f"Loading data for primary client (48804 - LPG1L) from:")
    print(f"  Train: {path_train}")
    print(f"  Test: {path_test}")

    # Check if files exist
    required_files = [path_train, path_test]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("Please run the data preparation script first.")
        return None

    # Load data
    df_train = pd.read_csv(path_train)
    df_test = pd.read_csv(path_test)

    print(f"Shape of primary client train (48804 - LPG1L): {df_train.shape}")
    print(f"Shape of primary client test (48804 - LPG1L): {df_test.shape}")

    return df_train, df_test


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics without AUC."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1


def train_neural_network(client_name, train_df, test_df, target_col, num_classes,
                        hidden_dims=[128, 128, 32], learning_rate=0.0001, 
                        local_epochs=5, global_rounds=20, batch_size=32, device='cpu',
                        unified_encoder=None, unified_scaler=None):
    """Train Neural Network using FedAvg architecture with preprocessing."""
    print(f"\n--- Training Neural Network for {client_name} ---")
    
    # Apply unified standardization
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    # Create copies to avoid modifying original data
    train_df_processed = train_df.copy()
    test_df_processed = test_df.copy()
    
    # Apply standardization using unified scaler
    if unified_scaler is not None:
        train_features = train_df_processed[feature_cols].fillna(0)
        test_features = test_df_processed[feature_cols].fillna(0)
        
        train_df_processed[feature_cols] = unified_scaler.transform(train_features)
        test_df_processed[feature_cols] = unified_scaler.transform(test_features)
        
        print(f"  Applied unified standardization to {len(feature_cols)} features")
    
    # Apply unified label encoding
    if unified_encoder is not None:
        train_df_processed[target_col] = unified_encoder.transform(train_df_processed[target_col].astype(str))
        test_df_processed[target_col] = unified_encoder.transform(test_df_processed[target_col].astype(str))
        print(f"  Applied unified label encoding")
    
    # Create single-client data dictionary
    train_data = {client_name: train_df_processed}
    test_data = {client_name: test_df_processed}
    
    # Initialize model
    model = FedAvg(
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        global_rounds=global_rounds,
        batch_size=batch_size,
        device=device
    )
    
    # Pre-set the label encoder to prevent FedAvg from creating its own
    if unified_encoder is not None:
        model.label_encoder = unified_encoder
    
    try:
        model.fit(train_data, target_col=target_col)
        results = model.eval(test_data, target_col=target_col)
        
        if client_name in results:
            accuracy, precision, recall, f1 = results[client_name]
            return (accuracy, precision, recall, f1)
        else:
            return None
    except Exception as e:
        print(f"Neural Network training failed: {e}")
        return None


def train_xgboost(client_name, train_df, test_df, target_col, num_classes, 
                 unified_encoder, unified_scaler=None):
    """Train XGBoost classifier with preprocessing."""
    print(f"\n--- Training XGBoost for {client_name} ---")
    
    try:
        # Prepare data
        X_train = train_df.drop(target_col, axis=1).fillna(0)
        y_train = train_df[target_col].dropna()
        X_test = test_df.drop(target_col, axis=1).fillna(0)
        y_test = test_df[target_col].dropna()
        
        # Apply unified standardization if available
        if unified_scaler is not None:
            X_train = unified_scaler.transform(X_train)
            X_test = unified_scaler.transform(X_test)
            print(f"  Applied unified standardization")
        
        # Use unified label encoder
        y_train_encoded = unified_encoder.transform(y_train.astype(str))
        y_test_encoded = unified_encoder.transform(y_test.astype(str))
        
        # --- START: LABEL RE-MAPPING LOGIC (same as run_individual_clients.py) ---
        
        # 1. Get the unique classes actually present in the local training data
        local_train_classes = np.unique(y_train_encoded)
        
        # 2. Create a mapping from the global label to a new, local, contiguous label
        #    e.g., global [0, 2, 3, 4] -> local [0, 1, 2, 3]
        global_to_local_map = {global_label: local_label for local_label, global_label in enumerate(local_train_classes)}
        
        # 3. Create the inverse mapping to convert predictions back to the global space
        local_to_global_map = {local_label: global_label for global_label, local_label in global_to_local_map.items()}
        
        # 4. Apply the mapping to the local training labels
        y_train_local_encoded = np.array([global_to_local_map[global_label] for global_label in y_train_encoded])
        
        # --- END: LABEL RE-MAPPING LOGIC ---

        # Inform the user about any classes in the test set that were not seen during local training
        train_classes_set = set(y_train_encoded)
        test_classes_set = set(y_test_encoded)
        unseen_in_train = test_classes_set - train_classes_set
        if unseen_in_train:
            print(f"  Warning: Test data contains classes not seen in training data: {sorted(list(unseen_in_train))}")
            print(f"  These will be treated as incorrect predictions")

        # Train model using the LOCAL number of classes and LOCAL labels
        local_num_classes = len(local_train_classes)
        
        if local_num_classes <= 1:
            print("  Skipping training: Client has only one class in its training data.")
            return None

        model = xgb.XGBClassifier(
            objective='multi:softmax',
            # Use the number of locally present classes
            num_class=local_num_classes,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            device='cuda'
        )
        
        # Fit the model on the re-mapped local labels
        model.fit(X_train, y_train_local_encoded)
        
        # Predict on the test set. Predictions will be in the LOCAL label space.
        y_pred_local = model.predict(X_test)
        
        # Map the local predictions back to the GLOBAL label space for evaluation
        y_pred_global = np.array([local_to_global_map.get(p, -1) for p in y_pred_local])
        
        # Handle unseen classes in test data - mark as incorrect
        if unseen_in_train:
            # For unseen test classes, they will already be incorrect since the model
            # can't predict classes it wasn't trained on
            pass
        
        # Calculate metrics using the global predictions and global true labels
        return calculate_metrics(y_test_encoded, y_pred_global)
        
    except Exception as e:
        print(f"XGBoost training failed: {e}")
        return None


def train_random_forest(client_name, train_df, test_df, target_col, num_classes, 
                       unified_encoder, unified_scaler=None):
    """Train Random Forest classifier with preprocessing."""
    print(f"\n--- Training Random Forest for {client_name} ---")
    
    try:
        # Prepare data
        X_train = train_df.drop(target_col, axis=1).fillna(0)
        y_train = train_df[target_col].dropna()
        X_test = test_df.drop(target_col, axis=1).fillna(0)
        y_test = test_df[target_col].dropna()
        
        # Apply unified standardization if available (though RF doesn't require it)
        if unified_scaler is not None:
            X_train = unified_scaler.transform(X_train)
            X_test = unified_scaler.transform(X_test)
            print(f"  Applied unified standardization")
        
        # Use unified label encoder
        y_train_encoded = unified_encoder.transform(y_train.astype(str))
        y_test_encoded = unified_encoder.transform(y_test.astype(str))
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train_encoded)
        y_pred = model.predict(X_test)
        
        return calculate_metrics(y_test_encoded, y_pred)
        
    except Exception as e:
        print(f"Random Forest training failed: {e}")
        return None


def train_logistic_regression(client_name, train_df, test_df, target_col, num_classes, 
                             unified_encoder, unified_scaler=None):
    """Train Logistic Regression classifier with preprocessing."""
    print(f"\n--- Training Logistic Regression for {client_name} ---")
    
    try:
        # Prepare data
        X_train = train_df.drop(target_col, axis=1).fillna(0)
        y_train = train_df[target_col].dropna()
        X_test = test_df.drop(target_col, axis=1).fillna(0)
        y_test = test_df[target_col].dropna()
        
        # Apply unified standardization (important for LR)
        if unified_scaler is not None:
            X_train = unified_scaler.transform(X_train)
            X_test = unified_scaler.transform(X_test)
            print(f"  Applied unified standardization")
        else:
            # If no unified scaler provided, create local scaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            print(f"  Applied local standardization")
        
        # Use unified label encoder
        y_train_encoded = unified_encoder.transform(y_train.astype(str))
        y_test_encoded = unified_encoder.transform(y_test.astype(str))
        
        # Train model
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='ovr' if num_classes > 2 else 'auto'
        )
        
        model.fit(X_train, y_train_encoded)
        y_pred = model.predict(X_test)
        
        return calculate_metrics(y_test_encoded, y_pred)
        
    except Exception as e:
        print(f"Logistic Regression training failed: {e}")
        return None


def train_svm(client_name, train_df, test_df, target_col, num_classes, 
             unified_encoder, unified_scaler=None):
    """Train SVM classifier with preprocessing."""
    print(f"\n--- Training SVM for {client_name} ---")
    
    try:
        # Prepare data
        X_train = train_df.drop(target_col, axis=1).fillna(0)
        y_train = train_df[target_col].dropna()
        X_test = test_df.drop(target_col, axis=1).fillna(0)
        y_test = test_df[target_col].dropna()
        
        # Apply unified standardization (important for SVM)
        if unified_scaler is not None:
            X_train = unified_scaler.transform(X_train)
            X_test = unified_scaler.transform(X_test)
            print(f"  Applied unified standardization")
        else:
            # If no unified scaler provided, create local scaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            print(f"  Applied local standardization")
        
        # Use unified label encoder
        y_train_encoded = unified_encoder.transform(y_train.astype(str))
        y_test_encoded = unified_encoder.transform(y_test.astype(str))
        
        # Train model
        model = SVC(
            kernel='rbf',
            random_state=42,
            probability=True  # Enable probability estimates for better compatibility
        )
        
        model.fit(X_train, y_train_encoded)
        y_pred = model.predict(X_test)
        
        return calculate_metrics(y_test_encoded, y_pred)
        
    except Exception as e:
        print(f"SVM training failed: {e}")
        return None


def test_models_on_primary_client(client_name, train_df, test_df, target_col, num_classes, 
                                 experiment_params, selected_models, unified_encoder, unified_scaler):
    """Test selected models on the primary client."""
    print(f"\n{'='*80}")
    print(f"Testing Selected Models on Primary Client: {client_name}")
    print(f"Selected models: {', '.join(selected_models)}")
    print(f"{'='*80}")
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Unique classes in train: {train_df[target_col].nunique()}")
    print(f"Unique classes in test: {test_df[target_col].nunique()}")
    
    results = {}
    
    # Neural Network
    if 'neural_network' in selected_models or 'nn' in selected_models:
        result = train_neural_network(
            client_name, train_df, test_df, target_col, num_classes,
            hidden_dims=experiment_params['hidden_dims'],
            learning_rate=experiment_params['learning_rate'],
            local_epochs=experiment_params['local_epochs'],
            global_rounds=experiment_params['global_rounds'],
            batch_size=experiment_params['batch_size'],
            device=experiment_params.get('device', 'cpu'),
            unified_encoder=unified_encoder,
            unified_scaler=unified_scaler
        )
        if result:
            results['Neural_Network'] = result
    
    # XGBoost
    if 'xgboost' in selected_models or 'xgb' in selected_models:
        result = train_xgboost(client_name, train_df, test_df, target_col, num_classes, 
                              unified_encoder, unified_scaler)
        if result:
            results['XGBoost'] = result
    
    # Random Forest
    if 'random_forest' in selected_models or 'rf' in selected_models:
        result = train_random_forest(client_name, train_df, test_df, target_col, num_classes, 
                                    unified_encoder, unified_scaler)
        if result:
            results['Random_Forest'] = result
    
    # Logistic Regression
    if 'logistic_regression' in selected_models or 'lr' in selected_models:
        result = train_logistic_regression(client_name, train_df, test_df, target_col, num_classes, 
                                          unified_encoder, unified_scaler)
        if result:
            results['Logistic_Regression'] = result
    
    # SVM
    if 'svm' in selected_models:
        result = train_svm(client_name, train_df, test_df, target_col, num_classes, 
                          unified_encoder, unified_scaler)
        if result:
            results['SVM'] = result
    
    return results


def save_results(results, experiment_params, selected_models, output_dir="results/primary_client"):
    """Save results to JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp and models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_str = "_".join(selected_models)
    filename = f"primary_client_48804_{models_str}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data for saving
    save_data = {
        "timestamp": timestamp,
        "client": "48804_LPG1L",
        "experiment_params": experiment_params,
        "selected_models": selected_models,
        "results": results,
        "summary": {}
    }
    
    # Add summary statistics
    if results:
        best_model = max(results.keys(), key=lambda x: results[x][3])  # F1 score
        save_data["summary"] = {
            "best_model": best_model,
            "best_f1": results[best_model][3],
            "all_models": {model: {"f1": metrics[3]} for model, metrics in results.items()}
        }
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def save_primary_results(results, experiment_params, selected_models, seed=0):
    """Save primary client results to JSON file in table-generation compatible format."""
    from pathlib import Path
    
    try:
        project_root = Path(get_project_root())
        results_dir = project_root / "results" / "primary_client"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create result data structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result_data = {
            "experiment_type": "primary_client",
            "client": "48804_LPG1L",
            "timestamp": timestamp,
            "seed": seed,
            "experiment_params": experiment_params,
            "selected_models": selected_models,
            "results": results
        }
        
        # Add summary statistics
        if results:
            best_model = max(results.keys(), key=lambda x: results[x][3])  # F1 score
            result_data["summary"] = {
                "best_model": best_model,
                "best_f1": results[best_model][3],
                "all_models": {model: {"f1": metrics[3]} for model, metrics in results.items()}
            }
        
        # Save to file with predictable naming including model names: primary_client_model_48804_seed.json
        # Create model string from selected models
        if len(selected_models) == 1:
            # Single model case
            model_map = {
                'neural_network': 'nn',
                'xgboost': 'xgb', 
                'random_forest': 'rf',
                'logistic_regression': 'lr',
                'svm': 'svm'
            }
            model_string = model_map.get(selected_models[0], selected_models[0])
        else:
            # Multiple models case - use abbreviated names
            model_abbrevs = []
            model_map = {
                'neural_network': 'nn',
                'xgboost': 'xgb', 
                'random_forest': 'rf',
                'logistic_regression': 'lr',
                'svm': 'svm'
            }
            for model in selected_models:
                model_abbrevs.append(model_map.get(model, model))
            model_string = "_".join(sorted(model_abbrevs))
        
        filename = f"primary_client_{model_string}_48804_seed{seed}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\n✅ Primary client results saved to: {filepath}")
        return str(filepath)
        
    except Exception as e:
        print(f"\n❌ Error saving primary client results: {e}")
        return None


def parse_models(models_str):
    """Parse model names from command line argument."""
    if not models_str:
        return ['neural_network', 'xgboost', 'random_forest', 'logistic_regression', 'svm']
    
    # Map abbreviations to full names
    model_map = {
        'nn': 'neural_network',
        'neural_network': 'neural_network',
        'xgb': 'xgboost', 
        'xgboost': 'xgboost',
        'rf': 'random_forest',
        'random_forest': 'random_forest',
        'lr': 'logistic_regression',
        'logistic_regression': 'logistic_regression',
        'svm': 'svm'
    }
    
    models = [m.strip().lower() for m in models_str.split(',')]
    selected_models = []
    
    for model in models:
        if model in model_map:
            selected_models.append(model_map[model])
        else:
            print(f"Warning: Unknown model '{model}'. Available models: {list(model_map.keys())}")
    
    return selected_models if selected_models else ['neural_network', 'xgboost', 'random_forest', 'logistic_regression', 'svm']


def main():
    parser = argparse.ArgumentParser(description='Run primary client training with different models')
    parser.add_argument('-m', '--models', type=str, default=None,
                       help='Comma-separated list of models to run. Options: nn/neural_network, xgb/xgboost, rf/random_forest, lr/logistic_regression, svm. Default: all models')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility and consistent file naming')
    
    args = parser.parse_args()
    
    # Parse selected models
    selected_models = parse_models(args.models)
    print(f"Selected models: {selected_models}")
    
    # Set parameters
    DEVICE = 'cuda:1'
    TARGET_COL = 'lpg_Uni_Prot_Protein_Id'  # Primary client label column
    CLIENT_NAME = 'primary_48804'
    
    # Load primary client data
    print("Loading primary client data...")
    train_df, test_df = load_primary_client_data()
    
    if train_df is None or test_df is None:
        print("Failed to load data. Exiting.")
        sys.exit(1)
    
    # Create unified label encoder from primary client data
    all_labels = set(train_df[TARGET_COL].unique()) | set(test_df[TARGET_COL].unique())
    all_labels = sorted([str(label) for label in all_labels])
    unified_encoder = LabelEncoder()
    unified_encoder.fit(all_labels)
    
    NUM_CLASSES = len(all_labels)
    print(f"Created unified label encoder with {NUM_CLASSES} classes: {all_labels[:5]}{'...' if len(all_labels) > 5 else ''}")
    
    # Create unified StandardScaler from training data
    print("Creating unified StandardScaler...")
    feature_cols = [col for col in train_df.columns if col != TARGET_COL]
    
    unified_scaler = StandardScaler()
    unified_scaler.fit(train_df[feature_cols].fillna(0))
    print(f"Created unified StandardScaler for {len(feature_cols)} features")
    
    # Experiment parameters
    experiment_params = {
        'hidden_dims': [64, 32],
        'learning_rate': 0.0001,
        'local_epochs': 5,
        'global_rounds': 20,
        'batch_size': 32,
        'device': DEVICE
    }
    
    # Test selected models on primary client
    results = test_models_on_primary_client(
        CLIENT_NAME, train_df, test_df, 
        TARGET_COL, NUM_CLASSES, experiment_params, selected_models, 
        unified_encoder, unified_scaler
    )
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("PRIMARY CLIENT MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n{CLIENT_NAME.upper()}:")
    print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 70)
    
    for model_name, metrics in results.items():
        accuracy, precision, recall, f1 = metrics
        print(f"{model_name:<20} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
    
    # Find best model
    print(f"\n{'='*80}")
    print("BEST PERFORMING MODEL")
    print(f"{'='*80}")
    
    if results:
        best_model = max(results.keys(), key=lambda x: results[x][3])  # F1 score
        best_f1 = results[best_model][3]
        
        print(f"\nBest Model: {best_model}")
        print(f"F1 Score: {best_f1:.4f}")
        
        # Show all models' F1 scores for comparison
        print(f"\nAll F1 Scores:")
        for model_name, metrics in results.items():
            f1_score = metrics[3]
            marker = "✓" if model_name == best_model else " "
            print(f"  {marker} {model_name}: F1={f1_score:.4f}")

    # Save results using the table-generation compatible function
    output_file = save_primary_results(results, experiment_params, selected_models, seed=args.seed)
    
    print(f"\n{'='*80}")
    print(f"Experiment completed.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()