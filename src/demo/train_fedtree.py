import pandas as pd
import os
import sys
import json
import argparse
import numpy as np
import tempfile
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from fedtree import FLClassifier, FLRegressor

# Add src to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def prepare_fedtree_data(data_dict, mode="horizontal"):
    """
    Prepare data for FedTree training using Python API - aligned with XGBoost preprocessing.
    
    Args:
        data_dict: Dictionary containing training and test data for clients
        mode: "horizontal" or "vertical" 
        
    Returns:
        X_train, y_train, X_test, y_test: Combined training and test data
        n_parties: Number of parties
    """
    if mode == "horizontal":
        # Horizontal FL: combine all client data
        X_train_list = []
        y_train_list = []
        X_test_list = []
        y_test_list = []
        
        n_parties = len([k for k in data_dict.keys() if 'train' in k])
        
        for i in range(n_parties):
            train_key = f'client_{i}_train'
            test_key = f'client_{i}_test'
            
            if train_key in data_dict and test_key in data_dict:
                train_df = data_dict[train_key]
                test_df = data_dict[test_key]
                
                # Separate features and labels
                X_train_list.append(train_df.iloc[:, 1:].values)  # Features (exclude first column - label)
                y_train_list.append(train_df.iloc[:, 0].values)   # Label (first column)
                
                X_test_list.append(test_df.iloc[:, 1:].values)
                y_test_list.append(test_df.iloc[:, 0].values)
        
        # Combine all data
        X_train = np.vstack(X_train_list)
        y_train = np.hstack(y_train_list)
        X_test = np.vstack(X_test_list) 
        y_test = np.hstack(y_test_list)
        
    else:  # vertical
        # Vertical FL: combine features horizontally
        party_keys = [k for k in data_dict.keys() if 'train' in k]
        n_parties = len(party_keys)
        
        X_train_parts = []
        X_test_parts = []
        y_train = None
        y_test = None
        
        for i, train_key in enumerate(party_keys):
            test_key = train_key.replace('train', 'test')
            
            if test_key in data_dict:
                train_df = data_dict[train_key].copy()
                test_df = data_dict[test_key].copy()
                
                # Identify and remove ALL constant columns from both train and test
                const_cols = []
                for col in train_df.columns:
                    if col != 'Common_Gene_ID':
                        # Check if constant in BOTH train and test
                        train_unique = train_df[col].nunique()
                        test_unique = test_df[col].nunique()
                        if train_unique <= 1 and test_unique <= 1:
                            const_cols.append(col)
                
                if const_cols:
                    print(f"Removing constant columns from {train_key}: {const_cols}")
                    train_df = train_df.drop(columns=const_cols)
                    test_df = test_df.drop(columns=const_cols)
                
                if i == 0:  # First party has labels
                    label_col = train_df.columns[0]
                    y_train = train_df[label_col].values
                    y_test = test_df[label_col].values
                    
                    # Features exclude label and Common_Gene_ID
                    feature_cols = [col for col in train_df.columns 
                                  if col not in ['Common_Gene_ID', label_col]]
                    if feature_cols:
                        X_train_parts.append(train_df[feature_cols].values)
                        X_test_parts.append(test_df[feature_cols].values)
                else:  # Other parties only have features  
                    # Features exclude Common_Gene_ID
                    feature_cols = [col for col in train_df.columns if col != 'Common_Gene_ID']
                    if feature_cols:
                        X_train_parts.append(train_df[feature_cols].values)
                        X_test_parts.append(test_df[feature_cols].values)
        
        # Combine features horizontally
        if len(X_train_parts) > 1:
            X_train = np.hstack(X_train_parts)
            X_test = np.hstack(X_test_parts)
        elif len(X_train_parts) == 1:
            X_train = X_train_parts[0]
            X_test = X_test_parts[0]
        else:
            raise ValueError("No features left after removing constant columns")
    
    # Check for and handle missing/invalid values
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        print("Warning: Found NaN/inf values in X_train, replacing with 0")
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    
    if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
        print("Warning: Found NaN/inf values in X_test, replacing with 0")
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_train, y_train, X_test, y_test, n_parties




def run_fedtree_training(X_train, y_train, X_test, y_test, n_parties, mode="horizontal", task_type="classification"):
    """
    Run FedTree training using Python API - aligned with XGBoost approach.
    
    Args:
        X_train, y_train: Training data and labels (raw labels like XGBoost)
        X_test, y_test: Test data and labels (raw labels like XGBoost)
        n_parties: Number of parties
        mode: "horizontal" or "vertical"
        task_type: "classification" or "regression"
        
    Returns:
        training_results: Dictionary containing training metrics
    """
    results = {}
    
    try:
        # Determine number of classes and configure FedTree like XGBoost
        if task_type == "classification":
            # Use only training labels for mapping (consistent with XGBoost approach)
            train_labels = np.unique(y_train)
            n_classes = len(train_labels)
            print(f"Detected {n_classes} classes in training labels: {train_labels}")
            
            # If labels don't start from 0, create a simple mapping (safer than LabelEncoder)
            if not np.array_equal(train_labels, np.arange(n_classes)):
                print(f"Converting labels to 0-indexed for FedTree compatibility")
                label_mapping = {label: idx for idx, label in enumerate(train_labels)}
                print(f"Label mapping: {label_mapping}")
                
                # Apply mapping to train labels
                y_train_mapped = np.array([label_mapping[label] for label in y_train])
                
                # For test labels: map known labels, mark unknown labels as -1 (will be wrong predictions)
                y_test_mapped = []
                for label in y_test:
                    if label in label_mapping:
                        y_test_mapped.append(label_mapping[label])
                    else:
                        # Unknown label - mark as -1 (guaranteed wrong prediction, like XGBoost)
                        y_test_mapped.append(-1)
                        print(f"Warning: Test label {label} not seen in training, marking as incorrect")
                
                y_test_mapped = np.array(y_test_mapped)
                
                print(f"Mapped train labels range: {np.min(y_train_mapped)} to {np.max(y_train_mapped)}")
                print(f"Mapped test labels range: {np.min(y_test_mapped)} to {np.max(y_test_mapped)}")
                
                # Use mapped labels
                y_train = y_train_mapped
                y_test = y_test_mapped
            else:
                print(f"Labels already 0-indexed: {train_labels}")
            
            if n_classes == 2:
                objective = "binary:logistic"
                num_class = 2
            else:
                objective = "multi:softmax"
                num_class = n_classes
                
            print(f"Using objective: {objective}, num_class: {num_class}")
                
            # Create FedTree classifier
            model = FLClassifier(
                n_trees=50,
                max_depth=6,
                max_num_bin=256,
                learning_rate=0.1,
                mode=mode,
                n_parties=n_parties,
                num_class=num_class,
                objective=objective,
                bagging=1,
                column_sampling_rate=1.0
            )
        else:
            # Create FedTree regressor
            model = FLRegressor(
                n_trees=50,
                max_depth=6,
                max_num_bin=256,
                learning_rate=0.1,
                mode=mode,
                n_parties=n_parties,
                objective="reg:linear",
                bagging=1,
                column_sampling_rate=1.0
            )
        
        # Train the model
        print(f"Training FedTree {task_type} model with {n_parties} parties in {mode} mode...")
        print(f"Training data shape: X_train={X_train.shape}, y_train={y_train.shape}")
        print(f"Feature value ranges: min={X_train.min():.2f}, max={X_train.max():.2f}")
        print(f"Label distribution: {np.bincount(y_train.astype(int))}")
        
        model.fit(X_train, y_train)
        
        # Make predictions - avoid using FedTree's predict_proba to prevent segfaults
        print("Making predictions...")
        y_pred = model.predict(X_test)
        
        if task_type == "classification":
            # Safely calculate metrics without using FedTree's evaluation
            print("Calculating classification metrics...")
            
            # Ensure predictions are valid integers
            y_pred = np.array(y_pred, dtype=int)
            y_test = np.array(y_test, dtype=int)
            
            print(f"Predictions range: {np.min(y_pred)} to {np.max(y_pred)}")
            print(f"Prediction distribution: {np.bincount(y_pred)}")
            
            # Calculate metrics safely
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        else:
            # Calculate regression metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
        
        results['status'] = 'success'
        results['metrics'] = metrics
        results['model'] = model
        
        print(f"Training completed successfully. Metrics: {metrics}")
        
    except Exception as e:
        import traceback
        results['status'] = 'error'
        results['error'] = str(e)
        results['error_traceback'] = traceback.format_exc()
        results['model'] = None
        print(f"Training failed: {e}")
        print("Full error traceback:")
        traceback.print_exc()
    
    return results




def load_horizontal_data(database_ids=None):
    """Load horizontal federated learning data."""
    if database_ids is None:
        database_ids = ["02799", "79665"]
    
    project_root = get_project_root()
    base_dir = project_root / "data" / "clean"
    
    data_dict = {}
    
    for i, db_id in enumerate(database_ids):
        train_path = base_dir / db_id / "train_data.csv"
        test_path = base_dir / db_id / "test_data.csv"
        
        if train_path.exists() and test_path.exists():
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            data_dict[f'client_{i}_train'] = train_df
            data_dict[f'client_{i}_test'] = test_df
        else:
            print(f"Warning: Data files not found for database {db_id}")
    
    return data_dict


def load_vertical_data(base_output_dir="data/clean"):
    """Load vertical federated learning data."""
    project_root = get_project_root()
    if not os.path.isabs(base_output_dir):
        base_output_dir = os.path.join(project_root, base_output_dir)
    
    # Party A (df_a - from DB 48804, prefixed lpg_, contains the LABEL)
    path_a_train = os.path.join(base_output_dir, "48804", "aligned", "lpg_aligned_train.csv")
    path_a_test = os.path.join(base_output_dir, "48804", "aligned", "lpg_aligned_test.csv")

    # Party B (df_b - from DB 00381, prefixed tc_)
    path_b_train = os.path.join(base_output_dir, "00381", "aligned", "tc_aligned_train.csv")
    path_b_test = os.path.join(base_output_dir, "00381", "aligned", "tc_aligned_test.csv")

    # Check if files exist
    required_files = [path_a_train, path_a_test, path_b_train, path_b_test]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Error: Missing required files for vertical FL:")
        for f in missing_files:
            print(f"  {f}")
        return None

    # Load data
    try:
        df_a_train = pd.read_csv(path_a_train)
        df_a_test = pd.read_csv(path_a_test)
        df_b_train = pd.read_csv(path_b_train)
        df_b_test = pd.read_csv(path_b_test)
        
        return {
            'party_a_train': df_a_train,
            'party_a_test': df_a_test,
            'party_b_train': df_b_train,
            'party_b_test': df_b_test
        }
        
    except Exception as e:
        print(f"Error loading vertical data: {e}")
        return None


def save_results(results, algorithm_name="fedtree", seed=0, database_ids=None, mode="horizontal"):
    """Save experiment results to JSON file."""
    try:
        if database_ids is None:
            if mode == "horizontal":
                database_ids = ["02799", "79665"]
            else:
                database_ids = ["48804", "00381"]
            
        project_root = get_project_root()
        results_dir = project_root / "results" / mode
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create result data structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result_data = {
            "experiment_type": mode,
            "algorithm": algorithm_name,
            "timestamp": timestamp,
            "seed": seed,
            "database_ids": database_ids,
            "weighted_metrics": results.get("FedTree", {}),
            "individual_metrics": results.get("individual_metrics", {}),
            "training_info": results.get("training_info", {})
        }
        
        # Save to file
        if mode == "horizontal":
            filename = f"{algorithm_name}_{'_'.join(database_ids)}_seed{seed}.json"
        else:
            filename = f"{algorithm_name}_vertical_seed{seed}.json"
            
        output_path = results_dir / filename
        
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=4)
        
        print(f"Results saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"Error saving results: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Train FedTree model for federated learning')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--databases', nargs='+', default=['02799', '79665'], 
                        help='Database IDs for horizontal FL (default: 02799 79665)')
    parser.add_argument('--mode', choices=['horizontal', 'vertical'], default='horizontal',
                        help='Federated learning mode (default: horizontal)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    
    print(f"Starting FedTree {args.mode} federated learning with seed {args.seed}")
    print(f"Database IDs: {args.databases}")
    
    try:
        # Load data based on mode
        if args.mode == "horizontal":
            data_dict = load_horizontal_data(args.databases)
        else:
            data_dict = load_vertical_data()
        
        if data_dict is None or len(data_dict) == 0:
            print("Error: Failed to load data")
            return
        
        print(f"Loaded data for {len(data_dict)//2} parties")
        
        # Prepare FedTree data
        X_train, y_train, X_test, y_test, n_parties = prepare_fedtree_data(data_dict, args.mode)
        print(f"Prepared data: X_train shape {X_train.shape}, y_train shape {y_train.shape}")
        print(f"Test data: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
        
        # Run FedTree training
        print("Starting FedTree training...")
        training_results = run_fedtree_training(X_train, y_train, X_test, y_test, n_parties, args.mode)
        
        if training_results['status'] == 'success':
            print("Training completed successfully")
            
            # Get evaluation results from training
            evaluation_results = training_results['metrics']
            
            # Prepare final results
            final_results = {
                "FedTree": evaluation_results,
                "individual_metrics": evaluation_results,
                "training_info": {
                    "mode": args.mode,
                    "n_parties": n_parties,
                    "training_status": training_results['status']
                }
            }
            
            # Save results
            save_results(final_results, "fedtree", args.seed, args.databases, args.mode)
            
        else:
            print(f"Training failed: {training_results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()