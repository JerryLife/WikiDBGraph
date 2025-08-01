import pandas as pd
import os
import sys
import numpy as np
import argparse
import json
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add src to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.SplitNN import SplitNN


def get_project_root():
    """Get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up from src/demo to project root
    return os.path.dirname(os.path.dirname(current_dir))

def load_vertical_data(base_output_dir="data/clean"):
    """
    Load aligned vertical federated learning data.
    Returns data dictionaries for training and testing.
    """
    # Get absolute paths
    project_root = get_project_root()
    if not os.path.isabs(base_output_dir):
        base_output_dir = os.path.join(project_root, base_output_dir)
    
    # Party A (df_a - from DB 48804, prefixed lpg_, contains the LABEL)
    path_a_train = os.path.join(base_output_dir, "48804", "aligned", "lpg_aligned_train.csv")
    path_a_test = os.path.join(base_output_dir, "48804", "aligned", "lpg_aligned_test.csv")

    # Party B (df_b - from DB 00381, prefixed tc_)
    path_b_train = os.path.join(base_output_dir, "00381", "aligned", "tc_aligned_train.csv")
    path_b_test = os.path.join(base_output_dir, "00381", "aligned", "tc_aligned_test.csv")

    print(f"Loading data for party A (48804 - LPG1L) from: {path_a_train}, {path_a_test}")
    print(f"Loading data for party B (00381 - TC) from: {path_b_train}, {path_b_test}")

    # Check if files exist
    required_files = [path_a_train, path_a_test, path_b_train, path_b_test]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("Please run the data preparation script first.")
        return None, None

    # Load data
    df_a_train = pd.read_csv(path_a_train)
    df_a_test = pd.read_csv(path_a_test)
    df_b_train = pd.read_csv(path_b_train)
    df_b_test = pd.read_csv(path_b_test)

    print(f"Shape of df_a_train (48804 - LPG1L): {df_a_train.shape}")
    print(f"Shape of df_b_train (00381 - TC): {df_b_train.shape}")

    # Create data dictionaries
    train_data = {
        "client_A": df_a_train,
        "client_B": df_b_train
    }
    
    test_data = {
        "client_A": df_a_test,
        "client_B": df_b_test
    }
    
    return train_data, test_data


def train_splitnn_model(train_data, test_data, target_col, num_classes=2,
                       client_hidden_dims=[[64, 32], [64, 32]],
                       server_hidden_dims=[32, 16],
                       learning_rate=0.001,
                       epochs=50,
                       batch_size=32,
                       device='cpu',
                       linkage_col='Common_Gene_ID'):
    """
    Train and evaluate SplitNN model.
    """
    print(f"\n=== Training SplitNN Model ===")
    print(f"Client hidden dims: {client_hidden_dims}")
    print(f"Server hidden dims: {server_hidden_dims}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Linkage column: {linkage_col}")
    print(f"Target column: {target_col}")
    print(f"Number of classes (specified): {num_classes}")
    
    # Verify target column exists
    label_client = None
    for client_name, df in train_data.items():
        if target_col in df.columns:
            unique_classes = df[target_col].nunique()
            label_client = client_name
            print(f"Found target column '{target_col}' in {client_name} with {unique_classes} unique classes in data")
            break
    
    if label_client is None:
        raise ValueError(f"Target column '{target_col}' not found in any client")
    
    # Initialize SplitNN model
    model = SplitNN(
        client_hidden_dims=client_hidden_dims,
        server_hidden_dims=server_hidden_dims,
        num_classes=num_classes,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        device=device
    )
    
    # Train the model
    print("\n--- Training Phase ---")
    try:
        model.fit(train_data, target_col=target_col, linkage_col=linkage_col)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        return None
    
    # Evaluate on test data
    print("\n--- Evaluation Phase ---")
    try:
        accuracy, precision, recall, f1 = model.eval(test_data, target_col=target_col)
        
        print(f"\nSplitNN Test Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        return {
            'model': model,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        }
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        return None


def compare_with_centralized(train_data, test_data, target_col, num_classes=2, linkage_col='Common_Gene_ID'):
    """
    Compare SplitNN with centralized learning baseline.
    FIXED: Refit standardization on merged data for fair comparison.
    """
    print(f"\n=== Centralized Baseline Comparison ===")
    
    # Merge data for centralized training
    train_dfs = []
    test_dfs = []
    
    for client_name, df in train_data.items():
        train_dfs.append(df)
    
    for client_name, df in test_data.items():
        test_dfs.append(df)
    
    # Merge on linkage column
    if len(train_dfs) > 1:
        centralized_train = train_dfs[0]
        centralized_test = test_dfs[0]
        
        for i in range(1, len(train_dfs)):
            centralized_train = pd.merge(centralized_train, train_dfs[i], 
                                       on=linkage_col, how='inner')
            centralized_test = pd.merge(centralized_test, test_dfs[i], 
                                      on=linkage_col, how='inner')
    else:
        centralized_train = train_dfs[0]
        centralized_test = test_dfs[0]
    
    print(f"Centralized train shape: {centralized_train.shape}")
    print(f"Centralized test shape: {centralized_test.shape}")
    
    # CRITICAL FIX: Refit standardization on merged data
    print("Refitting standardization on merged features...")
    feature_cols = [col for col in centralized_train.columns 
                   if col != target_col and col != linkage_col]
    
    if feature_cols:
        # Create new scaler for the merged features
        centralized_scaler = StandardScaler()
        
        # Fit on combined training data only (avoid data leakage)
        train_features = centralized_train[feature_cols].fillna(0)
        centralized_scaler.fit(train_features)
        
        # Apply to both train and test
        centralized_train[feature_cols] = centralized_scaler.transform(train_features)
        
        test_features = centralized_test[feature_cols].fillna(0)
        centralized_test[feature_cols] = centralized_scaler.transform(test_features)
        
        print(f"Refitted standardization for {len(feature_cols)} merged features")
    
    # Train centralized model on properly preprocessed data
    centralized_data_train = {"centralized": centralized_train}
    centralized_data_test = {"centralized": centralized_test}
    
    centralized_model = SplitNN(
        client_hidden_dims=[[64]],  # Same architecture as federated experiments
        server_hidden_dims=[32],
        num_classes=num_classes,
        epochs=50,
        device='cpu'
    )
    
    try:
        centralized_model.fit(centralized_data_train, target_col=target_col, linkage_col=linkage_col)
        accuracy, precision, recall, f1 = centralized_model.eval(centralized_data_test, target_col=target_col)
        
        print(f"\nCentralized Baseline Results:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    except Exception as e:
        print(f"Centralized baseline failed: {e}")
        return None


def save_results(results, seed=0, database_ids=None):
    """Save SplitNN results to JSON file in table-generation compatible format."""
    try:
        if database_ids is None:
            database_ids = ["48804", "00381"]  # Default vertical FL databases
            
        project_root = Path(get_project_root())
        results_dir = project_root / "results" / "vertical"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create result data structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result_data = {
            "experiment_type": "vertical",
            "algorithm": "splitnn",
            "timestamp": timestamp,
            "seed": seed,
            "database_ids": database_ids,
            "splitnn_results": results.get("SplitNN", {}),
            "centralized_results": results.get("Centralized_Baseline", {})
        }
        
        # Save to file with predictable naming: splitnn_databases_seed.json
        db_string = "_".join(database_ids)
        filename = f"splitnn_{db_string}_seed{seed}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\n✅ SplitNN results saved to: {filepath}")
        return str(filepath)
        
    except Exception as e:
        print(f"\n❌ Error saving SplitNN results: {e}")
        return None


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train SplitNN for vertical federated learning')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility and consistent file naming')
    args = parser.parse_args()
    
    # Set parameters
    DEVICE = 'cpu'  # Change to 'cuda' if GPU available
    TARGET_COL = 'lpg_Uni_Prot_Protein_Id'  # Must match train_vertical.py
    
    print(f"Running SplitNN experiment with seed: {args.seed}")
    
    # Get project root for absolute paths
    project_root = get_project_root()
    
    # Calculate number of classes from actual data (like train_vertical.py)
    df_a_train = pd.read_csv(os.path.join(project_root, "data/clean/48804/aligned/lpg_aligned_train.csv"))
    df_a_test = pd.read_csv(os.path.join(project_root, "data/clean/48804/aligned/lpg_aligned_test.csv"))
    df_b_train = pd.read_csv(os.path.join(project_root, "data/clean/00381/aligned/tc_aligned_train.csv"))
    df_b_test = pd.read_csv(os.path.join(project_root, "data/clean/00381/aligned/tc_aligned_test.csv"))
    
    # Create unified label encoder with all possible labels
    all_labels = set(df_a_train[TARGET_COL].unique()) | set(df_a_test[TARGET_COL].unique())
    all_labels = sorted([str(label) for label in all_labels])
    unified_encoder = LabelEncoder()
    unified_encoder.fit(all_labels)
    
    NUM_CLASSES = len(all_labels)
    print(f"Calculated number of classes from data: {NUM_CLASSES}")
    print(f"Created unified label encoder with {NUM_CLASSES} classes: {all_labels[:5]}{'...' if len(all_labels) > 5 else ''}")
    
    # Load data first
    print("Loading vertical federated learning data...")
    train_data, test_data = load_vertical_data()
    
    if train_data is None or test_data is None:
        print("Failed to load data. Exiting.")
        sys.exit(1)
    
    # Create StandardScaler for each client's features (correct for vertical FL)
    print("Creating StandardScalers for each client...")
    unified_scalers = {}
    
    for client_name in train_data.keys():
        # Get feature columns (exclude target and linkage columns)
        feature_cols = [col for col in train_data[client_name].columns 
                       if col != TARGET_COL and col != 'Common_Gene_ID']
        
        if feature_cols:  # Only create scaler if there are features
            scaler = StandardScaler()
            scaler.fit(train_data[client_name][feature_cols].fillna(0))
            unified_scalers[client_name] = {'scaler': scaler, 'feature_cols': feature_cols}
            print(f"Created StandardScaler for {client_name} with {len(feature_cols)} features")
    
    # Apply standardization to all client data
    print("Applying standardization to all clients...")
    for client_name in train_data.keys():
        if client_name in unified_scalers:
            scaler_info = unified_scalers[client_name]
            scaler = scaler_info['scaler']
            feature_cols = scaler_info['feature_cols']
            
            # Scale training features
            train_features = train_data[client_name][feature_cols].fillna(0)
            train_data[client_name][feature_cols] = scaler.transform(train_features)
            
            # Scale test features
            test_features = test_data[client_name][feature_cols].fillna(0)
            test_data[client_name][feature_cols] = scaler.transform(test_features)
            
            print(f"Applied standardization to {client_name} ({len(feature_cols)} features)")
    
    print("Applied standardization to all clients")
    
    # Apply unified label encoding to the client with labels
    print("Applying unified label encoding...")
    for client_name in train_data.keys():
        if TARGET_COL in train_data[client_name].columns:
            # Encode training data labels
            train_data[client_name][TARGET_COL] = unified_encoder.transform(
                train_data[client_name][TARGET_COL].astype(str)
            )
            # Encode test data labels
            test_data[client_name][TARGET_COL] = unified_encoder.transform(
                test_data[client_name][TARGET_COL].astype(str)
            )
            print(f"Applied unified label encoding to {client_name}")
            break
    
    # Single experiment configuration (fixed split between two clients)
    experiment = {
        'name': 'SplitNN',
        'client_hidden_dims': [[64], [64]],
        'server_hidden_dims': [32],
        'epochs': 50
    }
    
    results = {}
    
    # Run single experiment
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment['name']}")
    print(f"{'='*60}")
    
    result = train_splitnn_model(
        train_data, test_data, TARGET_COL, NUM_CLASSES,
        client_hidden_dims=experiment['client_hidden_dims'],
        server_hidden_dims=experiment['server_hidden_dims'],
        epochs=experiment['epochs'],
        device=DEVICE,
        linkage_col='Common_Gene_ID'
    )
    
    if result:
        results[experiment['name']] = result['metrics']
    
    # Compare with centralized baseline
    baseline_result = compare_with_centralized(train_data, test_data, TARGET_COL, NUM_CLASSES, 'Common_Gene_ID')
    if baseline_result:
        results['Centralized_Baseline'] = baseline_result
    
    # Print summary
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 80)
    
    for method, metrics in results.items():
        print(f"{method:<25} {metrics['accuracy']:<10.4f} {metrics['precision']:<10.4f} "
              f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}")
    
    print(f"{'='*80}")
    
    # Find best performing method
    if results:
        best_method = max(results.keys(), key=lambda x: results[x]['f1'])
        print(f"\nBest performing method: {best_method}")
        print(f"F1 Score: {results[best_method]['f1']:.4f}")
    
    # Save results to file
    output_file = save_results(results, seed=args.seed, database_ids=["48804", "00381"])
    
    print(f"\n{'='*80}")
    print(f"Experiment completed.")
    if output_file:
        print(f"Results saved to: {output_file}")
    else:
        print("Failed to save results to file.")
    print(f"{'='*80}")