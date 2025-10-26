import pandas as pd
import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add src to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.SCAFFOLD import SCAFFOLD
from centralized_training import train_centralized_baseline


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def save_results(results, algorithm_name="scaffold", seed=0, database_ids=None):
    """Save experiment results to JSON file."""
    try:
        if database_ids is None:
            database_ids = ["02799", "79665"]  # Default horizontal FL databases
            
        project_root = get_project_root()
        results_dir = project_root / "results" / "horizontal"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create result data structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Map algorithm name to the key used in results
        algorithm_key_map = {
            "fedavg": "FedAvg",
            "fedprox": "FedProx", 
            "scaffold": "SCAFFOLD",
            "fedov": "FedOV"
        }
        result_key = algorithm_key_map.get(algorithm_name.lower(), algorithm_name.title())
        
        result_data = {
            "experiment_type": "horizontal",
            "algorithm": algorithm_name,
            "timestamp": timestamp,
            "seed": seed,
            "database_ids": database_ids,
            "weighted_metrics": results.get(result_key, {}),
            "centralized_baseline": results.get("Centralized_Baseline", {}),
            "all_results": results
        }
        
        # Save to file with predictable naming: algorithm_databases_seed.json
        db_string = "_".join(database_ids)
        filename = f"{algorithm_name.lower()}_{db_string}_seed{seed}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\n✅ Results saved to: {filepath}")
        return str(filepath)
        
    except Exception as e:
        print(f"\n❌ Error saving results: {e}")
        return None


def load_horizontal_data(base_output_dir="data/clean", database_ids=None):
    """
    Load horizontal federated learning data.
    Returns data dictionaries for training and testing.
    """
    if database_ids is None:
        database_ids = ["02799", "79665"]  # Default databases
    
    db_a, db_b = database_ids
    
    # Client A data
    path_a_train = os.path.join(base_output_dir, db_a, "train_data.csv")
    path_a_test = os.path.join(base_output_dir, db_a, "test_data.csv")

    # Client B data  
    path_b_train = os.path.join(base_output_dir, db_b, "train_data.csv")
    path_b_test = os.path.join(base_output_dir, db_b, "test_data.csv")

    print(f"Loading data for client A ({db_a}) from: {path_a_train}, {path_a_test}")
    print(f"Loading data for client B ({db_b}) from: {path_b_train}, {path_b_test}")

    # Check if files exist
    required_files = [path_a_train, path_a_test, path_b_train, path_b_test]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("Please run the horizontal data preparation script first.")
        return None, None

    # Load data
    df_a_train = pd.read_csv(path_a_train)
    df_a_test = pd.read_csv(path_a_test)
    df_b_train = pd.read_csv(path_b_train)
    df_b_test = pd.read_csv(path_b_test)

    print(f"Shape of client A train ({db_a}): {df_a_train.shape}")
    print(f"Shape of client B train ({db_b}): {df_b_train.shape}")

    # Create data dictionaries
    train_data = {
        f"client_{db_a}": df_a_train,
        f"client_{db_b}": df_b_train
    }
    
    test_data = {
        f"client_{db_a}": df_a_test,
        f"client_{db_b}": df_b_test
    }
    
    return train_data, test_data


def train_scaffold_model(train_data, test_data,
                        num_classes=2,
                        hidden_dims=[64, 32],
                        learning_rate=0.001,
                        local_epochs=5,
                        global_rounds=20,
                        batch_size=32,
                        device='cpu',
                        client_fraction=1.0,
                        target_col=None,
                        unified_encoder=None):
    """
    Train and evaluate SCAFFOLD model.
    """
    print(f"\n=== Training SCAFFOLD Model ===")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Learning rate: {learning_rate}")
    print(f"Local epochs: {local_epochs}")
    print(f"Global rounds: {global_rounds}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print(f"Client fraction: {client_fraction}")
    
    # Validate target column
    if target_col is None:
        raise ValueError("target_col must be specified")
    
    # Check if target column exists in all clients
    for client_name, df in train_data.items():
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in client {client_name}. Available columns: {df.columns.tolist()}")
    
    print(f"Target column: {target_col}")
    print(f"Number of classes (specified): {num_classes}")
    
    # Display data information
    for client_name, df in train_data.items():
        if target_col in df.columns:
            unique_classes = df[target_col].nunique()
            print(f"Client {client_name}: {len(df)} samples, {unique_classes} unique classes in data")
    
    # Initialize SCAFFOLD model
    model = SCAFFOLD(
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        global_rounds=global_rounds,
        batch_size=batch_size,
        device=device,
        client_fraction=client_fraction
    )
    
    # Pre-set the unified label encoder to prevent SCAFFOLD from creating its own
    if unified_encoder is not None:
        model.label_encoder = unified_encoder
        print(f"Applied unified label encoder to SCAFFOLD model")
    
    # Train the model
    print("\n--- Training Phase ---")
    try:
        model.fit(train_data, target_col=target_col, test_data_dict=test_data)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed with error: {e}")
        return None
    
    # Evaluate on test data
    print("\n--- Evaluation Phase ---")
    try:
        client_results = model.eval(test_data, target_col=target_col)
        
        print(f"\nSCAFFOLD Test Results:")
        total_samples = 0
        weighted_metrics = {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0}
        
        for client_name, (accuracy, precision, recall, f1) in client_results.items():
            client_samples = len(test_data[client_name])
            total_samples += client_samples
            
            print(f"{client_name}:")
            print(f"  Samples: {client_samples}")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1 Score:  {f1:.4f}")
            
            # Calculate weighted averages
            weighted_metrics['accuracy'] += accuracy * client_samples
            weighted_metrics['precision'] += precision * client_samples
            weighted_metrics['recall'] += recall * client_samples
            weighted_metrics['f1'] += f1 * client_samples
        
        # Normalize weighted averages
        for metric in weighted_metrics:
            weighted_metrics[metric] /= total_samples
        
        print(f"\nWeighted Average Results:")
        print(f"Accuracy:  {weighted_metrics['accuracy']:.4f}")
        print(f"Precision: {weighted_metrics['precision']:.4f}")
        print(f"Recall:    {weighted_metrics['recall']:.4f}")
        print(f"F1 Score:  {weighted_metrics['f1']:.4f}")
        
        return {
            'model': model,
            'client_results': client_results,
            'weighted_metrics': weighted_metrics
        }
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        return None


def compare_with_centralized(train_data, test_data, num_classes=2, target_col=None, unified_encoder=None, unified_scaler=None):
    """
    Compare SCAFFOLD with centralized learning baseline using simple neural network.
    """
    print(f"\n=== Centralized Baseline Comparison ===")
    
    # Combine all client data
    combined_train = pd.concat(list(train_data.values()), ignore_index=True)
    combined_test = pd.concat(list(test_data.values()), ignore_index=True)
    
    print(f"Combined train shape: {combined_train.shape}")
    print(f"Combined test shape: {combined_test.shape}")
    
    # Apply unified standardization to centralized data if available
    if unified_scaler is not None:
        feature_cols = [col for col in combined_train.columns if col != target_col]
        combined_train[feature_cols] = unified_scaler.transform(combined_train[feature_cols].fillna(0))
        combined_test[feature_cols] = unified_scaler.transform(combined_test[feature_cols].fillna(0))
        print("Applied unified standardization to centralized data")
    
    try:
        # Train centralized neural network baseline
        centralized_results = train_centralized_baseline(
            train_df=combined_train,
            test_df=combined_test,
            target_col=target_col,
            num_classes=num_classes,
            hidden_dims=[64, 32],  # Same architecture as federated experiments
            learning_rate=1e-4,
            epochs=100,  # More epochs for centralized training
            batch_size=32,
            device=DEVICE,
            unified_encoder=unified_encoder
        )
        
        return centralized_results
        
    except Exception as e:
        print(f"Centralized baseline failed: {e}")
        return None


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="SCAFFOLD federated learning experiment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--databases", type=str, nargs=2, default=["02799", "79665"], 
                       help="Database IDs for federated learning (default: 02799 79665)")
    args = parser.parse_args()
    
    # Set parameters
    DEVICE = 'cuda:1'  # Change to 'cuda' if GPU available
    TARGET_COL = 'details_encoded_protein'  # Must match train_horizontal.py
    
    # Load all data to create unified label encoder using specified databases
    db_a, db_b = args.databases
    df_a_train = pd.read_csv(f"data/clean/{db_a}/train_data.csv")
    df_a_test = pd.read_csv(f"data/clean/{db_a}/test_data.csv")
    df_b_train = pd.read_csv(f"data/clean/{db_b}/train_data.csv")
    df_b_test = pd.read_csv(f"data/clean/{db_b}/test_data.csv")
    
    # Create unified label encoder with all possible labels
    all_labels = set()
    for df in [df_a_train, df_a_test, df_b_train, df_b_test]:
        if TARGET_COL in df.columns:
            all_labels.update(df[TARGET_COL].unique())
    
    all_labels = sorted([str(label) for label in all_labels])
    unified_encoder = LabelEncoder()
    unified_encoder.fit(all_labels)
    
    NUM_CLASSES = len(all_labels)
    print(f"Created unified label encoder with {NUM_CLASSES} classes: {all_labels[:5]}{'...' if len(all_labels) > 5 else ''}")
    
    # Create unified StandardScaler from combined training data
    print("Creating unified StandardScaler...")
    combined_train_df = pd.concat([df_a_train, df_b_train], ignore_index=True)
    feature_cols = [col for col in combined_train_df.columns if col != TARGET_COL]
    
    # Fit the scaler ONLY on the combined training data to avoid data leakage
    unified_scaler = StandardScaler()
    unified_scaler.fit(combined_train_df[feature_cols].fillna(0))
    print(f"Created unified StandardScaler for {len(feature_cols)} features")
    
    # Load data
    print("Loading horizontal federated learning data...")
    train_data, test_data = load_horizontal_data(database_ids=args.databases)
    
    if train_data is None or test_data is None:
        print("Failed to load data. Exiting.")
        sys.exit(1)
    
    # Apply unified standardization to all client data
    print("Applying unified standardization to all clients...")
    for client_name in train_data.keys():
        # Scale training features
        train_features = train_data[client_name][feature_cols].fillna(0)
        train_data[client_name][feature_cols] = unified_scaler.transform(train_features)
        
        # Scale test features  
        test_features = test_data[client_name][feature_cols].fillna(0)
        test_data[client_name][feature_cols] = unified_scaler.transform(test_features)
    
    print(f"Applied unified standardization to all clients")
    
    # Apply unified label encoding to all client data
    print("Applying unified label encoding to all clients...")
    for client_name in train_data.keys():
        # Encode training data labels
        train_data[client_name][TARGET_COL] = unified_encoder.transform(
            train_data[client_name][TARGET_COL].astype(str)
        )
        # Encode test data labels
        test_data[client_name][TARGET_COL] = unified_encoder.transform(
            test_data[client_name][TARGET_COL].astype(str)
        )
    
    print(f"Applied unified label encoding across all clients")
    
    # Single experiment configuration (fixed split between two clients)
    experiment = {
        'name': 'SCAFFOLD',
        'hidden_dims': [64, 32],
        'local_epochs': 5,
        'global_rounds': 20
    }
    
    results = {}
    
    # Run single experiment
    print(f"\n{'='*60}")
    print(f"Running experiment: {experiment['name']}")
    print(f"{'='*60}")
    
    result = train_scaffold_model(
        train_data, test_data,
        learning_rate=1e-4,
        num_classes=NUM_CLASSES,
        hidden_dims=experiment['hidden_dims'],
        local_epochs=experiment['local_epochs'],
        global_rounds=experiment['global_rounds'],
        device=DEVICE,
        target_col=TARGET_COL,
        unified_encoder=unified_encoder
    )
    
    if result:
        results[experiment['name']] = result['weighted_metrics']
    
    # Compare with centralized baseline
    baseline_result = compare_with_centralized(train_data, test_data, NUM_CLASSES, TARGET_COL, unified_encoder, unified_scaler)
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
        
        # Compare with centralized (only federated methods, not centralized baseline)
        federated_methods = {k: v for k, v in results.items() if k != 'Centralized_Baseline'}
        if 'Centralized_Baseline' in results and federated_methods:
            best_fed_method = max(federated_methods.keys(), key=lambda x: federated_methods[x]['f1'])
            fed_f1 = federated_methods[best_fed_method]['f1']
            cent_f1 = results['Centralized_Baseline']['f1']
            diff = fed_f1 - cent_f1
            print(f"Best federated method: {best_fed_method}")
            print(f"Federated vs Centralized F1: {fed_f1:.4f} vs {cent_f1:.4f} (diff: {diff:+.4f})")
            if diff >= -0.05:  # Within 5% is considered good for FL
                print("✓ Federated learning performance is competitive!")
            else:
                print("⚠ Significant performance gap vs centralized learning.")
    
    # Save results to JSON file
    save_results(results, "scaffold", seed=args.seed, database_ids=args.databases)