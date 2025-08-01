"""
FedAvg Training Script for Automated FL Validation

This script runs FedAvg federated learning on automatically processed data pairs.
"""

import pandas as pd
import os
import sys
import json
import argparse
import numpy as np
import torch
import logging
import traceback
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder


# Add src to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.FedAvg import FedAvg


def load_auto_processed_data(data_dir: str, config: dict) -> tuple:
    """
    Load automatically processed data for FL training.
    
    Args:
        data_dir: Directory containing processed data
        config: Configuration dictionary with pair metadata
        
    Returns:
        Tuple of (train_data, test_data, num_features, task_info)
    """
    db_id1 = config['db_id1']
    db_id2 = config['db_id2']
    label_col = config['label_column']
    feature_cols = config['feature_columns']
    task_type = config['task_type']
    
    # Load training data
    train_a = pd.read_csv(os.path.join(data_dir, f"{db_id1:05d}_train.csv"))
    train_b = pd.read_csv(os.path.join(data_dir, f"{db_id2:05d}_train.csv"))
    
    # Load test data
    test_a = pd.read_csv(os.path.join(data_dir, f"{db_id1:05d}_test.csv"))
    test_b = pd.read_csv(os.path.join(data_dir, f"{db_id2:05d}_test.csv"))
    
    # Handle missing values by filling with 0
    print("Handling missing values...")
    
    # Count missing values before filling
    missing_train_a = train_a.isnull().sum().sum()
    missing_train_b = train_b.isnull().sum().sum()
    missing_test_a = test_a.isnull().sum().sum()
    missing_test_b = test_b.isnull().sum().sum()
    total_missing = missing_train_a + missing_train_b + missing_test_a + missing_test_b
    
    # Fill missing values and infinite values with 0
    train_a = train_a.fillna(0).replace([np.inf, -np.inf], 0)
    train_b = train_b.fillna(0).replace([np.inf, -np.inf], 0)
    test_a = test_a.fillna(0).replace([np.inf, -np.inf], 0)
    test_b = test_b.fillna(0).replace([np.inf, -np.inf], 0)
    
    if total_missing > 0:
        print(f"Found and filled {total_missing} missing values:")
        print(f"  Train A: {missing_train_a}, Train B: {missing_train_b}")
        print(f"  Test A: {missing_test_a}, Test B: {missing_test_b}")
    else:
        print("No missing values found in data")
    
    print(f"Loaded data shapes:")
    print(f"  Train A: {train_a.shape}, Train B: {train_b.shape}")
    print(f"  Test A: {test_a.shape}, Test B: {test_b.shape}")
    
    # Apply unified preprocessing like demo scripts
    print("Applying unified preprocessing...")
    
    # Create unified StandardScaler from combined training data (to avoid data leakage)
    combined_train = pd.concat([train_a, train_b], ignore_index=True)
    unified_scaler = StandardScaler()
    unified_scaler.fit(combined_train[feature_cols].fillna(0))
    print(f"Created unified StandardScaler for {len(feature_cols)} features")
    
    # Apply standardization to all datasets
    train_a[feature_cols] = unified_scaler.transform(train_a[feature_cols].fillna(0))
    train_b[feature_cols] = unified_scaler.transform(train_b[feature_cols].fillna(0))
    test_a[feature_cols] = unified_scaler.transform(test_a[feature_cols].fillna(0))
    test_b[feature_cols] = unified_scaler.transform(test_b[feature_cols].fillna(0))
    print("Applied unified standardization to all datasets")
    
    # Determine output dimension and data types based on task type
    if task_type == 'classification':
        n_classes = config['processing_params']['n_classes']
        output_dim = n_classes
        # For classification, labels should be integers
        label_dtype = np.int64
        print(f"Classification task detected: {n_classes} classes")
        
        # Create unified label encoder for classification tasks
        combined_train_test = pd.concat([train_a, train_b, test_a, test_b], ignore_index=True)
        all_labels = sorted([str(label) for label in combined_train_test[label_col].unique()])
        unified_encoder = LabelEncoder()
        unified_encoder.fit(all_labels)
        print(f"Created unified label encoder with {len(all_labels)} classes")
        
        # Apply label encoding to all datasets
        train_a[label_col] = unified_encoder.transform(train_a[label_col].astype(str))
        train_b[label_col] = unified_encoder.transform(train_b[label_col].astype(str))
        test_a[label_col] = unified_encoder.transform(test_a[label_col].astype(str))
        test_b[label_col] = unified_encoder.transform(test_b[label_col].astype(str))
        print("Applied unified label encoding to all datasets")
    else:  # regression
        output_dim = 1
        # For regression, labels should be floats
        label_dtype = np.float32
        print(f"Regression task detected - no label encoding needed")
    
    # Prepare data for FL training
    train_data = {
        'client_0': {
            'X': train_a[feature_cols].values.astype(np.float32),
            'y': train_a[label_col].values.astype(label_dtype)
        },
        'client_1': {
            'X': train_b[feature_cols].values.astype(np.float32),
            'y': train_b[label_col].values.astype(label_dtype)
        }
    }
    
    # Combine test data
    test_combined = pd.concat([test_a, test_b], ignore_index=True)
    test_data = {
        'X': test_combined[feature_cols].values.astype(np.float32),
        'y': test_combined[label_col].values.astype(label_dtype)
    }
    
    num_features = len(feature_cols)
    task_info = {
        'task_type': task_type,
        'output_dim': output_dim,
        'num_features': num_features
    }
    
    if task_type == 'classification':
        task_info['n_classes'] = n_classes
        task_info['class_names'] = config['processing_params'].get('class_names', [])
    
    print(f"Prepared FL data:")
    print(f"  Task type: {task_type}")
    print(f"  Client 0: {train_data['client_0']['X'].shape}")
    print(f"  Client 1: {train_data['client_1']['X'].shape}")
    print(f"  Test (combined): {test_data['X'].shape}")
    print(f"  Features: {num_features}")
    print(f"  Output dimension: {output_dim}")
    
    return train_data, test_data, num_features, task_info


def train_fedavg_model(train_data: dict, test_data: dict, task_info: dict,
                      hidden_dims: list = [64, 32],
                      learning_rate: float = 0.001,
                      local_epochs: int = 5,
                      global_rounds: int = 20,
                      device: str = 'cuda:0',
                      batch_size: int = 32) -> dict:
    """
    Train FedAvg model for classification or regression task.
    
    Args:
        train_data: Training data for FL clients
        test_data: Combined test data
        task_info: Task information (type, dimensions, etc.)
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate for training
        local_epochs: Number of local epochs per round
        global_rounds: Number of global rounds
        device: Device for training
        batch_size: Batch size for training
        
    Returns:
        Dictionary with training results
    """
    task_type = task_info['task_type']
    output_dim = task_info['output_dim']
    num_features = task_info['num_features']
    
    print(f"Training FedAvg with parameters:")
    print(f"  Task type: {task_type}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Local epochs: {local_epochs}")
    print(f"  Global rounds: {global_rounds}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    
    # Initialize FedAvg model
    fedavg = FedAvg(
        input_dim=num_features,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        device=device,
        batch_size=batch_size
    )
    
    # Prepare client data
    client_data = [
        (train_data['client_0']['X'], train_data['client_0']['y']),
        (train_data['client_1']['X'], train_data['client_1']['y'])
    ]
    
    # Train the model
    print("Starting FedAvg training...")
    training_history = fedavg.train(client_data, global_rounds)
    
    # Evaluate on test data
    print("Evaluating on test data...")
    test_predictions = fedavg.predict(test_data['X'])
    test_true = test_data['y']
    
    # Calculate metrics based on task type
    if task_type == 'classification':
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(test_true, test_predictions)
        precision = precision_score(test_true, test_predictions, average='weighted', zero_division=0)
        recall = recall_score(test_true, test_predictions, average='weighted', zero_division=0)
        f1 = f1_score(test_true, test_predictions, average='weighted', zero_division=0)
        
        results = {
            'task_type': 'classification',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'training_history': training_history
        }
        
        print(f"FedAvg Classification Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
    else:  # regression
        mse = mean_squared_error(test_true, test_predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_true, test_predictions)
        mae = np.mean(np.abs(test_true - test_predictions))
        
        results = {
            'task_type': 'regression',
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'training_history': training_history
        }
        
        print(f"FedAvg Regression Results:")
        print(f"  MSE: {mse:.6f}")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  MAE: {mae:.6f}")
        print(f"  R²: {r2:.6f}")
    
    return results


def save_results(results: dict, output_file: str):
    """Save experiment results to JSON file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")


def main():
    """Main function for FedAvg training."""
    parser = argparse.ArgumentParser(description="FedAvg training for automated FL validation")
    
    # Data arguments
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing processed data")
    parser.add_argument("--config-file", type=str, required=True,
                       help="Configuration file with pair metadata")
    
    # Model arguments
    parser.add_argument("--hidden-dims", type=int, nargs='+', default=[64, 32],
                       help="Hidden layer dimensions (default: [64, 32])")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate (default: 0.001)")
    parser.add_argument("--local-epochs", type=int, default=5,
                       help="Local epochs for FedAvg (default: 5)")
    parser.add_argument("--global-rounds", type=int, default=20,
                       help="Global rounds for FedAvg (default: 20)")
    
    # Training arguments
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for training (default: cuda:0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training (default: 32)")
    
    # Output arguments
    parser.add_argument("--output-file", type=str, required=True,
                       help="Output file for results")
    
    args = parser.parse_args()
    
    # Set up logging to file
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'fedavg_errors.log'
    
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),  # Overwrite each run
        ]
    )
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Load configuration
    with open(args.config_file, 'r') as f:
        config = json.load(f)
    
    if 'error' in config:
        print(f"Configuration contains error: {config['error']}")
        return 1
    
    print(f"Processing pair: {config['pair_id']}")
    print(f"Database IDs: {config['db_id1']}, {config['db_id2']}")
    print(f"Similarity: {config['similarity']:.4f}")
    print(f"Label column: {config['label_column']}")
    print(f"Feature columns: {len(config['feature_columns'])}")
    
    # Load data
    try:
        train_data, test_data, num_features, task_info = load_auto_processed_data(args.data_dir, config)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        print(f"Error: Failed to load data: {e}")
        print(f"Full error details logged to: {log_file}")
        return 1
    
    # Run FedAvg training
    print("\n" + "="*60)
    print(f"Running FedAvg Training ({task_info['task_type']})")
    print("="*60)
    
    try:
        fedavg_results = train_fedavg_model(
            train_data, test_data, task_info,
            hidden_dims=args.hidden_dims,
            learning_rate=args.learning_rate,
            local_epochs=args.local_epochs,
            global_rounds=args.global_rounds,
            device=args.device,
            batch_size=args.batch_size
        )
    except Exception as e:
        logging.error(f"An error occurred during FedAvg training: {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        print(f"Error: FedAvg training failed: {e}")
        print(f"Full error details logged to: {log_file}")
        return 1
    
    # Create final results
    try:
        experiment_results = {
            'pair_id': config['pair_id'],
            'db_id1': config['db_id1'],
            'db_id2': config['db_id2'],
            'similarity': config['similarity'],
            'task_type': task_info['task_type'],
            'label_column': config['label_column'],
            'num_features': num_features,
            'experiment_params': {
                'algorithm': 'fedavg',
                'task_type': task_info['task_type'],
                'output_dim': task_info['output_dim'],
                'hidden_dims': args.hidden_dims,
                'learning_rate': args.learning_rate,
                'local_epochs': args.local_epochs,
                'global_rounds': args.global_rounds,
                'device': args.device,
                'seed': args.seed,
                'batch_size': args.batch_size
            },
            'timestamp': datetime.now().isoformat(),
            'results': fedavg_results
        }
        
        # Save results
        save_results(experiment_results, args.output_file)
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        print(f"Error: Failed to save results: {e}")
        print(f"Full error details logged to: {log_file}")
        return 1
    
    # Print summary
    print("\n" + "="*60)
    print("FEDAVG EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Pair: {config['pair_id']}")
    print(f"Task type: {task_info['task_type']}")
    print(f"Similarity: {config['similarity']:.4f}")
    print(f"Features: {num_features}")
    
    if task_info['task_type'] == 'classification':
        print(f"Classes: {task_info['output_dim']}")
        print(f"FedAvg Accuracy: {fedavg_results['accuracy']:.4f}")
        print(f"FedAvg F1 Score: {fedavg_results['f1']:.4f}")
    else:
        print(f"FedAvg R²: {fedavg_results['r2']:.4f}")
        print(f"FedAvg RMSE: {fedavg_results['rmse']:.6f}")
    
    print(f"Results saved to: {args.output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())