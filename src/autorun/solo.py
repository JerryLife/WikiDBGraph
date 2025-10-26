"""
Solo/Combined Training Script for Automated FL Validation

This script runs either:
- Solo training: individual client training on automatically processed data pairs
- Combined training: centralized training on combined data from both clients
"""

import pandas as pd
import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import traceback
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset



# Add src to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.FedAvg import SimpleRegressor


def load_auto_processed_data(data_dir: str, config: dict, mode: str = 'solo') -> tuple:
    """
    Load automatically processed data for solo or combined training.
    
    Args:
        data_dir: Directory containing processed data
        config: Configuration dictionary with pair metadata
        mode: 'solo' for individual client training, 'combined' for centralized training
        
    Returns:
        Tuple of (train_data, test_data, num_features)
    """
    db_id1 = config['db_id1']
    db_id2 = config['db_id2']
    label_col = config['label_column']
    feature_cols = config['feature_columns']
    
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
    
    # Get task information from config
    task_type = config['task_type']
    if task_type == 'classification':
        output_dim = config['processing_params']['n_classes']
        
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
    else:
        output_dim = 1
        print("Regression task - no label encoding needed")
    
    num_features = len(feature_cols)
    
    if mode == 'combined':
        # Prepare data for combined/centralized training
        # Combine all training data
        combined_train = pd.concat([train_a, train_b], ignore_index=True)
        combined_test = pd.concat([test_a, test_b], ignore_index=True)
        
        train_data = {
            'X': combined_train[feature_cols].values.astype(np.float32),
            'y': combined_train[label_col].values.astype(np.float32)
        }
        
        test_data = {
            'X': combined_test[feature_cols].values.astype(np.float32),
            'y': combined_test[label_col].values.astype(np.float32)
        }
        
        print(f"Prepared Combined data:")
        print(f"  Combined train: {train_data['X'].shape}")
        print(f"  Combined test: {test_data['X'].shape}")
        print(f"  Features: {num_features}")
        
    else:
        # Prepare data for solo training
        train_data = {
            'client_0': {
                'X': train_a[feature_cols].values.astype(np.float32),
                'y': train_a[label_col].values.astype(np.float32)
            },
            'client_1': {
                'X': train_b[feature_cols].values.astype(np.float32),
                'y': train_b[label_col].values.astype(np.float32)
            }
        }
        
        # For fair comparison, use COMBINED test set for all approaches
        combined_test = pd.concat([test_a, test_b], ignore_index=True)
        combined_test_data = {
            'X': combined_test[feature_cols].values.astype(np.float32),
            'y': combined_test[label_col].values.astype(np.float32)
        }
        
        # Both clients will be tested on the same combined test set
        test_data = {
            'client_0': combined_test_data,
            'client_1': combined_test_data
        }
        
        print(f"Prepared Solo data:")
        print(f"  Client 0 train: {train_data['client_0']['X'].shape}")
        print(f"  Client 1 train: {train_data['client_1']['X'].shape}")
        print(f"  Combined test (for both clients): {test_data['client_0']['X'].shape}")
        print(f"  Features: {num_features}")
        print(f"  Note: Solo trains individually but tests on combined test set for fair comparison")
    
    task_info = {
        'task_type': task_type,
        'output_dim': output_dim,
        'num_features': num_features
    }
    
    return train_data, test_data, num_features, task_info


def train_solo_client(client_data: dict, test_data: dict, task_info: dict,
                     client_id: str,
                     hidden_dims: list = [64, 32],
                     learning_rate: float = 0.001,
                     epochs: int = 100,
                     device: str = 'cuda:0',
                     batch_size: int = 32) -> dict:
    """
    Train a single client model for solo learning (classification or regression) with mini-batch support.
    """
    task_type = task_info['task_type']
    output_dim = task_info['output_dim']
    num_features = task_info['num_features']
    print(f"Training Solo client {client_id} with parameters:")
    print(f"  Task type: {task_type}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Hidden dims: {hidden_dims}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {epochs}")
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    # Create model
    model = SimpleRegressor(
        input_dim=num_features,
        hidden_dims=hidden_dims,
        output_dim=output_dim
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Choose appropriate loss function and data types
    if task_type == 'classification':
        criterion = nn.CrossEntropyLoss()
        X_train = torch.FloatTensor(client_data['X'])
        y_train = torch.LongTensor(client_data['y'])
        X_test = torch.FloatTensor(test_data['X']).to(device)
        y_test = torch.LongTensor(test_data['y']).to(device)
    else:  # regression
        criterion = nn.MSELoss()
        X_train = torch.FloatTensor(client_data['X'])
        y_train = torch.FloatTensor(client_data['y'])
        X_test = torch.FloatTensor(test_data['X']).to(device)
        y_test = torch.FloatTensor(test_data['y']).to(device)
    # DataLoader for mini-batch
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model.train()
    training_history = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            if task_type == 'classification':
                loss = criterion(outputs, batch_y)
            else:
                predictions = outputs.squeeze()
                loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        avg_loss = epoch_loss / len(train_loader.dataset)
        training_history.append(float(avg_loss))
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    # Evaluate on test data
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        if task_type == 'classification':
            test_loss = criterion(test_outputs, y_test)
            _, test_predictions = torch.max(test_outputs, 1)
            test_predictions_np = test_predictions.cpu().numpy()
            y_test_np = y_test.cpu().numpy()
        else:
            test_predictions = test_outputs.squeeze()
            test_loss = criterion(test_predictions, y_test)
            test_predictions_np = test_predictions.cpu().numpy()
            y_test_np = y_test.cpu().numpy()
    # Calculate metrics based on task type
    if task_type == 'classification':
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_test_np, test_predictions_np)
        precision = precision_score(y_test_np, test_predictions_np, average='weighted', zero_division=0)
        recall = recall_score(y_test_np, test_predictions_np, average='weighted', zero_division=0)
        f1 = f1_score(y_test_np, test_predictions_np, average='weighted', zero_division=0)
        results = {
            'task_type': 'classification',
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'final_train_loss': float(training_history[-1]),
            'training_history': training_history
        }
        print(f"Solo Client {client_id} Classification Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
    else:
        mse = mean_squared_error(y_test_np, test_predictions_np)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_np, test_predictions_np)
        mae = np.mean(np.abs(y_test_np - test_predictions_np))
        results = {
            'task_type': 'regression',
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'final_train_loss': float(training_history[-1]),
            'training_history': training_history
        }
        print(f"Solo Client {client_id} Regression Results:")
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
    """Main function for solo/combined training."""
    parser = argparse.ArgumentParser(description="Solo/Combined training for automated FL validation")
    
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
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs (default: 100)")
    
    # Training arguments
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device for training (default: cuda:0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training (default: 32)")
    
    # Output arguments
    parser.add_argument("--output-file", type=str, required=True,
                       help="Output file for results")
    
    # Algorithm arguments
    parser.add_argument("--algorithm", type=str, default="solo",
                       choices=["solo", "combined"],
                       help="Training algorithm: 'solo' for individual clients, 'combined' for centralized")
    
    args = parser.parse_args()
    
    # Set up logging to file
    output_dir = Path(args.output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f'{args.algorithm}_errors.log'
    
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
        train_data, test_data, num_features, task_info = load_auto_processed_data(args.data_dir, config, mode=args.algorithm)
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        logging.error(f"Full traceback: {traceback.format_exc()}")
        print(f"Error: Failed to load data: {e}")
        print(f"Full error details logged to: {log_file}")
        return 1
    
    # Run training based on algorithm
    if args.algorithm == 'combined':
        print("\n" + "="*60)
        print("Running Combined Training")
        print("="*60)
        
        try:
            combined_results = train_solo_client(
                train_data, test_data, task_info,
                'combined',
                hidden_dims=args.hidden_dims,
                learning_rate=args.learning_rate,
                epochs=args.epochs,
                device=args.device,
                batch_size=args.batch_size
            )
            
            results = {'combined': combined_results}
        except Exception as e:
            logging.error(f"An error occurred during combined training: {e}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            print(f"Error: Combined training failed: {e}")
            print(f"Full error details logged to: {log_file}")
            return 1
        
    else:  # solo mode
        print("\n" + "="*60)
        print("Running Solo Training")
        print("="*60)
        
        try:
            solo_results = {}
            
            for client_id in ['client_0', 'client_1']:
                print(f"\nTraining {client_id}...")
                
                client_results = train_solo_client(
                    train_data[client_id], test_data[client_id], task_info,
                    client_id,
                    hidden_dims=args.hidden_dims,
                    learning_rate=args.learning_rate,
                    epochs=args.epochs,
                    device=args.device,
                    batch_size=args.batch_size
                )
                
                solo_results[client_id] = client_results
            
            results = solo_results
        except Exception as e:
            logging.error(f"An error occurred during solo training: {e}")
            logging.error(f"Full traceback: {traceback.format_exc()}")
            print(f"Error: Solo training failed: {e}")
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
                'algorithm': args.algorithm,
                'task_type': task_info['task_type'],
                'output_dim': task_info['output_dim'],
                'hidden_dims': args.hidden_dims,
                'learning_rate': args.learning_rate,
                'epochs': args.epochs,
                'device': args.device,
                'seed': args.seed,
                'batch_size': args.batch_size
            },
            'timestamp': datetime.now().isoformat(),
            'results': results
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
    if args.algorithm == 'combined':
        print("COMBINED EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Pair: {config['pair_id']}")
        print(f"Task type: {task_info['task_type']}")
        print(f"Similarity: {config['similarity']:.4f}")
        print(f"Features: {num_features}")
        print(f"Combined samples: {len(train_data['X']) + len(test_data['X'])}")
        
        if task_info['task_type'] == 'classification':
            print(f"Classes: {task_info['output_dim']}")
            print(f"Combined Accuracy: {results['combined']['accuracy']:.4f}")
            print(f"Combined F1 Score: {results['combined']['f1']:.4f}")
        else:
            print(f"Combined R²: {results['combined']['r2']:.4f}")
            print(f"Combined RMSE: {results['combined']['rmse']:.6f}")
    else:
        print("SOLO EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Pair: {config['pair_id']}")
        print(f"Task type: {task_info['task_type']}")
        print(f"Similarity: {config['similarity']:.4f}")
        print(f"Features: {num_features}")
        
        if task_info['task_type'] == 'classification':
            print(f"Classes: {task_info['output_dim']}")
            print(f"Client 0 Accuracy: {results['client_0']['accuracy']:.4f}")
            print(f"Client 1 Accuracy: {results['client_1']['accuracy']:.4f}")
            print(f"Client 0 F1 Score: {results['client_0']['f1']:.4f}")
            print(f"Client 1 F1 Score: {results['client_1']['f1']:.4f}")
        else:
            print(f"Client 0 R²: {results['client_0']['r2']:.4f}")
            print(f"Client 1 R²: {results['client_1']['r2']:.4f}")
            print(f"Client 0 RMSE: {results['client_0']['rmse']:.6f}")
            print(f"Client 1 RMSE: {results['client_1']['rmse']:.6f}")
    
    print(f"Results saved to: {args.output_file}")
    
    return 0


if __name__ == "__main__":
    exit(main())