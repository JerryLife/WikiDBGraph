import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Dict, List, Tuple, Optional
from torch.utils.data import TensorDataset, DataLoader


class CentralizedNN(nn.Module):
    """Simple Neural Network for centralized training baseline"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 num_classes: int = 2, dropout_rate: float = 0.0):
        super(CentralizedNN, self).__init__()
        
        # Two-layer MLP with LayerNorm (same as federated models)
        layers = []
        current_dim = input_dim
        
        # Hidden layers with LayerNorm
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            current_dim = hidden_dim
        
        # Output layer (no normalization before final layer)
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


def train_centralized_baseline(train_df: pd.DataFrame, 
                             test_df: pd.DataFrame,
                             target_col: str,
                             num_classes: int,
                             hidden_dims: List[int] = [64, 32],
                             learning_rate: float = 1e-4,
                             epochs: int = 100,  # More epochs for centralized training
                             batch_size: int = 32,
                             device: str = 'cuda:1',  # Default to GPU
                             unified_encoder: Optional[LabelEncoder] = None) -> Dict[str, float]:
    """
    Train a centralized neural network baseline.
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame  
        target_col: Target column name
        num_classes: Number of classes
        hidden_dims: Hidden layer dimensions
        learning_rate: Learning rate
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to train on (default: 'cuda:1')
        unified_encoder: Pre-fitted label encoder
    
    Returns:
        Dictionary with accuracy, precision, recall, f1 metrics
    """
    # Auto-detect GPU if cuda:1 is not available
    if device.startswith('cuda') and not torch.cuda.is_available():
        print(f"CUDA not available, falling back to CPU")
        device = 'cpu'
    elif device.startswith('cuda'):
        # Check if the specific GPU is available
        gpu_id = int(device.split(':')[1]) if ':' in device else 0
        if gpu_id >= torch.cuda.device_count():
            print(f"GPU {gpu_id} not available, using GPU 0")
            device = 'cuda:0'
    
    print(f"\n=== Training Centralized Neural Network Baseline ===")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Device: {device}")
    
    # Prepare features and labels
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    # Training data
    X_train = train_df[feature_cols].fillna(0).values.astype(np.float32)
    y_train = train_df[target_col].dropna().astype(str).values
    
    # Test data
    X_test = test_df[feature_cols].fillna(0).values.astype(np.float32)
    y_test = test_df[target_col].dropna().astype(str).values
    
    print(f"Features: {len(feature_cols)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Use unified encoder if provided, otherwise create new one
    if unified_encoder is not None:
        y_train_encoded = unified_encoder.transform(y_train)
        y_test_encoded = unified_encoder.transform(y_test)
        print("Using provided unified label encoder")
    else:
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        print("Created new label encoder")
    
    # Convert to tensors and move to device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long).to(device)
    
    # Create model and move to device
    input_dim = X_train.shape[1]
    model = CentralizedNN(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout_rate=0.0
    ).to(device)
    
    print(f"Model moved to device: {device}")
    if device.startswith('cuda'):
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    model.train()
    print("\n--- Training Progress ---")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            # Data is already on the correct device
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        train_accuracy = correct / total
        avg_loss = epoch_loss / len(train_loader)
        
        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}")
    
    # Evaluation on test set
    model.eval()
    print("\n--- Evaluation ---")
    
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, test_predicted = torch.max(test_outputs, 1)
        test_predicted_np = test_predicted.cpu().numpy()
        test_true_np = y_test_tensor.cpu().numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(test_true_np, test_predicted_np)
    precision = precision_score(test_true_np, test_predicted_np, average='weighted', zero_division=0)
    recall = recall_score(test_true_np, test_predicted_np, average='weighted', zero_division=0)
    f1 = f1_score(test_true_np, test_predicted_np, average='weighted', zero_division=0)
    
    print(f"Test Results:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    
    # Clean up GPU memory if using CUDA
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
        print(f"GPU memory freed")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }