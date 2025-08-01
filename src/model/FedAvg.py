import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional, Any, Union
import copy
import warnings
from torch.utils.data import TensorDataset, DataLoader


class SimpleRegressor(nn.Module):
    """Simple Neural Network for regression tasks"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 output_dim: int = 1, dropout_rate: float = 0.0):
        super(SimpleRegressor, self).__init__()
        
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
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class FedAvgClient(nn.Module):
    """Client network for FedAvg with LayerNorm"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 num_classes: int = 2, dropout_rate: float = 0):
        super(FedAvgClient, self).__init__()
        
        # Two-layer MLP with LayerNorm
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


class FedAvg:
    """
    FedAvg implementation for horizontal federated learning simulation.
    Supports both classification (num_classes > 1) and regression (num_classes = 1).
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: List[int] = [64, 32],
                 output_dim: int = 1,
                 learning_rate: float = 0.001,
                 local_epochs: int = 5,
                 device: str = 'cpu',
                 batch_size: int = 32):
        """
        Initialize FedAvg model.
        
        Args:
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension (1 for regression, >1 for classification)
            learning_rate: Learning rate for local training
            local_epochs: Number of local epochs per round
            device: Device to run on ('cpu' or 'cuda')
            batch_size: Batch size for training
        """
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.device = device
        self.batch_size = batch_size
        
        # Determine task type based on output_dim
        self.is_regression = (output_dim == 1)
        
        # Model components
        self.global_model = None
        self.client_models = {}
        self.label_encoder = None
        self.client_data_sizes = {}
        
        task_type = "regression" if self.is_regression else "classification"
        print(f"Initialized FedAvg for {task_type} task")
        print(f"  Input dim: {input_dim}")
        print(f"  Hidden dims: {hidden_dims}")
        print(f"  Output dim: {output_dim}")
        print(f"  Device: {device}")
    
    def _prepare_data(self, data_dict: Dict[str, pd.DataFrame], 
                     target_col: Optional[str] = None) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare data for horizontal federated learning.
        
        Args:
            data_dict: Dictionary with client names as keys and DataFrames as values
            target_col: Target column name (auto-detected if None)
        
        Returns:
            Dictionary with client names as keys and (features, labels) tensors as values
        """
        if not data_dict:
            raise ValueError("data_dict cannot be empty")
        
        # Auto-detect target column if not provided
        if target_col is None:
            # Look for common target column patterns
            first_df = list(data_dict.values())[0]
            potential_targets = []
            
            for col in first_df.columns:
                if (col.lower() in ['label', 'target', 'class', 'y'] or 
                    'protein' in col.lower() or 'gene' in col.lower() or
                    'classification' in col.lower()):
                    potential_targets.append(col)
            
            if potential_targets:
                target_col = potential_targets[0]
                print(f"Auto-detected target column: {target_col}")
            else:
                # Use the last column as target
                target_col = first_df.columns[-1]
                print(f"Using last column as target: {target_col}")
        
        # Prepare client datasets
        client_datasets = {}
        feature_dims = []
        
        if not self.is_regression:
            # For classification, collect all unique labels to set up label encoder
            all_labels = []
            for client_name, df in data_dict.items():
                if target_col in df.columns:
                    all_labels.extend(df[target_col].dropna().values.astype(str))
            
            unique_labels = sorted(set(all_labels))
            print(f"Found labels in data: {unique_labels}")
            
            # Set up label encoder
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                self.label_encoder.fit(unique_labels)
        
        for client_name, df in data_dict.items():
            if df.empty:
                warnings.warn(f"Client {client_name} has empty data")
                continue
            
            if target_col not in df.columns:
                warnings.warn(f"Target column '{target_col}' not found in client {client_name}")
                continue
            
            # Extract features and labels
            feature_cols = [col for col in df.columns if col != target_col]
            
            if not feature_cols:
                warnings.warn(f"No features found for client {client_name}")
                continue
            
            # Handle missing values and align features and labels
            valid_indices = ~df[target_col].isna()
            features = df[feature_cols].fillna(0).values.astype(np.float32)[valid_indices]
            
            if len(features) == 0:
                warnings.warn(f"No valid samples for client {client_name}")
                continue
            
            if self.is_regression:
                # For regression, keep labels as float
                labels = df[target_col].dropna().values.astype(np.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.float32, device=self.device)
            else:
                # For classification, encode labels as integers
                labels = df[target_col].dropna().values.astype(str)
                encoded_labels = self.label_encoder.transform(labels)
                labels_tensor = torch.tensor(encoded_labels.astype(int), dtype=torch.long, device=self.device)
            
            # Convert to tensors
            features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            
            client_datasets[client_name] = (features_tensor, labels_tensor)
            feature_dims.append(features.shape[1])
            self.client_data_sizes[client_name] = len(features)
        
        # Verify all clients have same feature dimension
        if len(set(feature_dims)) > 1:
            raise ValueError(f"Feature dimensions mismatch across clients: {feature_dims}")
        
        if feature_dims:
            self.input_dim = feature_dims[0]
        
        return client_datasets
    
    def _initialize_models(self, client_names: List[str]):
        """Initialize global and client models."""
        if self.input_dim is None:
            raise ValueError("Input dimension not set. Call _prepare_data first.")
        
        # Initialize global model
        self.global_model = FedAvgClient(
            self.input_dim, self.hidden_dims, self.output_dim, dropout_rate=0.0
        ).to(self.device)
        
        # Initialize client models as copies of global model
        self.client_models = {}
        for client_name in client_names:
            self.client_models[client_name] = FedAvgClient(
                self.input_dim, self.hidden_dims, self.output_dim, dropout_rate=0.0
            ).to(self.device)
            # Copy global model parameters to client
            self.client_models[client_name].load_state_dict(self.global_model.state_dict())
    
    def _client_update(self, client_name: str, features: torch.Tensor, 
                      labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Perform local training on a client.
        
        Returns:
            Updated model parameters
        """
        model = self.client_models[client_name]
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Choose appropriate loss function based on task type
        if self.is_regression:
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        # Create DataLoader for efficient batching
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Local training
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            
            for batch_features, batch_labels in dataloader:
                optimizer.zero_grad()
                
                outputs = model(batch_features)
                
                if self.is_regression:
                    # For regression, squeeze output to match label dimensions
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, batch_labels)
                else:
                    # For classification, use raw outputs
                    loss = criterion(outputs, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
        
        return model.state_dict()
    
    def _fedavg_aggregate(self, client_updates: Dict[str, Dict[str, torch.Tensor]]):
        """
        Aggregate client updates using FedAvg algorithm.
        
        Args:
            client_updates: Dictionary with client names as keys and state_dicts as values
        """
        # Calculate total data size
        total_size = sum(self.client_data_sizes[client] for client in client_updates.keys())
        
        # Initialize aggregated parameters
        global_dict = self.global_model.state_dict()
        
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
        
        # Weighted averaging
        for client_name, client_state in client_updates.items():
            client_weight = self.client_data_sizes[client_name] / total_size
            
            for key in global_dict.keys():
                # Weighted averaging of all parameters
                global_dict[key] += client_weight * client_state[key]
        
        # Update global model
        self.global_model.load_state_dict(global_dict)
        
        # Update all client models with new global parameters
        for client_name in self.client_models.keys():
            self.client_models[client_name].load_state_dict(global_dict)
    
    def train(self, client_data: List[Tuple[np.ndarray, np.ndarray]], 
              global_rounds: int) -> List[float]:
        """
        Train the FedAvg model.
        
        Args:
            client_data: List of (X, y) tuples for each client
            global_rounds: Number of global rounds
            
        Returns:
            Training history (list of losses)
        """
        if len(client_data) == 0:
            raise ValueError("No client data provided")
        
        # Convert client data to tensors
        client_datasets = {}
        client_names = []
        
        for i, (X, y) in enumerate(client_data):
            client_name = f"client_{i}"
            client_names.append(client_name)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            
            if self.is_regression:
                y_tensor = torch.FloatTensor(y).to(self.device)
            else:
                y_tensor = torch.LongTensor(y).to(self.device)
            
            client_datasets[client_name] = (X_tensor, y_tensor)
            self.client_data_sizes[client_name] = len(X)
        
        print(f"Training with {len(client_names)} clients for {global_rounds} rounds")
        
        # Initialize models
        self._initialize_models(client_names)
        
        # Training history
        training_history = []
        
        # Federated training
        for round_idx in range(global_rounds):
            print(f"Global Round {round_idx + 1}/{global_rounds}")
            
            # Collect client updates
            client_updates = {}
            round_loss = 0.0
            
            for client_name in client_names:
                features, labels = client_datasets[client_name]
                updated_params = self._client_update(client_name, features, labels)
                client_updates[client_name] = updated_params
                
                # Calculate training loss for monitoring
                with torch.no_grad():
                    self.client_models[client_name].eval()
                    outputs = self.client_models[client_name](features)
                    
                    if self.is_regression:
                        criterion = nn.MSELoss()
                        outputs = outputs.squeeze()
                        loss = criterion(outputs, labels)
                    else:
                        criterion = nn.CrossEntropyLoss()
                        loss = criterion(outputs, labels)
                    
                    round_loss += loss.item()
            
            # Aggregate updates
            self._fedavg_aggregate(client_updates)
            
            # Record average loss
            avg_loss = round_loss / len(client_names)
            training_history.append(avg_loss)
            
            if (round_idx + 1) % 5 == 0:
                print(f"  Round {round_idx + 1}: Avg Loss: {avg_loss:.6f}")
        
        return training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained FedAvg model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self.global_model is None:
            raise ValueError("Model not fitted yet")
        
        self.global_model.eval()
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.global_model(X_tensor)
            
            if self.is_regression:
                predictions = outputs.squeeze().cpu().numpy()
            else:
                _, predicted = torch.max(outputs, 1)
                predictions = predicted.cpu().numpy()
                
                # Decode predictions if label encoder exists
                if self.label_encoder is not None:
                    predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def fit(self, data_dict: Dict[str, pd.DataFrame], target_col: Optional[str] = None, 
            test_data_dict: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Fit the FedAvg model (legacy method for compatibility).
        
        Args:
            data_dict: Dictionary with client names as keys and DataFrames as values
            target_col: Target column name (auto-detected if None)
            test_data_dict: Optional test data for evaluation during training
        """
        # Prepare data
        client_datasets = self._prepare_data(data_dict, target_col)
        
        if not client_datasets:
            raise ValueError("No valid client datasets found")
        
        client_names = list(client_datasets.keys())
        print(f"Training FedAvg with {len(client_names)} clients: {client_names}")
        
        # Initialize models
        self._initialize_models(client_names)
        
        # Use default global rounds for this method
        global_rounds = 20
        
        # Federated training
        for round_idx in range(global_rounds):
            print(f"Global Round {round_idx + 1}/{global_rounds}")
            
            # Collect client updates
            client_updates = {}
            for client_name in client_names:
                features, labels = client_datasets[client_name]
                updated_params = self._client_update(client_name, features, labels)
                client_updates[client_name] = updated_params
            
            # Aggregate updates
            self._fedavg_aggregate(client_updates)
            
            # Evaluate if needed (simplified for this legacy method)
            if (round_idx + 1) % 5 == 0:
                print(f"  Round {round_idx + 1} completed")
    
    def eval(self, data_dict: Dict[str, pd.DataFrame], 
            target_col: Optional[str] = None) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Evaluate the model performance on each client's data.
        
        Args:
            data_dict: Dictionary with client names as keys and DataFrames as values
            target_col: Target column name
        
        Returns:
            Dictionary with client names as keys and metrics as values
            For regression: (mse, rmse, mae, r2)
            For classification: (accuracy, precision, recall, f1)
        """
        predictions = self.predict_dict(data_dict, target_col)
        results = {}
        
        for client_name, df in data_dict.items():
            if client_name not in predictions or df.empty:
                continue
            
            if target_col is None:
                # Use same auto-detection logic as in _prepare_data
                potential_targets = [col for col in df.columns 
                                   if col.lower() in ['label', 'target', 'class', 'y'] or 
                                   'protein' in col.lower() or 'gene' in col.lower()]
                if potential_targets:
                    target_col = potential_targets[0]
                else:
                    target_col = df.columns[-1]
            
            if target_col not in df.columns:
                continue
            
            true_values = df[target_col].dropna().values
            pred_values = predictions[client_name]
            
            if len(true_values) != len(pred_values):
                raise ValueError(f"Length mismatch in client {client_name}: {len(true_values)} true vs {len(pred_values)} predictions")
            
            if len(true_values) == 0:
                continue
            
            if self.is_regression:
                # Calculate regression metrics
                true_values = true_values.astype(np.float32)
                mse = mean_squared_error(true_values, pred_values)
                rmse = np.sqrt(mse)
                mae = np.mean(np.abs(true_values - pred_values))
                r2 = r2_score(true_values, pred_values)
                results[client_name] = (mse, rmse, mae, r2)
            else:
                # Calculate classification metrics
                true_values = true_values.astype(str)
                accuracy = accuracy_score(true_values, pred_values)
                precision = precision_score(true_values, pred_values, average='weighted', zero_division=0)
                recall = recall_score(true_values, pred_values, average='weighted', zero_division=0)
                f1 = f1_score(true_values, pred_values, average='weighted', zero_division=0)
                results[client_name] = (accuracy, precision, recall, f1)
        
        return results
    
    def predict_dict(self, data_dict: Dict[str, pd.DataFrame], 
                    target_col: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Make predictions for dictionary input (for compatibility).
        
        Args:
            data_dict: Dictionary with client names as keys and DataFrames as values
            target_col: Target column name (should match training)
        
        Returns:
            Dictionary with client names as keys and predictions as values
        """
        if self.global_model is None:
            raise ValueError("Model not fitted yet")
        
        # Use the global model for inference
        self.global_model.eval()
        
        predictions = {}
        
        with torch.no_grad():
            for client_name, df in data_dict.items():
                if df.empty:
                    continue
                
                # Prepare features (exclude target column if present)
                if target_col and target_col in df.columns:
                    feature_cols = [col for col in df.columns if col != target_col]
                else:
                    feature_cols = df.columns.tolist()
                
                if not feature_cols:
                    continue
                
                features = df[feature_cols].fillna(0).values.astype(np.float32)
                features_tensor = torch.tensor(features, device=self.device)
                
                # Make predictions
                outputs = self.global_model(features_tensor)
                
                if self.is_regression:
                    pred_values = outputs.squeeze().cpu().numpy()
                else:
                    _, predicted = torch.max(outputs, 1)
                    pred_values = predicted.cpu().numpy()
                    
                    # Decode predictions
                    if self.label_encoder is not None:
                        pred_values = self.label_encoder.inverse_transform(pred_values)
                
                predictions[client_name] = pred_values
        
        return predictions