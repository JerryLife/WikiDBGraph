import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional, Any
import copy
import warnings
from torch.utils.data import TensorDataset, DataLoader


class SCAFFOLDClient(nn.Module):
    """Client network for SCAFFOLD with LayerNorm"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 num_classes: int = 2, dropout_rate: float = 0.2):
        super(SCAFFOLDClient, self).__init__()
        
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


class SCAFFOLD:
    """
    SCAFFOLD implementation for horizontal federated learning simulation.
    
    SCAFFOLD (Stochastic Controlled Averaging for Federated Learning) uses control variates
    to reduce client drift and improve convergence in non-IID federated settings.
    
    Key concepts:
    - c_global: Global control variate (server-side)
    - c_local: Local control variates (client-side) 
    - Control variates help correct for client drift during local training
    
    Reference: Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning"
    """
    
    def __init__(self,
                 hidden_dims: List[int] = [64, 32],
                 num_classes: int = 2,
                 learning_rate: float = 1e-4,
                 local_epochs: int = 5,
                 global_rounds: int = 20,
                 batch_size: int = 32,
                 device: str = 'cpu',
                 dropout_rate: float = 0,
                 client_fraction: float = 1.0,
                 random_state: int = 42):
        """
        Initialize SCAFFOLD model.
        
        Args:
            hidden_dims: Hidden layer dimensions
            num_classes: Number of output classes
            learning_rate: Learning rate for local training
            local_epochs: Number of local epochs per round
            global_rounds: Number of global communication rounds
            batch_size: Batch size for local training
            device: Device to run on ('cpu' or 'cuda')
            dropout_rate: Dropout rate for regularization
            client_fraction: Fraction of clients to sample each round
            random_state: Random seed for reproducibility
        """
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.global_rounds = global_rounds
        self.batch_size = batch_size
        self.device = device
        self.dropout_rate = dropout_rate
        self.client_fraction = client_fraction
        self.random_state = random_state
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Model components
        self.global_model = None
        self.client_models = {}
        self.label_encoder = None
        self.input_dim = None
        self.client_data_sizes = {}
        
        # SCAFFOLD-specific: Control variates
        self.c_global = None  # Global control variate
        self.c_local = {}     # Local control variates for each client
        
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
        
        # Collect all unique labels to understand the data
        all_labels = []
        for client_name, df in data_dict.items():
            if target_col in df.columns:
                all_labels.extend(df[target_col].dropna().values.astype(str))
        
        unique_labels = sorted(set(all_labels))
        print(f"Found labels in data: {unique_labels}")
        
        # Set up label encoder if labels are not already integers
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(unique_labels)
        
        # Prepare client datasets
        client_datasets = {}
        feature_dims = []
        
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
            labels = df[target_col].dropna().values.astype(str)  # Ensure consistent string type
            
            if len(features) == 0:
                warnings.warn(f"No valid samples for client {client_name}")
                continue
            
            # Encode labels (already strings)
            encoded_labels = self.label_encoder.transform(labels)
            
            # Convert to tensors - ensure labels are integers
            features_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            labels_tensor = torch.tensor(encoded_labels.astype(int), dtype=torch.long, device=self.device)
            
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
        """Initialize global model, client models, and control variates."""
        if self.input_dim is None:
            raise ValueError("Input dimension not set. Call _prepare_data first.")
        
        # Initialize global model
        self.global_model = SCAFFOLDClient(
            self.input_dim, self.hidden_dims, self.num_classes, self.dropout_rate
        ).to(self.device)
        
        # Initialize global control variate (c_global) to zero
        self.c_global = SCAFFOLDClient(
            self.input_dim, self.hidden_dims, self.num_classes, self.dropout_rate
        ).to(self.device)
        
        # Zero out all parameters in c_global
        for param in self.c_global.parameters():
            param.data.zero_()
        
        # Initialize client models and local control variates
        self.client_models = {}
        self.c_local = {}
        
        for client_name in client_names:
            # Client model initialized as copy of global model
            self.client_models[client_name] = SCAFFOLDClient(
                self.input_dim, self.hidden_dims, self.num_classes, self.dropout_rate
            ).to(self.device)
            self.client_models[client_name].load_state_dict(self.global_model.state_dict())
            
            # Local control variate initialized to zero
            self.c_local[client_name] = SCAFFOLDClient(
                self.input_dim, self.hidden_dims, self.num_classes, self.dropout_rate
            ).to(self.device)
            
            # Zero out all parameters in c_local
            for param in self.c_local[client_name].parameters():
                param.data.zero_()
    
    def _client_update(self, client_name: str, features: torch.Tensor, 
                      labels: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Perform SCAFFOLD local training on a client.
        
        Returns:
            Tuple of (updated_model_parameters, c_delta_parameters)
        """
        model = self.client_models[client_name]
        c_local = self.c_local[client_name]
        model.train()
        
        # Store initial model state for control variate update
        initial_model_state = copy.deepcopy(model.state_dict())
        
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        # Create DataLoader for efficient batching
        dataset = TensorDataset(features, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Get control variate states
        c_global_state = self.c_global.state_dict()
        c_local_state = c_local.state_dict()
        
        total_steps = 0
        
        # Local training with SCAFFOLD correction
        for epoch in range(self.local_epochs):
            for batch_features, batch_labels in dataloader:
                optimizer.zero_grad()
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                # SCAFFOLD: Apply control variate correction to gradients
                for param_name, param in model.named_parameters():
                    if (param.grad is not None and 
                        param_name in c_global_state and 
                        param_name in c_local_state):
                        
                        # Get control variate correction: c_global - c_local
                        correction = c_global_state[param_name] - c_local_state[param_name]
                        
                        # Apply correction to gradient
                        param.grad.data += correction.data
                
                optimizer.step()
                total_steps += 1
        
        # Update local control variate using SCAFFOLD formula
        # c_i^{new} = c_i - c_global + (x_initial - x_final) / (K * lr)
        final_model_state = model.state_dict()
        new_c_local_state = copy.deepcopy(c_local_state)
        c_delta_state = {}
        
        for key in new_c_local_state.keys():
            if key in initial_model_state and key in final_model_state:
                # Get parameters
                x_init = initial_model_state[key]
                x_final = final_model_state[key] 
                c_local_old = c_local_state[key]
                c_global = c_global_state[key]
                
                # SCAFFOLD control variate update formula:
                # c_i^{new} = c_i - c_global + (x_init - x_final) / (K * lr)
                param_diff = (x_init - x_final) / (total_steps * self.learning_rate)
                new_c_local_value = c_local_old - c_global + param_diff
                
                new_c_local_state[key] = new_c_local_value
                c_delta_state[key] = new_c_local_value - c_local_old
        
        # Update local control variate
        c_local.load_state_dict(new_c_local_state)
        
        return model.state_dict(), c_delta_state
    
    def _scaffold_aggregate(self, client_updates: Dict[str, Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]]):
        """
        Aggregate client updates using SCAFFOLD algorithm.
        
        Args:
            client_updates: Dictionary with client names as keys and 
                          (model_state, c_delta_state) tuples as values
        """
        # Calculate total data size for weighted averaging
        total_size = sum(self.client_data_sizes[client] for client in client_updates.keys())
        
        # Initialize aggregated parameters
        global_dict = self.global_model.state_dict()
        c_global_dict = self.c_global.state_dict()
        
        # Zero out for weighted averaging
        for key in global_dict.keys():
            global_dict[key] = torch.zeros_like(global_dict[key])
        
        for key in c_global_dict.keys():
            c_global_dict[key] = torch.zeros_like(c_global_dict[key])
        
        # Weighted averaging of model parameters and control variates
        for client_name, (client_state, c_delta_state) in client_updates.items():
            client_weight = self.client_data_sizes[client_name] / total_size
            
            for key in global_dict.keys():
                # Weighted average of model parameters
                global_dict[key] += client_weight * client_state[key]
                
                # Update global control variate with weighted c_delta
                if key in c_global_dict and key in c_delta_state:
                    c_global_dict[key] += client_weight * c_delta_state[key]
        
        # Update global model and global control variate
        self.global_model.load_state_dict(global_dict)
        self.c_global.load_state_dict(c_global_dict)
        
        # Update all client models with new global parameters
        for client_name in self.client_models.keys():
            self.client_models[client_name].load_state_dict(global_dict)
    
    def fit(self, data_dict: Dict[str, pd.DataFrame], target_col: Optional[str] = None, 
            test_data_dict: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Fit the SCAFFOLD model.
        
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
        print(f"Training SCAFFOLD with {len(client_names)} clients: {client_names}")
        
        # Initialize models and control variates
        self._initialize_models(client_names)
        
        # SCAFFOLD federated training
        for round_idx in range(self.global_rounds):
            print(f"Global Round {round_idx + 1}/{self.global_rounds}")
            
            # Sample clients for this round
            num_selected = max(1, int(len(client_names) * self.client_fraction))
            if num_selected < len(client_names):
                selected_clients = np.random.choice(client_names, num_selected, replace=False)
            else:
                selected_clients = client_names
            
            # Collect client updates
            client_updates = {}
            for client_name in selected_clients:
                features, labels = client_datasets[client_name]
                updated_params, c_delta = self._client_update(client_name, features, labels)
                client_updates[client_name] = (updated_params, c_delta)
            
            # Aggregate updates using SCAFFOLD
            self._scaffold_aggregate(client_updates)
            
            # Evaluate training accuracy after each round
            train_results = self.eval(data_dict, target_col=target_col)
            train_acc_total = 0
            train_samples_total = 0
            
            for client_name, (accuracy, _, _, _) in train_results.items():
                client_samples = len(data_dict[client_name])
                train_acc_total += accuracy * client_samples
                train_samples_total += client_samples
            
            avg_train_acc = train_acc_total / train_samples_total if train_samples_total > 0 else 0
            
            # Evaluate test accuracy if test data provided
            if test_data_dict is not None:
                test_results = self.eval(test_data_dict, target_col=target_col)
                test_acc_total = 0
                test_samples_total = 0
                
                for client_name, (accuracy, _, _, _) in test_results.items():
                    client_samples = len(test_data_dict[client_name])
                    test_acc_total += accuracy * client_samples
                    test_samples_total += client_samples
                
                avg_test_acc = test_acc_total / test_samples_total if test_samples_total > 0 else 0
                print(f"  Round {round_idx + 1}: Train Acc: {avg_train_acc:.4f}, Test Acc: {avg_test_acc:.4f}")
            else:
                print(f"  Round {round_idx + 1}: Train Acc: {avg_train_acc:.4f}")
    
    def predict(self, data_dict: Dict[str, pd.DataFrame], 
               target_col: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Make predictions using the trained SCAFFOLD model.
        
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
                _, predicted = torch.max(outputs, 1)
                pred_labels = predicted.cpu().numpy()
                
                # Decode predictions
                if self.label_encoder is not None:
                    pred_labels = self.label_encoder.inverse_transform(pred_labels)
                
                predictions[client_name] = pred_labels
        
        return predictions
    
    def eval(self, data_dict: Dict[str, pd.DataFrame], 
            target_col: Optional[str] = None) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Evaluate the model performance on each client's data.
        
        Args:
            data_dict: Dictionary with client names as keys and DataFrames as values
            target_col: Target column name
        
        Returns:
            Dictionary with client names as keys and (accuracy, precision, recall, f1) as values
        """
        predictions = self.predict(data_dict, target_col)
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
            
            true_labels = df[target_col].dropna().astype(str).values
            pred_labels = predictions[client_name]
            
            # Align predictions with true labels (handle missing values)
            valid_indices = ~df[target_col].isna()
            true_labels = true_labels
            pred_labels = pred_labels[valid_indices]
            
            if len(true_labels) != len(pred_labels):
                raise ValueError(f"Length mismatch in client {client_name}: {len(true_labels)} true labels vs {len(pred_labels)} predictions")
            
            if len(true_labels) == 0:
                continue
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, pred_labels)
            precision = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
            recall = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
            f1 = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
            
            results[client_name] = (accuracy, precision, recall, f1)
        
        return results