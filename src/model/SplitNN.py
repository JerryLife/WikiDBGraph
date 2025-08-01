import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional, Any
import warnings
from torch.utils.data import TensorDataset, DataLoader


class SplitNNClient(nn.Module):
    """Client-side network for SplitNN"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32]):
        super(SplitNNClient, self).__init__()
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
            ])
            current_dim = hidden_dim
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.layers(x)


class SplitNNServer(nn.Module):
    """Server-side network for SplitNN"""
    
    def __init__(self, client_output_dims: List[int], hidden_dims: List[int] = [32, 16], 
                 num_classes: int = 2):
        super(SplitNNServer, self).__init__()
        
        # Concatenate client outputs
        total_input_dim = sum(client_output_dims)
        
        layers = []
        current_dim = total_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            current_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(current_dim, num_classes))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, client_outputs):
        # Concatenate client outputs
        x = torch.cat(client_outputs, dim=1)
        return self.layers(x)


class SplitNN:
    """
    SplitNN implementation for vertical federated learning simulation.
    Supports fit() and eval() methods like XGBoost.
    """
    
    def __init__(self, 
                 client_hidden_dims: List[List[int]] = None,
                 server_hidden_dims: List[int] = [32, 16],
                 num_classes: int = 2,
                 learning_rate: float = 0.001,
                 epochs: int = 100,
                 batch_size: int = 32,
                 device: str = 'cpu',
                 random_state: int = 42):
        """
        Initialize SplitNN model.
        
        Args:
            client_hidden_dims: List of hidden dimensions for each client
            server_hidden_dims: Hidden dimensions for server network
            num_classes: Number of output classes
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            batch_size: Batch size for training
            device: Device to run on ('cpu' or 'cuda')
            random_state: Random seed for reproducibility
        """
        self.client_hidden_dims = client_hidden_dims or [[64, 32], [64, 32]]
        self.server_hidden_dims = server_hidden_dims
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Model components
        self.clients = []
        self.server = None
        self.optimizers = []
        self.label_encoder = None
        self.client_feature_dims = []
        self.linkage_column = None
        
    def _prepare_aligned_data(self, data_dict: Dict[str, pd.DataFrame], 
                             target_col: str, linkage_col: str = 'Common_Gene_ID') -> Tuple[List[torch.Tensor], torch.Tensor, List[int]]:
        """
        Prepare pre-aligned data for vertical federated learning.
        Assumes data is already aligned by Common_Gene_ID.
        
        Args:
            data_dict: Dictionary with client names as keys and DataFrames as values
            target_col: Target column name (must exist in label client)
            linkage_col: Column name used for alignment (will be excluded from features)
        
        Returns:
            Tuple of (client_features, labels, client_feature_dims)
        """
        if not data_dict:
            raise ValueError("data_dict cannot be empty")
            
        client_names = list(data_dict.keys())
        
        # Find the client with the target column (label client)
        label_client = None
        for client_name, df in data_dict.items():
            if target_col in df.columns:
                label_client = client_name
                print(f"Found target column '{target_col}' in client {client_name}")
                break
        
        if label_client is None:
            raise ValueError(f"Target column '{target_col}' not found in any client")
        
        # Verify all clients have the same number of rows (aligned)
        n_samples = len(data_dict[client_names[0]])
        for client_name, df in data_dict.items():
            if len(df) != n_samples:
                raise ValueError(f"Data misalignment: {client_names[0]} has {n_samples} rows, "
                               f"{client_name} has {len(df)} rows")
        
        print(f"Using {n_samples} aligned samples from {len(client_names)} clients")
        
        # Extract aligned data for each client
        client_features = []
        client_feature_dims = []
        labels = None
        
        for client_name, df in data_dict.items():
            # Extract features (excluding linkage column and target if present)
            feature_cols = [col for col in df.columns 
                          if col != linkage_col and 
                          (client_name != label_client or col != target_col)]
            
            if not feature_cols:
                warnings.warn(f"No features found for client {client_name}")
                continue
                
            features = df[feature_cols].fillna(0).values.astype(np.float32)
            client_features.append(torch.tensor(features, device=self.device))
            client_feature_dims.append(features.shape[1])
            
            # Extract labels from label client
            if client_name == label_client:
                labels = df[target_col].values
        
        if labels is None:
            raise ValueError("No labels extracted from label client")
        
        # Encode labels - handle unseen labels in test data
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            unique_labels = sorted(set(labels))
            print(f"Found labels in data: {unique_labels}")
            encoded_labels = self.label_encoder.fit_transform([str(label) for label in labels])
        else:
            # Handle unseen labels by filtering or assigning to a default class
            labels_str = [str(label) for label in labels]
            known_labels = []
            unseen_count = 0
            for label in labels_str:
                if label in self.label_encoder.classes_:
                    known_labels.append(label)
                else:
                    # Use the most frequent class as default for unseen labels
                    known_labels.append(self.label_encoder.classes_[0])
                    unseen_count += 1
            
            if unseen_count > 0:
                print(f"Warning: {unseen_count} unseen labels found, mapped to {self.label_encoder.classes_[0]}")
            
            encoded_labels = self.label_encoder.transform(known_labels)
        
        labels_tensor = torch.tensor(encoded_labels, dtype=torch.long, device=self.device)
        
        return client_features, labels_tensor, client_feature_dims
    
    def fit(self, data_dict: Dict[str, pd.DataFrame], target_col: str, linkage_col: str = 'Common_Gene_ID'):
        """
        Fit the SplitNN model.
        
        Args:
            data_dict: Dictionary with client names as keys and DataFrames as values
            target_col: Target column name (must exist in label client)
            linkage_col: Column name used for alignment (will be excluded from features)
        """
        self.linkage_column = linkage_col
        
        # Prepare aligned data
        client_features, labels, client_feature_dims = self._prepare_aligned_data(
            data_dict, target_col, linkage_col)
        
        self.client_feature_dims = client_feature_dims
        
        # Initialize client networks
        self.clients = []
        self.optimizers = []
        
        for i, feature_dim in enumerate(client_feature_dims):
            hidden_dims = (self.client_hidden_dims[i] if i < len(self.client_hidden_dims) 
                          else self.client_hidden_dims[0])
            client = SplitNNClient(feature_dim, hidden_dims).to(self.device)
            self.clients.append(client)
            
            optimizer = optim.Adam(client.parameters(), lr=self.learning_rate)
            self.optimizers.append(optimizer)
        
        # Initialize server network
        client_output_dims = [hidden_dims[-1] for hidden_dims in self.client_hidden_dims[:len(self.clients)]]
        self.server = SplitNNServer(client_output_dims, self.server_hidden_dims, 
                                   self.num_classes).to(self.device)
        server_optimizer = optim.Adam(self.server.parameters(), lr=self.learning_rate)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Create DataLoader for efficient batching
        # We need to create a combined dataset of all client features and labels
        combined_features = torch.cat(client_features, dim=1)  # Concatenate features from all clients
        dataset = TensorDataset(combined_features, labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"Training SplitNN with {len(self.clients)} clients for {self.epochs} epochs...")
        
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            
            for batch_combined_features, batch_labels in dataloader:
                # Split the combined features back to individual client features
                batch_client_features = []
                start_idx = 0
                for client_idx, feature_dim in enumerate(self.client_feature_dims):
                    end_idx = start_idx + feature_dim
                    client_batch = batch_combined_features[:, start_idx:end_idx]
                    batch_client_features.append(client_batch)
                    start_idx = end_idx
                
                # Forward pass through clients
                client_outputs = []
                for client_idx, (client, features) in enumerate(zip(self.clients, batch_client_features)):
                    output = client(features)
                    client_outputs.append(output)
                
                # Forward pass through server
                server_output = self.server(client_outputs)
                
                # Compute loss
                loss = criterion(server_output, batch_labels)
                epoch_loss += loss.item()
                
                # Backward pass
                # Zero gradients
                server_optimizer.zero_grad()
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                
                # Backward pass
                loss.backward()
                
                # Update parameters
                server_optimizer.step()
                for optimizer in self.optimizers:
                    optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                avg_loss = epoch_loss / len(dataloader)
                print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}')
    
    def predict(self, data_dict: Dict[str, pd.DataFrame], target_col: str) -> np.ndarray:
        """
        Make predictions using the trained SplitNN model.
        
        Args:
            data_dict: Dictionary with client names as keys and DataFrames as values
            target_col: Target column name (for consistency, not used in prediction)
        
        Returns:
            Predicted class labels
        """
        if not self.clients or self.server is None:
            raise ValueError("Model not fitted yet")
        
        # Prepare data
        client_features, _, _ = self._prepare_aligned_data(data_dict, target_col, self.linkage_column)
        
        # Set models to evaluation mode
        for client in self.clients:
            client.eval()
        self.server.eval()
        
        with torch.no_grad():
            # Forward pass through clients
            client_outputs = []
            for client_idx, (client, features) in enumerate(zip(self.clients, client_features)):
                output = client(features)
                client_outputs.append(output)
            
            # Forward pass through server
            server_output = self.server(client_outputs)
            
            # Get predictions
            _, predicted = torch.max(server_output, 1)
            predictions = predicted.cpu().numpy()
            
            # Decode predictions
            if self.label_encoder is not None:
                predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions
    
    def eval(self, data_dict: Dict[str, pd.DataFrame], target_col: str) -> Tuple[float, float, float, float]:
        """
        Evaluate the model performance.
        
        Args:
            data_dict: Dictionary with client names as keys and DataFrames as values
            target_col: Target column name
        
        Returns:
            Tuple of (accuracy, precision, recall, f1)
        """
        # Get true labels and predictions
        _, true_labels_tensor, _ = self._prepare_aligned_data(data_dict, target_col, self.linkage_column)
        true_labels = self.label_encoder.inverse_transform(true_labels_tensor.cpu().numpy())
        
        predictions = self.predict(data_dict, target_col)
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        return accuracy, precision, recall, f1