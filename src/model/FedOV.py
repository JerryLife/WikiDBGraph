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
import random


class FedOVClient(nn.Module):
    """Client network for FedOV with LayerNorm and outlier detection"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 num_classes: int = 2, dropout_rate: float = 0.2):
        super(FedOVClient, self).__init__()
        
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
        
        # Pre-classification layer (for intermediate features)
        self.pre_layers = nn.Sequential(*layers)
        
        # Final classification layer (adds +1 for outlier class)
        self.later_layers = nn.Linear(current_dim, num_classes + 1)
        
    def forward(self, x):
        x_mid = self.pre_layers(x)
        x_out = self.later_layers(x_mid)
        return x_out, x_mid


class FedOV:
    """
    FedOV implementation for one-shot federated learning with voting mechanism.
    
    FedOV (Federated One-shot Voting) trains each client independently in a single round,
    then uses a voting mechanism with outlier detection for final predictions.
    
    Key concepts:
    - One-shot learning: Only one communication round (comm_round=1)
    - Outlier detection: Adds an extra class for detecting out-of-distribution samples
    - Voting mechanism: Multiple client models vote on predictions
    - Data augmentation: Generates synthetic outliers for better detection
    
    Reference: Diao et al. "Towards Addressing Label Skews in One-Shot Federated Learning"
    """
    
    def __init__(self,
                 hidden_dims: List[int] = [64, 32],
                 num_classes: int = 2,
                 learning_rate: float = 1e-4,
                 local_epochs: int = 10,  # More epochs since it's one-shot
                 global_rounds: int = 1,  # Always 1 for FedOV
                 batch_size: int = 32,
                 device: str = 'cpu',
                 dropout_rate: float = 0.2,
                 client_fraction: float = 1.0,
                 random_state: int = 42,
                 augmentation_rate: float = 0.3,  # FedOV specific
                 outlier_threshold: float = 0.5):  # FedOV specific
        """
        Initialize FedOV model.
        
        Args:
            hidden_dims: Hidden layer dimensions
            num_classes: Number of output classes (excluding outlier class)
            learning_rate: Learning rate for local training
            local_epochs: Number of local epochs (more than other FL methods)
            global_rounds: Number of global rounds (always 1 for FedOV)
            batch_size: Batch size for local training
            device: Device to run on ('cpu' or 'cuda')
            dropout_rate: Dropout rate for regularization
            client_fraction: Fraction of clients to sample (usually 1.0 for one-shot)
            random_state: Random seed for reproducibility
            augmentation_rate: Rate of data augmentation for outlier generation
            outlier_threshold: Threshold for outlier detection
        """
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.global_rounds = 1  # Always 1 for FedOV
        self.batch_size = batch_size
        self.device = device
        self.dropout_rate = dropout_rate
        self.client_fraction = client_fraction
        self.random_state = random_state
        self.augmentation_rate = augmentation_rate
        self.outlier_threshold = outlier_threshold
        
        # Set random seeds
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        random.seed(random_state)
        
        # Model components
        self.client_models = {}
        self.label_encoder = None
        self.input_dim = None
        self.client_data_sizes = {}
        self.client_thresholds = {}  # Store outlier detection thresholds per client
        
    def _generate_tabular_outliers(self, features: torch.Tensor) -> torch.Tensor:
        """
        Generate synthetic outliers for tabular data using various augmentation techniques.
        
        Args:
            features: Input features tensor
            
        Returns:
            Augmented outlier features
        """
        outliers = []
        num_outliers = int(len(features) * self.augmentation_rate)
        
        if num_outliers == 0:
            num_outliers = 1
            
        for _ in range(num_outliers):
            # Choose a random augmentation technique
            aug_type = random.choice(['noise', 'shuffle', 'interpolation', 'extreme'])
            
            # Select a random sample as base
            base_idx = random.randint(0, len(features) - 1)
            base_sample = features[base_idx].clone()
            
            if aug_type == 'noise':
                # Add random noise
                noise = torch.randn_like(base_sample) * 0.1
                outlier = base_sample + noise
                
            elif aug_type == 'shuffle':
                # Shuffle features
                perm = torch.randperm(base_sample.size(0), device=self.device)
                outlier = base_sample[perm]
                
            elif aug_type == 'interpolation':
                # Interpolate between two random samples
                other_idx = random.randint(0, len(features) - 1)
                other_sample = features[other_idx]
                alpha = random.uniform(0.2, 0.8)
                outlier = alpha * base_sample + (1 - alpha) * other_sample
                
            else:  # extreme
                # Create extreme values
                outlier = base_sample.clone()
                # Randomly modify some features to extreme values
                num_to_modify = random.randint(1, min(5, len(outlier)))
                indices = torch.randperm(len(outlier), device=self.device)[:num_to_modify]
                for idx in indices:
                    if random.random() > 0.5:
                        outlier[idx] = outlier[idx] + 3 * torch.std(features[:, idx])
                    else:
                        outlier[idx] = outlier[idx] - 3 * torch.std(features[:, idx])
            
            outliers.append(outlier)
        
        return torch.stack(outliers)
    
    def _prepare_data(self, data_dict: Dict[str, pd.DataFrame], 
                     target_col: Optional[str] = None) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Prepare data for FedOV federated learning.
        
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
            labels = df[target_col].dropna().values.astype(str)
            
            if len(features) == 0:
                warnings.warn(f"No valid samples for client {client_name}")
                continue
            
            # Encode labels
            encoded_labels = self.label_encoder.transform(labels)
            
            # Convert to tensors
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
        """Initialize client models (no global model in FedOV)."""
        if self.input_dim is None:
            raise ValueError("Input dimension not set. Call _prepare_data first.")
        
        # Initialize client models (each trains independently)
        self.client_models = {}
        for client_name in client_names:
            self.client_models[client_name] = FedOVClient(
                self.input_dim, self.hidden_dims, self.num_classes, self.dropout_rate
            ).to(self.device)
    
    def _client_update(self, client_name: str, features: torch.Tensor, 
                      labels: torch.Tensor) -> Dict[str, Any]:
        """
        Perform FedOV local training on a client with outlier detection.
        
        Returns:
            Dictionary containing model state and outlier detection statistics
        """
        model = self.client_models[client_name]
        model.train()
        
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss(ignore_index=self.num_classes)  # Ignore outlier class in regular loss
        
        # Generate synthetic outliers
        outlier_features = self._generate_tabular_outliers(features)
        outlier_labels = torch.full((len(outlier_features),), self.num_classes, 
                                  dtype=torch.long, device=self.device)
        
        # Combine original and outlier data
        combined_features = torch.cat([features, outlier_features], dim=0)
        combined_labels = torch.cat([labels, outlier_labels], dim=0)
        
        # Create DataLoader
        dataset = TensorDataset(combined_features, combined_labels)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training statistics
        total_loss = 0
        correct_predictions = 0
        total_samples = 0
        outlier_confidences = []
        
        # Local training
        for epoch in range(self.local_epochs):
            epoch_loss = 0
            
            for batch_features, batch_labels in dataloader:
                optimizer.zero_grad()
                
                outputs, mid_features = model(batch_features)
                
                # Standard cross-entropy loss
                loss = criterion(outputs, batch_labels)
                
                # Additional outlier detection loss
                # Encourage high confidence for outlier class on synthetic outliers
                outlier_mask = (batch_labels == self.num_classes)
                if outlier_mask.sum() > 0:
                    outlier_outputs = outputs[outlier_mask]
                    outlier_loss = nn.CrossEntropyLoss()(outlier_outputs, batch_labels[outlier_mask])
                    loss += outlier_loss
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Track statistics
                with torch.no_grad():
                    _, predicted = torch.max(outputs, 1)
                    correct_predictions += (predicted == batch_labels).sum().item()
                    total_samples += batch_labels.size(0)
                    
                    # Track outlier confidence for threshold calculation
                    outlier_probs = torch.softmax(outputs, dim=1)[:, self.num_classes]
                    outlier_confidences.extend(outlier_probs.cpu().numpy())
            
            total_loss += epoch_loss
        
        # Calculate outlier detection threshold
        outlier_confidences = np.array(outlier_confidences)
        threshold = np.percentile(outlier_confidences, 90)  # 90th percentile as threshold
        max_prob = np.max(outlier_confidences)
        avg_max = np.mean(outlier_confidences)
        
        training_accuracy = correct_predictions / total_samples
        
        print(f"Client {client_name} - Training Acc: {training_accuracy:.4f}, "
              f"Threshold: {threshold:.4f}, Max Prob: {max_prob:.4f}")
        
        return {
            'model_state': model.state_dict(),
            'threshold': threshold,
            'max_prob': max_prob,
            'avg_max': avg_max,
            'training_accuracy': training_accuracy
        }
    
    def _voting_prediction(self, client_outputs: List[torch.Tensor], 
                          client_thresholds: List[Dict[str, float]], 
                          accepted_votes: int = None) -> torch.Tensor:
        """
        Perform voting-based prediction using multiple client models.
        
        Args:
            client_outputs: List of output tensors from different clients
            client_thresholds: List of threshold dictionaries for each client
            accepted_votes: Number of votes to accept (if None, use all)
            
        Returns:
            Final predictions
        """
        if accepted_votes is None:
            accepted_votes = len(client_outputs)
        
        batch_size = client_outputs[0].size(0)
        predictions = []
        
        for i in range(batch_size):
            # Get outputs for this sample from all clients
            sample_outputs = [output[i] for output in client_outputs]
            
            # Apply softmax and normalize outlier scores
            normalized_outputs = []
            for j, output in enumerate(sample_outputs):
                probs = torch.softmax(output, dim=0)
                
                # Normalize outlier score using threshold
                threshold_info = client_thresholds[j]
                outlier_prob = probs[self.num_classes].item()
                
                # Normalize outlier probability
                normalized_outlier = (np.log(outlier_prob + 1e-8) - threshold_info['threshold']) / \
                                   (threshold_info['max_prob'] - threshold_info['threshold'] + 1e-8)
                normalized_outlier = max(0, min(1, normalized_outlier))
                
                # Adjust class probabilities
                class_probs = probs[:self.num_classes].clone()
                class_probs = class_probs * (1 - normalized_outlier)
                
                # Combine normalized probabilities
                final_probs = torch.cat([class_probs, torch.tensor([normalized_outlier], device=self.device)])
                normalized_outputs.append(final_probs)
            
            # Convert to numpy for easier manipulation
            vote_matrix = torch.stack(normalized_outputs).cpu().numpy()
            
            # Sort by outlier confidence (ascending - less outlier-like first)
            outlier_scores = vote_matrix[:, -1]
            sorted_indices = np.argsort(outlier_scores)
            
            # Take top accepted_votes
            selected_votes = vote_matrix[sorted_indices[:accepted_votes]]
            
            # Sum class probabilities (exclude outlier class)
            final_class_votes = np.sum(selected_votes[:, :-1], axis=0)
            
            # Predict class with highest vote
            prediction = int(np.argmax(final_class_votes))
            predictions.append(prediction)
        
        return torch.tensor(predictions, dtype=torch.long, device=self.device)
    
    def fit(self, data_dict: Dict[str, pd.DataFrame], target_col: Optional[str] = None, 
            test_data_dict: Optional[Dict[str, pd.DataFrame]] = None):
        """
        Fit the FedOV model using one-shot federated learning.
        
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
        print(f"Training FedOV with {len(client_names)} clients: {client_names}")
        
        # Initialize models
        self._initialize_models(client_names)
        
        # One-shot federated training
        print(f"Global Round 1/1 (One-shot training)")
        
        # Train all clients independently
        client_results = {}
        for client_name in client_names:
            features, labels = client_datasets[client_name]
            result = self._client_update(client_name, features, labels)
            client_results[client_name] = result
            
            # Store threshold information
            self.client_thresholds[client_name] = {
                'threshold': result['threshold'],
                'max_prob': result['max_prob'],
                'avg_max': result['avg_max']
            }
        
        print("One-shot training completed!")
        
        # Evaluate if test data provided
        if test_data_dict is not None:
            test_results = self.eval(test_data_dict, target_col=target_col)
            test_acc_total = 0
            test_samples_total = 0
            
            for client_name, (accuracy, _, _, _) in test_results.items():
                client_samples = len(test_data_dict[client_name])
                test_acc_total += accuracy * client_samples
                test_samples_total += client_samples
            
            avg_test_acc = test_acc_total / test_samples_total if test_samples_total > 0 else 0
            print(f"Overall Test Accuracy: {avg_test_acc:.4f}")
    
    def predict(self, data_dict: Dict[str, pd.DataFrame], 
               target_col: Optional[str] = None, accepted_votes: int = None) -> Dict[str, np.ndarray]:
        """
        Make predictions using the trained FedOV model with voting.
        
        Args:
            data_dict: Dictionary with client names as keys and DataFrames as values
            target_col: Target column name (should match training)
            accepted_votes: Number of client votes to accept (if None, use all)
        
        Returns:
            Dictionary with client names as keys and predictions as values
        """
        if not self.client_models:
            raise ValueError("Model not fitted yet")
        
        # Set all models to evaluation mode
        for model in self.client_models.values():
            model.eval()
        
        predictions = {}
        
        with torch.no_grad():
            for data_client_name, df in data_dict.items():
                if df.empty:
                    continue
                
                # Prepare features
                if target_col and target_col in df.columns:
                    feature_cols = [col for col in df.columns if col != target_col]
                else:
                    feature_cols = df.columns.tolist()
                
                if not feature_cols:
                    continue
                
                features = df[feature_cols].fillna(0).values.astype(np.float32)
                features_tensor = torch.tensor(features, device=self.device)
                
                # Get predictions from all client models
                client_outputs = []
                client_threshold_list = []
                
                for client_name, model in self.client_models.items():
                    outputs, _ = model(features_tensor)
                    client_outputs.append(outputs)
                    client_threshold_list.append(self.client_thresholds[client_name])
                
                # Perform voting
                voted_predictions = self._voting_prediction(client_outputs, client_threshold_list, accepted_votes)
                pred_labels = voted_predictions.cpu().numpy()
                
                # Decode predictions
                if self.label_encoder is not None:
                    pred_labels = self.label_encoder.inverse_transform(pred_labels)
                
                predictions[data_client_name] = pred_labels
        
        return predictions
    
    def eval(self, data_dict: Dict[str, pd.DataFrame], 
            target_col: Optional[str] = None, accepted_votes: int = None) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Evaluate the FedOV model performance using voting mechanism.
        
        Args:
            data_dict: Dictionary with client names as keys and DataFrames as values
            target_col: Target column name
            accepted_votes: Number of client votes to accept
        
        Returns:
            Dictionary with client names as keys and (accuracy, precision, recall, f1) as values
        """
        predictions = self.predict(data_dict, target_col, accepted_votes)
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