"""
FedGNN: Property-Aware GNN Aggregator for Federated Learning.

This module implements a GNN-based federated learning aggregator that:
1. Uses graph structure (from WikiDBGraph) to determine neighbor relationships
2. Leverages node/edge properties to compute attention weights
3. Supports masked aggregation for handling missing features (union schema)

Property modes:
- none: Standard FedAvg-style uniform aggregation
- edge_only: Use edge properties for attention
- node_only: Use node properties for conditioning
- both: Use both node and edge properties (default)
"""

import os
import sys
import copy
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LocalModel(nn.Module):
    """
    Simple MLP model for local client training.
    Supports masked forward pass for union schema.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass with optional feature mask.
        
        Args:
            x: Input features [batch_size, input_dim]
            mask: Feature mask [input_dim], 1.0 for present features, 0.0 for missing
        """
        if mask is not None:
            # Apply mask to zero out missing features
            x = x * mask.unsqueeze(0)
        return self.net(x)


class EdgeAttentionNet(nn.Module):
    """
    Computes attention scores based on edge properties.
    """
    
    def __init__(self, edge_feat_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, edge_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_features: [num_edges, edge_feat_dim]
        Returns:
            Attention scores [num_edges, 1]
        """
        return self.net(edge_features)


class NodeConditioningNet(nn.Module):
    """
    Computes conditioning factors based on node properties.
    """
    
    def __init__(self, node_feat_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
    
    def forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_features: [num_nodes, node_feat_dim]
        Returns:
            Conditioning factors [num_nodes, 1]
        """
        return self.net(node_features)


class MaskedGraphAttentionAggregator(nn.Module):
    """
    Property-aware graph attention aggregator with masked aggregation.
    
    Handles the "low overlap" problem by:
    1. Using union schema (all columns from all DBs)
    2. Masking out missing features during aggregation
    3. Normalizing by contributing count, not total count
    """
    
    def __init__(
        self,
        edge_feat_dim: int = 3,
        node_feat_dim: int = 8,
        property_mode: str = "both"
    ):
        """
        Args:
            edge_feat_dim: Dimension of edge features
            node_feat_dim: Dimension of node features
            property_mode: 'none', 'edge_only', 'node_only', or 'both'
        """
        super().__init__()
        self.property_mode = property_mode
        
        # Edge attention network
        if property_mode in ["edge_only", "both"]:
            self.edge_att = EdgeAttentionNet(edge_feat_dim)
        else:
            self.edge_att = None
        
        # Node conditioning network
        if property_mode in ["node_only", "both"]:
            self.node_cond = NodeConditioningNet(node_feat_dim)
        else:
            self.node_cond = None
    
    def compute_attention_weights(
        self,
        target_id: int,
        neighbors: List[int],
        edge_features: Dict[Tuple[int, int], torch.Tensor],
        node_features: Dict[int, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention weights for neighbors.
        
        Returns:
            Normalized attention weights [num_neighbors]
        """
        if self.property_mode == "none" or not neighbors:
            # Uniform weights
            return torch.ones(len(neighbors)) / len(neighbors)
        
        scores = []
        
        for n_id in neighbors:
            score = torch.tensor(1.0)
            
            # Edge-based attention
            if self.edge_att is not None:
                edge_key = (target_id, n_id)
                if edge_key in edge_features:
                    edge_feat = edge_features[edge_key]
                    edge_score = self.edge_att(edge_feat.unsqueeze(0)).squeeze()
                    score = score * torch.sigmoid(edge_score)
                else:
                    score = score * 0.5  # Default for missing edges
            
            # Node-based conditioning
            if self.node_cond is not None and node_features is not None:
                if n_id in node_features:
                    node_feat = node_features[n_id]
                    node_score = self.node_cond(node_feat.unsqueeze(0)).squeeze()
                    score = score * node_score
            
            scores.append(score)
        
        scores = torch.stack(scores)
        # Softmax normalization
        return F.softmax(scores, dim=0)
    
    def aggregate(
        self,
        target_id: int,
        neighbors: List[int],
        client_models: Dict[int, nn.Module],
        client_masks: Dict[int, torch.Tensor],
        edge_features: Dict[Tuple[int, int], torch.Tensor],
        node_features: Dict[int, torch.Tensor] = None,
        self_weight: float = 0.7
    ) -> OrderedDict:
        """
        Perform masked aggregation of model weights.
        
        Args:
            target_id: ID of target client
            neighbors: List of neighbor client IDs
            client_models: Dict of client models
            client_masks: Dict of feature masks per client [input_dim]
            edge_features: Dict of edge features
            node_features: Optional dict of node features
            self_weight: Weight for self model (residual connection)
            
        Returns:
            Aggregated state dict for target client
        """
        neighbor_weight = 1.0 - self_weight
        
        # Get attention weights
        attention_weights = self.compute_attention_weights(
            target_id, neighbors, edge_features, node_features
        )
        
        target_state = client_models[target_id].state_dict()
        target_mask = client_masks[target_id].float()
        
        new_state = OrderedDict()
        
        for key in target_state:
            param = target_state[key]
            
            # Check if this is the input layer (needs masking)
            if "net.0.weight" in key:
                # Shape: [hidden_dim, input_dim]
                weighted_sum = torch.zeros_like(param)
                norm_factor = torch.zeros_like(param) + 1e-8
                
                # Add target's own weights (with its mask)
                mask_matrix = target_mask.unsqueeze(0).expand_as(param)
                weighted_sum += self_weight * param * mask_matrix
                norm_factor += self_weight * mask_matrix
                
                # Add neighbor contributions (masked)
                for i, n_id in enumerate(neighbors):
                    n_param = client_models[n_id].state_dict()[key]
                    n_mask = client_masks[n_id].float()
                    n_mask_matrix = n_mask.unsqueeze(0).expand_as(n_param)
                    
                    weight = neighbor_weight * attention_weights[i]
                    weighted_sum += weight * n_param * n_mask_matrix
                    norm_factor += weight * n_mask_matrix
                
                # Coordinate-wise normalization
                new_state[key] = weighted_sum / norm_factor
                
            else:
                # Standard weighted aggregation for other layers
                weighted_param = self_weight * param
                
                for i, n_id in enumerate(neighbors):
                    n_param = client_models[n_id].state_dict()[key]
                    weighted_param += neighbor_weight * attention_weights[i] * n_param
                
                new_state[key] = weighted_param
        
        return new_state


class MaskedFedAvg:
    """
    Coordinate-wise FedAvg with masked aggregation.
    
    Instead of dividing by total clients N, divides by the count
    of clients who actually possess each specific feature.
    """
    
    @staticmethod
    def aggregate(
        client_models: Dict[int, nn.Module],
        client_masks: Dict[int, torch.Tensor],
        client_ids: List[int] = None
    ) -> OrderedDict:
        """
        Perform coordinate-wise (masked) FedAvg aggregation.
        
        Args:
            client_models: Dict of client models
            client_masks: Dict of feature masks per client [input_dim]
            client_ids: Optional list of client IDs to aggregate
            
        Returns:
            Aggregated global state dict
        """
        if client_ids is None:
            client_ids = list(client_models.keys())
        
        if not client_ids:
            raise ValueError("No clients to aggregate")
        
        # Get reference state dict
        ref_state = client_models[client_ids[0]].state_dict()
        n_clients = len(client_ids)
        
        global_state = OrderedDict()
        
        for key in ref_state:
            ref_param = ref_state[key]
            
            if "net.0.weight" in key:
                # Coordinate-wise aggregation for input layer
                # Shape: [hidden_dim, input_dim]
                sum_weights = torch.zeros_like(ref_param)
                sum_counts = torch.zeros_like(ref_param) + 1e-8
                
                for cid in client_ids:
                    w = client_models[cid].state_dict()[key]
                    m = client_masks[cid].float()  # [input_dim]
                    
                    # Expand mask to [hidden_dim, input_dim]
                    m_expanded = m.unsqueeze(0).expand_as(w)
                    
                    sum_weights += w
                    sum_counts += m_expanded  # Count contributing clients
                
                # Divide by contributing count (not N)
                global_state[key] = sum_weights / sum_counts
                
            else:
                # Standard averaging for other layers
                sum_weights = torch.zeros_like(ref_param)
                for cid in client_ids:
                    sum_weights += client_models[cid].state_dict()[key]
                global_state[key] = sum_weights / n_clients
        
        return global_state


class FedGNN:
    """
    High-level FedGNN trainer combining:
    - Local client training
    - Property-aware GNN aggregation with masking
    - Graph structure from WikiDBSubgraph
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        edge_feat_dim: int = 3,
        node_feat_dim: int = 8,
        property_mode: str = "both",
        self_weight: float = 0.7,
        device: str = "cpu"
    ):
        """
        Args:
            input_dim: Union schema dimension (total unique columns)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for binary classification)
            edge_feat_dim: Dimension of edge properties
            node_feat_dim: Dimension of node properties
            property_mode: 'none', 'edge_only', 'node_only', 'both'
            self_weight: Weight for self in aggregation (residual)
            device: 'cpu' or 'cuda'
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.property_mode = property_mode
        self.self_weight = self_weight
        self.device = device
        
        # GNN aggregator
        self.aggregator = MaskedGraphAttentionAggregator(
            edge_feat_dim=edge_feat_dim,
            node_feat_dim=node_feat_dim,
            property_mode=property_mode
        ).to(device)
        
        # Client models (initialized later)
        self.client_models: Dict[int, LocalModel] = {}
        self.client_masks: Dict[int, torch.Tensor] = {}
        
        # Graph structure (set externally)
        self.neighbors: Dict[int, List[int]] = {}
        self.edge_features: Dict[Tuple[int, int], torch.Tensor] = {}
        self.node_features: Dict[int, torch.Tensor] = {}
    
    def initialize_clients(
        self,
        client_ids: List[int],
        feature_masks: Dict[int, List[bool]]
    ):
        """
        Initialize client models with shared architecture.
        
        Args:
            client_ids: List of client IDs
            feature_masks: Dict mapping client_id -> boolean mask list
        """
        # Create shared initial model
        init_model = LocalModel(
            self.input_dim, 
            self.hidden_dim, 
            self.output_dim
        )
        init_state = init_model.state_dict()
        
        for cid in client_ids:
            # Create model with same initial weights
            model = LocalModel(
                self.input_dim,
                self.hidden_dim,
                self.output_dim
            ).to(self.device)
            model.load_state_dict(copy.deepcopy(init_state))
            
            self.client_models[cid] = model
            
            # Convert mask to tensor
            mask = torch.tensor(feature_masks[cid], dtype=torch.float32).to(self.device)
            self.client_masks[cid] = mask
    
    def set_graph_structure(
        self,
        neighbors: Dict[int, List[int]],
        edge_features: Dict[Tuple[int, int], torch.Tensor] = None,
        node_features: Dict[int, torch.Tensor] = None
    ):
        """
        Set the graph structure for aggregation.
        
        Args:
            neighbors: Dict mapping client_id -> list of neighbor client_ids
            edge_features: Dict mapping (src, dst) -> edge feature tensor
            node_features: Dict mapping client_id -> node feature tensor
        """
        self.neighbors = neighbors
        self.edge_features = edge_features or {}
        self.node_features = node_features or {}
    
    def local_update(
        self,
        client_id: int,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        epochs: int = 1,
        lr: float = 0.01
    ) -> float:
        """
        Perform local training on a client.
        
        Args:
            client_id: Client ID
            train_data: Tuple of (X, y) tensors
            epochs: Number of local epochs
            lr: Learning rate
            
        Returns:
            Final training loss
        """
        model = self.client_models[client_id]
        mask = self.client_masks[client_id]
        
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        X, y = train_data
        X = X.to(self.device)
        y = y.to(self.device)
        
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(X, mask)
            loss = criterion(pred.squeeze(), y.float())
            loss.backward()
            optimizer.step()
        
        return loss.item()
    
    def aggregate_round(self):
        """
        Perform one round of GNN-based aggregation.
        """
        new_states = {}
        
        for cid in self.client_models:
            neighbors = self.neighbors.get(cid, [])
            
            if not neighbors:
                # No neighbors, keep own weights
                new_states[cid] = self.client_models[cid].state_dict()
            else:
                new_states[cid] = self.aggregator.aggregate(
                    target_id=cid,
                    neighbors=neighbors,
                    client_models=self.client_models,
                    client_masks=self.client_masks,
                    edge_features=self.edge_features,
                    node_features=self.node_features,
                    self_weight=self.self_weight
                )
        
        # Update all models
        for cid, state in new_states.items():
            self.client_models[cid].load_state_dict(state)
    
    def evaluate(
        self,
        client_id: int,
        test_data: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Evaluate a client model on test data.
        
        Args:
            client_id: Client ID
            test_data: Tuple of (X, y) tensors
            
        Returns:
            Dict with 'accuracy', 'loss' metrics
        """
        model = self.client_models[client_id]
        mask = self.client_masks[client_id]
        
        model.eval()
        X, y = test_data
        X = X.to(self.device)
        y = y.to(self.device)
        
        with torch.no_grad():
            logits = model(X, mask)
            preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
            
            accuracy = (preds == y).float().mean().item()
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(), y.float()
            ).item()
        
        return {'accuracy': accuracy, 'loss': loss}


def main():
    """Test the FedGNN implementation."""
    print("Testing FedGNN with masked aggregation...\n")
    
    # Simulate 4 clients with different feature coverage
    n_clients = 4
    union_dim = 147  # From column matcher
    hidden_dim = 32
    
    # Simulate feature masks (varying coverage)
    np.random.seed(42)
    feature_masks = {
        0: [True] * 14 + [False] * 133,   # DB54379: 14/147
        1: [True] * 123 + [False] * 24,   # DB37176: 123/147 
        2: [True] * 41 + [False] * 106,   # DB85770: 41/147
        3: [True] * 58 + [False] * 89,    # DB50469: 58/147
    }
    
    # Shuffle to simulate realistic distribution
    for cid in feature_masks:
        mask = feature_masks[cid]
        np.random.shuffle(mask)
        feature_masks[cid] = mask
    
    # Initialize FedGNN
    fedgnn = FedGNN(
        input_dim=union_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        property_mode="both"
    )
    
    # Initialize clients
    fedgnn.initialize_clients(list(range(n_clients)), feature_masks)
    
    # Set graph structure (fully connected for demo)
    neighbors = {i: [j for j in range(n_clients) if j != i] for i in range(n_clients)}
    fedgnn.set_graph_structure(neighbors)
    
    print(f"Initialized {n_clients} clients with union dimension {union_dim}")
    print("Feature coverage per client:")
    for cid in range(n_clients):
        coverage = sum(feature_masks[cid])
        print(f"  Client {cid}: {coverage}/{union_dim} ({100*coverage/union_dim:.1f}%)")
    
    # Generate dummy data
    n_samples = 100
    dummy_data = {}
    for cid in range(n_clients):
        mask = torch.tensor(feature_masks[cid], dtype=torch.float32)
        X = torch.randn(n_samples, union_dim) * mask  # Masked input
        y = (X.sum(dim=1) > 0).float()  # Simple binary label
        dummy_data[cid] = (X, y)
    
    # Training loop
    print("\n--- Training ---")
    for round_idx in range(5):
        # Local updates
        losses = []
        for cid in range(n_clients):
            loss = fedgnn.local_update(cid, dummy_data[cid], epochs=3, lr=0.01)
            losses.append(loss)
        
        # GNN aggregation
        fedgnn.aggregate_round()
        
        # Evaluate
        accs = []
        for cid in range(n_clients):
            metrics = fedgnn.evaluate(cid, dummy_data[cid])
            accs.append(metrics['accuracy'])
        
        print(f"Round {round_idx+1}: Loss={np.mean(losses):.4f}, Acc={np.mean(accs):.4f}")
    
    # Final per-client accuracy
    print("\n--- Final Per-Client Accuracy ---")
    for cid in range(n_clients):
        metrics = fedgnn.evaluate(cid, dummy_data[cid])
        print(f"  Client {cid}: {metrics['accuracy']:.4f}")
    
    print("\n✓ FedGNN test completed successfully!")


if __name__ == "__main__":
    main()
