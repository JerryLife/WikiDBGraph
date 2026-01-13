"""
FedGTA: Federated Graph Topology-Aware Aggregation for Global FL.

This module implements FedGTA adapted for global federated learning on WikiDB
subgraphs with missing value handling via coordinate-wise masked aggregation.

Key differences from original personalized FedGTA:
1. Global model broadcast (not per-client personalized models)
2. All clients contribute (no similarity-based filtering)
3. Coordinate-wise masked aggregation for heterogeneous schemas

Reference: FedGTA paper and baseline/FedGTA implementation
"""

import os
import sys
import copy
import math
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class NonParametricLP:
    """
    Non-parametric Label Propagation for topology-aware soft labels.
    
    Matches baseline implementation in gnn_model/label_propagation.py.
    Uses iterative propagation: Y = alpha * (A_norm @ Y) + (1 - alpha) * Y_init
    """
    
    def __init__(
        self,
        adj: torch.Tensor,
        prop_steps: int = 5,
        alpha: float = 0.5,
        r: float = 0.5,
        device: str = "cpu"
    ):
        """
        Args:
            adj: Adjacency matrix (dense or sparse)
            prop_steps: Number of propagation steps (K in paper)
            alpha: Weight for neighbor aggregation (1-alpha for initial features)
            r: Power for symmetric normalization D^(r-1) A D^(-r)
            device: Device for computation
        """
        self.prop_steps = prop_steps
        self.alpha = alpha
        self.r = r
        self.device = device
        
        # Compute symmetric normalized adjacency
        self.adj_norm = self._symmetric_normalize(adj, r)
    
    def _symmetric_normalize(self, adj: torch.Tensor, r: float) -> torch.Tensor:
        """
        Compute symmetric normalized adjacency: D^(r-1) A D^(-r).
        Matches baseline gm.py adj_to_symmetric_norm.
        """
        if isinstance(adj, sp.spmatrix):
            adj = adj.tocoo()
            adj_t = torch.sparse_coo_tensor(
                indices=torch.LongTensor([adj.row, adj.col]),
                values=torch.FloatTensor(adj.data),
                size=adj.shape
            ).to(self.device)
        elif isinstance(adj, np.ndarray):
            adj_t = torch.tensor(adj, dtype=torch.float32, device=self.device)
        else:
            adj_t = adj.to(self.device)
        
        # Add self-loops
        n = adj_t.shape[0]
        if adj_t.is_sparse:
            adj_t = adj_t.to_dense()
        adj_t = adj_t + torch.eye(n, device=self.device)
        
        # Compute degree
        degrees = adj_t.sum(dim=1)
        
        # D^(r-1) and D^(-r)
        d_pow_r_1 = torch.pow(degrees, r - 1)
        d_pow_neg_r = torch.pow(degrees, -r)
        
        # Handle inf
        d_pow_r_1[torch.isinf(d_pow_r_1)] = 0.0
        d_pow_neg_r[torch.isinf(d_pow_neg_r)] = 0.0
        
        # D^(r-1) A D^(-r)
        adj_norm = (d_pow_r_1.unsqueeze(1) * adj_t) * d_pow_neg_r.unsqueeze(0)
        
        return adj_norm
    
    def propagate(
        self,
        initial_labels: torch.Tensor,
        labeled_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Perform label propagation.
        
        Matches baseline label_propagation.py line 44-56:
        feat_temp = alpha * feat_temp + (1 - alpha) * feature
        
        Args:
            initial_labels: Initial soft labels [n_nodes, n_classes]
            labeled_mask: Optional mask for nodes to keep fixed
            
        Returns:
            Propagated labels [n_nodes, n_classes]
        """
        Y = initial_labels.clone().to(self.device)
        Y_init = initial_labels.clone().to(self.device)
        
        for _ in range(self.prop_steps):
            # Propagate: Y = alpha * (A @ Y) + (1 - alpha) * Y_init
            Y = self.alpha * torch.mm(self.adj_norm, Y) + (1 - self.alpha) * Y_init
            
            # Keep labeled nodes fixed (as in baseline line 55)
            if labeled_mask is not None:
                Y[labeled_mask] = Y_init[labeled_mask]
        
        return Y


def compute_smoothing_confidence(
    soft_labels: torch.Tensor,
    degrees: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Compute smoothing confidence (aggregation weight).
    
    Matches baseline fedgta_client.py line 13-14:
    info_entropy_rev = (num_neig.sum()) * vec.shape[1] * exp(-1) 
                       + sum(num_neig * sum(vec * log(vec), dim=1))
    
    Uses vec * log(vec) directly (negative values) to reward low entropy.
    
    Args:
        soft_labels: Softmax probabilities [n_nodes, n_classes]
        degrees: Node degrees [n_nodes]
        eps: Small constant for numerical stability
        
    Returns:
        Scalar smoothing confidence value
    """
    n_nodes, n_classes = soft_labels.shape
    
    # vec * log(vec) - this is NEGATIVE for probabilities < 1
    # More confident (entropy closer to 0) means this is less negative
    raw_entropy_per_node = torch.sum(
        soft_labels * torch.log(soft_labels + eps), dim=1
    )  # Shape: [n_nodes], negative values
    
    # Weighted sum by degrees
    weighted_entropy = torch.sum(degrees * raw_entropy_per_node)
    
    # Add positive constant (max possible value)
    max_term = degrees.sum() * n_classes * math.exp(-1)
    
    return max_term + weighted_entropy


def compute_moments(
    soft_labels: torch.Tensor,
    num_moments: int = 5,
    moment_type: str = "origin"
) -> torch.Tensor:
    """
    Compute mixed moments of soft labels.
    
    Matches baseline gm.py compute_moment with dim="v" (vertical, per-class).
    
    Args:
        soft_labels: Propagated labels [n_nodes, n_classes]
        num_moments: Number of moment orders (1 to num_moments)
        moment_type: "origin" (raw moments), "mean" (central moments), "hybrid"
        
    Returns:
        Moment vector [num_moments * n_classes] flattened
    """
    moments = []
    
    for order in range(1, num_moments + 1):
        if moment_type == "origin":
            # Origin moment: E[X^k]
            moment = torch.mean(torch.pow(soft_labels, order), dim=0)
        elif moment_type == "mean":
            # Central moment: E[(X - mean)^k]
            mean = torch.mean(soft_labels, dim=0)
            centered = soft_labels - mean.unsqueeze(0)
            moment = torch.mean(torch.pow(centered, order), dim=0)
        else:  # hybrid
            origin = torch.mean(torch.pow(soft_labels, order), dim=0)
            mean = torch.mean(soft_labels, dim=0)
            centered = soft_labels - mean.unsqueeze(0)
            central = torch.mean(torch.pow(centered, order), dim=0)
            moment = torch.cat([origin, central])
        
        moments.append(moment.view(1, -1))
    
    return torch.cat(moments).view(-1)


class LocalModel(nn.Module):
    """
    MLP model for local client training with BatchNorm for stability.
    Supports masked forward pass for union schema.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
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
            x = x * mask.unsqueeze(0)
        return self.net(x)


class FedGTAClient:
    """
    FedGTA Client for local operations.
    
    Performs:
    1. Local model training with masked features
    2. Non-parametric LP on predictions
    3. Smoothing confidence computation
    4. Mixed moment computation for similarity
    """
    
    def __init__(
        self,
        client_id: str,
        model: LocalModel,
        mask: torch.Tensor,
        lp_prop_steps: int = 5,
        lp_alpha: float = 0.5,
        num_moments: int = 5,
        moment_type: str = "origin",
        device: str = "cpu"
    ):
        """
        Args:
            client_id: Unique client identifier
            model: Local model instance
            mask: Feature mask [input_dim]
            lp_prop_steps: LP propagation steps
            lp_alpha: LP alpha parameter
            num_moments: Number of moment orders
            moment_type: Type of moments ("origin", "mean", "hybrid")
            device: Computation device
        """
        self.client_id = client_id
        self.model = model.to(device)
        self.mask = mask.to(device)
        self.lp_prop_steps = lp_prop_steps
        self.lp_alpha = lp_alpha
        self.num_moments = num_moments
        self.moment_type = moment_type
        self.device = device
        
        # Will be set when data is available
        self.lp = None
        self.degrees = None
    
    def setup_lp(self, adj: torch.Tensor, degrees: torch.Tensor):
        """Setup label propagation with adjacency matrix."""
        self.lp = NonParametricLP(
            adj=adj,
            prop_steps=self.lp_prop_steps,
            alpha=self.lp_alpha,
            device=self.device
        )
        self.degrees = degrees.to(self.device)
    
    def local_train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 3,
        lr: float = 0.01
    ) -> float:
        """
        Perform local training.
        
        Args:
            X: Features [n_samples, input_dim]
            y: Labels [n_samples]
            epochs: Number of local epochs
            lr: Learning rate
            
        Returns:
            Final training loss
        """
        X = X.to(self.device)
        y = y.to(self.device)
        
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()
        
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = self.model(X, self.mask)
            loss = criterion(pred.squeeze(), y.float())
            loss.backward()
            optimizer.step()
        
        return loss.item()
    
    def compute_topology_metrics(
        self,
        X: torch.Tensor,
        labeled_mask: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute topology-aware metrics for aggregation.
        
        Args:
            X: Features for LP initialization [n_nodes, input_dim]
            labeled_mask: Mask for labeled nodes
            
        Returns:
            Dict with 'smoothing_conf', 'moments', 'predictions'
        """
        X = X.to(self.device)
        
        # Get model predictions as soft labels
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X, self.mask)
            # For binary classification, create 2-class soft labels
            probs = torch.sigmoid(logits.squeeze())
            soft_labels = torch.stack([1 - probs, probs], dim=1)
        
        # Apply label propagation
        if self.lp is not None:
            propagated = self.lp.propagate(soft_labels, labeled_mask)
            # Ensure valid probabilities
            propagated = F.softmax(propagated, dim=1)
        else:
            propagated = soft_labels
        
        # Compute smoothing confidence
        if self.degrees is not None:
            smoothing_conf = compute_smoothing_confidence(
                propagated, self.degrees
            )
        else:
            # Fallback: uniform confidence
            smoothing_conf = torch.tensor(1.0, device=self.device)
        
        # Compute moments
        moments = compute_moments(
            propagated, 
            num_moments=self.num_moments,
            moment_type=self.moment_type
        )
        
        return {
            'smoothing_conf': smoothing_conf,
            'moments': moments,
            'predictions': soft_labels
        }
    
    def get_state_dict(self) -> OrderedDict:
        """Get model state dict."""
        return copy.deepcopy(self.model.state_dict())
    
    def set_state_dict(self, state_dict: OrderedDict):
        """Set model state dict."""
        self.model.load_state_dict(copy.deepcopy(state_dict))


class FedGTAServer:
    """
    FedGTA Server for global aggregation.
    
    Performs:
    1. Collect client metrics (weights, confidence, moments)
    2. Compute similarity matrix (for logging)
    3. Confidence-weighted masked aggregation
    4. Broadcast global model to all clients
    
    Property modes:
    - none: Standard FedAvg-style uniform aggregation
    - edge: Use edge-based moments for similarity (LP-based)
    - node: Use node-based smoothing confidence for weighting
    - both: Use both LP moments and smoothing confidence (default)
    """
    
    def __init__(self, property_mode: str = "both", device: str = "cpu"):
        """
        Args:
            property_mode: 'none', 'edge', 'node', or 'both'
            device: Computation device
        """
        self.property_mode = property_mode
        self.device = device
        self.similarity_matrix = None
    
    def compute_similarity_matrix(
        self,
        moments: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute pairwise cosine similarity of moment vectors.
        
        Args:
            moments: Dict mapping client_id -> moment vector
            
        Returns:
            Similarity matrix [n_clients, n_clients]
        """
        client_ids = list(moments.keys())
        n = len(client_ids)
        sim = torch.zeros(n, n, device=self.device)
        
        for i, c1 in enumerate(client_ids):
            for j, c2 in enumerate(client_ids):
                sim[i, j] = F.cosine_similarity(
                    moments[c1].unsqueeze(0),
                    moments[c2].unsqueeze(0)
                )
        
        self.similarity_matrix = sim
        return sim
    
    def aggregate(
        self,
        client_states: Dict[str, OrderedDict],
        client_masks: Dict[str, torch.Tensor],
        client_confs: Dict[str, torch.Tensor],
        client_moments: Dict[str, torch.Tensor] = None
    ) -> OrderedDict:
        """
        Perform confidence-weighted masked aggregation.
        
        Global FL: All clients contribute (no similarity filtering).
        
        Args:
            client_states: Dict of client model state dicts
            client_masks: Dict of feature masks [input_dim]
            client_confs: Dict of smoothing confidence values
            client_moments: Optional moments for similarity logging
            
        Returns:
            Aggregated global state dict
        """
        client_ids = list(client_states.keys())
        
        if not client_ids:
            raise ValueError("No clients to aggregate")
        
        # Compute similarity for logging if edge or both mode
        if client_moments is not None and self.property_mode in ["edge", "both"]:
            sim = self.compute_similarity_matrix(client_moments)
            print(f"  Similarity matrix: min={sim.min():.3f}, max={sim.max():.3f}, mean={sim.mean():.3f}")
        
        # Determine weights based on property_mode
        if self.property_mode == "none":
            # Uniform weighting
            n_clients = len(client_ids)
            norm_confs = {cid: 1.0 / n_clients for cid in client_ids}
        elif self.property_mode == "edge":
            # Uniform weighting (moments used for logging only in global FL)
            n_clients = len(client_ids)
            norm_confs = {cid: 1.0 / n_clients for cid in client_ids}
        elif self.property_mode in ["node", "both"]:
            # Use smoothing confidence for weighting
            confs = {cid: max(client_confs[cid].item(), 1e-8) for cid in client_ids}
            total_conf = sum(confs.values())
            norm_confs = {cid: confs[cid] / total_conf for cid in client_ids}
        else:
            raise ValueError(f"Unknown property_mode: {self.property_mode}")
        
        # Reference state for structure
        ref_state = client_states[client_ids[0]]
        global_state = OrderedDict()
        
        for key in ref_state:
            ref_param = ref_state[key]
            
            if "net.0.weight" in key:
                # Coordinate-wise masked aggregation for input layer
                # Shape: [hidden_dim, input_dim]
                weighted_sum = torch.zeros_like(ref_param).to(self.device)
                conf_sum = torch.zeros_like(ref_param).to(self.device) + 1e-8
                
                for cid in client_ids:
                    w_c = norm_confs[cid]
                    param = client_states[cid][key].to(self.device)
                    mask = client_masks[cid].float().to(self.device)
                    
                    # Expand mask: [input_dim] -> [hidden_dim, input_dim]
                    mask_expanded = mask.unsqueeze(0).expand_as(param)
                    
                    weighted_sum += w_c * param * mask_expanded
                    conf_sum += w_c * mask_expanded
                
                global_state[key] = weighted_sum / conf_sum
                
            else:
                # Standard weighted aggregation for other layers
                # Skip BatchNorm running stats (they are not weights)
                if "num_batches_tracked" in key:
                    # Just use first client's value
                    global_state[key] = client_states[client_ids[0]][key].clone()
                elif "running_" in key:
                    # Average running stats (running_mean, running_var)
                    sum_param = torch.zeros_like(ref_param, dtype=torch.float32).to(self.device)
                    for cid in client_ids:
                        sum_param += client_states[cid][key].float().to(self.device)
                    global_state[key] = sum_param / len(client_ids)
                else:
                    # Standard weighted aggregation
                    weighted_param = torch.zeros_like(ref_param, dtype=torch.float32).to(self.device)
                    
                    for cid in client_ids:
                        w_c = norm_confs[cid]
                        param = client_states[cid][key].float().to(self.device)
                        weighted_param += w_c * param
                    
                    global_state[key] = weighted_param
        
        return global_state


class FedGTA:
    """
    High-level FedGTA trainer for global federated learning.
    
    Combines:
    - Local client training with masked features
    - Non-parametric LP for topology-aware metrics
    - Confidence-weighted masked aggregation
    - Global test set evaluation
    
    Property modes:
    - none: Standard FedAvg-style uniform aggregation
    - edge: Use LP moments (edge/neighborhood structure)
    - node: Use smoothing confidence (node prediction quality)
    - both: Use both LP and confidence (full FedGTA)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 1,
        property_mode: str = "both",
        lp_prop_steps: int = 5,
        lp_alpha: float = 0.5,
        num_moments: int = 5,
        moment_type: str = "origin",
        device: str = "cpu"
    ):
        """
        Args:
            input_dim: Union schema dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            property_mode: 'none', 'edge', 'node', or 'both'
            lp_prop_steps: LP propagation steps
            lp_alpha: LP alpha parameter
            num_moments: Number of moment orders
            moment_type: Moment type
            device: Computation device
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.property_mode = property_mode
        self.lp_prop_steps = lp_prop_steps
        self.lp_alpha = lp_alpha
        self.num_moments = num_moments
        self.moment_type = moment_type
        self.device = device
        
        self.clients: Dict[str, FedGTAClient] = {}
        self.server = FedGTAServer(property_mode=property_mode, device=device)
    
    def initialize_clients(
        self,
        client_ids: List[str],
        feature_masks: Dict[str, List[bool]]
    ):
        """
        Initialize clients with shared initial model.
        
        Args:
            client_ids: List of client IDs
            feature_masks: Dict mapping client_id -> boolean mask list
        """
        # Create shared initial model
        init_model = LocalModel(self.input_dim, self.hidden_dim, self.output_dim)
        init_state = init_model.state_dict()
        
        for cid in client_ids:
            model = LocalModel(self.input_dim, self.hidden_dim, self.output_dim)
            model.load_state_dict(copy.deepcopy(init_state))
            
            mask = torch.tensor(feature_masks[cid], dtype=torch.float32)
            
            self.clients[cid] = FedGTAClient(
                client_id=cid,
                model=model,
                mask=mask,
                lp_prop_steps=self.lp_prop_steps,
                lp_alpha=self.lp_alpha,
                num_moments=self.num_moments,
                moment_type=self.moment_type,
                device=self.device
            )
    
    def setup_client_lp(
        self,
        client_id: str,
        adj: torch.Tensor,
        degrees: torch.Tensor
    ):
        """Setup LP for a specific client."""
        if client_id in self.clients:
            self.clients[client_id].setup_lp(adj, degrees)
    
    def local_update(
        self,
        client_id: str,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 3,
        lr: float = 0.01
    ) -> float:
        """Perform local training for a client."""
        return self.clients[client_id].local_train(X, y, epochs, lr)
    
    def aggregate_round(
        self,
        client_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ):
        """
        Perform one round of FedGTA aggregation.
        
        Args:
            client_data: Dict mapping client_id -> (X_train, y_train)
        """
        # Collect client outputs
        client_states = {}
        client_masks = {}
        client_confs = {}
        client_moments = {}
        
        for cid, client in self.clients.items():
            # Get topology metrics
            X, _ = client_data.get(cid, (None, None))
            if X is not None:
                metrics = client.compute_topology_metrics(X)
                client_confs[cid] = metrics['smoothing_conf']
                client_moments[cid] = metrics['moments']
            else:
                client_confs[cid] = torch.tensor(1.0, device=self.device)
                client_moments[cid] = torch.zeros(
                    self.num_moments * 2, device=self.device
                )
            
            client_states[cid] = client.get_state_dict()
            client_masks[cid] = client.mask
        
        # Server aggregation
        global_state = self.server.aggregate(
            client_states, client_masks, client_confs, client_moments
        )
        
        # Broadcast to all clients
        for cid in self.clients:
            self.clients[cid].set_state_dict(global_state)
    
    def evaluate(
        self,
        client_id: str,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> Dict[str, float]:
        """
        Evaluate a client's model.
        
        Args:
            client_id: Client ID
            X: Test features
            y: Test labels
            
        Returns:
            Dict with accuracy and loss
        """
        client = self.clients[client_id]
        client.model.eval()
        
        X = X.to(self.device)
        y = y.to(self.device)
        
        with torch.no_grad():
            logits = client.model(X, client.mask)
            preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
            
            accuracy = (preds == y).float().mean().item()
            loss = F.binary_cross_entropy_with_logits(
                logits.squeeze(), y.float()
            ).item()
        
        return {'accuracy': accuracy, 'loss': loss}
    
    def evaluate_global(
        self,
        test_data: Dict[str, Tuple[torch.Tensor, torch.Tensor]]
    ) -> Dict[str, float]:
        """
        Evaluate on combined global test set.
        
        Args:
            test_data: Dict mapping client_id -> (X_test, y_test)
            
        Returns:
            Dict with global accuracy and loss
        """
        all_preds = []
        all_labels = []
        all_losses = []
        
        for cid, (X, y) in test_data.items():
            if cid not in self.clients:
                continue
            
            client = self.clients[cid]
            client.model.eval()
            
            X = X.to(self.device)
            y = y.to(self.device)
            
            with torch.no_grad():
                logits = client.model(X, client.mask)
                preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(), y.float()
                ).item()
            
            all_preds.append(preds)
            all_labels.append(y)
            all_losses.append(loss * len(y))
        
        if not all_preds:
            return {'accuracy': 0.0, 'loss': 0.0}
        
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        total_samples = len(all_labels)
        
        global_accuracy = (all_preds == all_labels).float().mean().item()
        global_loss = sum(all_losses) / total_samples
        
        return {'accuracy': global_accuracy, 'loss': global_loss}


def main():
    """Test the FedGTA implementation."""
    print("Testing FedGTA with topology-aware aggregation...\n")
    
    # Simulate 3 clients with different feature coverage
    n_clients = 3
    union_dim = 50
    hidden_dim = 32
    n_samples = 100
    
    # Simulate feature masks
    np.random.seed(42)
    torch.manual_seed(42)
    
    feature_masks = {
        "client0": [True] * 30 + [False] * 20,
        "client1": [True] * 20 + [False] * 10 + [True] * 10 + [False] * 10,
        "client2": [False] * 20 + [True] * 30,
    }
    
    # Initialize FedGTA
    fedgta = FedGTA(
        input_dim=union_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        lp_prop_steps=3,
        num_moments=3,
        moment_type="origin"
    )
    
    fedgta.initialize_clients(list(feature_masks.keys()), feature_masks)
    
    print(f"Initialized {n_clients} clients with union dimension {union_dim}")
    print("Feature coverage per client:")
    for cid, mask in feature_masks.items():
        coverage = sum(mask)
        print(f"  {cid}: {coverage}/{union_dim} ({100*coverage/union_dim:.1f}%)")
    
    # Generate dummy data
    client_data = {}
    test_data = {}
    
    for cid, mask in feature_masks.items():
        mask_t = torch.tensor(mask, dtype=torch.float32)
        X = torch.randn(n_samples, union_dim) * mask_t
        y = (X.sum(dim=1) > 0).float()
        
        # Split train/test
        X_train, y_train = X[:80], y[:80]
        X_test, y_test = X[80:], y[80:]
        
        client_data[cid] = (X_train, y_train)
        test_data[cid] = (X_test, y_test)
        
        # Setup simple k-NN graph for LP
        n = len(X_train)
        adj = torch.eye(n) + 0.1 * torch.randn(n, n).abs()
        adj = (adj + adj.T) / 2  # Symmetric
        degrees = adj.sum(dim=1)
        fedgta.setup_client_lp(cid, adj, degrees)
    
    # Training loop
    print("\n--- Training ---")
    for round_idx in range(5):
        # Local updates
        for cid, (X, y) in client_data.items():
            fedgta.local_update(cid, X, y, epochs=3, lr=0.01)
        
        # Aggregation
        fedgta.aggregate_round(client_data)
        
        # Evaluate
        global_metrics = fedgta.evaluate_global(test_data)
        print(f"Round {round_idx+1}: Global Acc={global_metrics['accuracy']:.4f}")
    
    # Final per-client accuracy
    print("\n--- Final Per-Client Accuracy ---")
    for cid in client_data:
        X_test, y_test = test_data[cid]
        metrics = fedgta.evaluate(cid, X_test, y_test)
        print(f"  {cid}: {metrics['accuracy']:.4f}")
    
    print("\n✓ FedGTA test completed successfully!")


if __name__ == "__main__":
    main()
