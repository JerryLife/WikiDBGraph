"""
Train FedGTA: Training script for Topology-Aware Federated Learning.

This script trains and evaluates the FedGTA model on horizontally-partitioned
WikiDB databases with global FL testing (not personalized).

Usage:
    python src/demo/train_fedgta.py --db_ids 37176,85770,50469
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Tuple
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.FedGTA import FedGTA, LocalModel, compute_smoothing_confidence, compute_moments
from model.FedGNN import MaskedFedAvg, LocalModel as FedGNNLocalModel
from model.WKDataset import WKDataset
from analysis.WikiDBSubgraph import WikiDBSubgraph
from analysis.semantic_column_matcher import match_columns_across_databases


def load_database_data(
    wk: WKDataset,
    db_id: str,
    union_columns: Dict[str, Dict[str, str]],
    target_candidates: List[str] = None
) -> Tuple[pd.DataFrame, List[bool]]:
    """
    Load and align database data to union schema using proper table joins.
    
    Args:
        wk: WKDataset instance
        db_id: Database ID
        union_columns: Union column mapping from semantic_column_matcher
        target_candidates: List of target column names (lowercase) to find base table
        
    Returns:
        Tuple of (aligned DataFrame, feature mask)
    """
    if target_candidates is None:
        target_candidates = ['entitytype', 'placetype', 'citytype']
    
    # Load schema to get foreign key relationships
    schema = wk.load_database(db_id)
    
    # Load ALL tables WITHOUT sampling
    table_dfs = wk.load_csv_data(db_id, sample=False)
    
    if not table_dfs:
        return pd.DataFrame(), []
    
    # Find the table containing a target column (this will be the base table)
    base_table_name = None
    base_target_col = None
    
    for candidate in target_candidates:
        for table_name, df in table_dfs.items():
            for col in df.columns:
                if col.lower() == candidate:
                    base_table_name = table_name
                    base_target_col = col
                    break
            if base_table_name:
                break
        if base_table_name:
            break
    
    if base_table_name is None:
        # Fallback: use largest table as base
        base_table_name = max(table_dfs.keys(), key=lambda k: len(table_dfs[k]))
    
    base_df = table_dfs[base_table_name].copy()
    print(f"    Base table: {base_table_name} ({len(base_df)} rows)")
    
    # Build FK lookup from schema
    fk_map = {}  # table_name -> list of {col, ref_table, ref_col}
    for table_info in schema.get('tables', []):
        t_name = table_info['table_name']
        fk_map[t_name] = []
        for fk in table_info.get('foreign_keys', []):
            fk_map[t_name].append({
                'column': fk['column_name'],
                'ref_table': fk['reference_table_name'],
                'ref_col': fk['reference_column_name']
            })
    
    # Perform LEFT JOINs from base table to other tables via FKs
    joined_df = base_df.copy()
    joined_tables = {base_table_name}
    
    # Try to join other tables via FK relationships (BFS from base)
    for fk in fk_map.get(base_table_name, []):
        ref_table = fk['ref_table']
        if ref_table in joined_tables or ref_table not in table_dfs:
            continue
        
        left_col = fk['column']
        right_col = fk['ref_col']
        
        if left_col not in joined_df.columns:
            continue
        if right_col not in table_dfs[ref_table].columns:
            continue
        
        # Perform LEFT JOIN, adding suffix to avoid column name conflicts
        ref_df = table_dfs[ref_table].copy()
        # Rename columns to avoid conflicts (except join key)
        rename_map = {}
        for col in ref_df.columns:
            if col != right_col and col in joined_df.columns:
                rename_map[col] = f"{col}_{ref_table}"
        ref_df = ref_df.rename(columns=rename_map)
        
        joined_df = joined_df.merge(
            ref_df, 
            left_on=left_col, 
            right_on=right_col, 
            how='left',
            suffixes=('', f'_{ref_table}')
        )
        joined_tables.add(ref_table)
        print(f"    Joined {ref_table} via {left_col}->{right_col} ({len(joined_df)} rows)")
    
    # Get column mapping for this database
    db_col_map = {}  # normalized_name -> original_name
    for norm_name, mapping in union_columns.items():
        if mapping.get(db_id) is not None:
            db_col_map[norm_name] = mapping[db_id]
    
    # Create aligned DataFrame with union schema
    sorted_cols = sorted(union_columns.keys())
    aligned_data = {}
    feature_mask = []
    
    for norm_name in sorted_cols:
        orig_name = db_col_map.get(norm_name)
        if orig_name and orig_name in joined_df.columns:
            aligned_data[norm_name] = joined_df[orig_name].values
            feature_mask.append(True)
        else:
            # Missing column - fill with zeros
            aligned_data[norm_name] = np.zeros(len(joined_df))
            feature_mask.append(False)
    
    aligned_df = pd.DataFrame(aligned_data)
    return aligned_df, feature_mask


def prepare_client_data(
    wk: WKDataset,
    db_ids: List[str],
    union_columns: Dict[str, Dict[str, str]],
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Dict]:
    """
    Prepare training and test data for all clients.
    
    Returns:
        Dict mapping db_id -> {
            'X_train': tensor, 'y_train': tensor,
            'X_test': tensor, 'y_test': tensor,
            'mask': list of bool
        }
    """
    client_data = {}
    
    for db_id in db_ids:
        print(f"  Loading data for DB{db_id}...")
        aligned_df, feature_mask = load_database_data(wk, db_id, union_columns)
        
        if aligned_df.empty:
            print(f"    Warning: No data for DB{db_id}, skipping")
            continue
        
        cols = aligned_df.columns.tolist()
        if len(cols) < 2:
            print(f"    Warning: Not enough columns for DB{db_id}, skipping")
            continue
        
        # Target Column Selection - "City vs Non-City" classification
        target_candidates = ['entitytype', 'placetype', 'citytype']
        target_col = None
        target_idx = None
        
        for candidate in target_candidates:
            for i, col in enumerate(cols):
                if col.lower() == candidate:
                    col_present = feature_mask[i] if i < len(feature_mask) else False
                    if col_present:
                        target_col = col
                        target_idx = i
                        break
            if target_col:
                break
        
        if target_col is None:
            print(f"    Warning: No PlaceType column found for DB{db_id}, using hash-based labels")
            feature_cols = cols
            feature_mask_final = feature_mask
            hash_vals = aligned_df.iloc[:, 0].astype(str).apply(lambda x: hash(x) % 2)
            y = hash_vals.values.astype(np.float32)
        else:
            # LEAKAGE PREVENTION: Remove ALL target candidate columns from features
            feature_cols = [c for c in cols if c.lower() not in target_candidates]
            feature_mask_final = [m for c, m in zip(cols, feature_mask) if c.lower() not in target_candidates]
            
            # Create binary labels: 1 if "city" in value, 0 otherwise
            target_data = aligned_df[target_col].astype(str).fillna('')
            y = target_data.str.contains('city', case=False).astype(np.float32).values
            
            n_city = int(y.sum())
            print(f"    Target: {target_col} -> City={n_city}/{len(y)} ({100*n_city/len(y):.1f}%)")
        
        # Process features: encode categorical columns
        X_list = []
        for col in feature_cols:
            col_data = aligned_df[col]
            
            # Try to convert to numeric
            numeric_data = pd.to_numeric(col_data, errors='coerce')
            
            if numeric_data.isna().sum() / len(numeric_data) > 0.5:
                # More than 50% failed to convert - treat as categorical
                le = LabelEncoder()
                try:
                    encoded = le.fit_transform(col_data.astype(str).fillna('__MISSING__'))
                    X_list.append(encoded.astype(np.float32))
                except Exception:
                    X_list.append(np.zeros(len(col_data), dtype=np.float32))
            else:
                X_list.append(numeric_data.fillna(0).values.astype(np.float32))
        
        X = np.column_stack(X_list) if X_list else np.zeros((len(aligned_df), 0))
        
        # Handle edge cases
        if X.shape[1] == 0:
            print(f"    Warning: No valid features for DB{db_id}, skipping")
            continue
        
        # Ensure label variation (flip some if all same)
        if np.std(y) == 0:
            print(f"    Warning: All labels same, adding variation")
            n_flip = max(1, int(len(y) * 0.2))
            flip_idx = np.random.choice(len(y), n_flip, replace=False)
            y[flip_idx] = 1 - y[flip_idx]
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Split with stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError:
            # Stratification failed (too few samples in a class)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        client_data[db_id] = {
            'X_train': torch.tensor(X_train, dtype=torch.float32),
            'y_train': torch.tensor(y_train, dtype=torch.float32),
            'X_test': torch.tensor(X_test, dtype=torch.float32),
            'y_test': torch.tensor(y_test, dtype=torch.float32),
            'mask': feature_mask_final,
            'n_samples': len(X)
        }
        print(f"    Loaded {len(X)} samples, {X.shape[1]} features")
    
    return client_data


def build_local_knn_graph(X: torch.Tensor, k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a k-NN graph for local subgraph.
    
    Args:
        X: Features [n_samples, n_features]
        k: Number of nearest neighbors
        
    Returns:
        Tuple of (adjacency matrix, node degrees)
    """
    n = X.shape[0]
    
    # Compute pairwise distances
    X_norm = X / (X.norm(dim=1, keepdim=True) + 1e-8)
    sim = torch.mm(X_norm, X_norm.T)
    
    # Get k nearest neighbors
    _, indices = sim.topk(min(k + 1, n), dim=1)  # +1 for self
    
    # Build adjacency matrix
    adj = torch.zeros(n, n)
    for i in range(n):
        for j in indices[i]:
            if i != j:
                adj[i, j] = 1
                adj[j, i] = 1  # Symmetric
    
    # Add self-loops
    adj = adj + torch.eye(n)
    
    # Compute degrees
    degrees = adj.sum(dim=1)
    
    return adj, degrees


def build_global_test_set(
    client_data: Dict[str, Dict],
    input_dim: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a combined global test set from all clients.
    
    Missing features are handled with zero interpolation (already zeros in the data).
    
    Args:
        client_data: Dict mapping db_id -> client data dict
        input_dim: Feature dimension
        
    Returns:
        Tuple of (X_global, y_global) tensors
    """
    all_X = []
    all_y = []
    
    for db_id, data in client_data.items():
        # X_test already has zeros for missing features (from prepare_client_data)
        all_X.append(data['X_test'])
        all_y.append(data['y_test'])
    
    X_global = torch.cat(all_X, dim=0)
    y_global = torch.cat(all_y, dim=0)
    
    print(f"  Global test set: {X_global.shape[0]} samples from {len(client_data)} clients")
    
    return X_global, y_global


def compute_global_metrics(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor = None
) -> Dict[str, float]:
    """
    Compute global metrics: Accuracy, AUC, F1.
    
    Args:
        model: Trained model
        X: Features [n_samples, input_dim]
        y: Labels [n_samples]
        mask: Optional feature mask (if None, use all features)
        
    Returns:
        Dict with 'accuracy', 'auc', 'f1', 'loss'
    """
    model.eval()
    
    with torch.no_grad():
        # Apply mask for zero interpolation (zeros already in data, mask zeros out missing)
        if mask is not None:
            X_masked = X * mask.unsqueeze(0)
        else:
            X_masked = X
        
        logits = model(X_masked, mask)
        probs = torch.sigmoid(logits.squeeze())
        preds = (probs > 0.5).float()
        
        # Convert to numpy for sklearn metrics
        y_np = y.numpy()
        probs_np = probs.numpy()
        preds_np = preds.numpy()
        
        # Accuracy
        accuracy = (preds == y).float().mean().item()
        
        # AUC (handle edge case where only one class exists)
        try:
            auc = roc_auc_score(y_np, probs_np)
        except ValueError:
            auc = 0.5  # Default when only one class
        
        # F1
        f1 = f1_score(y_np, preds_np, zero_division=0)
        
        # Loss
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(), y.float()
        ).item()
    
    return {
        'accuracy': accuracy,
        'auc': auc,
        'f1': f1,
        'loss': loss
    }


def train_solo(
    client_data: Dict[str, Dict],
    input_dim: int,
    hidden_dim: int = 64,
    epochs: int = 50,
    lr: float = 0.01
) -> LocalModel:
    """
    Train Solo model on pooled data from all clients.
    
    This is NOT federatd - it's centralized training for comparison.
    All client data is pooled together with zero interpolation for missing features.
    
    Returns:
        Trained model
    """
    # Pool all training data
    all_X = []
    all_y = []
    
    for db_id, data in client_data.items():
        # Data already has zeros for missing features
        all_X.append(data['X_train'])
        all_y.append(data['y_train'])
    
    X_train = torch.cat(all_X, dim=0)
    y_train = torch.cat(all_y, dim=0)
    
    # Create and train model (no mask needed - data already zero-interpolated)
    model = LocalModel(input_dim, hidden_dim, output_dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train, None)  # No mask - already zero-interpolated
        loss = criterion(pred.squeeze(), y_train)
        loss.backward()
        optimizer.step()
    
    return model


def train_fedavg(
    client_data: Dict[str, Dict],
    input_dim: int,
    hidden_dim: int = 64,
    global_rounds: int = 10,
    local_epochs: int = 3,
    lr: float = 0.01
) -> LocalModel:
    """
    Train with Masked FedAvg (coordinate-wise averaging).
    
    Returns:
        Global model after FL training
    """
    # Initialize models
    models = {}
    masks = {}
    
    init_model = LocalModel(input_dim, hidden_dim, output_dim=1)
    init_state = init_model.state_dict()
    
    for db_id, data in client_data.items():
        model = LocalModel(input_dim, hidden_dim, output_dim=1)
        model.load_state_dict(init_state.copy())
        models[db_id] = model
        masks[db_id] = torch.tensor(data['mask'], dtype=torch.float32)
    
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # Training loop
    for round_idx in range(global_rounds):
        # Local training
        for db_id, data in client_data.items():
            model = models[db_id]
            mask = masks[db_id]
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            
            model.train()
            for _ in range(local_epochs):
                optimizer.zero_grad()
                pred = model(data['X_train'], mask)
                loss = criterion(pred.squeeze(), data['y_train'])
                loss.backward()
                optimizer.step()
        
        # Masked FedAvg aggregation
        global_state = MaskedFedAvg.aggregate(models, masks)
        
        # Distribute global model
        for db_id in models:
            models[db_id].load_state_dict(global_state)
    
    # Return global model (all clients have same model after aggregation)
    global_model = LocalModel(input_dim, hidden_dim, output_dim=1)
    global_model.load_state_dict(global_state)
    
    return global_model


def train_fedgta(
    client_data: Dict[str, Dict],
    input_dim: int,
    hidden_dim: int = 64,
    global_rounds: int = 10,
    local_epochs: int = 3,
    lr: float = 0.01,
    property_mode: str = "both",
    lp_prop_steps: int = 5,
    lp_alpha: float = 0.5,
    num_moments: int = 5,
    moment_type: str = "origin",
    knn_k: int = 5
) -> LocalModel:
    """
    Train with FedGTA (Topology-Aware Aggregation).
    
    Args:
        property_mode: 'none', 'edge', 'node', or 'both'
    
    Returns:
        Global model after FL training
    """
    db_ids = list(client_data.keys())
    
    # Initialize FedGTA
    fedgta = FedGTA(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        property_mode=property_mode,
        lp_prop_steps=lp_prop_steps,
        lp_alpha=lp_alpha,
        num_moments=num_moments,
        moment_type=moment_type
    )
    
    # Initialize clients with feature masks
    feature_masks = {db_id: client_data[db_id]['mask'] for db_id in db_ids}
    fedgta.initialize_clients(db_ids, feature_masks)
    
    # Setup LP for each client using local k-NN graph
    for db_id in db_ids:
        X_train = client_data[db_id]['X_train']
        adj, degrees = build_local_knn_graph(X_train, k=knn_k)
        fedgta.setup_client_lp(db_id, adj, degrees)
    
    # Prepare data dicts
    train_data = {db_id: (client_data[db_id]['X_train'], client_data[db_id]['y_train']) 
                  for db_id in db_ids}
    
    # Training loop
    for round_idx in range(global_rounds):
        # Local updates
        for db_id in db_ids:
            X, y = train_data[db_id]
            fedgta.local_update(db_id, X, y, epochs=local_epochs, lr=lr)
        
        # FedGTA aggregation
        print(f"  Round {round_idx + 1}/{global_rounds}:")
        fedgta.aggregate_round(train_data)
    
    # Return global model (all clients have same model after FedGTA aggregation)
    # Get model from first client (all have same weights)
    first_client_id = db_ids[0]
    global_model = LocalModel(input_dim, hidden_dim, output_dim=1)
    global_model.load_state_dict(fedgta.clients[first_client_id].model.state_dict())
    
    return global_model


def main():
    parser = argparse.ArgumentParser(description="Train FedGTA model")
    parser.add_argument(
        "--db_ids",
        type=str,
        default="37176,85770,50469",  # 3-client subgraph with PlaceType columns
        help="Comma-separated database IDs"
    )
    parser.add_argument(
        "--global_rounds",
        type=int,
        default=20,
        help="Number of global communication rounds"
    )
    parser.add_argument(
        "--local_epochs",
        type=int,
        default=3,
        help="Number of local training epochs per round"
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Hidden layer dimension"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--property_mode",
        type=str,
        choices=["none", "edge", "node", "both"],
        default="both",
        help="Property mode for aggregation: none (FedAvg), edge (LP moments), node (confidence), both (full FedGTA)"
    )
    parser.add_argument(
        "--lp_prop_steps",
        type=int,
        default=5,
        help="Label propagation steps"
    )
    parser.add_argument(
        "--lp_alpha",
        type=float,
        default=0.5,
        help="LP alpha (neighbor vs initial weight)"
    )
    parser.add_argument(
        "--num_moments",
        type=int,
        default=5,
        help="Number of moment orders"
    )
    parser.add_argument(
        "--moment_type",
        type=str,
        choices=["origin", "mean", "hybrid"],
        default="origin",
        help="Type of moments"
    )
    parser.add_argument(
        "--knn_k",
        type=int,
        default=5,
        help="K for k-NN graph construction"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/fedgta",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Parse database IDs
    db_ids = [db_id.strip().zfill(5) for db_id in args.db_ids.split(",")]
    
    print("=" * 60)
    print("FedGTA Training Pipeline (Global FL)")
    print("=" * 60)
    print(f"Databases: {db_ids}")
    print(f"Global rounds: {args.global_rounds}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Property mode: {args.property_mode}")
    print(f"LP steps: {args.lp_prop_steps}, alpha: {args.lp_alpha}")
    print(f"Moments: {args.num_moments} ({args.moment_type})")
    print()
    
    # Step 1: Column matching (union schema)
    print("Step 1: Matching columns across databases...")
    wk = WKDataset(schema_dir="data/schema", csv_base_dir="data/unzip")
    union_columns, omitted_columns, feature_masks = match_columns_across_databases(wk, db_ids)
    input_dim = len(union_columns) - 1  # Exclude target column
    print(f"  Union schema size: {len(union_columns)} columns")
    
    # Step 2: Prepare client data
    print("\nStep 2: Preparing client data...")
    client_data = prepare_client_data(wk, db_ids, union_columns)
    
    if not client_data:
        print("Error: No client data loaded. Exiting.")
        return
    
    # Update input_dim based on actual feature count
    actual_input_dim = client_data[list(client_data.keys())[0]]['X_train'].shape[1]
    print(f"  Actual input dimension: {actual_input_dim}")
    
    # Step 3: Build global test set
    print("\nStep 3: Building global test set...")
    X_global, y_global = build_global_test_set(client_data, actual_input_dim)
    
    # Step 4: Train all methods
    print("\n" + "=" * 60)
    print("Training Models...")
    print("=" * 60)
    
    # Solo (centralized on pooled data)
    print("\n>>> Training Solo (centralized, pooled data)...")
    solo_model = train_solo(
        client_data, actual_input_dim, args.hidden_dim,
        epochs=args.global_rounds * args.local_epochs, lr=args.lr
    )
    
    # FedAvg
    print("\n>>> Training Masked FedAvg...")
    fedavg_model = train_fedavg(
        client_data, actual_input_dim, args.hidden_dim,
        args.global_rounds, args.local_epochs, args.lr
    )
    
    # FedGTA
    print(f"\n>>> Training FedGTA (mode: {args.property_mode}, LP steps: {args.lp_prop_steps}, moments: {args.num_moments})...")
    fedgta_model = train_fedgta(
        client_data, actual_input_dim, args.hidden_dim,
        args.global_rounds, args.local_epochs, args.lr,
        args.property_mode, args.lp_prop_steps, args.lp_alpha, args.num_moments,
        args.moment_type, args.knn_k
    )
    
    # Step 5: Evaluate all methods on global test set
    print("\n" + "=" * 60)
    print("Evaluating on Global Test Set...")
    print("=" * 60)
    
    # Evaluate Solo
    print("\n>>> Evaluating Solo...")
    solo_metrics = compute_global_metrics(solo_model, X_global, y_global, mask=None)
    
    # Evaluate FedAvg
    print(">>> Evaluating FedAvg...")
    fedavg_metrics = compute_global_metrics(fedavg_model, X_global, y_global, mask=None)
    
    # Evaluate FedGTA
    print(">>> Evaluating FedGTA...")
    fedgta_metrics = compute_global_metrics(fedgta_model, X_global, y_global, mask=None)
    
    # Step 6: Report results (Global Test Only)
    print("\n" + "=" * 70)
    print("GLOBAL TEST SET RESULTS")
    print("=" * 70)
    print(f"{'Method':<12} {'Accuracy':<12} {'AUC':<12} {'F1':<12} {'Loss':<12}")
    print("-" * 60)
    print(f"{'Solo':<12} {solo_metrics['accuracy']:<12.4f} {solo_metrics['auc']:<12.4f} {solo_metrics['f1']:<12.4f} {solo_metrics['loss']:<12.4f}")
    print(f"{'FedAvg':<12} {fedavg_metrics['accuracy']:<12.4f} {fedavg_metrics['auc']:<12.4f} {fedavg_metrics['f1']:<12.4f} {fedavg_metrics['loss']:<12.4f}")
    print(f"{'FedGTA':<12} {fedgta_metrics['accuracy']:<12.4f} {fedgta_metrics['auc']:<12.4f} {fedgta_metrics['f1']:<12.4f} {fedgta_metrics['loss']:<12.4f}")
    print("-" * 60)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, 
        f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    output = {
        'config': vars(args),
        'db_ids': db_ids,
        'union_schema_size': len(union_columns),
        'global_test_samples': X_global.shape[0],
        'global_test': {
            'solo': solo_metrics,
            'fedavg': fedavg_metrics,
            'fedgta': fedgta_metrics
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
