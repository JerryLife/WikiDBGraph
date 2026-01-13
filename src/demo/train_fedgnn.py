"""
Train FedGNN: Training script for Property-Aware GNN Federated Learning.

This script trains and evaluates the FedGNN model on horizontally-partitioned
WikiDB databases, comparing against Solo and FedAvg baselines.

Usage:
    python src/demo/train_fedgnn.py --db_ids 54379,37176,85770,50469
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

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.FedGNN import FedGNN, MaskedFedAvg, LocalModel
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
        
        # ---------------------------------------------------------------------
        # DATASET-SPECIFIC SETTINGS: Target Column Selection
        # 
        # "City vs Non-City" classification (Urban vs Rural Historic Places)
        # Label 1: "city" in value, Label 0: otherwise (town, village, county, etc.)
        # Priority: entitytype first (better balance ~30% city)
        # ---------------------------------------------------------------------
        target_candidates = ['entitytype', 'placetype', 'citytype']
        target_col = None
        target_idx = None
        
        # Iterate by priority order (not alphabetical column order)
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


def train_solo(
    client_data: Dict[str, Dict],
    input_dim: int,
    hidden_dim: int = 64,
    epochs: int = 50,
    lr: float = 0.01
) -> Dict[str, Dict]:
    """
    Train Solo models (no federation - each client trains independently).
    
    Returns:
        Dict mapping db_id -> {'accuracy': float, 'loss': float}
    """
    results = {}
    
    for db_id, data in client_data.items():
        model = LocalModel(input_dim, hidden_dim, output_dim=1)
        mask = torch.tensor(data['mask'], dtype=torch.float32)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.BCEWithLogitsLoss()
        
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(data['X_train'], mask)
            loss = criterion(pred.squeeze(), data['y_train'])
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            pred = model(data['X_test'], mask)
            preds = (torch.sigmoid(pred.squeeze()) > 0.5).float()
            accuracy = (preds == data['y_test']).float().mean().item()
            test_loss = criterion(pred.squeeze(), data['y_test']).item()
        
        results[db_id] = {'accuracy': accuracy, 'loss': test_loss}
    
    return results


def train_fedavg(
    client_data: Dict[str, Dict],
    input_dim: int,
    hidden_dim: int = 64,
    global_rounds: int = 10,
    local_epochs: int = 3,
    lr: float = 0.01
) -> Dict[str, Dict]:
    """
    Train with Masked FedAvg (coordinate-wise averaging).
    
    Returns:
        Dict mapping db_id -> {'accuracy': float, 'loss': float}
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
    
    # Evaluate
    results = {}
    for db_id, data in client_data.items():
        model = models[db_id]
        mask = masks[db_id]
        
        model.eval()
        with torch.no_grad():
            pred = model(data['X_test'], mask)
            preds = (torch.sigmoid(pred.squeeze()) > 0.5).float()
            accuracy = (preds == data['y_test']).float().mean().item()
            test_loss = criterion(pred.squeeze(), data['y_test']).item()
        
        results[db_id] = {'accuracy': accuracy, 'loss': test_loss}
    
    return results


def train_fedgnn(
    client_data: Dict[str, Dict],
    subgraph_data: Dict,
    input_dim: int,
    hidden_dim: int = 64,
    global_rounds: int = 10,
    local_epochs: int = 3,
    lr: float = 0.01,
    property_mode: str = "both"
) -> Dict[str, Dict]:
    """
    Train with FedGNN (Property-Aware GNN Aggregation).
    
    Returns:
        Dict mapping db_id -> {'accuracy': float, 'loss': float}
    """
    db_ids = list(client_data.keys())
    node_id_map = subgraph_data['node_id_map']
    
    # Build neighbor dict
    edges_src = subgraph_data['edges_src']
    edges_dst = subgraph_data['edges_dst']
    
    neighbors = {i: [] for i in range(len(db_ids))}
    for s, d in zip(edges_src, edges_dst):
        if d not in neighbors[s]:
            neighbors[s].append(d)
    
    # Map back to db_ids
    idx_to_db = {v: k for k, v in node_id_map.items()}
    neighbors_by_db = {}
    for db_id in db_ids:
        idx = node_id_map.get(db_id)
        if idx is not None:
            neighbors_by_db[db_id] = [idx_to_db[n] for n in neighbors.get(idx, [])]
        else:
            neighbors_by_db[db_id] = []
    
    # Build edge features dict
    edge_features = {}
    edge_props = subgraph_data.get('edge_props', {})
    if edge_props and 'similarity' in edge_props:
        for i, (s, d) in enumerate(zip(edges_src, edges_dst)):
            src_db = idx_to_db[s]
            dst_db = idx_to_db[d]
            sim = edge_props['similarity'][i]
            jac_col = edge_props.get('jaccard_columns', np.ones(len(edges_src)))[i]
            jac_tbl = edge_props.get('jaccard_table_names', np.ones(len(edges_src)))[i]
            edge_features[(src_db, dst_db)] = torch.tensor([sim, jac_col, jac_tbl], dtype=torch.float32)
    
    # Build node features dict
    node_features = {}
    node_props = subgraph_data.get('node_props', {})
    if node_props:
        prop_names = sorted(node_props.keys())
        for db_id in db_ids:
            idx = node_id_map.get(db_id)
            if idx is not None:
                feat = [node_props[p][idx] for p in prop_names]
                node_features[db_id] = torch.tensor(feat, dtype=torch.float32)
    
    # Initialize FedGNN
    edge_feat_dim = 3
    node_feat_dim = len(node_props) if node_props else 8
    
    fedgnn = FedGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=1,
        edge_feat_dim=edge_feat_dim,
        node_feat_dim=node_feat_dim,
        property_mode=property_mode
    )
    
    # Initialize clients
    feature_masks = {db_id: client_data[db_id]['mask'] for db_id in db_ids}
    fedgnn.initialize_clients(db_ids, feature_masks)
    fedgnn.set_graph_structure(neighbors_by_db, edge_features, node_features)
    
    # Training loop
    for round_idx in range(global_rounds):
        # Local updates
        for db_id in db_ids:
            train_data = (client_data[db_id]['X_train'], client_data[db_id]['y_train'])
            fedgnn.local_update(db_id, train_data, epochs=local_epochs, lr=lr)
        
        # GNN aggregation
        fedgnn.aggregate_round()
    
    # Evaluate
    results = {}
    for db_id in db_ids:
        test_data = (client_data[db_id]['X_test'], client_data[db_id]['y_test'])
        metrics = fedgnn.evaluate(db_id, test_data)
        results[db_id] = metrics
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train FedGNN model")
    parser.add_argument(
        "--db_ids",
        type=str,
        default="37176,85770,50469",  # 3-client subgraph with PlaceType columns
        help="Comma-separated database IDs"
    )
    parser.add_argument(
        "--property_mode",
        type=str,
        choices=["none", "edge_only", "node_only", "both"],
        default="both",
        help="Property mode for GNN aggregation"
    )
    parser.add_argument(
        "--global_rounds",
        type=int,
        default=10,
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
        default=64,
        help="Hidden layer dimension"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate"
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
        default="results/fedgnn",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Parse database IDs
    db_ids = [db_id.strip().zfill(5) for db_id in args.db_ids.split(",")]
    
    print("=" * 60)
    print("FedGNN Training Pipeline")
    print("=" * 60)
    print(f"Databases: {db_ids}")
    print(f"Property mode: {args.property_mode}")
    print(f"Global rounds: {args.global_rounds}")
    print(f"Local epochs: {args.local_epochs}")
    print()
    
    # Step 1: Column matching (union schema)
    print("Step 1: Matching columns across databases...")
    wk = WKDataset(schema_dir="data/schema", csv_base_dir="data/unzip")
    union_columns, omitted_columns, feature_masks = match_columns_across_databases(wk, db_ids)
    input_dim = len(union_columns) - 1  # Exclude target column
    print(f"  Union schema size: {len(union_columns)} columns")
    
    # Step 2: Load subgraph
    print("\nStep 2: Loading graph structure...")
    subgraph = WikiDBSubgraph()
    subgraph_data = subgraph.load_or_construct(db_ids)
    print(f"  Nodes: {subgraph_data['n_nodes']}, Edges: {subgraph_data['n_edges']}")
    
    # Step 3: Prepare client data
    print("\nStep 3: Preparing client data...")
    client_data = prepare_client_data(wk, db_ids, union_columns)
    
    if not client_data:
        print("Error: No client data loaded. Exiting.")
        return
    
    # Update input_dim based on actual feature count
    actual_input_dim = client_data[list(client_data.keys())[0]]['X_train'].shape[1]
    print(f"  Actual input dimension: {actual_input_dim}")
    
    # Step 4: Train all methods
    print("\n" + "=" * 60)
    print("Training Models...")
    print("=" * 60)
    
    # Solo
    print("\n>>> Training Solo (no federation)...")
    solo_results = train_solo(
        client_data, actual_input_dim, args.hidden_dim,
        epochs=args.global_rounds * args.local_epochs, lr=args.lr
    )
    
    # FedAvg
    print("\n>>> Training Masked FedAvg...")
    fedavg_results = train_fedavg(
        client_data, actual_input_dim, args.hidden_dim,
        args.global_rounds, args.local_epochs, args.lr
    )
    
    # FedGNN
    print(f"\n>>> Training FedGNN (mode: {args.property_mode})...")
    fedgnn_results = train_fedgnn(
        client_data, subgraph_data, actual_input_dim, args.hidden_dim,
        args.global_rounds, args.local_epochs, args.lr, args.property_mode
    )
    
    # Step 5: Report results
    print("\n" + "=" * 60)
    print("RESULTS: Per-Database Test Accuracy")
    print("=" * 60)
    print(f"{'Database':<12} {'Solo':<12} {'FedAvg':<12} {'FedGNN':<12}")
    print("-" * 48)
    
    all_results = {}
    for db_id in client_data.keys():
        solo_acc = solo_results.get(db_id, {}).get('accuracy', 0)
        fedavg_acc = fedavg_results.get(db_id, {}).get('accuracy', 0)
        fedgnn_acc = fedgnn_results.get(db_id, {}).get('accuracy', 0)
        
        print(f"DB{db_id:<10} {solo_acc:<12.4f} {fedavg_acc:<12.4f} {fedgnn_acc:<12.4f}")
        
        all_results[db_id] = {
            'solo': solo_acc,
            'fedavg': fedavg_acc,
            'fedgnn': fedgnn_acc
        }
    
    # Averages
    avg_solo = np.mean([r['solo'] for r in all_results.values()])
    avg_fedavg = np.mean([r['fedavg'] for r in all_results.values()])
    avg_fedgnn = np.mean([r['fedgnn'] for r in all_results.values()])
    
    print("-" * 48)
    print(f"{'Average':<12} {avg_solo:<12.4f} {avg_fedavg:<12.4f} {avg_fedgnn:<12.4f}")
    
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
        'n_edges': subgraph_data['n_edges'],
        'per_database': all_results,
        'averages': {
            'solo': avg_solo,
            'fedavg': avg_fedavg,
            'fedgnn': avg_fedgnn
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
