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
import hashlib
import pickle
from typing import Dict, List, Tuple

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch.nn.functional as F

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.FedGNN import FedGNN, MaskedFedAvg, LocalModel
from model.WKDataset import WKDataset
from analysis.WikiDBSubgraph import WikiDBSubgraph
from analysis.semantic_column_matcher import match_columns_across_databases


def normalize_col_name(name: str) -> str:
    """Standard normalization for column names."""
    if not name:
        return ""
    # Remove table prefix if present, lowercase and strip
    name = name.split(".")[-1]
    return "".join(c for c in name.lower() if c.isalnum())


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
        # Keep only first match for each join key to prevent row explosion
        ref_df = ref_df.drop_duplicates(subset=[right_col], keep='first')
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
        # Safeguard: if join results in a massive explosion, cap it or stop
        if len(joined_df) > 100000:
            print(f"    Warning: Join with {ref_table} resulted in {len(joined_df)} rows. Capping to 100k.")
            joined_df = joined_df.head(100000)
            
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
    target_candidates: List[str] = None,
    positive_token: str = 'city',
    test_size: float = 0.2,
    random_state: int = 42,
    num_classes: int = 2,
    cache_dir: str = "data/cache/raw_data",
    drop_other: bool = False
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
    # Cache logic: Cache only the raw aligned data per DB (db_ids + union_columns)
    # This allows reusing the expensive join results across different experiments
    cache_params = {
        'db_ids': sorted(db_ids),
        'target_candidates': sorted(target_candidates) if target_candidates else None,
        'union_columns_hash': hashlib.md5(json.dumps(sorted(union_columns.items()), sort_keys=True).encode()).hexdigest()
    }
    param_str = json.dumps(cache_params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()[:12]
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"raw_data_{param_hash}.pkl")
    
    raw_target_data = None
    if os.path.exists(cache_path):
        print(f"  Loading cached raw data from {cache_path}")
        try:
            with open(cache_path, 'rb') as f:
                raw_target_data = pickle.load(f)
        except Exception as e:
            print(f"    Warning: Failed to load cache: {e}. Recomputing...")
            raw_target_data = None
    
    # If cache miss, load from database
    if raw_target_data is None:
        raw_target_data = {}
    
        for db_id in db_ids:
            print(f"  Loading data for DB{db_id}...")
            aligned_df, feature_mask = load_database_data(wk, db_id, union_columns, target_candidates=target_candidates)
            
            if aligned_df.empty:
                print(f"    Warning: No data for DB{db_id}, skipping")
                continue
            
            cols = aligned_df.columns.tolist()
            if len(cols) < 2:
                print(f"    Warning: Not enough columns for DB{db_id}, skipping")
                continue
            
            # Use provided target candidates or fall back to default
            if target_candidates is None:
                active_target_candidates = ['entitytype', 'placetype', 'citytype']
            else:
                active_target_candidates = target_candidates
                
            target_col = None
            target_idx = None
            
            # Iterate by priority order (not alphabetical column order)
            for candidate in active_target_candidates:
                # Try exact match first, then try normalized match
                for i, col in enumerate(cols):
                    if col.lower() == candidate.lower() or normalize_col_name(col) == normalize_col_name(candidate):
                        col_present = feature_mask[i] if i < len(feature_mask) else False
                        if col_present:
                            target_col = col
                            target_idx = i
                            break
                if target_col:
                    break
            
            if target_col is None:
                print(f"    Warning: No target column found for DB{db_id}. Skipping this database.")
                continue
            
            # Store raw target values and target_col for later encoding
            raw_target_data[db_id] = {
                'aligned_df': aligned_df,
                'feature_mask': feature_mask,
                'target_col': target_col,
                'y_raw': aligned_df[target_col].values
            }
        
        # Save raw data cache
        print(f"  Caching raw data to {cache_path}")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(raw_target_data, f)
        except Exception as e:
            print(f"    Warning: Failed to save cache: {e}")
    
    client_data = {}
    
    # Multi-class global label encoding
    if num_classes > 2:
        from collections import Counter
        all_labels = Counter()
        for db_id, data in raw_target_data.items():
            y_raw = data['y_raw']
            for val in y_raw:
                if pd.notna(val):
                    label = str(val).lower().strip()
                    all_labels[label] += 1

        if drop_other:
            # Use top K labels and drop everything else
            top_labels = [label for label, _ in all_labels.most_common(num_classes)]
            global_label_map = {label: i for i, label in enumerate(top_labels)}
            print(f"  Multi-Class: Top {num_classes} global labels: {top_labels}")
        else:
            # Reserve class 0 for "Other"
            top_labels = [label for label, _ in all_labels.most_common(num_classes - 1)]
            global_label_map = {label: i + 1 for i, label in enumerate(top_labels)}
            global_label_map['__OTHER__'] = 0  # "Other" class
            print(f"  Multi-Class: Top {num_classes-1} global labels: {top_labels}")
    else:
        global_label_map = None
    
    # Second pass: Process each DB with the global label map
    for db_id, data in raw_target_data.items():
        aligned_df = data['aligned_df']
        feature_mask = data['feature_mask']
        target_col = data['target_col']
        y_raw = data['y_raw']
        
        cols = aligned_df.columns.tolist()
        
        # Use provided target candidates or fall back to default
        if target_candidates is None:
            active_target_candidates = ['entitytype', 'placetype', 'citytype']
        else:
            active_target_candidates = target_candidates
        
        if num_classes == 2:
            # Binary labels based on positive_token
            def is_positive(val):
                if pd.isna(val): return False
                val_str = str(val).lower()
                token_str = str(positive_token).lower()
                return token_str in val_str or val_str in token_str

            y = np.array([1 if is_positive(val) else 0 for val in y_raw])
            
            # If all labels are the same (e.g. 0), we need some variation for training
            if len(np.unique(y)) < 2:
                print(f"    Warning: All labels same, adding variation")
                if len(y) > 1:
                    y[0] = 1 - y[0]
        else:
            # Multi-class encoding using global_label_map
            if drop_other:
                y = np.array([
                    global_label_map.get(str(val).lower().strip(), -1) if pd.notna(val) else -1
                    for val in y_raw
                ])
            else:
                y = np.array([
                    global_label_map.get(str(val).lower().strip(), 0) if pd.notna(val) else 0
                    for val in y_raw
                ])

            if drop_other:
                keep_mask = y >= 0
                if keep_mask.sum() == 0:
                    print(f"    Warning: No valid labels after drop_other for DB{db_id}, skipping")
                    continue
                aligned_df = aligned_df.loc[keep_mask].reset_index(drop=True)
                y = y[keep_mask]
        
        # LEAKAGE PREVENTION: Drop target candidates from features
        # and drop any columns that have zero variance
        X_df = aligned_df.copy()
        for candidate in active_target_candidates:
            norm_cand = normalize_col_name(candidate)
            cols_to_drop = [c for c in X_df.columns if normalize_col_name(c) == norm_cand]
            X_df = X_df.drop(columns=cols_to_drop, errors='ignore') # Use errors='ignore' if column not found
        
        # Also drop columns that have zero variance (after target columns are removed)
        # and convert categorical to numerical
        processed_X_list = []
        feature_mask_final = []
        original_cols_after_target_removal = [col for col in aligned_df.columns if col not in [c for cand in active_target_candidates for c in aligned_df.columns if normalize_col_name(c) == normalize_col_name(cand)]]

        for i, col in enumerate(original_cols_after_target_removal):
            col_data = aligned_df[col] # Use original aligned_df for feature mask
            
            # Check if this column was part of the original feature mask
            original_col_idx = aligned_df.columns.get_loc(col)
            is_present_in_db = feature_mask[original_col_idx]

            if not is_present_in_db:
                # If column was not present in the DB, it's all zeros, so skip or add zeros
                processed_X_list.append(np.zeros(len(aligned_df), dtype=np.float32))
                feature_mask_final.append(False)
                continue

            # Try to convert to numeric
            numeric_data = pd.to_numeric(col_data, errors='coerce')
            
            if numeric_data.isna().sum() / len(numeric_data) > 0.5:
                # More than 50% failed to convert - treat as categorical
                le = LabelEncoder()
                try:
                    encoded = le.fit_transform(col_data.astype(str).fillna('__MISSING__'))
                    processed_X_list.append(encoded.astype(np.float32))
                except Exception:
                    processed_X_list.append(np.zeros(len(col_data), dtype=np.float32))
            else:
                processed_X_list.append(numeric_data.fillna(0).values.astype(np.float32))
            feature_mask_final.append(True) # Mark as present and processed

        X = np.column_stack(processed_X_list) if processed_X_list else np.zeros((len(aligned_df), 0))

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
        
        
    return client_data


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray, num_classes: int) -> float:
    """Compute AUC with safe fallbacks for small/degenerate label sets."""
    try:
        if num_classes == 2:
            if len(np.unique(y_true)) < 2:
                return None
            return roc_auc_score(y_true, y_score)
        if len(np.unique(y_true)) < 2:
            return None
        return roc_auc_score(y_true, y_score, multi_class='ovr', average='macro')
    except Exception:
        return None


def _compute_metrics_and_preds(
    logits: torch.Tensor,
    y_true: torch.Tensor,
    num_classes: int
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """Compute metrics and return predictions + scores."""
    y_true_np = y_true.detach().cpu().numpy()

    if num_classes == 2:
        probs = torch.sigmoid(logits.squeeze()).detach().cpu().numpy()
        y_pred = (probs > 0.5).astype(int)
        auc = _safe_auc(y_true_np, probs, num_classes)
        metrics = {
            'accuracy': accuracy_score(y_true_np, y_pred),
            'f1': f1_score(y_true_np, y_pred),
            'auc': auc
        }
        return metrics, y_pred, probs

    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    y_pred = np.argmax(probs, axis=1)
    auc = _safe_auc(y_true_np, probs, num_classes)
    metrics = {
        'accuracy': accuracy_score(y_true_np, y_pred),
        'f1': f1_score(y_true_np, y_pred, average='macro'),
        'auc': auc
    }
    return metrics, y_pred, probs


def _save_predictions_csv(
    output_path: str,
    db_ids: List[str],
    preds_by_db: Dict[str, Dict[str, np.ndarray]],
    num_classes: int
):
    """Save predictions to a CSV file."""
    rows = []
    for db_id in db_ids:
        preds = preds_by_db.get(db_id)
        if not preds:
            continue
        y_true = preds['y_true']
        y_pred = preds['y_pred']
        y_score = preds['y_score']
        for idx, (yt, yp) in enumerate(zip(y_true, y_pred)):
            row = {
                'db_id': db_id,
                'index': idx,
                'y_true': int(yt),
                'y_pred': int(yp)
            }
            if num_classes == 2:
                row['prob_pos'] = float(y_score[idx])
            else:
                for c in range(num_classes):
                    row[f'prob_{c}'] = float(y_score[idx][c])
            rows.append(row)
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)


def _build_run_tag(args: argparse.Namespace, seed: int) -> str:
    """Build a filename-safe tag for results/predictions."""
    lr_tag = f"{args.lr:g}".replace(".", "p")
    drop_tag = "drop1" if args.drop_other else "drop0"
    return (
        f"seed{seed}_r{args.global_rounds}_e{args.local_epochs}_"
        f"h{args.hidden_dim}_lr{lr_tag}_c{args.num_classes}_"
        f"pm{args.property_mode}_{drop_tag}"
    )


def _aggregate_metric(values: List[float]) -> float:
    valid = [v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None


def train_solo(
    client_data: Dict[str, Dict],
    input_dim: int,
    hidden_dim: int = 64,
    epochs: int = 50,
    lr: float = 0.01,
    device: str = 'cpu',
    use_batchnorm: bool = False,
    num_classes: int = 2
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Train Solo models (no federation - each client trains independently).
    
    Returns:
        Tuple: (metrics_by_db, preds_by_db)
    """
    results = {}
    preds_by_db = {}
    output_dim = 1 if num_classes == 2 else num_classes
    
    for db_id, data in client_data.items():
        model = LocalModel(input_dim, hidden_dim, output_dim=output_dim, use_batchnorm=use_batchnorm).to(device)
        mask = torch.tensor(data['mask'], dtype=torch.float32).to(device)
        
        X_train = data['X_train'].to(device)
        y_train = data['y_train'].to(device)
        X_test = data['X_test'].to(device)
        y_test = data['y_test'].to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        if num_classes == 2:
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            criterion = torch.nn.CrossEntropyLoss()
        
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            pred = model(X_train, mask)
            if num_classes == 2:
                loss = criterion(pred.squeeze(), y_train.float())
            else:
                loss = criterion(pred, y_train.long())
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            pred = model(X_test, mask)
            if num_classes == 2:
                test_loss = criterion(pred.squeeze(), y_test.float()).item()
            else:
                test_loss = criterion(pred, y_test.long()).item()
            metrics, y_pred, y_score = _compute_metrics_and_preds(pred, y_test, num_classes)
        
        metrics['loss'] = test_loss
        results[db_id] = metrics
        preds_by_db[db_id] = {
            'y_true': y_test.detach().cpu().numpy(),
            'y_pred': y_pred,
            'y_score': y_score
        }
    
    return results, preds_by_db


def train_fedavg(
    client_data: Dict[str, Dict],
    input_dim: int,
    hidden_dim: int = 64,
    global_rounds: int = 10,
    local_epochs: int = 3,
    lr: float = 0.01,
    device: str = 'cpu',
    use_batchnorm: bool = False,
    num_classes: int = 2
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Train with Masked FedAvg (coordinate-wise averaging).
    
    Returns:
        Tuple: (metrics_by_db, preds_by_db)
    """
    # Initialize models
    models = {}
    masks = {}
    output_dim = 1 if num_classes == 2 else num_classes
    
    init_model = LocalModel(input_dim, hidden_dim, output_dim=output_dim, use_batchnorm=use_batchnorm).to(device)
    init_state = init_model.state_dict()
    
    for db_id, data in client_data.items():
        model = LocalModel(input_dim, hidden_dim, output_dim=output_dim, use_batchnorm=use_batchnorm).to(device)
        model.load_state_dict(init_state.copy())
        models[db_id] = model
        masks[db_id] = torch.tensor(data['mask'], dtype=torch.float32).to(device)
    
    if num_classes == 2:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    for round_idx in range(global_rounds):
        # Local training
        for db_id, data in client_data.items():
            model = models[db_id].to(device)
            mask = masks[db_id].to(device)
            X_train = data['X_train'].to(device)
            y_train = data['y_train'].to(device)
            
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            
            model.train()
            for _ in range(local_epochs):
                optimizer.zero_grad()
                pred = model(X_train, mask)
                if num_classes == 2:
                    loss = criterion(pred.squeeze(), y_train.float())
                else:
                    loss = criterion(pred, y_train.long())
                loss.backward()
                optimizer.step()
        
        # Masked FedAvg aggregation
        global_state = MaskedFedAvg.aggregate(models, masks)
        
        # Distribute global model
        for db_id in models:
            models[db_id].load_state_dict(global_state)
    
    # Evaluate
    results = {}
    preds_by_db = {}
    for db_id, data in client_data.items():
        model = models[db_id]
        mask = masks[db_id]
        X_test = data['X_test'].to(device)
        y_test = data['y_test'].to(device)
        
        model.eval()
        with torch.no_grad():
            pred = model(X_test, mask)
            if num_classes == 2:
                test_loss = criterion(pred.squeeze(), y_test.float()).item()
            else:
                test_loss = criterion(pred, y_test.long()).item()
            metrics, y_pred, y_score = _compute_metrics_and_preds(pred, y_test, num_classes)
        
        metrics['loss'] = test_loss
        results[db_id] = metrics
        preds_by_db[db_id] = {
            'y_true': y_test.detach().cpu().numpy(),
            'y_pred': y_pred,
            'y_score': y_score
        }
    
    return results, preds_by_db


def train_fedgnn(
    client_data: Dict[str, Dict],
    subgraph_data: Dict,
    input_dim: int,
    hidden_dim: int = 64,
    global_rounds: int = 10,
    local_epochs: int = 3,
    lr: float = 0.01,
    property_mode: str = "both",
    target_candidates: List[str] = None,
    positive_token: str = 'city',
    device: str = 'cpu',
    use_batchnorm: bool = False,
    num_classes: int = 2
) -> Tuple[Dict[str, Dict], Dict[str, Dict]]:
    """
    Train with FedGNN (Property-Aware GNN Aggregation).
    
    Returns:
        Tuple: (metrics_by_db, preds_by_db)
    """
    db_ids = list(client_data.keys())
    node_id_map = subgraph_data['node_id_map']
    
    # Build neighbor dict
    edges_src = subgraph_data['edges_src']
    edges_dst = subgraph_data['edges_dst']
    
    neighbors = {idx: [] for idx in node_id_map.values()}
    for s, d in zip(edges_src, edges_dst):
        if d not in neighbors[s]:
            neighbors[s].append(d)
    
    # Map back to db_ids (filtering to only those in the current set)
    idx_to_db = {v: k for k, v in node_id_map.items()}
    neighbors_by_db = {}
    db_id_set = set(db_ids)
    for db_id in db_ids:
        idx = node_id_map.get(db_id)
        if idx is not None:
            # Only include neighbors that are also in our current db_ids list
            all_neighbors = [idx_to_db[n] for n in neighbors.get(idx, [])]
            neighbors_by_db[db_id] = [n for n in all_neighbors if n in db_id_set]
        else:
            neighbors_by_db[db_id] = []
    
    # Build edge features dict with Normalization
    edge_features = {}
    edge_props = subgraph_data.get('edge_props', {})
    
    raw_edge_feats = []
    edge_keys = []

    if edge_props and 'similarity' in edge_props:
        for i, (s, d) in enumerate(zip(edges_src, edges_dst)):
            src_db = idx_to_db[s]
            dst_db = idx_to_db[d]
            
            # Only process edges relevant to the current training set
            if src_db in db_id_set and dst_db in db_id_set:
                sim = edge_props['similarity'][i]
                jac_col = edge_props.get('jaccard_columns', np.ones(len(edges_src)))[i]
                jac_tbl = edge_props.get('jaccard_table_names', np.ones(len(edges_src)))[i]
                
                raw_edge_feats.append([sim, jac_col, jac_tbl])
                edge_keys.append((src_db, dst_db))
    
    # Apply Normalization to edge features
    if raw_edge_feats:
        print(f"  Normalizing {len(raw_edge_feats)} edges...")
        edge_scaler = StandardScaler()
        norm_edge_feats = edge_scaler.fit_transform(raw_edge_feats)
        norm_edge_feats = np.nan_to_num(norm_edge_feats, nan=0.0, posinf=0.0, neginf=0.0)
        
        for k, feat_arr in zip(edge_keys, norm_edge_feats):
            edge_features[k] = torch.tensor(feat_arr, dtype=torch.float32).to(device)
    else:
        print("  Warning: No edges found between selected databases.")
    
    # Build node features dict
    node_features = {}
    node_props = subgraph_data.get('node_props', {})
    if node_props:
        prop_names = sorted(node_props.keys())
        # First, collect raw features into a matrix for normalization
        raw_feats = []
        db_ids_with_feats = []
        for db_id in db_ids:
            idx = node_id_map.get(db_id)
            if idx is not None:
                feat = [node_props[p][idx] for p in prop_names]
                raw_feats.append(feat)
                db_ids_with_feats.append(db_id)
        
        # Apply StandardScaler to normalize node features
        if raw_feats:
            raw_feats_np = np.array(raw_feats, dtype=np.float32)
            scaler = StandardScaler()
            scaled_feats = scaler.fit_transform(raw_feats_np)
            scaled_feats = np.nan_to_num(scaled_feats, nan=0.0, posinf=0.0, neginf=0.0)
            
            for i, db_id in enumerate(db_ids_with_feats):
                node_features[db_id] = torch.tensor(scaled_feats[i], dtype=torch.float32).to(device)
    
    # Initialize FedGNN
    edge_feat_dim = 3
    node_feat_dim = len(node_props) if node_props else 8
    output_dim = 1 if num_classes == 2 else num_classes
    
    fedgnn = FedGNN(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        edge_feat_dim=edge_feat_dim,
        node_feat_dim=node_feat_dim,
        property_mode=property_mode,
        device=device,
        use_batchnorm=use_batchnorm
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
    preds_by_db = {}
    for db_id in db_ids:
        model = fedgnn.client_models[db_id]
        mask = fedgnn.client_masks[db_id]
        X_test = client_data[db_id]['X_test'].to(device)
        y_test = client_data[db_id]['y_test'].to(device)

        model.eval()
        with torch.no_grad():
            logits = model(X_test, mask)
            if num_classes == 2:
                test_loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(), y_test.float()
                ).item()
            else:
                test_loss = F.cross_entropy(logits, y_test.long()).item()
            metrics, y_pred, y_score = _compute_metrics_and_preds(logits, y_test, num_classes)

        metrics['loss'] = test_loss
        results[db_id] = metrics
        preds_by_db[db_id] = {
            'y_true': y_test.detach().cpu().numpy(),
            'y_pred': y_pred,
            'y_score': y_score
        }
    
    return results, preds_by_db


def main():
    parser = argparse.ArgumentParser(description="Train FedGNN model")
    parser.add_argument(
        "--db_ids",
        type=str,
        default="01318,15832,26192,34036,52953,67222,79114,84208",  # Candidate #26 (Figure Skating)
        help="Comma-separated database IDs"
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="RailwayTrafficSide",
        help="Target column name(s), comma-separated"
    )
    parser.add_argument(
        "--positive_token",
        type=str,
        default='right',
        help="Token in target column that represents the positive class"
    )
    parser.add_argument(
        "--property_mode",
        type=str,
        choices=["none", "edge_only", "node_only", "both", "all"],
        default="all",
        help="Property mode for GNN aggregation ('all' runs ablation with all modes)"
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
        default=2,
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
        "--output_dir",
        type=str,
        default="results/fedgnn",
        help="Output directory for results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="results/cache/raw_data",
        help="Cache directory for raw aligned data"
    )
    parser.add_argument(
        "--subgraph_cache_dir",
        type=str,
        default="results/subgraph_cache",
        help="Cache directory for constructed subgraphs"
    )
    
    parser.add_argument(
        "--use_batchnorm",
        action="store_false",
        dest="use_batchnorm",
        help="Disable BatchNorm in models (default is enabled)"
    )
    parser.set_defaults(use_batchnorm=True)
    
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes for classification (2 for binary, >2 for multi-class)"
    )
    parser.add_argument(
        "--drop_other",
        action="store_true",
        help="For multi-class, drop labels outside the top-K instead of using an '__OTHER__' class"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="accuracy,f1,auc",
        help="Comma-separated metrics to report (accuracy,f1,auc)"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated random seeds to run"
    )
    
    args = parser.parse_args()
    # Parse database IDs
    db_ids = [db_id.strip().zfill(5) for db_id in args.db_ids.split(",") if db_id.strip()]
    if not db_ids:
        print("Error: No database IDs provided. Exiting.")
        return

    # Parse metrics to report
    requested_metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    allowed_metrics = {"accuracy", "f1", "auc", "loss"}
    metrics_to_report = [m for m in requested_metrics if m in allowed_metrics]
    if not metrics_to_report:
        metrics_to_report = ["accuracy", "f1", "auc"]

    # Parse seeds
    try:
        seeds = sorted({int(s.strip()) for s in args.seeds.split(",") if s.strip()})
    except ValueError:
        print(f"Error: Invalid --seeds value: {args.seeds}")
        return
    if not seeds:
        seeds = [0, 1, 2, 3, 4]

    print("=" * 60)
    print("FedGNN Training Pipeline")
    print("=" * 60)
    print(f"Databases: {db_ids}")
    print(f"Property mode: {args.property_mode}")
    print(f"Global rounds: {args.global_rounds}")
    print(f"Local epochs: {args.local_epochs}")
    print(f"Seeds: {seeds}")
    print(f"Metrics: {metrics_to_report}")
    print()

    # Step 1: Column matching (union schema)
    print("Step 1: Matching columns across databases...")
    wk = WKDataset(schema_dir="data/schema", csv_base_dir="data/unzip")
    union_columns, omitted_columns, feature_masks = match_columns_across_databases(wk, db_ids)
    print(f"  Union schema size: {len(union_columns)} columns")

    # Step 2: Load subgraph
    print("\nStep 2: Loading graph structure...")
    subgraph = WikiDBSubgraph(cache_dir=args.subgraph_cache_dir)
    subgraph_data = subgraph.load_or_construct(db_ids)
    print(f"  Nodes: {subgraph_data['n_nodes']}, Edges: {subgraph_data['n_edges']}")

    # Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Output directory per dataset set
    dataset_tag = "-".join(db_ids)
    output_root = os.path.join(args.output_dir, dataset_tag)
    os.makedirs(output_root, exist_ok=True)

    def format_metric(value: float) -> str:
        return "NA" if value is None else f"{value:.4f}"

    property_modes = ["both", "node_only", "edge_only", "none"] if args.property_mode == "all" else [args.property_mode]
    target_cands = args.target_col.split(",") if args.target_col else None
    all_metrics = ["accuracy", "f1", "auc", "loss"]

    for seed in seeds:
        print("\n" + "=" * 70)
        print(f"Running seed {seed}")
        print("=" * 70)

        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Step 3: Prepare client data
        print("\nStep 3: Preparing client data...")
        client_data = prepare_client_data(
            wk, db_ids, union_columns,
            target_candidates=target_cands,
            positive_token=args.positive_token,
            num_classes=args.num_classes,
            cache_dir=args.cache_dir,
            drop_other=args.drop_other,
            random_state=seed
        )

        if not client_data:
            print(f"Error: No client data loaded for seed {seed}. Skipping.")
            continue

        # Update input_dim based on actual feature count
        actual_input_dim = client_data[list(client_data.keys())[0]]['X_train'].shape[1]
        print(f"  Actual input dimension: {actual_input_dim}")

        # Step 4: Train all methods
        print("\n" + "=" * 60)
        print("Training Models...")
        print("=" * 60)

        print("\n>>> Training Solo (no federation)...")
        solo_results, solo_preds = train_solo(
            client_data, actual_input_dim, args.hidden_dim,
            epochs=args.global_rounds * args.local_epochs, lr=args.lr, device=device,
            use_batchnorm=args.use_batchnorm, num_classes=args.num_classes
        )

        print("\n>>> Training Masked FedAvg...")
        fedavg_results, fedavg_preds = train_fedavg(
            client_data, actual_input_dim, args.hidden_dim,
            args.global_rounds, args.local_epochs, args.lr, device=device,
            use_batchnorm=args.use_batchnorm, num_classes=args.num_classes
        )

        fedgnn_results_by_mode = {}
        fedgnn_preds_by_mode = {}
        for mode in property_modes:
            print(f"\n>>> Training FedGNN (mode: {mode})...")
            fedgnn_results, fedgnn_preds = train_fedgnn(
                client_data, subgraph_data, actual_input_dim, args.hidden_dim,
                args.global_rounds, args.local_epochs, args.lr, mode,
                target_candidates=target_cands,
                positive_token=args.positive_token,
                device=device,
                use_batchnorm=args.use_batchnorm,
                num_classes=args.num_classes
            )
            fedgnn_results_by_mode[mode] = fedgnn_results
            fedgnn_preds_by_mode[mode] = fedgnn_preds

        methods_order = ["solo", "fedavg"] + [f"fedgnn_{mode}" for mode in property_modes]
        method_metrics = {
            "solo": solo_results,
            "fedavg": fedavg_results,
        }
        method_preds = {
            "solo": solo_preds,
            "fedavg": fedavg_preds,
        }
        for mode in property_modes:
            method_metrics[f"fedgnn_{mode}"] = fedgnn_results_by_mode[mode]
            method_preds[f"fedgnn_{mode}"] = fedgnn_preds_by_mode[mode]

        # Step 5: Report results
        print("\n" + "=" * 80)
        for metric in metrics_to_report:
            print(f"RESULTS: Per-Database Test {metric.upper()}")
            header = f"{'Database':<12}"
            for method in methods_order:
                header += f" {method:<12}"
            print(header)
            print("-" * (12 + 13 * len(methods_order)))

            for db_id in client_data.keys():
                row = f"DB{db_id:<10}"
                for method in methods_order:
                    value = method_metrics.get(method, {}).get(db_id, {}).get(metric)
                    row += f" {format_metric(value):<12}"
                print(row)

            averages = {}
            avg_row = f"{'Average':<12}"
            for method in methods_order:
                values = [method_metrics.get(method, {}).get(db_id, {}).get(metric) for db_id in client_data.keys()]
                avg_value = _aggregate_metric(values)
                averages.setdefault(method, {})
                averages[method][metric] = avg_value
                avg_row += f" {format_metric(avg_value):<12}"
            print("-" * (12 + 13 * len(methods_order)))
            print(avg_row)
            print("-" * 80)

        # Save predictions and metrics
        run_tag = _build_run_tag(args, seed)
        metrics_rows = []
        predictions_paths = {}
        for method in methods_order:
            preds_path = os.path.join(output_root, f"predictions_{method}_{run_tag}.csv")
            _save_predictions_csv(preds_path, list(client_data.keys()), method_preds.get(method, {}), args.num_classes)
            predictions_paths[method] = preds_path

            for db_id in client_data.keys():
                metrics_entry = {
                    'db_id': db_id,
                    'method': method
                }
                for metric_name in all_metrics:
                    metrics_entry[metric_name] = method_metrics.get(method, {}).get(db_id, {}).get(metric_name)
                metrics_rows.append(metrics_entry)

        metrics_csv = os.path.join(output_root, f"metrics_{run_tag}.csv")
        if metrics_rows:
            pd.DataFrame(metrics_rows).to_csv(metrics_csv, index=False)

        # Save results JSON (no timestamp)
        output_file = os.path.join(output_root, f"results_{run_tag}.json")
        output = {
            'config': vars(args),
            'seed': seed,
            'db_ids': db_ids,
            'union_schema_size': len(union_columns),
            'n_edges': subgraph_data['n_edges'],
            'property_modes': property_modes,
            'metrics_reported': metrics_to_report,
            'per_database': {
                db_id: {
                    method: method_metrics.get(method, {}).get(db_id, {})
                    for method in methods_order
                }
                for db_id in client_data.keys()
            },
            'averages': {
                method: {
                    metric: _aggregate_metric(
                        [method_metrics.get(method, {}).get(db_id, {}).get(metric) for db_id in client_data.keys()]
                    )
                    for metric in all_metrics
                }
                for method in methods_order
            },
            'predictions': predictions_paths,
            'metrics_csv': metrics_csv
        }

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {output_file}")
        print(f"Metrics CSV saved to: {metrics_csv}")


if __name__ == "__main__":
    main()
