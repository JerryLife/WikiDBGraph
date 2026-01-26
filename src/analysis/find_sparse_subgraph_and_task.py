#!/usr/bin/env python3
"""
Find a sparse subgraph of WikiDBs, select a balanced label, and evaluate FedGNN.

This script loads a DGL graph, searches for a high-sparsity node subset (not
necessarily connected), prints the database list, selects a meaningful and
balanced label, and then runs Solo/FedAvg/FedGNN. If FedGNN does not outperform
both baselines on every database, it moves to the next candidate subset.

Key features:
- Dynamic label discovery: Scans ALL tables for suitable binary classification columns
- Candidate output: Saves subgraph candidates to temp file for manual review
- Database-specific stats: Each database gets its own column_stats file
"""

import argparse
import os
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import dgl
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

from model.WKDataset import WKDataset
from analysis.semantic_column_matcher import match_columns_across_databases
from analysis.WikiDBSubgraph import WikiDBSubgraph
from demo.train_fedgnn import prepare_client_data, train_fedavg, train_fedgnn, train_solo


@dataclass
class ColumnLabelInfo:
    """Information about a column that could serve as a label."""
    table_name: str
    column_name: str
    n_samples: int
    n_unique: int
    num_classes: int  # Number of classes (2 for binary, >2 for multi-class)
    class_distribution: Dict[str, int]
    balance_ratio: float  # min_class / max_class (only meaningful for binary)
    positive_class: str  # The class to use as positive (minority class for binary)


@dataclass
class DbLabelInfo:
    db_id: str
    table_name: str
    column_name: str
    n_samples: int
    pos_ratio: float
    feature_cols: int
    signal: float
    positive_class: str = ""
    num_classes: int = 2


@dataclass(frozen=True)
class SubgraphCandidate:
    node_ids: Tuple[int, ...]
    edges: int
    density: float
    sparsity: float


@dataclass
class CandidateWithLabels:
    """A subgraph candidate with its associated label information."""
    candidate: SubgraphCandidate
    db_ids: List[str]
    db_names: List[str]
    common_label_columns: List[Tuple[str, str]]  # List of (table_pattern, column_name) pairs
    label_details: Dict[str, List[ColumnLabelInfo]] = field(default_factory=dict)


def load_graph(graph_path: str) -> dgl.DGLGraph:
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph file not found: {graph_path}")
    graphs, _ = dgl.load_graphs(graph_path)
    g = graphs[0].to("cpu")
    return g


def load_schema_index(schema_dir: str) -> Dict[str, str]:
    mapping = {}
    for filename in os.listdir(schema_dir):
        if not filename.endswith(".json"):
            continue
        db_id, _, name_part = filename.partition("_")
        name = os.path.splitext(name_part)[0] if name_part else ""
        if db_id.isdigit():
            mapping[db_id.zfill(5)] = name
    return mapping


def normalize_col_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def discover_labels_in_db(
    wk: WKDataset,
    db_id: str,
    min_samples: int = 200,
    min_ratio: float = 0.2,
    max_ratio: float = 0.8,
    max_unique: int = 10,
) -> List[ColumnLabelInfo]:
    """
    Discover columns suitable for classification in ALL tables of a database.
    Supports both binary and multi-class labels.
    
    Args:
        wk: WKDataset instance
        db_id: Database ID
        min_samples: Minimum number of samples required
        min_ratio: Minimum class balance ratio (minority/majority) - for binary only
        max_ratio: Maximum positive class ratio - for binary only
        max_unique: Maximum unique values for a column to be considered
        
    Returns:
        List of ColumnLabelInfo for suitable columns
    """
    try:
        table_dfs = wk.load_csv_data(db_id, sample=False)
    except Exception:
        return []
    
    if not table_dfs:
        return []
    
    candidates = []
    
    for table_name, df in table_dfs.items():
        if len(df) < min_samples:
            continue
            
        for col in df.columns:
            series = df[col].dropna()
            if len(series) < min_samples:
                continue
            
            # Get unique values
            value_counts = series.astype(str).value_counts()
            n_unique = len(value_counts)
            
            # Skip columns with too many or too few unique values
            if n_unique < 2 or n_unique > max_unique:
                continue
            
            # Calculate balance for binary, or use num_classes for multi-class
            top_classes = value_counts.head(2)
            majority_count = top_classes.iloc[0]
            minority_count = top_classes.iloc[1] if len(top_classes) > 1 else 0
            
            # Calculate balance ratio (only meaningful for binary)
            balance_ratio = minority_count / majority_count if majority_count > 0 else 0
            
            # For binary (n_unique == 2), enforce balance constraints
            if n_unique == 2:
                if balance_ratio < min_ratio or balance_ratio > max_ratio:
                    continue
                total = len(series)
                pos_ratio = minority_count / total
                if pos_ratio < min_ratio or pos_ratio > (1 - min_ratio):
                    continue
            
            candidates.append(ColumnLabelInfo(
                table_name=table_name,
                column_name=col,
                n_samples=len(series),
                n_unique=n_unique,
                num_classes=n_unique,
                class_distribution=value_counts.to_dict(),
                balance_ratio=balance_ratio,
                positive_class=top_classes.index[1] if len(top_classes) > 1 else top_classes.index[0],
            ))
    
    # Sort by: fewer classes first (binary preferred), then by balance ratio
    candidates.sort(key=lambda x: (x.num_classes, -x.balance_ratio))
    
    return candidates


def save_db_column_stats(
    wk: WKDataset,
    db_id: str,
    output_dir: str,
) -> None:
    """Save column statistics for a specific database."""
    try:
        table_dfs = wk.load_csv_data(db_id, sample=False)
    except Exception:
        return
    
    if not table_dfs:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{db_id}_column_stats.txt")
    
    with open(output_path, "w") as f:
        f.write(f"Column Statistics for Database {db_id}\n")
        f.write("=" * 60 + "\n\n")
        
        for table_name, df in table_dfs.items():
            f.write(f"Table: {table_name} ({len(df)} rows)\n")
            f.write("-" * 40 + "\n")
            
            for col in df.columns:
                series = df[col].dropna()
                n_unique = series.nunique()
                f.write(f"  {col}: {n_unique} unique values, {len(series)} non-null\n")
                
                # Show top 5 values for low-cardinality columns
                if n_unique <= 10:
                    top_vals = series.astype(str).value_counts().head(5)
                    for val, count in top_vals.items():
                        f.write(f"    - {val}: {count} ({100*count/len(series):.1f}%)\n")
            
            f.write("\n")


def compute_degrees(g: dgl.DGLGraph, node_ids: Iterable[int]) -> Dict[int, int]:
    node_ids_tensor = torch.tensor(list(node_ids), dtype=torch.int64)
    in_deg = g.in_degrees(node_ids_tensor).cpu().numpy()
    out_deg = g.out_degrees(node_ids_tensor).cpu().numpy()
    degrees = {}
    for idx, node_id in enumerate(node_ids_tensor.tolist()):
        degrees[node_id] = int(in_deg[idx] + out_deg[idx])
    return degrees


def count_induced_edges(g: dgl.DGLGraph, nodes: Sequence[int]) -> int:
    """Count the number of edges in the induced subgraph formed by nodes."""
    nodes_tensor = torch.tensor(list(nodes), dtype=torch.int64)
    # dgl.node_subgraph returns a graph with consecutive IDs 0..k-1
    sg = dgl.node_subgraph(g, nodes_tensor)
    # DGL graphs of WikiDBs are usually bidirected, so we divide by 2
    return sg.num_edges() // 2


def candidate_subgraphs(
    g: dgl.DGLGraph,
    seed_nodes: Sequence[int],
    sizes: Sequence[int],
    samples_per_size: int,
    min_edges: int,
    rng: random.Random,
) -> List[SubgraphCandidate]:
    """
    Find candidate subgraphs that are connected.
    
    A "sparse but connected" subgraph has:
    - All nodes reachable from each other (connected component)
    - Lower density = sparser (fewer redundant edges)
    
    Uses a growth-based approach (randomly expanding from a seed) to guarantee connectivity.
    """
    unique = {}
    
    # We want to explore various sizes
    for target_size in sizes:
        for _ in range(samples_per_size):
            # Pick a random start node from our seeds
            start_node = rng.choice(seed_nodes)
            current_nodes = {start_node}
            
            # Maintain a frontier of potential nodes to add (neighbors of current set in FULL graph)
            # We use set(succ) | set(pred) to handle (potentially) directed graphs as undirected
            frontier = set(g.successors(start_node).tolist()) | set(g.predecessors(start_node).tolist())
            frontier.discard(start_node)
            
            # Grow the subgraph until target_size
            while len(current_nodes) < target_size and frontier:
                # Pick a random node from frontier to maintain connectivity
                next_node = rng.choice(list(frontier))
                current_nodes.add(next_node)
                
                # Update frontier with neighbors of the new node
                new_neighbors = set(g.successors(next_node).tolist()) | set(g.predecessors(next_node).tolist())
                frontier = (frontier | new_neighbors) - current_nodes
                
            if len(current_nodes) != target_size:
                continue
                
            subset = tuple(sorted(current_nodes))
            if subset in unique:
                continue
                
            # subset is guaranteed to be connected by construction
            edges = count_induced_edges(g, subset)
            if edges < min_edges:
                continue
                
            possible = target_size * (target_size - 1) // 2
            density = edges / possible if possible > 0 else 0.0
            sparsity = 1.0 - density
            unique[subset] = SubgraphCandidate(subset, edges, density, sparsity)

    # Sort by: lower density first (sparser), then more edges, then smaller size
    # Density 0.0 means it's a tree (minimal edges for connectivity)
    return sorted(unique.values(), key=lambda c: (c.density, -c.edges, len(c.node_ids)))


def find_common_label_columns(
    label_infos: Dict[str, List[ColumnLabelInfo]],
) -> List[Tuple[str, str]]:
    """
    Find columns that appear in all databases with similar characteristics.
    
    Returns list of (table_pattern, column_name) tuples.
    """
    if not label_infos:
        return []
    
    # Collect all column names per database (normalized)
    col_sets = []
    for db_id, infos in label_infos.items():
        cols = {normalize_col_name(info.column_name) for info in infos}
        col_sets.append(cols)
    
    if not col_sets:
        return []
    
    # Find intersection
    common = col_sets[0]
    for cols in col_sets[1:]:
        common = common & cols
    
    # Map back to original column names and tables
    results = []
    for norm_col in common:
        # Use the first database's info as reference
        first_db = next(iter(label_infos.values()))
        for info in first_db:
            if normalize_col_name(info.column_name) == norm_col:
                results.append((info.table_name, info.column_name))
                break
    
    return results


def save_candidates_to_file(
    candidates: List[CandidateWithLabels],
    schema_index: Dict[str, str],
    output_path: str,
) -> None:
    """Save candidate subgraphs to a file for manual review."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("Subgraph Candidates for FedGNN Evaluation\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, cand in enumerate(candidates, start=1):
            f.write(f"Candidate #{idx}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Size: {len(cand.db_ids)} databases\n")
            f.write(f"Edges: {cand.candidate.edges}\n")
            f.write(f"Sparsity: {cand.candidate.sparsity:.4f}\n")
            f.write("\nDatabases:\n")
            
            for db_id, db_name in zip(cand.db_ids, cand.db_names):
                f.write(f"  - {db_id}: {db_name}\n")
            
            f.write("\nCommon Label Columns:\n")
            if cand.common_label_columns:
                for table, col in cand.common_label_columns:
                    f.write(f"  - {table}.{col}\n")
            else:
                f.write("  (none found)\n")
            
            f.write("\nLabel Details per Database:\n")
            for db_id, infos in cand.label_details.items():
                f.write(f"  DB {db_id}:\n")
                for info in infos[:5]:  # Show top 5
                    f.write(f"    - {info.table_name}.{info.column_name}: "
                           f"n={info.n_samples}, "
                           f"unique={info.n_unique}, "
                           f"balance={info.balance_ratio:.3f}\n")
                    # Show class distribution
                    for cls, cnt in list(info.class_distribution.items())[:3]:
                        f.write(f"        {cls}: {cnt}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
        
        f.write(f"\nTotal candidates: {len(candidates)}\n")


def compute_feature_signal(
    df: "pd.DataFrame",
    label: np.ndarray,
    exclude_cols: Sequence[str],
    max_features: int,
    sample_size: int,
    rng: np.random.RandomState,
) -> Tuple[int, float]:
    if np.std(label) < 1e-6:
        return 0, 0.0

    exclude_norm = {normalize_col_name(col) for col in exclude_cols}
    usable_cols = []
    for col in df.columns:
        if normalize_col_name(col) in exclude_norm:
            continue
        series = df[col]
        if series.nunique(dropna=True) <= 1:
            continue
        usable_cols.append(col)

    if not usable_cols:
        return 0, 0.0

    if len(usable_cols) > max_features:
        usable_cols = rng.choice(usable_cols, size=max_features, replace=False).tolist()

    if len(df) > sample_size:
        sample_idx = rng.choice(len(df), size=sample_size, replace=False)
        df = df.iloc[sample_idx]
        label = label[sample_idx]

    best_signal = 0.0
    for col in usable_cols:
        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_ratio = numeric.notna().mean()

        if numeric_ratio >= 0.9:
            values = numeric.fillna(0).values.astype(np.float32)
            if np.std(values) > 1e-6:
                corr = float(np.corrcoef(values, label)[0, 1])
                best_signal = max(best_signal, abs(corr))
            continue

        le = LabelEncoder()
        encoded = le.fit_transform(series.astype(str).fillna("__MISSING__"))
        mi = mutual_info_classif(encoded.reshape(-1, 1), label, discrete_features=True)
        if len(mi):
            best_signal = max(best_signal, float(mi[0]))

    return len(usable_cols), best_signal


def evaluate_label_for_db_dynamic(
    wk: WKDataset,
    db_id: str,
    table_name: str,
    column_name: str,
    positive_class: str,
    min_samples: int,
    min_ratio: float,
    max_ratio: float,
    min_feature_cols: int,
    min_signal: float,
    max_features: int,
    signal_sample_size: int,
    rng: np.random.RandomState,
) -> Optional[DbLabelInfo]:
    """Evaluate a specific column as a label for a database."""
    try:
        table_dfs = wk.load_csv_data(db_id, sample=False)
    except Exception:
        return None
    
    if not table_dfs or table_name not in table_dfs:
        return None

    df = table_dfs[table_name]
    if column_name not in df.columns:
        return None

    series = df[column_name].fillna("").astype(str)
    label = (series == positive_class).astype(np.int32).values
    n_samples = len(label)
    if n_samples < min_samples:
        return None

    pos_ratio = float(label.mean()) if n_samples > 0 else 0.0
    if not (min_ratio <= pos_ratio <= max_ratio):
        return None

    exclude = {column_name}
    feature_cols, signal = compute_feature_signal(
        df=df,
        label=label,
        exclude_cols=exclude,
        max_features=max_features,
        sample_size=signal_sample_size,
        rng=rng,
    )

    if feature_cols < min_feature_cols or signal < min_signal:
        return None

    return DbLabelInfo(
        db_id=db_id,
        table_name=table_name,
        column_name=column_name,
        n_samples=n_samples,
        pos_ratio=pos_ratio,
        feature_cols=feature_cols,
        signal=signal,
        positive_class=positive_class,
    )


def print_db_list(db_ids: Sequence[str], name_map: Dict[str, str]) -> None:
    print("Databases in candidate subgraph:")
    for db_id in db_ids:
        name = name_map.get(db_id, "")
        suffix = f" ({name})" if name else ""
        print(f"  - {db_id}{suffix}")


def evaluate_fedgnn(
    db_ids: Sequence[str],
    property_mode: str,
    global_rounds: int,
    local_epochs: int,
    hidden_dim: int,
    lr: float,
) -> Optional[Dict[str, Dict[str, float]]]:
    wk = WKDataset(schema_dir="data/schema", csv_base_dir="data/unzip")
    union_columns, _, _ = match_columns_across_databases(wk, list(db_ids))
    client_data = prepare_client_data(wk, list(db_ids), union_columns)

    if len(client_data) < 2:
        return None

    actual_input_dim = next(iter(client_data.values()))["X_train"].shape[1]
    subgraph = WikiDBSubgraph()
    subgraph_data = subgraph.load_or_construct(list(db_ids))

    solo_results, _ = train_solo(
        client_data, actual_input_dim, hidden_dim,
        epochs=global_rounds * local_epochs, lr=lr
    )
    fedavg_results, _ = train_fedavg(
        client_data, actual_input_dim, hidden_dim,
        global_rounds, local_epochs, lr
    )
    fedgnn_results, _ = train_fedgnn(
        client_data, subgraph_data, actual_input_dim, hidden_dim,
        global_rounds, local_epochs, lr, property_mode
    )

    merged = {}
    for db_id in client_data.keys():
        merged[db_id] = {
            "solo": solo_results.get(db_id, {}).get("accuracy", 0.0),
            "fedavg": fedavg_results.get(db_id, {}).get("accuracy", 0.0),
            "fedgnn": fedgnn_results.get(db_id, {}).get("accuracy", 0.0),
        }
    return merged


def fedgnn_outperforms(results: Dict[str, Dict[str, float]]) -> bool:
    for metrics in results.values():
        if not (metrics["fedgnn"] > metrics["solo"] and metrics["fedgnn"] > metrics["fedavg"]):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Find sparse subgraph and evaluate FedGNN")
    parser.add_argument("--graph-path", default="data/graph/graph_raw_0.94.dgl")
    parser.add_argument("--schema-dir", default="data/schema")
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--max-size", type=int, default=20)
    parser.add_argument("--candidate-pool", type=int, default=500)
    parser.add_argument("--samples-per-size", type=int, default=200)
    parser.add_argument("--min-edges", type=int, default=2,
                       help="Minimum edges (for 3+ nodes, need at least 2 to be connected)")
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--min-ratio", type=float, default=0.2)
    parser.add_argument("--max-ratio", type=float, default=0.8)
    parser.add_argument("--min-feature-cols", type=int, default=3)
    parser.add_argument("--min-signal", type=float, default=0.01)
    parser.add_argument("--signal-sample-size", type=int, default=2000)
    parser.add_argument("--signal-max-features", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--property-mode", default="both")
    parser.add_argument("--global-rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dry-run", action="store_true", help="Skip training")
    
    # New arguments for scan-only mode
    parser.add_argument("--scan-only", action="store_true", 
                       help="Only scan and save candidates, don't train")
    parser.add_argument("--output-dir", default="data/analysis",
                       help="Output directory for candidate files")
    parser.add_argument("--max-candidates", type=int, default=50,
                       help="Maximum number of candidates to save")
    parser.add_argument("--save-column-stats", action="store_true",
                       help="Save per-database column statistics")

    args = parser.parse_args()
    rng = random.Random(args.seed)
    np_rng = np.random.RandomState(args.seed)

    g = load_graph(args.graph_path)
    schema_index = load_schema_index(args.schema_dir)

    print(f"Graph: {g.num_nodes()} nodes, {g.num_edges()} edges")
    print(f"Schema DBs: {len(schema_index)}")

    # Get all available nodes sorted by degree
    # For connected subgraphs, we need nodes that HAVE edges - prioritize moderate degree nodes
    available_nodes = sorted(int(db_id) for db_id in schema_index.keys())
    degrees = compute_degrees(g, available_nodes)
    
    # Filter nodes that have at least 1 edge and sort by degree (moderate degrees preferred)
    # Nodes with 0 edges can't form connected subgraphs
    nodes_with_edges = [n for n in available_nodes if degrees.get(n, 0) > 0]
    
    # Sort by degree - we want nodes that are connected but not hyper-connected
    # Take a range of degrees to ensure variety
    sorted_nodes = sorted(nodes_with_edges, key=lambda n: degrees.get(n, 0))
    
    # Use nodes with edges as candidates
    candidate_pool = sorted_nodes[:args.candidate_pool]
    
    avg_degree = np.mean([degrees.get(n, 0) for n in candidate_pool]) if candidate_pool else 0
    print(f"Candidate pool: {len(candidate_pool)} nodes (with edges, avg degree: {avg_degree:.1f})")

    sizes = list(range(args.min_size, args.max_size + 1))
    candidates = candidate_subgraphs(
        g=g,
        seed_nodes=candidate_pool,
        sizes=sizes,
        samples_per_size=args.samples_per_size,
        min_edges=args.min_edges,
        rng=rng,
    )

    if not candidates:
        if args.min_edges > 0:
            print("No candidates with min_edges > 0; retrying with min_edges=0.")
            candidates = candidate_subgraphs(
                g=g,
                seed_nodes=candidate_pool,
                sizes=sizes,
                samples_per_size=args.samples_per_size,
                min_edges=0,
                rng=rng,
            )
        if not candidates:
            print("No candidate subgraphs found with the given constraints.")
            return

    print(f"Found {len(candidates)} candidate subgraphs")
    
    wk = WKDataset(schema_dir=args.schema_dir, csv_base_dir="data/unzip")
    
    # Scan databases for label columns
    print("\nScanning databases for binary classification columns...")
    candidates_with_labels: List[CandidateWithLabels] = []
    
    for idx, candidate in enumerate(candidates[:args.max_candidates], start=1):
        db_ids = [str(node).zfill(5) for node in candidate.node_ids]
        db_names = [schema_index.get(db_id, "") for db_id in db_ids]
        
        # Discover labels for each database
        label_details: Dict[str, List[ColumnLabelInfo]] = {}
        for db_id in db_ids:
            labels = discover_labels_in_db(
                wk, db_id,
                min_samples=args.min_samples,
                min_ratio=args.min_ratio,
                max_ratio=args.max_ratio,
            )
            if labels:
                label_details[db_id] = labels
                
                # Save column stats if requested
                if args.save_column_stats:
                    save_db_column_stats(wk, db_id, args.output_dir)
        
        # Find common label columns
        common_cols = find_common_label_columns(label_details)
        
        if label_details:  # Only save if at least one DB has labels
            candidates_with_labels.append(CandidateWithLabels(
                candidate=candidate,
                db_ids=db_ids,
                db_names=db_names,
                common_label_columns=common_cols,
                label_details=label_details,
            ))
        
        if idx % 10 == 0:
            print(f"  Scanned {idx}/{min(len(candidates), args.max_candidates)} candidates...")
    
    print(f"Found {len(candidates_with_labels)} candidates with potential labels")
    
    # Save candidates to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"subgraph_candidates_{timestamp}.txt")
    
    os.makedirs(args.output_dir, exist_ok=True)
    save_candidates_to_file(candidates_with_labels, schema_index, output_path)
    print(f"\nCandidates saved to: {output_path}")
    
    if args.scan_only:
        print("\nScan-only mode: Skipping training.")
        print(f"Review candidates in {output_path} and run training manually.")
        return
    
    # Continue with training for candidates with common labels
    for cand in candidates_with_labels:
        if not cand.common_label_columns:
            continue
        
        print("\n" + "=" * 70)
        print(f"Evaluating: size={len(cand.db_ids)} edges={cand.candidate.edges} "
              f"sparsity={cand.candidate.sparsity:.4f}")
        print_db_list(cand.db_ids, schema_index)
        print(f"Common labels: {cand.common_label_columns}")

        if args.dry_run:
            print("Dry run enabled; skipping training.")
            return

        results = evaluate_fedgnn(
            db_ids=cand.db_ids,
            property_mode=args.property_mode,
            global_rounds=args.global_rounds,
            local_epochs=args.local_epochs,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
        )

        if results is None:
            print("Skipping: insufficient client data after preparation.")
            continue

        print("Results (accuracy):")
        for db_id, metrics in results.items():
            print(
                f"  DB{db_id} solo={metrics['solo']:.4f} "
                f"fedavg={metrics['fedavg']:.4f} fedgnn={metrics['fedgnn']:.4f}"
            )

        if fedgnn_outperforms(results):
            print("FedGNN outperforms Solo and FedAvg on all databases!")
            return

        print("FedGNN did not outperform both baselines; trying next candidate...")

    print("No candidate subset satisfied the FedGNN criterion.")


if __name__ == "__main__":
    main()
