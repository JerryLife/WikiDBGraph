"""
Compare GNN vs FedAvg: Analysis and comparison script.

This script runs comprehensive experiments comparing Solo, FedAvg, and FedGNN
on horizontally-partitioned WikiDB databases with multiple seeds.

Usage:
    python src/analysis/compare_gnn_fedavg.py --db_ids 54379,37176,85770,50469 --seeds 0,1,2
"""

import os
import sys
import json
import argparse
from typing import Dict, List
from datetime import datetime

import numpy as np
import pandas as pd

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from train_fedgnn
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'demo'))


def run_experiment(
    db_ids: List[str],
    property_mode: str = "both",
    global_rounds: int = 10,
    local_epochs: int = 3,
    hidden_dim: int = 64,
    lr: float = 0.01,
    seed: int = 42
) -> Dict:
    """
    Run a single experiment with the given configuration.
    
    Returns:
        Dict with per-database results for all methods
    """
    import torch
    from model.FedGNN import FedGNN, MaskedFedAvg, LocalModel
    from model.WKDataset import WKDataset
    from analysis.WikiDBSubgraph import WikiDBSubgraph
    from analysis.semantic_column_matcher import match_columns_across_databases
    from demo.train_fedgnn import (
        prepare_client_data, train_solo, train_fedavg, train_fedgnn
    )
    
    # Set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Column matching
    wk = WKDataset(schema_dir="data/schema", csv_base_dir="data/unzip")
    union_columns, omitted_columns, feature_masks = match_columns_across_databases(wk, db_ids)
    
    # Load subgraph
    subgraph = WikiDBSubgraph()
    subgraph_data = subgraph.load_or_construct(db_ids)
    
    # Prepare client data
    client_data = prepare_client_data(wk, db_ids, union_columns, random_state=seed)
    
    if not client_data:
        return {'error': 'No client data loaded'}
    
    actual_input_dim = client_data[list(client_data.keys())[0]]['X_train'].shape[1]
    
    # Train all methods
    solo_results = train_solo(
        client_data, actual_input_dim, hidden_dim,
        epochs=global_rounds * local_epochs, lr=lr
    )
    
    fedavg_results = train_fedavg(
        client_data, actual_input_dim, hidden_dim,
        global_rounds, local_epochs, lr
    )
    
    fedgnn_results = train_fedgnn(
        client_data, subgraph_data, actual_input_dim, hidden_dim,
        global_rounds, local_epochs, lr, property_mode
    )
    
    # Compile results
    results = {
        'seed': seed,
        'per_database': {},
        'summary': {}
    }
    
    for db_id in client_data.keys():
        results['per_database'][db_id] = {
            'solo': solo_results.get(db_id, {}).get('accuracy', 0),
            'fedavg': fedavg_results.get(db_id, {}).get('accuracy', 0),
            'fedgnn': fedgnn_results.get(db_id, {}).get('accuracy', 0)
        }
    
    # Compute averages
    for method in ['solo', 'fedavg', 'fedgnn']:
        accs = [r[method] for r in results['per_database'].values()]
        results['summary'][method] = {
            'mean': np.mean(accs),
            'std': np.std(accs)
        }
    
    return results


def run_multi_seed_experiment(
    db_ids: List[str],
    seeds: List[int],
    property_mode: str = "both",
    global_rounds: int = 10,
    local_epochs: int = 3,
    hidden_dim: int = 64,
    lr: float = 0.01,
    quick_test: bool = False
) -> Dict:
    """
    Run experiments with multiple seeds for statistical significance.
    """
    if quick_test:
        global_rounds = 2
        local_epochs = 1
        seeds = seeds[:1]
    
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running experiment with seed={seed}")
        print('='*60)
        
        result = run_experiment(
            db_ids, property_mode, global_rounds, local_epochs,
            hidden_dim, lr, seed
        )
        all_results.append(result)
    
    # Aggregate across seeds
    aggregated = {
        'db_ids': db_ids,
        'seeds': seeds,
        'config': {
            'property_mode': property_mode,
            'global_rounds': global_rounds,
            'local_epochs': local_epochs,
            'hidden_dim': hidden_dim,
            'lr': lr
        },
        'per_seed': all_results,
        'final_summary': {}
    }
    
    # Compute cross-seed statistics
    for method in ['solo', 'fedavg', 'fedgnn']:
        means = [r['summary'][method]['mean'] for r in all_results if 'summary' in r]
        if means:
            aggregated['final_summary'][method] = {
                'mean': np.mean(means),
                'std': np.std(means),
                'min': np.min(means),
                'max': np.max(means)
            }
    
    return aggregated


def print_comparison_report(results: Dict):
    """Print a formatted comparison report."""
    print("\n" + "=" * 70)
    print("COMPARISON REPORT: Solo vs FedAvg vs FedGNN")
    print("=" * 70)
    
    config = results.get('config', {})
    print(f"\nConfiguration:")
    print(f"  Databases: {results.get('db_ids', [])}")
    print(f"  Seeds: {results.get('seeds', [])}")
    print(f"  Property mode: {config.get('property_mode', 'N/A')}")
    print(f"  Global rounds: {config.get('global_rounds', 'N/A')}")
    print(f"  Local epochs: {config.get('local_epochs', 'N/A')}")
    
    # Cross-seed summary
    summary = results.get('final_summary', {})
    if summary:
        print("\n" + "-" * 70)
        print("FINAL RESULTS (averaged across seeds):")
        print("-" * 70)
        print(f"{'Method':<15} {'Mean Acc':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
        print("-" * 63)
        
        for method in ['solo', 'fedavg', 'fedgnn']:
            if method in summary:
                s = summary[method]
                print(f"{method.upper():<15} {s['mean']:.4f}       {s['std']:.4f}       {s['min']:.4f}       {s['max']:.4f}")
    
    # Performance comparison
    if summary and 'fedavg' in summary and 'fedgnn' in summary:
        fedavg_mean = summary['fedavg']['mean']
        fedgnn_mean = summary['fedgnn']['mean']
        improvement = (fedgnn_mean - fedavg_mean) * 100
        
        print("\n" + "-" * 70)
        print("ANALYSIS:")
        print("-" * 70)
        
        if improvement > 0:
            print(f"  ✓ FedGNN outperforms FedAvg by {improvement:.2f} percentage points")
        elif improvement < 0:
            print(f"  ✗ FedAvg outperforms FedGNN by {-improvement:.2f} percentage points")
        else:
            print(f"  = FedGNN and FedAvg have similar performance")
        
        if 'solo' in summary:
            solo_mean = summary['solo']['mean']
            fed_improvement = (fedgnn_mean - solo_mean) * 100
            if fed_improvement > 0:
                print(f"  ✓ Federation (FedGNN) improves over Solo by {fed_improvement:.2f} pp")
            else:
                print(f"  ✗ Solo outperforms FedGNN by {-fed_improvement:.2f} pp")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Compare GNN vs FedAvg")
    parser.add_argument(
        "--db_ids",
        type=str,
        default="54379,37176,85770,50469",
        help="Comma-separated database IDs"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated random seeds"
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
        "--quick_test",
        action="store_true",
        help="Run quick test with reduced settings"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/gnn_vs_fedavg",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Parse inputs
    db_ids = [db_id.strip().zfill(5) for db_id in args.db_ids.split(",")]
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    
    print("=" * 70)
    print("GNN vs FedAvg Comparison Experiment")
    print("=" * 70)
    
    # Run experiments
    results = run_multi_seed_experiment(
        db_ids=db_ids,
        seeds=seeds,
        property_mode=args.property_mode,
        global_rounds=args.global_rounds,
        local_epochs=args.local_epochs,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        quick_test=args.quick_test
    )
    
    # Print report
    print_comparison_report(results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir,
        f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
