#!/usr/bin/env python3
"""
Centralized Training Script for All ML Models

This script trains machine learning models on combined training data from multiple clients/databases.
It supports both horizontal and vertical federated learning scenarios.

For HFL: Combines training data from all clients (same features, different samples)
For VFL: Combines features from all databases (different features, same samples)

Usage:
    python run_centralized_training.py --scenario hfl --model neural_network --seed 0
    python run_centralized_training.py --scenario vfl --model xgboost --seed 1
    python run_centralized_training.py --scenario both --model all --seed 0
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

# Add src to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from centralized_training import train_centralized_baseline


def get_project_root():
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def load_data_for_centralized_training(scenario: str):
    """Load and preprocess data for centralized training (same as train_fedavg.py)."""
    print(f"\n--- Loading data for centralized training ({scenario}) ---")
    
    project_root = get_project_root()
    
    if scenario == "hfl":
        # Horizontal FL: Load data from both clients
        data_dir = project_root / "data" / "clean"
        df_a_train = pd.read_csv(data_dir / "02799" / "train_data.csv")
        df_a_test = pd.read_csv(data_dir / "02799" / "test_data.csv")
        df_b_train = pd.read_csv(data_dir / "79665" / "train_data.csv")
        df_b_test = pd.read_csv(data_dir / "79665" / "test_data.csv")
        
        # Combine training and test data from both clients
        combined_train = pd.concat([df_a_train, df_b_train], ignore_index=True)
        combined_test = pd.concat([df_a_test, df_b_test], ignore_index=True)
        
        database_ids = ["02799", "79665"]
        
    elif scenario == "vfl":
        # Vertical FL: Load data from both feature databases (aligned subdirectories)
        data_dir = project_root / "data" / "clean"
        df_a_train = pd.read_csv(data_dir / "48804" / "aligned" / "lpg_aligned_train.csv")
        df_a_test = pd.read_csv(data_dir / "48804" / "aligned" / "lpg_aligned_test.csv")
        df_b_train = pd.read_csv(data_dir / "00381" / "aligned" / "tc_aligned_train.csv")
        df_b_test = pd.read_csv(data_dir / "00381" / "aligned" / "tc_aligned_test.csv")
        
        # For VFL, merge on shared instances (combine features)
        # The target column is in df_a (48804) - lpg_Uni_Prot_Protein_Id
        # Remove Common_Gene_ID from df_b to avoid duplication, keep all other columns
        df_b_features_train = df_b_train.drop('Common_Gene_ID', axis=1)
        df_b_features_test = df_b_test.drop('Common_Gene_ID', axis=1)
        
        combined_train = pd.concat([df_a_train, df_b_features_train], axis=1)
        combined_test = pd.concat([df_a_test, df_b_features_test], axis=1)
        
        database_ids = ["48804", "00381"]
    
    return combined_train, combined_test, database_ids


def auto_detect_target_column(df: pd.DataFrame) -> str:
    """Auto-detect target column using FedAvg logic."""
    potential_targets = []
    
    for col in df.columns:
        if (col.lower() in ['label', 'target', 'class', 'y'] or 
            'protein' in col.lower() or 'gene' in col.lower() or
            'classification' in col.lower()):
            potential_targets.append(col)
    
    if potential_targets:
        target_col = potential_targets[0]
        print(f"Auto-detected target column: {target_col}")
    else:
        # Use the last column as target
        target_col = df.columns[-1]
        print(f"Using last column as target: {target_col}")
    
    return target_col

def create_unified_preprocessing(combined_train: pd.DataFrame, combined_test: pd.DataFrame, target_col: str = None):
    """Create unified preprocessing (same as FedAvg.py logic)."""
    print("Creating unified preprocessing...")
    
    # Auto-detect target column if not provided
    if target_col is None:
        target_col = auto_detect_target_column(combined_train)
    
    # Collect all unique labels to understand the data (same as FedAvg)
    all_labels = []
    for df in [combined_train, combined_test]:
        if target_col in df.columns:
            all_labels.extend(df[target_col].dropna().values.astype(str))
    
    unique_labels = sorted(set(all_labels))
    print(f"Found labels in data: {unique_labels}")
    
    # Set up label encoder (same as FedAvg)
    unified_encoder = LabelEncoder()
    unified_encoder.fit(unique_labels)
    
    num_classes = len(unique_labels)
    print(f"Created unified label encoder with {num_classes} classes")
    
    # Create unified StandardScaler from combined training data
    feature_cols = [col for col in combined_train.columns if col != target_col]
    unified_scaler = StandardScaler()
    unified_scaler.fit(combined_train[feature_cols].fillna(0))
    print(f"Created unified StandardScaler for {len(feature_cols)} features")
    
    return unified_encoder, unified_scaler, num_classes, feature_cols, target_col


def train_centralized_models(scenario: str, model: str, seed: int = 0):
    """Train centralized models for the specified scenario and model."""
    print(f"\n--- Training Centralized {model.upper()} for {scenario.upper()} (seed {seed}) ---")
    
    # Load and preprocess data
    combined_train, combined_test, database_ids = load_data_for_centralized_training(scenario)
    unified_encoder, unified_scaler, num_classes, feature_cols, target_col = create_unified_preprocessing(
        combined_train, combined_test
    )
    
    # Prepare data with unified preprocessing
    X_train = unified_scaler.transform(combined_train[feature_cols].fillna(0))
    X_test = unified_scaler.transform(combined_test[feature_cols].fillna(0))
    y_train = unified_encoder.transform(combined_train[target_col].astype(str))
    y_test = unified_encoder.transform(combined_test[target_col].astype(str))
    
    results = {}
    
    if model == "neural_network":
        # Use existing centralized training function
        nn_results = train_centralized_baseline(
            train_df=combined_train,
            test_df=combined_test,
            target_col=target_col,
            num_classes=num_classes,
            hidden_dims=[64, 32],
            learning_rate=1e-4,
            epochs=100,
            batch_size=32,
            device='cuda:1',
            unified_encoder=unified_encoder
        )
        results["Neural_Network"] = [nn_results["accuracy"], nn_results["precision"], nn_results["recall"], nn_results["f1"]]
        
    elif model == "xgboost":
        # Train XGBoost
        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(random_state=seed, device='cuda')
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results["XGBoost"] = [accuracy, precision, recall, f1]
        print(f"XGBoost F1: {f1:.4f}")
        
    elif model == "random_forest":
        # Train Random Forest
        print("Training Random Forest...")
        rf_model = RandomForestClassifier(random_state=seed)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results["Random_Forest"] = [accuracy, precision, recall, f1]
        print(f"Random Forest F1: {f1:.4f}")
        
    elif model == "logistic_regression":
        # Train Logistic Regression
        print("Training Logistic Regression...")
        lr_model = LogisticRegression(random_state=seed, max_iter=1000)
        lr_model.fit(X_train, y_train)
        y_pred = lr_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results["Logistic_Regression"] = [accuracy, precision, recall, f1]
        print(f"Logistic Regression F1: {f1:.4f}")
        
    elif model == "svm":
        # Train SVM
        print("Training SVM...")
        svm_model = SVC(random_state=seed, kernel='rbf')
        svm_model.fit(X_train, y_train)
        y_pred = svm_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results["SVM"] = [accuracy, precision, recall, f1]
        print(f"SVM F1: {f1:.4f}")
    
    # Save centralized results
    save_centralized_results(results, scenario, model, seed, database_ids)
    
    return results


def save_centralized_results(results: dict, scenario: str, model: str, seed: int, database_ids: list):
    """Save centralized training results to JSON file."""
    project_root = get_project_root()
    results_dir = project_root / "results"
    centralized_dir = results_dir / "centralized"
    centralized_dir.mkdir(parents=True, exist_ok=True)
    
    # Create result data structure
    result_data = {
        "experiment_type": "centralized",
        "scenario": scenario,
        "model": model,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "seed": seed,
        "database_ids": database_ids,
        "results": results
    }
    
    # Create filename with model abbreviation
    model_abbrev_map = {
        "neural_network": "nn",
        "xgboost": "xgb", 
        "random_forest": "rf",
        "logistic_regression": "lr",
        "svm": "svm"
    }
    model_abbrev = model_abbrev_map.get(model, model)
    db_string = "_".join(database_ids)
    filename = f"centralized_{model_abbrev}_{db_string}_seed{seed}.json"
    filepath = centralized_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"✅ Centralized {model} results saved: {filepath}")
    return str(filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Train centralized models for federated learning comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train centralized neural network for HFL
  python run_centralized_training.py --scenario hfl --model neural_network --seed 0
  
  # Train centralized XGBoost for VFL
  python run_centralized_training.py --scenario vfl --model xgboost --seed 1
  
  # Train all models for both scenarios
  python run_centralized_training.py --scenario both --model all --seed 0
        """
    )
    
    parser.add_argument("--scenario", choices=["hfl", "vfl", "both"], 
                       default="hfl", help="Scenario type (default: hfl)")
    parser.add_argument("--model", choices=["neural_network", "xgboost", "random_forest", "logistic_regression", "svm", "all"],
                       default="svm", help="Model type (default: svm)")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed (default: 0)")
    
    args = parser.parse_args()
    
    # Determine scenarios and models to process
    scenarios = ["hfl", "vfl"] if args.scenario == "both" else [args.scenario]
    models = ["neural_network", "xgboost", "random_forest", "logistic_regression", "svm"] if args.model == "all" else [args.model]
    
    for scenario in scenarios:
        print(f"\n{'='*80}")
        print(f"CENTRALIZED TRAINING FOR {scenario.upper()} SCENARIO")
        print(f"{'='*80}")
        
        for model in models:
            try:
                results = train_centralized_models(scenario, model, args.seed)
                print(f"✅ Successfully trained centralized {model} for {scenario}")
            except Exception as e:
                print(f"❌ Failed to train centralized {model} for {scenario}: {e}")
    
    print(f"\n{'='*80}")
    print("🎉 CENTRALIZED TRAINING COMPLETED!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()