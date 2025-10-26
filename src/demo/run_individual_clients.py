import pandas as pd
import os
import sys
import numpy as np
import argparse
import json
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Add src to path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.FedAvg import FedAvg

"""
Individual Client Training and Evaluation Script

This script trains different machine learning models on each client's data individually
and evaluates them in two ways:

1. Individual Evaluation: Each model is tested on the same client's test data
2. Combined Evaluation: All trained models are tested on a combined test dataset
   containing data from all clients

The combined evaluation helps understand:
- How well models trained on individual clients generalize to the broader population
- Which client's data produces the most generalizable models
- Generalization gap between individual and combined performance

Usage:
    python run_individual_clients.py -m nn,xgb,rf,lr
    python run_individual_clients.py --skip-combined  # Skip combined evaluation
    python run_individual_clients.py -m nn,xgb        # Run only specific models
"""


def load_horizontal_data(base_output_dir="data/clean"):
    """
    Load horizontal federated learning data.
    Returns data dictionaries for training and testing.
    """
    # Get absolute paths
    project_root = get_project_root()
    if not os.path.isabs(base_output_dir):
        base_output_dir = os.path.join(project_root, base_output_dir)
    
    # Client A data (DB 02799)
    path_a_train = os.path.join(base_output_dir, "02799", "train_data.csv")
    path_a_test = os.path.join(base_output_dir, "02799", "test_data.csv")

    # Client B data (DB 79665)  
    path_b_train = os.path.join(base_output_dir, "79665", "train_data.csv")
    path_b_test = os.path.join(base_output_dir, "79665", "test_data.csv")

    print(f"Loading data for client A (02799) from: {path_a_train}, {path_a_test}")
    print(f"Loading data for client B (79665) from: {path_b_train}, {path_b_test}")

    # Check if files exist
    required_files = [path_a_train, path_a_test, path_b_train, path_b_test]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("Error: Missing required files:")
        for f in missing_files:
            print(f"  - {f}")
        print("Please run the horizontal data preparation script first.")
        return None

    # Load data
    df_a_train = pd.read_csv(path_a_train)
    df_a_test = pd.read_csv(path_a_test)
    df_b_train = pd.read_csv(path_b_train)
    df_b_test = pd.read_csv(path_b_test)

    print(f"Shape of client A train (02799): {df_a_train.shape}")
    print(f"Shape of client B train (79665): {df_b_train.shape}")

    return {
        "client_02799": {"train": df_a_train, "test": df_a_test},
        "client_79665": {"train": df_b_train, "test": df_b_test}
    }


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics without AUC."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return accuracy, precision, recall, f1


def train_neural_network(client_name, train_df, test_df, target_col, num_classes,
                        hidden_dims=[128, 128, 32], learning_rate=0.0001, 
                        local_epochs=5, global_rounds=20, batch_size=32, device='cpu',
                        unified_encoder=None):
    """Train Neural Network using FedAvg architecture."""
    result = train_neural_network_with_model(client_name, train_df, test_df, target_col, num_classes,
                                            hidden_dims, learning_rate, local_epochs, global_rounds,
                                            batch_size, device, unified_encoder)
    return result[0] if result else None


def train_neural_network_with_model(client_name, train_df, test_df, target_col, num_classes,
                                   hidden_dims=[128, 128, 32], learning_rate=0.0001, 
                                   local_epochs=5, global_rounds=20, batch_size=32, device='cpu',
                                   unified_encoder=None, unified_scaler=None):
    """Train Neural Network using FedAvg architecture and return both results and model."""
    print(f"\n--- Training Neural Network for {client_name} ---")
    
    # Use unified StandardScaler (consistent with federated learning scripts)
    feature_cols = [col for col in train_df.columns if col != target_col]
    
    if unified_scaler is None:
        raise ValueError("unified_scaler must be provided")
    
    train_features = train_df[feature_cols].fillna(0)
    test_features = test_df[feature_cols].fillna(0)
    
    # Apply unified standardization to both training and test data
    train_features_scaled = unified_scaler.transform(train_features)
    test_features_scaled = unified_scaler.transform(test_features)
    
    # Create standardized DataFrames - preserve target column exactly
    train_df_scaled = pd.DataFrame(train_features_scaled, columns=feature_cols, index=train_df.index)
    train_df_scaled[target_col] = train_df[target_col].copy()  # Preserve original labels
    
    test_df_scaled = pd.DataFrame(test_features_scaled, columns=feature_cols, index=test_df.index)
    test_df_scaled[target_col] = test_df[target_col].copy()  # Preserve original labels
    
    print(f"  Applied unified standardization to {len(feature_cols)} features")
    
    # Use unified label encoder for consistent encoding across all clients and models
    if unified_encoder is None:
        raise ValueError("unified_encoder must be provided")
    
    # Encode labels using the unified encoder
    train_df_scaled[target_col] = unified_encoder.transform(train_df_scaled[target_col].astype(str))
    test_df_scaled[target_col] = unified_encoder.transform(test_df_scaled[target_col].astype(str))
    
    print(f"  Applied unified label encoding")
    
    # Create single-client data dictionary with standardized data
    train_data = {client_name: train_df_scaled}
    test_data = {client_name: test_df_scaled}
    
    # Initialize model
    model = FedAvg(
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        global_rounds=global_rounds,
        batch_size=batch_size,
        device=device
    )
    
    # Pre-set the label encoder to prevent FedAvg from creating its own
    model.label_encoder = unified_encoder
    
    try:
        model.fit(train_data, target_col=target_col)
        results = model.eval(test_data, target_col=target_col)
        
        if client_name in results:
            accuracy, precision, recall, f1 = results[client_name]
            model_info = {
                'model': model,
                'scaler': unified_scaler,
                'feature_cols': feature_cols,
                'target_col': target_col
            }
            return (accuracy, precision, recall, f1), model_info
        else:
            return None
    except Exception as e:
        print(f"Neural Network training failed: {e}")
        return None


def train_xgboost(client_name, train_df, test_df, target_col, num_classes, unified_encoder):
    """Train XGBoost classifier."""
    result = train_xgboost_with_model(client_name, train_df, test_df, target_col, num_classes, unified_encoder, None)
    return result[0] if result else None


def train_xgboost_with_model(client_name, train_df, test_df, target_col, num_classes, unified_encoder, unified_scaler):
    """Train XGBoost classifier and return both results and model."""
    print(f"\n--- Training XGBoost for {client_name} ---")
    
    try:
        # Prepare data with optional unified scaling
        X_train_raw = train_df.drop(target_col, axis=1).fillna(0)
        X_test_raw = test_df.drop(target_col, axis=1).fillna(0)
        
        if unified_scaler is not None:
            # Use unified scaling (preferred for consistency)
            X_train = unified_scaler.transform(X_train_raw)
            X_test = unified_scaler.transform(X_test_raw)
        else:
            # Fallback to no scaling (XGBoost can handle raw features)
            X_train = X_train_raw.values
            X_test = X_test_raw.values
        
        y_train = train_df[target_col].dropna()
        y_test = test_df[target_col].dropna()
        
        # Use unified label encoder to get labels in the global space
        y_train_encoded = unified_encoder.transform(y_train.astype(str))
        y_test_encoded = unified_encoder.transform(y_test.astype(str))
        
        # --- START: NEW LABEL RE-MAPPING LOGIC ---
        
        # 1. Get the unique classes actually present in the local training data
        local_train_classes = np.unique(y_train_encoded)
        
        # 2. Create a mapping from the global label to a new, local, contiguous label
        #    e.g., global [0, 2, 3, 4] -> local [0, 1, 2, 3]
        global_to_local_map = {global_label: local_label for local_label, global_label in enumerate(local_train_classes)}
        
        # 3. Create the inverse mapping to convert predictions back to the global space
        local_to_global_map = {local_label: global_label for global_label, local_label in global_to_local_map.items()}
        
        # 4. Apply the mapping to the local training labels
        y_train_local_encoded = np.array([global_to_local_map[global_label] for global_label in y_train_encoded])
        
        # --- END: NEW LABEL RE-MAPPING LOGIC ---

        # Inform the user about any classes in the test set that were not seen during local training
        train_classes_set = set(y_train_encoded)
        test_classes_set = set(y_test_encoded)
        unseen_in_train = test_classes_set - train_classes_set
        if unseen_in_train:
            print(f"  Warning: Test data contains classes not seen in this client's training data: {sorted(list(unseen_in_train))}")
            print(f"  Model was trained on classes: {sorted(list(train_classes_set))}")

        # Train model using the LOCAL number of classes and LOCAL labels
        local_num_classes = len(local_train_classes)
        
        if local_num_classes <= 1:
            print("  Skipping training: Client has only one class in its training data.")
            return None, None # Cannot train a classifier on one class

        model = xgb.XGBClassifier(
            objective='multi:softmax',
            # Use the number of locally present classes
            num_class=local_num_classes,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            device='cuda'
        )
        
        # Fit the model on the re-mapped local labels
        model.fit(X_train, y_train_local_encoded)
        
        # Predict on the test set. Predictions will be in the LOCAL label space.
        y_pred_local = model.predict(X_test)
        
        # Map the local predictions back to the GLOBAL label space for evaluation
        y_pred_global = np.array([local_to_global_map.get(p, -1) for p in y_pred_local]) # Use -1 for safety, though it shouldn't happen
        
        model_info = {
            'model': model,
            'target_col': target_col,
            # Store maps for potential future use or debugging
            'local_to_global_map': local_to_global_map,
            'global_to_local_map': global_to_local_map,
            'trained_classes': sorted(list(train_classes_set))
        }
        
        # Calculate metrics using the global predictions and global true labels
        return calculate_metrics(y_test_encoded, y_pred_global), model_info
        
    except Exception as e:
        print(f"XGBoost training failed: {e}")
        return None, None


def train_random_forest(client_name, train_df, test_df, target_col, num_classes, unified_encoder):
    """Train Random Forest classifier."""
    result = train_random_forest_with_model(client_name, train_df, test_df, target_col, num_classes, unified_encoder, None)
    return result[0] if result else None


def train_random_forest_with_model(client_name, train_df, test_df, target_col, num_classes, unified_encoder, unified_scaler):
    """Train Random Forest classifier and return both results and model."""
    print(f"\n--- Training Random Forest for {client_name} ---")
    
    try:
        # Prepare data with optional unified scaling
        X_train_raw = train_df.drop(target_col, axis=1).fillna(0)
        X_test_raw = test_df.drop(target_col, axis=1).fillna(0)
        
        if unified_scaler is not None:
            # Use unified scaling (preferred for consistency)
            X_train = unified_scaler.transform(X_train_raw)
            X_test = unified_scaler.transform(X_test_raw)
        else:
            # Fallback to no scaling (Random Forest can handle raw features)
            X_train = X_train_raw.values
            X_test = X_test_raw.values
        
        y_train = train_df[target_col].dropna()
        y_test = test_df[target_col].dropna()
        
        # Use unified label encoder
        y_train_encoded = unified_encoder.transform(y_train.astype(str))
        y_test_encoded = unified_encoder.transform(y_test.astype(str))
        
        # Check for class distribution mismatch
        train_classes = set(y_train_encoded)
        test_classes = set(y_test_encoded)
        unseen_classes = test_classes - train_classes
        
        if unseen_classes:
            print(f"  Warning: Test data contains unseen classes: {sorted(list(unseen_classes))}")
            print(f"  Unseen classes will be treated as incorrect predictions")
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train_encoded)
        
        # Handle predictions for samples with unseen classes
        if unseen_classes:
            # Separate test samples into seen and unseen classes
            seen_mask = np.isin(y_test_encoded, list(train_classes))
            unseen_mask = ~seen_mask
            
            # Initialize predictions array
            y_pred = np.full(len(y_test_encoded), -1)
            
            # Predict only for samples with seen classes
            if np.any(seen_mask):
                X_test_seen = X_test[seen_mask]
                y_pred_seen = model.predict(X_test_seen)
                y_pred[seen_mask] = y_pred_seen
            
            # For unseen classes, predict the most common class from training
            if np.any(unseen_mask):
                most_common_class = np.bincount(y_train_encoded).argmax()
                y_pred[unseen_mask] = most_common_class
                print(f"  Predicted {np.sum(unseen_mask)} samples with unseen classes as class {most_common_class}")
        else:
            # Normal prediction when all test classes were seen in training
            y_pred = model.predict(X_test)
        
        model_info = {
            'model': model,
            'target_col': target_col,
            'trained_classes': sorted(list(train_classes))
        }
        
        return calculate_metrics(y_test_encoded, y_pred), model_info
        
    except Exception as e:
        print(f"Random Forest training failed: {e}")
        return None


def train_logistic_regression(client_name, train_df, test_df, target_col, num_classes, unified_encoder):
    """Train Logistic Regression classifier."""
    result = train_logistic_regression_with_model(client_name, train_df, test_df, target_col, num_classes, unified_encoder, None)
    return result[0] if result else None


def train_logistic_regression_with_model(client_name, train_df, test_df, target_col, num_classes, unified_encoder, unified_scaler):
    """Train Logistic Regression classifier and return both results and model."""
    print(f"\n--- Training Logistic Regression for {client_name} ---")
    
    try:
        # Prepare data with optional unified scaling
        X_train = train_df.drop(target_col, axis=1).fillna(0)
        y_train = train_df[target_col].dropna()
        X_test = test_df.drop(target_col, axis=1).fillna(0)
        y_test = test_df[target_col].dropna()
        
        # Apply scaling (required for Logistic Regression)
        if unified_scaler is not None:
            # Use unified scaling (preferred for consistency)
            X_train_scaled = unified_scaler.transform(X_train)
            X_test_scaled = unified_scaler.transform(X_test)
        else:
            # Fallback to local scaling (not recommended but necessary for compatibility)
            local_scaler = StandardScaler()
            X_train_scaled = local_scaler.fit_transform(X_train)
            X_test_scaled = local_scaler.transform(X_test)
        
        # Use unified label encoder
        y_train_encoded = unified_encoder.transform(y_train.astype(str))
        y_test_encoded = unified_encoder.transform(y_test.astype(str))
        
        # Check for class distribution mismatch
        train_classes = set(y_train_encoded)
        test_classes = set(y_test_encoded)
        unseen_classes = test_classes - train_classes
        
        if unseen_classes:
            print(f"  Warning: Test data contains unseen classes: {sorted(list(unseen_classes))}")
            print(f"  Unseen classes will be treated as incorrect predictions")
        
        # Train model
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='ovr' if num_classes > 2 else 'auto'
        )
        
        model.fit(X_train_scaled, y_train_encoded)
        
        # Handle predictions for samples with unseen classes
        if unseen_classes:
            # Separate test samples into seen and unseen classes
            seen_mask = np.isin(y_test_encoded, list(train_classes))
            unseen_mask = ~seen_mask
            
            # Initialize predictions array
            y_pred = np.full(len(y_test_encoded), -1)
            
            # Predict only for samples with seen classes
            if np.any(seen_mask):
                X_test_seen = X_test_scaled[seen_mask]
                y_pred_seen = model.predict(X_test_seen)
                y_pred[seen_mask] = y_pred_seen
            
            # For unseen classes, predict the most common class from training
            if np.any(unseen_mask):
                most_common_class = np.bincount(y_train_encoded).argmax()
                y_pred[unseen_mask] = most_common_class
                print(f"  Predicted {np.sum(unseen_mask)} samples with unseen classes as class {most_common_class}")
        else:
            # Normal prediction when all test classes were seen in training
            y_pred = model.predict(X_test_scaled)
        
        model_info = {
            'model': model,
            'scaler': unified_scaler if unified_scaler is not None else local_scaler,
            'target_col': target_col,
            'trained_classes': sorted(list(train_classes))
        }
        
        return calculate_metrics(y_test_encoded, y_pred), model_info
        
    except Exception as e:
        print(f"Logistic Regression training failed: {e}")
        return None, None


def train_svm_with_model(client_name, train_df, test_df, target_col, num_classes, unified_encoder, unified_scaler):
    """Train SVM classifier and return both results and model."""
    print(f"\n--- Training SVM for {client_name} ---")
    
    try:
        # Prepare data with optional unified scaling
        X_train = train_df.drop(target_col, axis=1).fillna(0)
        y_train = train_df[target_col].dropna()
        X_test = test_df.drop(target_col, axis=1).fillna(0)
        y_test = test_df[target_col].dropna()
        
        # Apply scaling (required for SVM)
        if unified_scaler is not None:
            # Use unified scaling (preferred for consistency)
            X_train_scaled = unified_scaler.transform(X_train)
            X_test_scaled = unified_scaler.transform(X_test)
        else:
            # Fallback to local scaling (not recommended but necessary for compatibility)
            local_scaler = StandardScaler()
            X_train_scaled = local_scaler.fit_transform(X_train)
            X_test_scaled = local_scaler.transform(X_test)
        
        # Use unified label encoder
        y_train_encoded = unified_encoder.transform(y_train.astype(str))
        y_test_encoded = unified_encoder.transform(y_test.astype(str))
        
        # Check for class distribution mismatch
        train_classes = set(y_train_encoded)
        test_classes = set(y_test_encoded)
        unseen_classes = test_classes - train_classes
        
        if unseen_classes:
            print(f"  Warning: Test data contains unseen classes: {sorted(list(unseen_classes))}")
            print(f"  Unseen classes will be treated as incorrect predictions")
        
        # Train model
        model = SVC(
            kernel='rbf',
            random_state=42,
            probability=True  # Enable probability estimates for better compatibility
        )
        
        model.fit(X_train_scaled, y_train_encoded)
        
        # Handle predictions for samples with unseen classes
        if unseen_classes:
            # Separate test samples into seen and unseen classes
            seen_mask = np.isin(y_test_encoded, list(train_classes))
            unseen_mask = ~seen_mask
            
            # Initialize predictions array
            y_pred = np.full(len(y_test_encoded), -1)
            
            # Predict only for samples with seen classes
            if np.any(seen_mask):
                X_test_seen = X_test_scaled[seen_mask]
                y_pred_seen = model.predict(X_test_seen)
                y_pred[seen_mask] = y_pred_seen
            
            # For unseen classes, predict the most common class from training
            if np.any(unseen_mask):
                most_common_class = np.bincount(y_train_encoded).argmax()
                y_pred[unseen_mask] = most_common_class
                print(f"  Predicted {np.sum(unseen_mask)} samples with unseen classes as class {most_common_class}")
        else:
            # Normal prediction when all test classes were seen in training
            y_pred = model.predict(X_test_scaled)
        
        model_info = {
            'model': model,
            'scaler': unified_scaler if unified_scaler is not None else local_scaler,
            'target_col': target_col,
            'trained_classes': sorted(list(train_classes))
        }
        
        return calculate_metrics(y_test_encoded, y_pred), model_info
        
    except Exception as e:
        print(f"SVM training failed: {e}")
        return None, None


def test_models_on_client(client_name, train_df, test_df, target_col, num_classes, 
                         experiment_params, selected_models, unified_encoder, unified_scaler):
    """Test selected models on a single client."""
    print(f"\n{'='*80}")
    print(f"Testing Selected Models on Client: {client_name}")
    print(f"Selected models: {', '.join(selected_models)}")
    print(f"{'='*80}")
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Unique classes in train: {train_df[target_col].nunique()}")
    print(f"Unique classes in test: {test_df[target_col].nunique()}")
    
    results = {}
    trained_models = {}  # Store trained models for combined test evaluation
    
    # Neural Network
    if 'neural_network' in selected_models or 'nn' in selected_models:
        nn_result, nn_model = train_neural_network_with_model(
            client_name, train_df, test_df, target_col, num_classes,
            hidden_dims=experiment_params['hidden_dims'],
            learning_rate=experiment_params['learning_rate'],
            local_epochs=experiment_params['local_epochs'],
            global_rounds=experiment_params['global_rounds'],
            batch_size=experiment_params['batch_size'],
            device=experiment_params.get('device', 'cpu'),
            unified_encoder=unified_encoder,
            unified_scaler=unified_scaler
        )
        if nn_result:
            results['Neural_Network'] = nn_result
            trained_models['Neural_Network'] = nn_model
    
    # XGBoost
    if 'xgboost' in selected_models or 'xgb' in selected_models:
        xgb_result, xgb_model = train_xgboost_with_model(client_name, train_df, test_df, target_col, num_classes, unified_encoder, unified_scaler)
        if xgb_result:
            results['XGBoost'] = xgb_result
            trained_models['XGBoost'] = xgb_model
    
    # Random Forest
    if 'random_forest' in selected_models or 'rf' in selected_models:
        rf_result, rf_model = train_random_forest_with_model(client_name, train_df, test_df, target_col, num_classes, unified_encoder, unified_scaler)
        if rf_result:
            results['Random_Forest'] = rf_result
            trained_models['Random_Forest'] = rf_model
    
    # Logistic Regression
    if 'logistic_regression' in selected_models or 'lr' in selected_models:
        lr_result, lr_model = train_logistic_regression_with_model(client_name, train_df, test_df, target_col, num_classes, unified_encoder, unified_scaler)
        if lr_result:
            results['Logistic_Regression'] = lr_result
            trained_models['Logistic_Regression'] = lr_model
    
    # SVM
    if 'svm' in selected_models:
        svm_result, svm_model = train_svm_with_model(client_name, train_df, test_df, target_col, num_classes, unified_encoder, unified_scaler)
        if svm_result:
            results['SVM'] = svm_result
            trained_models['SVM'] = svm_model
    
    return results, trained_models


def evaluate_model_on_combined_data(model_info, combined_test_df, target_col, unified_encoder):
    """Evaluate a trained model on combined test data."""
    try:
        model_type = None
        if 'model' in model_info:
            model = model_info['model']
            
            # Determine model type and prepare data accordingly
            if hasattr(model, 'predict') and hasattr(model, 'eval'):  # Neural Network (FedAvg)
                model_type = 'neural_network'
                
                # Apply the same preprocessing as used during training
                if 'scaler' in model_info and 'feature_cols' in model_info:
                    scaler = model_info['scaler']
                    feature_cols = model_info['feature_cols']
                    
                    # Prepare features with same preprocessing
                    test_features = combined_test_df[feature_cols].fillna(0)
                    test_features_scaled = scaler.transform(test_features)
                    
                    # Create scaled DataFrame
                    test_df_scaled = pd.DataFrame(test_features_scaled, columns=feature_cols, index=combined_test_df.index)
                    test_df_scaled[target_col] = unified_encoder.transform(combined_test_df[target_col].astype(str))
                    
                    # Evaluate using the model's eval method
                    test_data = {"combined": test_df_scaled}
                    results = model.eval(test_data, target_col=target_col)
                    
                    if "combined" in results:
                        return results["combined"]
                        
            elif hasattr(model, 'predict') and hasattr(model, 'fit'):  # Sklearn-style models
                # Prepare features
                X_test = combined_test_df.drop(target_col, axis=1).fillna(0)
                y_test = unified_encoder.transform(combined_test_df[target_col].astype(str))
                
                # Apply scaling if model uses it
                if 'scaler' in model_info:
                    scaler = model_info['scaler']
                    X_test = scaler.transform(X_test)
                
                # Handle XGBoost models with local-to-global mapping
                if 'local_to_global_map' in model_info and 'global_to_local_map' in model_info:
                    # This is an XGBoost model with local label mapping
                    trained_classes = set(model_info['trained_classes'])
                    test_classes = set(y_test)
                    unseen_classes = test_classes - trained_classes
                    
                    if unseen_classes:
                        print(f"    Combined evaluation: unseen classes {sorted(list(unseen_classes))} found")
                        
                        # Separate test samples into seen and unseen classes
                        seen_mask = np.isin(y_test, list(trained_classes))
                        unseen_mask = ~seen_mask
                        
                        # Initialize predictions array
                        y_pred = np.full(len(y_test), -1)
                        
                        # Predict only for samples with seen classes
                        if np.any(seen_mask):
                            X_test_seen = X_test[seen_mask] if hasattr(X_test, '__getitem__') else X_test.iloc[seen_mask]
                            
                            # Convert global test labels to local for prediction
                            y_test_seen_global = y_test[seen_mask]
                            global_to_local_map = model_info['global_to_local_map']
                            local_to_global_map = model_info['local_to_global_map']
                            
                            # Predict in local space
                            y_pred_local = model.predict(X_test_seen)
                            
                            # Map back to global space
                            y_pred_seen_global = np.array([local_to_global_map.get(p, -1) for p in y_pred_local])
                            y_pred[seen_mask] = y_pred_seen_global
                        
                        # For unseen classes, predict most common training class (will be wrong)
                        if np.any(unseen_mask):
                            most_common_class = min(trained_classes)  # Use a consistent fallback
                            y_pred[unseen_mask] = most_common_class
                            print(f"    Predicted {np.sum(unseen_mask)} samples with unseen classes as class {most_common_class}")
                    else:
                        # Normal prediction: predict in local space then map to global
                        y_pred_local = model.predict(X_test)
                        local_to_global_map = model_info['local_to_global_map']
                        y_pred = np.array([local_to_global_map.get(p, -1) for p in y_pred_local])
                
                # Handle other models with potential unseen classes  
                elif 'trained_classes' in model_info:
                    # This model tracks which classes it was trained on
                    trained_classes = set(model_info['trained_classes'])
                    test_classes = set(y_test)
                    unseen_classes = test_classes - trained_classes
                    
                    if unseen_classes:
                        print(f"    Combined evaluation: unseen classes {sorted(list(unseen_classes))} found")
                        
                        # Separate test samples into seen and unseen classes
                        seen_mask = np.isin(y_test, list(trained_classes))
                        unseen_mask = ~seen_mask
                        
                        # Initialize predictions array
                        y_pred = np.full(len(y_test), -1)
                        
                        # Predict only for samples with seen classes
                        if np.any(seen_mask):
                            X_test_seen = X_test[seen_mask] if hasattr(X_test, '__getitem__') else X_test.iloc[seen_mask]
                            y_pred_seen = model.predict(X_test_seen)
                            y_pred[seen_mask] = y_pred_seen
                        
                        # For unseen classes, predict most common training class (will be wrong)
                        if np.any(unseen_mask):
                            most_common_class = min(trained_classes)  # Use a consistent fallback
                            y_pred[unseen_mask] = most_common_class
                            print(f"    Predicted {np.sum(unseen_mask)} samples with unseen classes as class {most_common_class}")
                    else:
                        # Normal prediction
                        y_pred = model.predict(X_test)
                else:
                    # Models without tracked training classes (shouldn't happen with our updates)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                return calculate_metrics(y_test, y_pred)
        
        return None
        
    except Exception as e:
        print(f"Error evaluating model on combined data: {e}")
        return None


def evaluate_all_models_on_combined_data(all_trained_models, combined_test_df, target_col, unified_encoder):
    """Evaluate all trained models on the combined test dataset."""
    print(f"\n{'='*80}")
    print("EVALUATING ALL MODELS ON COMBINED TEST DATASET")
    print(f"{'='*80}")
    print(f"Combined test data shape: {combined_test_df.shape}")
    print(f"Unique classes in combined test: {combined_test_df[target_col].nunique()}")
    
    combined_results = {}
    
    for client_name, client_models in all_trained_models.items():
        print(f"\n--- Evaluating models trained on {client_name} ---")
        client_combined_results = {}
        
        for model_name, model_info in client_models.items():
            print(f"  Testing {model_name}...")
            result = evaluate_model_on_combined_data(model_info, combined_test_df, target_col, unified_encoder)
            
            if result:
                accuracy, precision, recall, f1 = result
                client_combined_results[model_name] = result
                print(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            else:
                print(f"    Failed to evaluate {model_name}")
        
        if client_combined_results:
            combined_results[client_name] = client_combined_results
    
    return combined_results


def print_combined_evaluation_summary(combined_results, all_results):
    """Print a comprehensive summary comparing individual vs combined dataset performance."""
    print(f"\n{'='*80}")
    print("INDIVIDUAL vs COMBINED DATASET PERFORMANCE COMPARISON")
    print(f"{'='*80}")
    
    for client_name in combined_results.keys():
        print(f"\n{client_name.upper()} TRAINED MODELS:")
        print(f"{'Model':<20} {'Individual F1':<15} {'Combined F1':<15} {'Generalization':<15}")
        print("-" * 75)
        
        if client_name in all_results:
            individual_results = all_results[client_name]
            combined_client_results = combined_results[client_name]
            
            for model_name in individual_results.keys():
                individual_f1 = individual_results[model_name][3]  # F1 score
                
                if model_name in combined_client_results:
                    combined_f1 = combined_client_results[model_name][3]  # F1 score
                    generalization = combined_f1 - individual_f1
                    generalization_str = f"{generalization:+.4f}"
                    
                    if generalization >= 0:
                        generalization_str += " ✓"
                    else:
                        generalization_str += " ↓"
                        
                    print(f"{model_name:<20} {individual_f1:<15.4f} {combined_f1:<15.4f} {generalization_str:<15}")
                else:
                    print(f"{model_name:<20} {individual_f1:<15.4f} {'N/A':<15} {'N/A':<15}")
    
    # Find best generalizing models
    print(f"\n{'='*80}")
    print("BEST GENERALIZING MODELS (Combined Dataset Performance)")
    print(f"{'='*80}")
    
    all_combined_scores = []
    for client_name, client_models in combined_results.items():
        for model_name, metrics in client_models.items():
            f1_score = metrics[3]
            all_combined_scores.append((f1_score, f"{client_name}_{model_name}", client_name, model_name))
    
    # Sort by F1 score (descending)
    all_combined_scores.sort(reverse=True)
    
    print(f"{'Rank':<5} {'Client_Model':<30} {'F1 Score':<10} {'Accuracy':<10}")
    print("-" * 65)
    
    for rank, (f1_score, full_name, client_name, model_name) in enumerate(all_combined_scores[:10], 1):
        accuracy = combined_results[client_name][model_name][0]
        print(f"{rank:<5} {full_name:<30} {f1_score:<10.4f} {accuracy:<10.4f}")


def save_results(all_results, combined_results, experiment_params, selected_models, output_dir="results/horizontal"):
    """Save results to JSON file including combined dataset evaluation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with timestamp and models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_str = "_".join(selected_models)
    filename = f"individual_clients_{models_str}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    # Prepare data for saving
    save_data = {
        "timestamp": timestamp,
        "experiment_params": experiment_params,
        "selected_models": selected_models,
        "individual_results": all_results,
        "combined_results": combined_results,
        "summary": {}
    }
    
    # Add summary statistics
    for client_name, client_results in all_results.items():
        if client_results:
            best_model = max(client_results.keys(), key=lambda x: client_results[x][3])  # F1 score
            client_summary = {
                "individual_performance": {
                    "best_model": best_model,
                    "best_f1": client_results[best_model][3],
                    "all_models": {model: {"f1": metrics[3]} for model, metrics in client_results.items()}
                }
            }
            
            # Add combined dataset performance if available
            if client_name in combined_results:
                combined_client_results = combined_results[client_name]
                best_combined_model = max(combined_client_results.keys(), key=lambda x: combined_client_results[x][3])
                client_summary["combined_performance"] = {
                    "best_model": best_combined_model,
                    "best_f1": combined_client_results[best_combined_model][3],
                    "all_models": {model: {"f1": metrics[3]} for model, metrics in combined_client_results.items()}
                }
                
                # Add generalization analysis
                client_summary["generalization"] = {}
                for model_name in client_results.keys():
                    if model_name in combined_client_results:
                        individual_f1 = client_results[model_name][3]
                        combined_f1 = combined_client_results[model_name][3]
                        generalization = combined_f1 - individual_f1
                        client_summary["generalization"][model_name] = {
                            "individual_f1": individual_f1,
                            "combined_f1": combined_f1,
                            "generalization_gap": generalization
                        }
            
            save_data["summary"][client_name] = client_summary
    
    # Add overall best performers on combined dataset
    if combined_results:
        all_combined_scores = []
        for client_name, client_models in combined_results.items():
            for model_name, metrics in client_models.items():
                f1_score = metrics[3]
                all_combined_scores.append({
                    "client": client_name,
                    "model": model_name,
                    "f1_score": f1_score,
                    "accuracy": metrics[0]
                })
        
        # Sort by F1 score (descending)
        all_combined_scores.sort(key=lambda x: x["f1_score"], reverse=True)
        save_data["summary"]["overall_best_on_combined"] = all_combined_scores[:5]
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath


def parse_models(models_str):
    """Parse model names from command line argument."""
    if not models_str:
        return ['neural_network', 'xgboost', 'random_forest', 'logistic_regression', 'svm']
    
    # Map abbreviations to full names
    model_map = {
        'nn': 'neural_network',
        'neural_network': 'neural_network',
        'xgb': 'xgboost', 
        'xgboost': 'xgboost',
        'rf': 'random_forest',
        'random_forest': 'random_forest',
        'lr': 'logistic_regression',
        'logistic_regression': 'logistic_regression',
        'svm': 'svm'
    }
    
    models = [m.strip().lower() for m in models_str.split(',')]
    selected_models = []
    
    for model in models:
        if model in model_map:
            selected_models.append(model_map[model])
        else:
            print(f"Warning: Unknown model '{model}'. Available models: {list(model_map.keys())}")
    
    return selected_models if selected_models else ['neural_network', 'xgboost', 'random_forest', 'logistic_regression', 'svm']


def get_project_root():
    """Get the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up from src/demo to project root
    return os.path.dirname(os.path.dirname(current_dir))


def save_individual_results(all_results, combined_results, experiment_params, selected_models, seed=0, database_ids=None):
    """Save individual client results to JSON file in FL format."""
    from pathlib import Path
    
    try:
        if database_ids is None:
            database_ids = ["02799", "79665"]  # Default horizontal FL databases
            
        project_root = Path(get_project_root())
        results_dir = project_root / "results" / "horizontal"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create result data structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        result_data = {
            "experiment_type": "horizontal",
            "algorithm": "individual_clients",
            "timestamp": timestamp,
            "seed": seed,
            "database_ids": database_ids,
            "experiment_params": experiment_params,
            "selected_models": selected_models,
            "individual_results": all_results,
            "combined_results": combined_results
        }
        
        # Save to file with predictable naming including model names: individual_clients_model_databases_seed.json
        db_string = "_".join(database_ids)
        
        # Create model string from selected models
        if len(selected_models) == 1:
            # Single model case
            model_map = {
                'neural_network': 'nn',
                'xgboost': 'xgb', 
                'random_forest': 'rf',
                'logistic_regression': 'lr',
                'svm': 'svm'
            }
            model_string = model_map.get(selected_models[0], selected_models[0])
        else:
            # Multiple models case - use abbreviated names
            model_abbrevs = []
            model_map = {
                'neural_network': 'nn',
                'xgboost': 'xgb', 
                'random_forest': 'rf',
                'logistic_regression': 'lr',
                'svm': 'svm'
            }
            for model in selected_models:
                model_abbrevs.append(model_map.get(model, model))
            model_string = "_".join(sorted(model_abbrevs))
        
        filename = f"individual_clients_{model_string}_{db_string}_seed{seed}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\n✅ Individual client results saved to: {filepath}")
        return str(filepath)
        
    except Exception as e:
        print(f"\n❌ Error saving individual client results: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Run individual client training with different models')
    parser.add_argument('-m', '--models', type=str, default=None,
                       help='Comma-separated list of models to run. Options: nn/neural_network, xgb/xgboost, rf/random_forest, lr/logistic_regression, svm. Default: all models')
    parser.add_argument('--skip-combined', action='store_true',
                       help='Skip evaluation on combined test dataset (default: False)')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility and consistent file naming')
    
    args = parser.parse_args()
    
    # Parse selected models
    selected_models = parse_models(args.models)
    print(f"Selected models: {selected_models}")
    
    if args.skip_combined:
        print("Combined dataset evaluation: DISABLED")
    else:
        print("Combined dataset evaluation: ENABLED")
    
    # Set parameters
    DEVICE = 'cuda:1'
    TARGET_COL = 'details_encoded_protein'
    
    # Get project root for absolute paths
    project_root = get_project_root()
    
    # Load all data to create unified label encoder and scaler
    df_a_train = pd.read_csv(os.path.join(project_root, "data/clean/02799/train_data.csv"))
    df_a_test = pd.read_csv(os.path.join(project_root, "data/clean/02799/test_data.csv"))
    df_b_train = pd.read_csv(os.path.join(project_root, "data/clean/79665/train_data.csv"))
    df_b_test = pd.read_csv(os.path.join(project_root, "data/clean/79665/test_data.csv"))
    
    # Create unified label encoder with all possible labels
    all_labels = set()
    for df in [df_a_train, df_a_test, df_b_train, df_b_test]:
        if TARGET_COL in df.columns:
            all_labels.update(df[TARGET_COL].unique())
    
    all_labels = sorted([str(label) for label in all_labels])
    unified_encoder = LabelEncoder()
    unified_encoder.fit(all_labels)
    
    NUM_CLASSES = len(all_labels)
    print(f"Created unified label encoder with {NUM_CLASSES} classes: {all_labels[:5]}{'...' if len(all_labels) > 5 else ''}")
    
    # Create unified StandardScaler from combined training data (like federated learning scripts)
    print("Creating unified StandardScaler...")
    combined_train_df = pd.concat([df_a_train, df_b_train], ignore_index=True)
    feature_cols = [col for col in combined_train_df.columns if col != TARGET_COL]
    
    # Fit the scaler ONLY on the combined training data to avoid data leakage
    unified_scaler = StandardScaler()
    unified_scaler.fit(combined_train_df[feature_cols].fillna(0))
    print(f"Created unified StandardScaler for {len(feature_cols)} features")
    
    # Load data
    print("Loading horizontal federated learning data...")
    client_data = load_horizontal_data()
    
    if client_data is None:
        print("Failed to load data. Exiting.")
        sys.exit(1)
    
    # Experiment parameters
    experiment_params = {
        'hidden_dims': [64, 32],
        'learning_rate': 0.0001,
        'local_epochs': 5,
        'global_rounds': 20,
        'batch_size': 32,
        'device': DEVICE
    }
    
    # Create combined test set for consistent evaluation
    df_a_test = client_data["client_02799"]["test"]
    df_b_test = client_data["client_79665"]["test"]
    combined_test_df = pd.concat([df_a_test, df_b_test], ignore_index=True)
    
    print(f"\n📊 Using combined test set for consistent evaluation")
    print(f"Individual test set sizes: A={len(df_a_test)}, B={len(df_b_test)}")
    print(f"Combined test set size: {len(combined_test_df)}")
    
    all_results = {}
    all_trained_models = {}  # Store all trained models for combined evaluation
    
    # Test selected models on each client using combined test set
    for client_name, data in client_data.items():
        print(f"\n💻 Training on {client_name}, testing on combined test set")
        client_results, trained_models = test_models_on_client(
            client_name, data['train'], combined_test_df, 
            TARGET_COL, NUM_CLASSES, experiment_params, selected_models, unified_encoder, unified_scaler
        )
        all_results[client_name] = client_results
        all_trained_models[client_name] = trained_models
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    for client_name, client_results in all_results.items():
        print(f"\n{client_name.upper()}:")
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        print("-" * 70)
        
        for model_name, metrics in client_results.items():
            accuracy, precision, recall, f1 = metrics
            print(f"{model_name:<20} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
    
    # Find best model for each client
    print(f"\n{'='*80}")
    print("BEST PERFORMING MODELS")
    print(f"{'='*80}")
    
    for client_name, client_results in all_results.items():
        if client_results:
            best_model = max(client_results.keys(), key=lambda x: client_results[x][3])  # F1 score
            best_f1 = client_results[best_model][3]
            
            print(f"\n{client_name}:")
            print(f"  Best Model: {best_model}")
            print(f"  F1 Score: {best_f1:.4f}")
            
            # Show all models' F1 scores for comparison
            print(f"  All F1 Scores:")
            for model_name, metrics in client_results.items():
                f1_score = metrics[3]
                marker = "✓" if model_name == best_model else " "
                print(f"    {marker} {model_name}: F1={f1_score:.4f}")
    
    # Evaluate all trained models on combined test data (using same test set as individual evaluation)
    combined_results = {}
    if not args.skip_combined:
        combined_results = evaluate_all_models_on_combined_data(all_trained_models, combined_test_df, TARGET_COL, unified_encoder)
        print_combined_evaluation_summary(combined_results, all_results)
    else:
        print(f"\n{'='*80}")
        print("COMBINED DATASET EVALUATION SKIPPED")
        print(f"{'='*80}")

    # Save results using the table-generation compatible function
    output_file = save_individual_results(all_results, combined_results, experiment_params, selected_models, 
                                        seed=args.seed, database_ids=["02799", "79665"])
    
    print(f"\n{'='*80}")
    print(f"Experiment completed.")
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()