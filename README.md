# WikiDBGraph: Federated Learning System for Database Analysis

WikiDBGraph is an enterprise-grade system for graph-based database analysis and federated learning on distributed database pairs. It provides comprehensive tools for analyzing database schemas, computing structural similarities, and training machine learning models using various federated learning algorithms across databases with similar structures.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Directory Structure](#directory-structure)
- [Core Components](#core-components)
- [Quick Start Guide](#quick-start-guide)
- [Usage Examples](#usage-examples)
- [Automated Validation System](#automated-validation-system)
- [Configuration Options](#configuration-options)
- [Output and Results](#output-and-results)
- [Development Guide](#development-guide)
- [Troubleshooting](#troubleshooting)
- [Requirements](#requirements)
- [License](#license)

## Overview

WikiDBGraph is designed for researchers and practitioners working with multiple databases that share structural similarities. The system constructs graph representations of database schemas, computes semantic and structural similarities, and enables federated learning experiments across database pairs without centralizing data.

The system supports both horizontal and vertical federated learning paradigms, implements multiple state-of-the-art FL algorithms, and provides automated validation pipelines for large-scale experimentation.

## Key Features

### Graph-Based Analysis
- Automatic graph construction from database schemas
- Semantic similarity computation using BGE embeddings
- Community detection and connected component analysis
- GPU-accelerated join cardinality estimation
- Statistical and semantic property analysis for nodes and edges

### Federated Learning Algorithms
- **FedAvg**: Standard federated averaging
- **FedProx**: Federated learning with proximal term regularization
- **SCAFFOLD**: Control variates for improved convergence
- **FedOV**: One-shot federated learning with outlier detection
- **FedTree**: Tree-based federated learning for gradient boosting
- **SplitNN**: Split neural networks for vertical federated learning

### Automated Validation Pipeline
- Intelligent database pair sampling based on similarity thresholds
- Automated data preprocessing and quality validation
- Multi-GPU task scheduling with load balancing
- Parallel experiment execution across multiple algorithm types
- Comprehensive result aggregation and reporting

### Visualization and Analysis
- Performance distribution plots for all metrics
- Comparative analysis between federated and centralized training
- Database clustering visualization
- Graph structure and community analysis plots
- LaTeX table generation for research papers

### GPU Acceleration
- RAPIDS cuML ecosystem integration for high-performance computing
- Multi-GPU support for parallel training
- GPU-accelerated graph operations using cuGraph
- Efficient memory management for large-scale experiments

## System Architecture

### Data Flow Pipeline

```
Database Schemas → Graph Construction → Similarity Analysis → Pair Selection
                                                                    ↓
Results ← FL Training ← Data Preprocessing ← Pair Validation ← Pair Sampling
```

### Component Interaction

1. **Analysis Layer**: Processes database schemas and builds graph structures
2. **Similarity Layer**: Computes structural and semantic similarities
3. **Preprocessing Layer**: Prepares data for federated learning
4. **Training Layer**: Executes FL algorithms on distributed data
5. **Aggregation Layer**: Collects results and generates visualizations

### Execution Modes

- **Manual Mode**: Individual script execution for specific experiments
- **Automated Mode**: End-to-end pipeline with batch processing
- **Interactive Mode**: Jupyter notebooks for exploratory analysis

## Installation

### Prerequisites

- Linux/Unix environment
- Python 3.12 or higher
- CUDA 12.1+ compatible GPU (required for GPU acceleration)
- Minimum 16GB RAM (32GB+ recommended for large datasets)
- Sufficient disk space for data and results (100GB+ recommended)

### Installation Steps

```bash
# Clone the repository
git clone <repository-url>
cd wikidbs

# Install dependencies using pip
pip install -e .

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import cudf; print('RAPIDS installed successfully')"
```

### Dependencies

The project includes the following major dependencies (see `pyproject.toml` for complete list):

- **Deep Learning**: PyTorch 2.4.0, DGL with CUDA support
- **GPU Acceleration**: RAPIDS cuML ecosystem (cuDF, cuML, cuGraph, cuSpatial)
- **Data Processing**: pandas, numpy, scikit-learn
- **Visualization**: matplotlib, seaborn
- **Utilities**: orjson, tqdm, xgboost, pydot

### Environment Setup

```bash
# Set Python path for running scripts
export PYTHONPATH=$PWD/src

# Verify GPU availability
nvidia-smi
```

## Directory Structure

### Root Level

```
wikidbs/
├── run_automated_fl_validation.sh    # Main automation orchestrator
├── pyproject.toml                     # Package configuration
├── LICENSE                            # Apache 2.0 license
├── README.md                          # This file
├── CLAUDE.md                          # Development guidance
├── src/                               # Main source code
├── data/                              # Data storage
├── out/                               # Experiment outputs
├── fig/                               # Generated visualizations
├── FedOV/                             # External FedOV framework
├── FedTree/                           # External FedTree framework
└── NIID-Bench/                        # Non-IID benchmark framework
```

### Source Code Organization

```
src/
├── analysis/          # Graph analysis and similarity computation
├── autorun/           # Automated FL validation system
├── demo/              # Manual FL experiment scripts
├── model/             # Neural networks and FL algorithms
├── summary/           # Visualization and result aggregation
├── train/             # Model training utilities
├── test/              # Testing and validation scripts
└── utils/             # Helper functions and data structures
```

### Data Directories

```
data/
├── raw/               # Original database dumps
├── schema/            # Database schema files
├── graph/             # Graph similarity data
├── encoders/          # Trained encoder models
├── clean/             # Cleaned database files
├── auto/              # Automated validation preprocessed data
└── unzip/             # Unzipped database archives
```

### Output Directories

```
out/
├── autorun/           # Automated validation results
│   ├── logs/          # Execution logs
│   ├── results/       # Training results JSON files
│   └── sampled_pairs.json
├── graphs/            # Graph analysis outputs
└── *.csv              # Statistical summaries
```

### Figures Directory

```
fig/
├── fl_performance_*.{pdf,png}     # FL performance distributions
├── fl_delta_*.{pdf,png}           # Improvement over solo training
├── database_cluster.png           # Database similarity clustering
├── community_*.png                # Community detection results
├── component_*.png                # Connected component analysis
└── degree_distribution_*.png      # Graph degree distributions
```

## Core Components

### 1. Analysis Module (`src/analysis/`)

The analysis module provides tools for database graph construction and similarity computation.

#### Primary Files

**`build_raw_graph.py`**
- Constructs graph representations from database schemas
- Nodes represent tables, edges represent foreign key relationships
- Extracts metadata including column types, constraints, and indexes
- Usage: `python src/analysis/build_raw_graph.py`

**`CommunityDetection.py`**
- Implements Louvain algorithm for community detection
- Groups similar databases into clusters
- Identifies structural patterns across database collections
- Supports multiple resolution parameters

**`NodeSemantic.py`**
- Computes semantic embeddings for table and column names
- Uses BGE (BAAI General Embedding) models
- Generates contextual representations for schema elements
- Supports batch processing for efficiency

**`NodeStatistical.py`**
- Calculates statistical properties of nodes
- Computes degree centrality, betweenness, and clustering coefficients
- Analyzes data distribution statistics

**`EdgeProperties.py`**
- Computes edge weights based on relationship strength
- Analyzes foreign key cardinality and referential integrity
- Calculates edge-level similarity metrics

**`compute_all_join_size_fast.py`**
- GPU-accelerated join cardinality estimation
- Uses RAPIDS cuDF for high-performance computation
- Handles large-scale database pairs efficiently
- Usage: `python src/analysis/compute_all_join_size_fast.py`

**`analysis_graph_components.py`**
- Analyzes connected components in database graphs
- Identifies isolated subgraphs and their properties
- Generates component size distributions
- Usage: `python src/analysis/analysis_graph_components.py`

**`collect_similar_database_pairs.py`**
- Filters database pairs by similarity threshold
- Ranks pairs based on structural and semantic similarity
- Outputs candidate pairs for FL experiments

#### Additional Analysis Utilities

- `add_properties.py`: Adds computed properties to graph nodes
- `analyze_degree_nodes.py`: Analyzes node degree distribution
- `calculate_thresholds.py`: Computes optimal similarity thresholds
- `check_triplets_overlap.py`: Validates triplet consistency
- `compare_edge_files.py`: Compares edge properties across versions
- `create_symlinks.py`: Creates symbolic links for data organization
- `filter_edges.py`: Filters edges based on criteria
- `find_small_cc.py`: Identifies small connected components
- `print_similar_database.py`: Displays similar database information
- `upload_huggingface.py`: Uploads datasets to Hugging Face Hub

### 2. Automated Validation System (`src/autorun/`)

The automated system orchestrates large-scale federated learning validation experiments.

#### Core Scripts

**`pair_sampler.py`**
- Samples database pairs within specified similarity ranges
- Validates data quality requirements (minimum rows, common columns)
- Outputs JSON file with sampled pairs and metadata
- Usage: `python src/autorun/pair_sampler.py --min-similarity 0.98 --sample-size 2000`
- Key parameters:
  - `--min-similarity`, `--max-similarity`: Similarity range
  - `--min-rows`: Minimum table rows requirement
  - `--sample-size`: Number of pairs to sample
  - `--output`: Output JSON file path

**`data_preprocessor.py`**
- Loads and joins tables from each database
- Identifies common columns between database pairs
- Automatically selects regression labels with sufficient variance
- Cleans data (handles missing values, removes invalid records)
- Normalizes labels to [0,1] range
- Splits data into train/test sets with stratification
- Outputs CSV files and configuration JSON
- Usage: `python src/autorun/data_preprocessor.py --input pairs.json --output-dir data/auto`
- Key parameters:
  - `--test-size`: Test set proportion (default: 0.2)
  - `--min-label-variance`: Minimum variance for label selection
  - `--max-missing-ratio`: Maximum allowed missing values

**`gpu_scheduler.py`**
- Manages parallel execution across multiple GPUs
- Implements load balancing and task queuing
- Monitors GPU utilization and memory usage
- Handles task failures with automatic retry
- Generates execution reports with timing statistics
- Usage: `python src/autorun/gpu_scheduler.py --num-gpus 4 --max-concurrent-per-gpu 5`
- Key parameters:
  - `--num-gpus`: Number of GPUs to use
  - `--max-concurrent-per-gpu`: Concurrent tasks per GPU
  - `--timeout`: Task timeout in seconds
  - `--task-types`: Algorithms to run (fedavg, fedprox, scaffold, fedov, fedtree, solo, combined)

**`fedavg.py`**
- FedAvg algorithm implementation for automated pipeline
- Supports customizable number of clients and rounds
- Implements secure aggregation
- Usage: Called by GPU scheduler

**`fedprox.py`**
- FedProx with proximal term regularization
- Configurable mu parameter for proximal term strength
- Handles heterogeneous client data distributions
- Usage: Called by GPU scheduler

**`scaffold.py`**
- SCAFFOLD algorithm with control variates
- Maintains control variates for each client
- Improves convergence in non-IID settings
- Usage: Called by GPU scheduler

**`fedov.py`**
- FedOV one-shot federated learning
- Implements outlier detection and voting mechanisms
- Suitable for extreme non-IID scenarios
- Usage: Called by GPU scheduler

**`fedtree.py`**
- Integration with FedTree framework
- Tree-based federated learning for gradient boosting
- Supports homomorphic encryption
- Usage: Called by GPU scheduler

**`solo.py`**
- Individual client training (solo mode)
- Centralized training on combined data (combined mode)
- Provides baseline comparisons for FL algorithms
- Usage: Called by GPU scheduler with `--mode solo` or `--mode combined`

**`README.md`**
- Detailed documentation for the automated validation system
- Component descriptions and usage examples
- Parameter explanations and troubleshooting guide

### 3. Manual Experiment Scripts (`src/demo/`)

Scripts for running individual federated learning experiments with full control.

#### Training Scripts

**`train_fedavg.py`**
- FedAvg horizontal federated learning
- Usage: `python src/demo/train_fedavg.py --seed 0 --databases 02799 79665`
- Parameters:
  - `--databases`: Two database IDs to train on
  - `--seed`: Random seed for reproducibility
  - `--local-epochs`: Local training epochs
  - `--global-rounds`: Global communication rounds
  - `--learning-rate`: Learning rate
  - `--batch-size`: Batch size

**`train_fedprox.py`**
- FedProx with proximal regularization
- Usage: `python src/demo/train_fedprox.py --seed 1 --databases 02799 79665 --mu 0.01`
- Additional parameter:
  - `--mu`: Proximal term coefficient

**`train_scaffold.py`**
- SCAFFOLD with control variates
- Usage: `python src/demo/train_scaffold.py --seed 2 --databases 02799 79665`
- Maintains control variates automatically

**`train_fedov.py`**
- FedOV one-shot federated learning
- Usage: `python src/demo/train_fedov.py --seed 3 --databases 02799 79665`
- Implements outlier detection and voting

**`train_fedtree.py`**
- FedTree integration for tree-based models
- Usage: `python src/demo/train_fedtree.py --databases 02799 79665`
- Supports gradient boosting decision trees

**`train_splitnn.py`**
- Split neural network for vertical FL
- Usage: `python src/demo/train_splitnn.py --seed 4`
- Splits model between clients and server

**`run_individual_clients.py`**
- Trains individual models for each client
- Usage: `python src/demo/run_individual_clients.py -m nn --seed 0`
- Parameters:
  - `-m`: Model type (nn, lr, rf, xgb)
  - `--seed`: Random seed

**`centralized_training.py`**
- Centralized baseline on combined data
- Usage: `python src/demo/centralized_training.py`
- Provides upper bound for FL performance

#### Data Preparation Scripts

**`prepare_horizontal_data.py`**
- Prepares data splits for horizontal FL
- Distributes samples across clients
- Ensures consistent feature spaces
- Usage: `python src/demo/prepare_horizontal_data.py`

**`prepare_vertical_data.py`**
- Prepares data splits for vertical FL
- Distributes features across clients
- Maintains sample alignment
- Usage: `python src/demo/prepare_vertical_data.py`

#### Additional Experiment Utilities

- `run_centralized_training.py`: Batch centralized training
- `run_primary_client.py`: Primary client experiments
- `train_horizontal.py`: Generic horizontal FL trainer
- `train_vertical.py`: Generic vertical FL trainer

### 4. Model Implementations (`src/model/`)

Neural network architectures and federated learning algorithm implementations.

#### Federated Learning Algorithms

**`FedAvg.py`**
- Core FedAvg implementation
- Server-side model aggregation
- Client-side local training
- Supports weighted averaging based on dataset size

**`FedProx.py`**
- FedProx algorithm implementation
- Adds proximal term to loss function
- Configurable mu parameter
- Handles system heterogeneity

**`SCAFFOLD.py`**
- SCAFFOLD algorithm with control variates
- Maintains server and client control variates
- Reduces client drift in non-IID settings
- Improved convergence guarantees

**`FedOV.py`**
- FedOV one-shot federated learning
- Outlier detection using ensemble methods
- Voting mechanism for robust aggregation
- Suitable for extreme heterogeneity

**`SplitNN.py`**
- Split neural network implementation
- Model split between clients and server
- Forward/backward pass coordination
- Privacy-preserving by design

#### Model Components

**`BGEEmbedder.py`**
- Wrapper for BGE (BAAI General Embedding) models
- Generates semantic embeddings for text
- Supports batch processing
- GPU acceleration support

**`WKDataset.py`**
- PyTorch dataset implementation for WikiDB data
- Handles tabular data loading
- Supports train/test splitting
- Efficient memory usage

**`column_encoder.py`**
- Column-level encoding for tabular data
- Handles categorical and numerical features
- One-hot encoding and normalization
- Preserves feature semantics

**`col_embedding_model.py`**
- Column embedding model for feature learning
- Learns representations for columns
- Supports pre-training and fine-tuning

**`cal_sim.py`**
- Similarity calculation utilities
- Cosine similarity, Euclidean distance
- GPU-accelerated computation

### 5. Visualization and Analysis (`src/summary/`)

Tools for result aggregation, visualization, and LaTeX table generation.

#### Table Generation Scripts

**`run_and_generate_horizontal_table.py`**
- Generates LaTeX tables for horizontal FL experiments
- Automatically detects and runs missing experiments
- Aggregates results across multiple seeds
- Computes mean and standard deviation
- Usage: `python src/summary/run_and_generate_horizontal_table.py --show-std --num-seeds 5`
- Parameters:
  - `--show-std`: Include standard deviations
  - `--num-seeds`: Number of seeds to aggregate
  - `--skip-runs`: Skip running missing experiments

**`run_and_generate_vertical_table.py`**
- Generates LaTeX tables for vertical FL experiments
- Similar functionality to horizontal table generator
- Usage: `python src/summary/run_and_generate_vertical_table.py --show-std --num-seeds 5`

**`run_and_generate_ml_model_tables.py`**
- Generates comparison tables for different ML models
- Includes neural networks, random forests, XGBoost
- Usage: `python src/summary/run_and_generate_ml_model_tables.py --runs 5`

**`generate_federated_learning_tables.py`**
- Generic table generator for FL results
- Supports custom metrics and algorithms
- Flexible formatting options

#### Visualization Scripts

**`plot_auto_horizontal.py`**
- Plots FL performance distributions
- Generates box plots and histograms
- Compares federated vs. solo training
- Outputs PDF and PNG formats
- Usage: `python src/summary/plot_auto_horizontal.py`

**`plot_datasets_info.py`**
- Visualizes dataset statistics
- Shows feature distributions and correlations
- Displays data quality metrics
- Usage: `python src/summary/plot_datasets_info.py`

**`plot_graph.py`**
- Visualizes database graph structures
- Shows community detection results
- Displays node and edge properties
- Usage: `python src/summary/plot_graph.py`

**`plot_db_cluster.py`**
- Creates database clustering visualizations
- Uses dimensionality reduction (t-SNE, UMAP)
- Shows similarity-based groupings
- Usage: `python src/summary/plot_db_cluster.py`

**`plot_component_distribution.py`**
- Plots connected component size distributions
- Shows power-law behavior
- Usage: `python src/summary/plot_component_distribution.py`

**`plot_graph_similarity_distribution.py`**
- Visualizes similarity score distributions
- Shows histogram of pairwise similarities
- Usage: `python src/summary/plot_graph_similarity_distribution.py`

**`plot_feature_overlap.py`**
- Plots feature overlap between databases
- Shows Venn diagrams and heatmaps
- Usage: `python src/summary/plot_feature_overlap.py`

**`plot_feature_skew.py`**
- Visualizes feature distribution skew
- Analyzes non-IID characteristics
- Usage: `python src/summary/plot_feature_skew.py`

**`plot_sim_distribution.py`**
- Plots similarity score distributions
- Multiple similarity metrics supported
- Usage: `python src/summary/plot_sim_distribution.py`

**`plot_matched_ratio.py`**
- Visualizes schema matching ratios
- Shows percentage of matched elements
- Usage: `python src/summary/plot_matched_ratio.py`

#### Additional Analysis Tools

- `print_auto_horizontal.py`: Prints horizontal FL results to console
- `print_edge_properties.py`: Displays edge property statistics
- `print_graph_connectivity.py`: Shows graph connectivity metrics
- `print_node_property.py`: Displays node property statistics
- `plot_test_results.py`: Plots test set performance
- `plot_test_results_fullneg.py`: Plots results with negative examples
- `plot_time_test.py`: Visualizes execution time comparisons
- `plot_methods_results_for_sim_cal.py`: Compares similarity methods
- `test_result.py`: Result validation and testing
- `test.sh`: Automated testing script

### 6. Training Utilities (`src/train/`)

Model training and data preparation utilities.

**`train_bge_softmax.py`**
- Trains BGE embedding model with softmax classifier
- Fine-tunes on domain-specific data
- Supports multi-GPU training
- Usage: `python src/train/train_bge_softmax.py`

**`embed_full.sh`**
- Shell script for full embedding generation
- Processes entire database collection
- Parallelizes across multiple GPUs
- Usage: `./src/train/embed_full.sh`

**`build_graph.py`**
- Constructs training graphs from schemas
- Prepares graph data for GNN training
- Usage: `python src/train/build_graph.py`

**`split_dataset.py`**
- Splits datasets for training and evaluation
- Supports stratified splitting
- Usage: `python src/train/split_dataset.py`

**`generate_random_test_dataset.py`**
- Generates random test datasets
- For validation and testing
- Usage: `python src/train/generate_random_test_dataset.py`

**`scaleup_test_dataset.py`**
- Scales up test datasets for stress testing
- Generates large-scale evaluation sets
- Usage: `python src/train/scaleup_test_dataset.py`

**`train.sh`**
- Master training script
- Coordinates multiple training runs
- Usage: `./src/train/train.sh`

### 7. Testing and Validation (`src/test/`)

Testing utilities and validation notebooks.

**`bge_m3_similarity.py`**
- Tests BGE-M3 model similarity computation
- Validates embedding quality
- Usage: `python src/test/bge_m3_similarity.py`

**`check.py`**
- General validation checks
- Data integrity verification
- Usage: `python src/test/check.py`

**`cluster.py`**
- Clustering validation and testing
- Evaluates clustering quality
- Usage: `python src/test/cluster.py`

**`convert2pt.py`**
- Converts models to PyTorch format
- Model serialization utilities
- Usage: `python src/test/convert2pt.py`

**`print_sim.py`**
- Prints similarity matrices
- Debugging utility
- Usage: `python src/test/print_sim.py`

**`test.sh`**
- Automated test suite
- Runs all validation tests
- Usage: `./src/test/test.sh`

**Notebooks**
- `print_schema.ipynb`: Schema inspection notebook
- `test_sim.ipynb`: Similarity testing notebook

### 8. Utilities (`src/utils/`)

Helper functions and data structures.

**`wikidbs.py`**
- Core data structures for database schemas
- Classes: `Schema`, `Table`, `Column`, `ForeignKey`
- Methods for schema manipulation and querying
- Serialization and deserialization utilities

**`schema_formatter.py`**
- Schema formatting and normalization
- Converts between different schema formats
- Validates schema consistency

**`llm_judger.py`**
- LLM-based evaluation for semantic similarity
- Uses language models to judge schema matches
- Provides qualitative assessments

**`load_from_uci.py`**
- Loads datasets from UCI Machine Learning Repository
- Standardizes UCI data formats
- Usage: `python src/utils/load_from_uci.py`

**`print_schema.py`**
- Pretty-prints database schemas
- Console visualization of schema structure
- Usage: `python src/utils/print_schema.py`

**`copy_schema.sh`**
- Shell script for schema backup
- Copies schemas to backup locations
- Usage: `./src/utils/copy_schema.sh`

### 9. External Frameworks

#### FedOV Framework

Located in `FedOV/` directory, this is an external framework for one-shot federated learning.

**Purpose**: Implements FedOV algorithm for addressing label skew in one-shot FL scenarios.

**Key Files**:
- `experiments.py`: Main experiment runner
- `model.py`: Neural network models
- `datasets.py`: Dataset loaders
- `utils.py`: Utility functions
- `attack.py`: Adversarial attack implementations
- `cutpaste.py`: Data augmentation techniques
- `run.sh`: Quick start script

**Usage**: See `FedOV/README.md` for detailed instructions.

#### FedTree Framework

Located in `FedTree/` directory, this is an external framework for tree-based federated learning.

**Purpose**: Fast, effective, and secure tree-based federated learning system supporting gradient boosting decision trees.

**Key Features**:
- Parallel computing on CPUs and GPUs
- Homomorphic encryption support
- Secure aggregation and differential privacy
- Classification and regression support

**Key Directories**:
- `src/`: C++ source code
- `python/`: Python bindings
- `include/`: Header files
- `examples/`: Example configurations
- `build/`: Compiled binaries (after building)

**Building**:
```bash
cd FedTree
mkdir build && cd build
cmake ..
make -j
```

**Usage**: See `FedTree/README.md` for detailed instructions.

#### NIID-Bench Framework

Located in `NIID-Bench/` directory, this is a benchmark for federated learning on non-IID data.

**Purpose**: Comprehensive benchmark for comparing FL algorithms under various non-IID data distribution scenarios.

**Non-IID Settings Supported**:
- Label distribution skew (quantity-based and distribution-based)
- Feature distribution skew (noise-based, synthetic, real-world)
- Quantity skew
- Mixed skew scenarios

**Key Files**:
- `experiments.py`: Main experiment runner
- `partition.py`: Data partitioning utilities
- `datasets.py`: Dataset implementations
- `model.py`: Model architectures
- `utils.py`: Helper functions
- `run.sh`: Example run script

**Usage**: See `NIID-Bench/README.md` for detailed instructions.

## Quick Start Guide

### 1. Basic Graph Analysis

Analyze database schemas and construct similarity graphs:

```bash
# Set Python path
export PYTHONPATH=$PWD/src

# Build graphs from database schemas
python src/analysis/build_raw_graph.py

# Analyze graph components
python src/analysis/analysis_graph_components.py

# Generate visualizations
python src/summary/plot_graph.py
python src/summary/plot_datasets_info.py
```

### 2. Running a Simple FL Experiment

Train FedAvg on a database pair:

```bash
# Train FedAvg with seed 0
python src/demo/train_fedavg.py --seed 0 --databases 02799 79665

# Train FedProx for comparison
python src/demo/train_fedprox.py --seed 0 --databases 02799 79665 --mu 0.01

# Run individual client training (baseline)
python src/demo/run_individual_clients.py -m nn --seed 0
```

### 3. Automated Large-Scale Validation

Run the automated validation pipeline:

```bash
# Run with default parameters (2000 pairs, similarity 0.98-1.0)
./run_automated_fl_validation.sh

# Run with custom parameters
./run_automated_fl_validation.sh \
    --min-similarity 0.95 \
    --max-similarity 0.99 \
    --sample-size 500 \
    --num-gpus 4 \
    --task-types "fedprox scaffold fedov"

# Run specific algorithms only
./run_automated_fl_validation.sh --task-types "fedprox solo combined"
```

### 4. Generating Result Tables

Generate LaTeX tables from experiment results:

```bash
# Generate horizontal FL tables
python src/summary/run_and_generate_horizontal_table.py --show-std --num-seeds 5

# Generate vertical FL tables
python src/summary/run_and_generate_vertical_table.py --show-std --num-seeds 5

# Generate ML model comparison tables
python src/summary/run_and_generate_ml_model_tables.py --runs 5
```

### 5. Creating Visualizations

Generate plots and figures:

```bash
# Plot FL performance distributions
python src/summary/plot_auto_horizontal.py

# Plot database clustering
python src/summary/plot_db_cluster.py

# Plot component distributions
python src/summary/plot_component_distribution.py
```

## Usage Examples

### Example 1: Manual FL Experiment with Multiple Algorithms

```bash
# Set environment
export PYTHONPATH=$PWD/src

# Database pair: 02799 and 79665
DB1="02799"
DB2="79665"
SEED=0

# Run FedAvg
python src/demo/train_fedavg.py --seed $SEED --databases $DB1 $DB2 \
    --local-epochs 5 --global-rounds 20 --learning-rate 0.001

# Run FedProx with mu=0.01
python src/demo/train_fedprox.py --seed $SEED --databases $DB1 $DB2 \
    --mu 0.01 --local-epochs 5 --global-rounds 20

# Run SCAFFOLD
python src/demo/train_scaffold.py --seed $SEED --databases $DB1 $DB2 \
    --local-epochs 5 --global-rounds 20

# Run individual clients (baseline)
python src/demo/run_individual_clients.py -m nn --seed $SEED

# Run centralized training (upper bound)
python src/demo/centralized_training.py
```

### Example 2: Automated Pipeline with Custom Filters

```bash
# Sample high-similarity pairs with strict quality requirements
./run_automated_fl_validation.sh \
    --min-similarity 0.99 \
    --max-similarity 1.0 \
    --min-rows 1000 \
    --sample-size 100 \
    --seed 42 \
    --num-gpus 4 \
    --max-concurrent 3 \
    --task-types "fedprox scaffold fedov solo combined"

# Check results
ls -lh out/autorun/results/
```

### Example 3: Resuming After Interruption

```bash
# Run was interrupted after preprocessing
# Resume automatically (detects existing outputs)
./run_automated_fl_validation.sh

# Force fresh run (delete all previous data)
./run_automated_fl_validation.sh --force-rerun

# Skip only sampling (use existing pairs)
./run_automated_fl_validation.sh --skip-sampling

# Skip sampling and preprocessing (run only training)
./run_automated_fl_validation.sh --skip-sampling --skip-preprocessing
```

### Example 4: Using Specific GPUs

```bash
# Use only GPU 1 (e.g., other GPUs are busy)
./run_automated_fl_validation.sh --gpu-ids "1"

# Use GPUs 0, 2, and 5
./run_automated_fl_validation.sh --gpu-ids "0,2,5"

# Use GPUs 2 and 3 with high concurrency
./run_automated_fl_validation.sh --gpu-ids "2,3" --max-concurrent 8
```

### Example 5: Graph Analysis Workflow

```bash
export PYTHONPATH=$PWD/src

# Step 1: Build graphs
python src/analysis/build_raw_graph.py

# Step 2: Compute node semantics
python src/analysis/NodeSemantic.py

# Step 3: Detect communities
python src/analysis/CommunityDetection.py

# Step 4: Analyze components
python src/analysis/analysis_graph_components.py

# Step 5: Collect similar pairs
python src/analysis/collect_similar_database_pairs.py

# Step 6: Visualize results
python src/summary/plot_graph.py
python src/summary/plot_db_cluster.py
```

### Example 6: Training BGE Embeddings

```bash
export PYTHONPATH=$PWD/src

# Train BGE model on domain data
python src/train/train_bge_softmax.py

# Generate embeddings for all databases
./src/train/embed_full.sh

# Validate embedding quality
python src/test/bge_m3_similarity.py
```

## Automated Validation System

The automated validation system provides end-to-end orchestration of large-scale FL experiments.

### System Components

#### 1. Orchestrator Script

**File**: `run_automated_fl_validation.sh`

**Purpose**: Main entry point that coordinates the entire pipeline.

**Workflow**:
1. Validate environment and dependencies
2. Sample database pairs based on similarity criteria
3. Preprocess data for each pair
4. Schedule and execute FL training tasks
5. Aggregate results and generate reports

**Key Features**:
- Automatic step detection (resumes from last successful step)
- Comprehensive logging to `out/autorun/logs/`
- Error handling with partial success support
- GPU utilization monitoring
- Parallel execution across multiple GPUs

#### 2. Pair Sampler

**File**: `src/autorun/pair_sampler.py`

**Purpose**: Intelligently samples database pairs for experiments.

**Selection Criteria**:
- Similarity range (default: 0.98-1.0)
- Minimum row count (default: 100)
- Common column requirements
- Data quality thresholds

**Output**: `out/autorun/sampled_pairs.json`

```json
{
  "pairs": [
    {
      "db_id1": 2799,
      "db_id2": 79665,
      "similarity": 0.9894,
      "common_columns": 15,
      "db1_rows": 5430,
      "db2_rows": 8712
    }
  ],
  "metadata": {
    "total_candidates": 50000,
    "sampled": 2000,
    "min_similarity": 0.98,
    "max_similarity": 1.0
  }
}
```

#### 3. Data Preprocessor

**File**: `src/autorun/data_preprocessor.py`

**Purpose**: Prepares data for federated learning.

**Processing Steps**:
1. Load tables from both databases
2. Join tables within each database
3. Identify common columns
4. Select regression labels (numeric columns with variance > threshold)
5. Clean data (handle missing values, remove outliers)
6. Normalize features
7. Split into train/test sets (80/20)
8. Save as CSV files

**Output Structure**:
```
data/auto/{db_id1}_{db_id2}/
├── config.json              # Pair configuration
├── {db_id1}_train.csv      # Client 1 training data
├── {db_id1}_test.csv       # Client 1 test data
├── {db_id2}_train.csv      # Client 2 training data
└── {db_id2}_test.csv       # Client 2 test data
```

**Configuration File** (`config.json`):
```json
{
  "pair_id": "02799_79665",
  "db_id1": 2799,
  "db_id2": 79665,
  "similarity": 0.9894,
  "common_columns": ["feature1", "feature2", ...],
  "label_column": "target_variable",
  "num_features": 15,
  "train_samples": {"client_0": 4344, "client_1": 6970},
  "test_samples": {"client_0": 1086, "client_1": 1742}
}
```

#### 4. GPU Scheduler

**File**: `src/autorun/gpu_scheduler.py`

**Purpose**: Manages parallel execution across multiple GPUs.

**Features**:
- Load balancing across GPUs
- Configurable concurrency per GPU
- Task queuing with priority
- Automatic retry on failure
- Timeout management
- Real-time status monitoring

**Task Types**:
- `fedavg`: FedAvg algorithm
- `fedprox`: FedProx algorithm
- `scaffold`: SCAFFOLD algorithm
- `fedov`: FedOV algorithm
- `fedtree`: FedTree algorithm
- `solo`: Individual client training
- `combined`: Centralized training on combined data

**Execution Flow**:
1. Load preprocessed pair list
2. Generate task queue (pair × task_type combinations)
3. Distribute tasks to GPU queues
4. Monitor and execute tasks in parallel
5. Collect results and handle failures
6. Generate execution report

**Output**: `out/autorun/results/execution_report.json`

```json
{
  "summary": {
    "total_tasks": 6000,
    "completed": 5847,
    "failed": 153,
    "execution_time_hours": 12.5
  },
  "completed_tasks": [...],
  "failed_tasks": [...]
}
```

### Usage Guide

#### Basic Usage

Run with default parameters:

```bash
./run_automated_fl_validation.sh
```

This will:
- Sample 2000 database pairs with similarity 0.98-1.0
- Preprocess data for all pairs
- Run FedProx, SCAFFOLD, and FedOV with solo and combined baselines
- Use 4 GPUs with up to 5 concurrent tasks per GPU

#### Custom Similarity Range

```bash
./run_automated_fl_validation.sh \
    --min-similarity 0.95 \
    --max-similarity 0.99 \
    --sample-size 500
```

#### Selecting Specific Algorithms

Run only FedProx:
```bash
./run_automated_fl_validation.sh --task-types "fedprox solo combined"
```

Run all algorithms including FedTree:
```bash
./run_automated_fl_validation.sh --task-types all
```

Run specific subset:
```bash
./run_automated_fl_validation.sh --task-types "fedprox scaffold fedov"
```

#### GPU Configuration

Use specific GPUs:
```bash
# Use only GPU 1
./run_automated_fl_validation.sh --gpu-ids "1"

# Use GPUs 0, 2, and 5
./run_automated_fl_validation.sh --gpu-ids "0,2,5"

# Use GPUs 2-3 with high concurrency
./run_automated_fl_validation.sh --gpu-ids "2,3" --max-concurrent 8
```

#### Step Control

Skip specific steps:
```bash
# Skip sampling (use existing pairs)
./run_automated_fl_validation.sh --skip-sampling

# Skip sampling and preprocessing
./run_automated_fl_validation.sh --skip-sampling --skip-preprocessing

# Skip training
./run_automated_fl_validation.sh --skip-training
```

Force fresh run:
```bash
# Delete all existing data and start over
./run_automated_fl_validation.sh --force-rerun
```

#### Advanced Options

```bash
./run_automated_fl_validation.sh \
    --min-similarity 0.98 \
    --max-similarity 1.0 \
    --min-rows 500 \
    --sample-size 1000 \
    --seed 42 \
    --num-gpus 4 \
    --max-concurrent 5 \
    --timeout 7200 \
    --task-types "fedprox scaffold fedov solo combined"
```

### Result Format

Each experiment generates a JSON file with comprehensive results:

**File**: `out/autorun/results/{pair_id}_{algorithm}_results.json`

```json
{
  "pair_id": "02799_79665",
  "db_id1": 2799,
  "db_id2": 79665,
  "similarity": 0.9894,
  "algorithm": "fedprox",
  "label_column": "target_variable",
  "num_features": 15,
  "hyperparameters": {
    "local_epochs": 5,
    "global_rounds": 20,
    "learning_rate": 0.001,
    "batch_size": 32,
    "mu": 0.01
  },
  "results": {
    "client_0": {
      "mse": 0.0456,
      "rmse": 0.2135,
      "mae": 0.1678,
      "r2": 0.7234
    },
    "client_1": {
      "mse": 0.0389,
      "rmse": 0.1972,
      "mae": 0.1543,
      "r2": 0.7891
    },
    "average": {
      "mse": 0.0423,
      "rmse": 0.2054,
      "mae": 0.1611,
      "r2": 0.7563
    }
  },
  "training_time_seconds": 245.7,
  "convergence_round": 18
}
```

## Configuration Options

### Global Parameters

**Environment Variables**:
```bash
export PYTHONPATH=$PWD/src           # Required for all scripts
export CUDA_VISIBLE_DEVICES="0,1"    # GPU selection
```

### Automated Pipeline Parameters

**Sampling Parameters**:
- `--min-similarity`: Minimum similarity threshold (float, default: 0.98)
- `--max-similarity`: Maximum similarity threshold (float, default: 1.0)
- `--min-rows`: Minimum table rows (int, default: 100)
- `--sample-size`: Number of pairs to sample (int, default: 2000)
- `--seed`: Random seed (int, default: 42)

**GPU Parameters**:
- `--num-gpus`: Number of GPUs (int, default: 4)
- `--gpu-ids`: Specific GPU IDs (string, e.g., "0,2,5")
- `--max-concurrent`: Concurrent tasks per GPU (int, default: 5)
- `--timeout`: Task timeout in seconds (int, default: 3600)

**Algorithm Selection**:
- `--task-types`: Algorithms to run (string, default: "fedprox scaffold fedov")
  - Options: fedavg, fedprox, scaffold, fedov, fedtree, solo, combined
  - Special: "all" runs all algorithms

**Step Control**:
- `--skip-sampling`: Skip pair sampling
- `--skip-preprocessing`: Skip data preprocessing
- `--skip-training`: Skip training
- `--force-rerun`: Delete existing data and start fresh

### Training Hyperparameters

**FedAvg/FedProx/SCAFFOLD**:
- `--local-epochs`: Local training epochs (int, default: 5)
- `--global-rounds`: Global communication rounds (int, default: 20)
- `--learning-rate`: Learning rate (float, default: 0.001)
- `--batch-size`: Batch size (int, default: 32)
- `--hidden-dims`: Hidden layer dimensions (list, default: [64, 32])

**FedProx Specific**:
- `--mu`: Proximal term coefficient (float, default: 0.01)

**Solo/Combined**:
- `--epochs`: Training epochs (int, default: 100)

**FedTree Specific**:
- `--max-depth`: Maximum tree depth (int, default: 6)
- `--num-trees`: Number of trees (int, default: 100)
- `--learning-rate`: Learning rate (float, default: 0.1)

## Output and Results

### Directory Structure

```
out/
├── autorun/
│   ├── sampled_pairs.json              # Sampled database pairs
│   ├── logs/                           # Execution logs
│   │   ├── sampling.log
│   │   ├── preprocessing.log
│   │   ├── training.log
│   │   └── {pair_id}_{task}_gpu{id}.log
│   └── results/                        # Training results
│       ├── {pair_id}_fedavg_results.json
│       ├── {pair_id}_fedprox_results.json
│       ├── {pair_id}_scaffold_results.json
│       ├── {pair_id}_fedov_results.json
│       ├── {pair_id}_solo_results.json
│       ├── {pair_id}_combined_results.json
│       └── execution_report.json
├── graphs/                             # Graph analysis outputs
│   ├── similarity_matrix.npy
│   ├── node_embeddings.npy
│   └── edge_properties.json
└── *.csv                               # Statistical summaries
```

### Result Files

#### Sampled Pairs

**File**: `out/autorun/sampled_pairs.json`

Contains list of sampled database pairs with metadata.

#### Training Results

**Files**: `out/autorun/results/{pair_id}_{algorithm}_results.json`

Each file contains:
- Pair metadata (database IDs, similarity)
- Hyperparameters used
- Performance metrics (MSE, RMSE, MAE, R²)
- Training time and convergence information

#### Execution Report

**File**: `out/autorun/results/execution_report.json`

Summary of entire execution:
- Total tasks attempted
- Completed vs. failed tasks
- Execution time statistics
- Resource utilization metrics

#### Preprocessing Summary

**File**: `data/auto/preprocessing_summary.json`

Summary of preprocessing:
- Number of pairs processed
- Success/failure counts
- Common failure reasons
- Data quality statistics

### Logs

#### Main Logs

- `out/autorun/logs/sampling.log`: Pair sampling execution log
- `out/autorun/logs/preprocessing.log`: Data preprocessing log
- `out/autorun/logs/training.log`: Overall training coordination log

#### Task Logs

- `out/autorun/logs/{pair_id}_{task_type}_gpu{id}.log`: Individual task logs

Each task log contains:
- Task start and end times
- Hyperparameters used
- Training progress (epoch-by-epoch)
- Final metrics
- Error messages (if failed)

### Figures

Generated figures are saved in `fig/` directory:

#### FL Performance Distributions

- `fl_performance_accuracy_distribution.pdf/.png`
- `fl_performance_precision_distribution.pdf/.png`
- `fl_performance_recall_distribution.pdf/.png`
- `fl_performance_f1_distribution.pdf/.png`

#### FL Delta (Improvement) Distributions

- `fl_delta_accuracy_distribution.pdf/.png`
- `fl_delta_precision_distribution.pdf/.png`
- `fl_delta_recall_distribution.pdf/.png`
- `fl_delta_f1_distribution.pdf/.png`

#### Graph Analysis

- `database_cluster.png`: Database similarity clustering
- `community_size_distribution_louvain_*.png`: Community detection
- `component_size_distribution_*.png`: Connected component analysis
- `degree_distribution_*.png`: Node degree distribution
- `small_cc_*.png`: Small connected component visualization

## Development Guide

### Project Layout

The project follows a modular architecture with clear separation of concerns:

- **Analysis Module**: Graph construction and similarity computation
- **Autorun Module**: Automated pipeline orchestration
- **Demo Module**: Manual experiment scripts
- **Model Module**: FL algorithm implementations
- **Summary Module**: Result aggregation and visualization
- **Utils Module**: Shared utilities and data structures

### Adding New FL Algorithms

To add a new federated learning algorithm:

1. **Create algorithm implementation** in `src/model/`:
   ```python
   # src/model/MyAlgorithm.py
   class MyAlgorithm:
       def __init__(self, ...):
           ...
       
       def train(self, client_data, ...):
           ...
       
       def aggregate(self, client_models):
           ...
   ```

2. **Create training script** in `src/demo/`:
   ```python
   # src/demo/train_myalgorithm.py
   from model.MyAlgorithm import MyAlgorithm
   
   def main():
       # Load data
       # Initialize algorithm
       # Train
       # Save results
   ```

3. **Create autorun trainer** in `src/autorun/`:
   ```python
   # src/autorun/myalgorithm.py
   # Parameterized version for automated pipeline
   ```

4. **Update GPU scheduler** in `src/autorun/gpu_scheduler.py`:
   - Add to `scripts` dictionary in `TrainingConfig`
   - Add to choices in `--task-types` argument

5. **Update orchestrator script** `run_automated_fl_validation.sh`:
   - Add to task types help text
   - Add to default task types if desired

### Code Style Guidelines

- Follow PEP 8 for Python code
- Use type hints for function signatures
- Include docstrings for all public functions
- Keep functions focused and modular
- Use meaningful variable names
- Add comments for complex logic

### Testing

Run the test suite:

```bash
# Run all tests
./src/test/test.sh

# Run specific tests
python src/test/bge_m3_similarity.py
python src/test/cluster.py
```

### Contributing Visualizations

To add new visualization scripts:

1. Create script in `src/summary/`
2. Follow existing naming convention: `plot_*.py`
3. Support both PDF and PNG output
4. Include command-line arguments for customization
5. Add usage example to this README

### Data Structure Extensions

When extending `wikidbs.py` data structures:

1. Maintain backward compatibility
2. Add serialization methods for new fields
3. Update schema validation logic
4. Document new fields in docstrings

## Troubleshooting

### Common Issues

#### Issue: No Database Pairs Found

**Symptoms**: Pair sampler returns empty list

**Solutions**:
1. Lower similarity thresholds:
   ```bash
   ./run_automated_fl_validation.sh --min-similarity 0.90
   ```
2. Reduce minimum row requirement:
   ```bash
   ./run_automated_fl_validation.sh --min-rows 50
   ```
3. Check graph similarity data exists:
   ```bash
   ls -lh data/graph/
   ```

#### Issue: GPU Out of Memory

**Symptoms**: CUDA out of memory errors during training

**Solutions**:
1. Reduce concurrent tasks per GPU:
   ```bash
   ./run_automated_fl_validation.sh --max-concurrent 2
   ```
2. Decrease batch size (modify training configs)
3. Use smaller model (reduce hidden dimensions)
4. Distribute across more GPUs:
   ```bash
   ./run_automated_fl_validation.sh --gpu-ids "0,1,2,3"
   ```

#### Issue: Preprocessing Failures

**Symptoms**: Many pairs fail preprocessing

**Solutions**:
1. Check preprocessing log:
   ```bash
   cat out/autorun/logs/preprocessing.log
   ```
2. Increase missing value tolerance:
   ```bash
   python src/autorun/data_preprocessor.py --max-missing-ratio 0.7
   ```
3. Decrease label variance requirement:
   ```bash
   python src/autorun/data_preprocessor.py --min-label-variance 0.001
   ```

#### Issue: Training Timeouts

**Symptoms**: Tasks timeout before completion

**Solutions**:
1. Increase timeout:
   ```bash
   ./run_automated_fl_validation.sh --timeout 7200
   ```
2. Reduce model complexity
3. Decrease number of epochs/rounds

#### Issue: Import Errors

**Symptoms**: `ModuleNotFoundError` when running scripts

**Solutions**:
1. Set PYTHONPATH:
   ```bash
   export PYTHONPATH=$PWD/src
   ```
2. Verify installation:
   ```bash
   pip install -e .
   ```
3. Check Python version:
   ```bash
   python --version  # Should be 3.12+
   ```

#### Issue: CUDA Not Available

**Symptoms**: Scripts report no GPU available

**Solutions**:
1. Verify CUDA installation:
   ```bash
   nvidia-smi
   ```
2. Check PyTorch CUDA support:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```
3. Reinstall PyTorch with correct CUDA version
4. Check CUDA_VISIBLE_DEVICES:
   ```bash
   echo $CUDA_VISIBLE_DEVICES
   ```

### Performance Optimization

#### Slow Graph Construction

- Use GPU-accelerated version: `compute_all_join_size_fast.py`
- Process databases in parallel
- Cache intermediate results

#### Slow Training

- Increase batch size (if memory allows)
- Use mixed precision training
- Reduce model complexity
- Parallelize across more GPUs

#### Slow Preprocessing

- Limit number of pairs processed
- Use faster file I/O (consider Parquet instead of CSV)
- Parallelize preprocessing (modify data_preprocessor.py)

### Debugging Tips

1. **Enable verbose logging**: Check individual task logs in `out/autorun/logs/`
2. **Test on small subset**: Use `--sample-size 10` for quick tests
3. **Run components individually**: Test pair_sampler, data_preprocessor separately
4. **Monitor resources**: Use `nvidia-smi` and `htop` to watch utilization
5. **Check data quality**: Inspect CSV files in `data/auto/` before training

### Getting Help

For additional support:

1. Check existing issues in the repository
2. Review log files for detailed error messages
3. Verify all requirements are met
4. Test with minimal configuration first
5. Consult component-specific README files

## Requirements

### System Requirements

- **Operating System**: Linux/Unix (tested on Ubuntu 20.04+)
- **CPU**: Multi-core processor (16+ cores recommended)
- **RAM**: 32GB minimum (64GB+ recommended for large-scale experiments)
- **GPU**: CUDA 12.1+ compatible GPU (NVIDIA A100/V100 recommended)
- **Storage**: 100GB+ free disk space
  - 20GB for raw data
  - 30GB for processed data
  - 50GB for results and logs

### Software Requirements

- **Python**: 3.12 or higher
- **CUDA Toolkit**: 12.1 or higher
- **CMake**: 3.15+ (for FedTree compilation)
- **GMP Library**: For FedTree homomorphic encryption
- **NTL Library**: For FedTree secure aggregation

### Python Dependencies

Core dependencies are managed in `pyproject.toml`:

**Deep Learning**:
- torch ~= 2.4.0
- torchvision
- torchdata
- dgl (with CUDA 12.1 support)

**GPU Acceleration**:
- cudf-cu12 == 25.4.*
- cuml-cu12 == 25.4.*
- cugraph-cu12 == 25.4.*
- cuspatial-cu12 == 25.4.*
- cucim-cu12 == 25.4.*
- pylibraft-cu12 == 25.4.*
- cuvs-cu12 == 25.4.*

**Data Processing**:
- numpy
- pandas
- scikit-learn
- orjson
- tqdm

**Visualization**:
- matplotlib
- seaborn
- pydot

**Machine Learning**:
- xgboost

### Data Requirements

- **WikiDB Database Dump**: Database files in `data/unzip/`
- **Schema Files**: Schema definitions in `data/schema/`
- **Graph Similarity Data**: Pre-computed similarities in `data/graph/`

### Hardware Recommendations

**For Development**:
- 1 GPU (GTX 1080 Ti or better)
- 16GB RAM
- 50GB storage

**For Production**:
- 4+ GPUs (A100 40GB or V100 32GB)
- 64GB+ RAM
- 200GB+ storage
- High-speed network for multi-node setups

**For Large-Scale Experiments**:
- 8+ GPUs across multiple nodes
- 128GB+ RAM per node
- 500GB+ shared storage
- InfiniBand or 10GbE networking

## License

This project is licensed under the Apache License 2.0.

```
Copyright 2025 Zhaomin Wu, Ziyang Wang, Bingsheng He

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

See the [LICENSE](LICENSE) file for the full license text.

### External Components

This project incorporates or references external frameworks:

- **FedOV**: Licensed under its own terms (see `FedOV/LICENSE`)
- **FedTree**: Licensed under its own terms (see `FedTree/LICENSE`)
- **NIID-Bench**: Licensed under its own terms (see `NIID-Bench/LICENSE`)

Please refer to individual LICENSE files in each framework directory for specific terms.

## Acknowledgments

This project builds upon several open-source frameworks and libraries. We acknowledge the developers and maintainers of PyTorch, RAPIDS cuML, DGL, and the federated learning research community for their invaluable contributions.
