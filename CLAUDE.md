# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WikiGraph is a graph-based database analysis system that performs similarity analysis and matching between WikiData databases. The project uses PyTorch, DGL for graph operations, and RAPIDS cuML for GPU-accelerated machine learning.

## Development Environment Setup

```bash
# Install dependencies (requires CUDA 12.1+ and Python 3.12+)
pip install -e .

# Set up environment variables for running scripts
cd src
export PYTHONPATH=.
```

## Common Commands

### Federated Learning Experiments
```bash
# Individual FL algorithm training (with seeded runs for reproducibility)
python src/demo/train_fedavg.py --seed 0 --databases 02799 79665
python src/demo/train_fedprox.py --seed 1 --databases 02799 79665  
python src/demo/train_scaffold.py --seed 2 --databases 02799 79665
python src/demo/train_fedov.py --seed 3 --databases 02799 79665
python src/demo/train_splitnn.py --seed 4

# Individual client training (Solo models)
python src/demo/run_individual_clients.py -m nn --seed 0

# Centralized baseline training
python src/demo/centralized_training.py

# Generate comprehensive LaTeX tables (automatically runs missing experiments)
python src/summary/run_and_generate_horizontal_table.py --show-std --num-seeds 5
python src/summary/run_and_generate_vertical_table.py --show-std --num-seeds 5

# Run all experiments with multiple seeds
python src/summary/run_and_generate_tables.py --runs 5
```

### Training and Model Building
```bash
# Train BGE softmax model
cd src && export PYTHONPATH=. && python -u script/train_bge_softmax.py

# Generate embeddings
./src/script/embed_full.sh

# Run untrained baseline matcher
cd src/script/untrained_version && ./run_all.sh
```

### Analysis and Visualization
```bash
# Run graph analysis components
cd src && export PYTHONPATH=. && python -u analysis/analysis_graph_components.py

# Generate plots and summaries
cd src && export PYTHONPATH=. && python -u summary/plot_datasets_info.py
cd src && export PYTHONPATH=. && python -u summary/plot_graph.py

# Compute join sizes (fast GPU version)
cd src && export PYTHONPATH=. && python -u analysis/compute_all_join_size_fast.py
```

### Testing
```bash
# Run tests (requires test environment setup)
cd src/test && ./test.sh
cd src/summary && ./test.sh
```

## Architecture Overview

### Core Components

- **`src/model/`** - Neural network models and federated learning algorithms
  - `BGEEmbedder.py` - BGE (BAAI General Embedding) model wrapper
  - `WKDataset.py` - Dataset handling for WikiData
  - `column_encoder.py` - Column-level encoding
  - `FedAvg.py` - FedAvg federated learning implementation
  - `FedProx.py` - FedProx federated learning with proximal regularization
  - `SCAFFOLD.py` - SCAFFOLD federated learning with control variates
  - `FedOV.py` - FedOV one-shot federated learning with outlier detection
  - `SplitNN.py` - Split Neural Network for vertical federated learning

- **`src/demo/`** - Federated learning experiments and training scripts
  - `train_fedavg.py` - FedAvg horizontal FL training
  - `train_fedprox.py` - FedProx horizontal FL training
  - `train_scaffold.py` - SCAFFOLD horizontal FL training
  - `train_fedov.py` - FedOV horizontal FL training
  - `train_splitnn.py` - SplitNN vertical FL training
  - `run_individual_clients.py` - Individual client (Solo) model training
  - `centralized_training.py` - Centralized baseline training
  - `prepare_horizontal_data.py` - Data preparation for horizontal FL

- **`src/summary/`** - Experiment aggregation and LaTeX table generation
  - `run_and_generate_horizontal_table.py` - Horizontal FL table generator (auto-runs missing experiments)
  - `run_and_generate_vertical_table.py` - Vertical FL table generator (auto-runs missing experiments)
  - `run_and_generate_tables.py` - Master script for running all FL experiments
  - `generate_latex_tables.py` - Legacy table generator
  - Various plotting scripts for analysis results

- **`src/analysis/`** - Graph analysis and database comparison
  - `build_raw_graph.py` - Graph construction from database schemas
  - `CommunityDetection.py` - Graph clustering algorithms
  - `NodeSemantic.py` - Semantic analysis of graph nodes
  - `compute_all_join_size*.py` - Join cardinality estimation

- **`src/utils/`** - Utilities and data structures
  - `wikidbs.py` - Core data structures for Schema, Table, Column, ForeignKey
  - `schema_formatter.py` - Schema processing utilities

### Data Flow

#### Graph Analysis Pipeline
1. Database schemas are loaded and processed using classes in `utils/wikidbs.py`
2. Graph structures are built from schema relationships
3. Node and edge embeddings are computed using BGE models
4. Similarity analysis is performed using GPU-accelerated operations
5. Results are visualized and exported for analysis

#### Federated Learning Pipeline
1. **Data Preparation**: `prepare_horizontal_data.py` splits databases into client data
2. **Training Phase**: FL algorithms train on distributed data with consistent seeding
3. **Evaluation Phase**: Models tested on held-out test sets with unified label encoding
4. **Aggregation Phase**: Results from multiple seed runs aggregated (mean ± std)
5. **Table Generation**: LaTeX tables generated with automatic missing experiment detection

#### File Naming Convention
All FL experiment results use predictable naming for reproducibility:
- Format: `{algorithm}_{database_ids}_seed{N}.json`
- Example: `fedavg_02799_79665_seed0.json`
- Enables `--skip-runs` functionality in batch execution scripts

### GPU Dependencies

The project heavily relies on RAPIDS cuML ecosystem:
- cuDF, cuML, cuGraph for GPU-accelerated data processing
- Requires CUDA 12.1+ compatible GPU
- DGL with CUDA support for graph neural networks

## Key Scripts for Development

- Run any Python script from `src/` directory with `export PYTHONPATH=.`
- Many scripts expect data in `data/` directory with specific structure
- Graph outputs are saved to `out/graphs/` directory
- Figures are generated in `fig/` directory

## Results Directory Structure

```
results/
├── horizontal/                 # Horizontal FL experiment results
│   ├── fedavg_02799_79665_seed*.json
│   ├── fedprox_02799_79665_seed*.json
│   ├── scaffold_02799_79665_seed*.json
│   ├── fedov_02799_79665_seed*.json
│   └── individual_clients_02799_79665_seed*.json
├── vertical/                   # Vertical FL experiment results
│   └── splitnn_*_seed*.json
├── primary_client/             # Primary client experiment results
│   └── primary_client_*_seed*.json
└── tables/                     # Generated LaTeX tables and aggregated results
    ├── horizontal_fl_*_aggregated.json
    ├── horizontal_fl_*_table.tex
    ├── vertical_fl_*_aggregated.json
    └── vertical_fl_*_table.tex
```

## Development Guidelines

- Every execution should be done on project root, should not use cd before executing script