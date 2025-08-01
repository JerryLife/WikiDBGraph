# WikiDBGraph: Graph-based Database Analysis and Federated Learning System

WikiDBGraph is a comprehensive system for analyzing and visualizing WikiData database relationships, with advanced federated learning capabilities for distributed machine learning on database pairs.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Federated Learning Experiments](#federated-learning-experiments)
- [Graph Analysis Pipeline](#graph-analysis-pipeline)
- [Automated Validation System](#automated-validation-system)
- [Training and Model Building](#training-and-model-building)
- [Analysis and Visualization](#analysis-and-visualization)
- [Architecture Overview](#architecture-overview)
- [Results and Figures](#results-and-figures)
- [Requirements](#requirements)
- [Citation](#citation)

## Project Overview

WikiDBGraph is a graph-based database analysis system that performs similarity analysis and matching between WikiData databases. The project uses PyTorch, DGL for graph operations, and RAPIDS cuML for GPU-accelerated machine learning. It features comprehensive federated learning algorithms for distributed training across database pairs.

### Key Features

- **Graph-based Analysis**: Structural and semantic analysis of database relationships
- **Federated Learning**: Multiple FL algorithms (FedAvg, FedProx, SCAFFOLD, FedOV, SplitNN)
- **Automated Validation**: Large-scale automated experiment pipeline
- **GPU Acceleration**: RAPIDS cuML ecosystem for high-performance computing
- **Comprehensive Visualization**: Rich plotting and analysis capabilities

## Installation

This project requires Python 3.12+ and CUDA 12.1+ for GPU acceleration.

```bash
# Clone the repository
git clone <repository-url>
cd WikiDBGraph

# Install dependencies
pip install -e .

# Set up environment variables for running scripts
cd src
export PYTHONPATH=.
```

### Dependencies

The project depends on the following main packages:
- torch ~=2.4.0
- torchvision, torchdata
- dgl (with CUDA 12.1 support)
- numpy, pandas, matplotlib, scikit-learn
- RAPIDS cuML ecosystem (cudf, cuml, cugraph, etc.)
- xgboost, seaborn, orjson, tqdm

See `pyproject.toml` for the complete dependency list.

## Project Structure

The project is organized into the following main directories:

```
src/
├── analysis/           # Graph analysis and database comparison
│   ├── build_raw_graph.py              # Graph construction from schemas
│   ├── CommunityDetection.py           # Graph clustering algorithms
│   ├── NodeSemantic.py                 # Semantic analysis of nodes
│   ├── analysis_graph_components.py    # Component analysis
│   └── compute_all_join_size_fast.py   # Join cardinality estimation
├── model/              # Neural networks and federated learning
│   ├── BGEEmbedder.py                  # BGE embedding model wrapper
│   ├── FedAvg.py                       # FedAvg algorithm
│   ├── FedProx.py                      # FedProx with proximal regularization
│   ├── SCAFFOLD.py                     # SCAFFOLD with control variates
│   ├── FedOV.py                        # FedOV one-shot federated learning
│   ├── SplitNN.py                      # Split Neural Network
│   └── column_encoder.py               # Column-level encoding
├── demo/               # Federated learning experiments
│   ├── train_fedavg.py                 # FedAvg training script
│   ├── train_fedprox.py                # FedProx training script
│   ├── train_scaffold.py               # SCAFFOLD training script
│   ├── train_fedov.py                  # FedOV training script
│   ├── train_splitnn.py                # SplitNN training script
│   ├── centralized_training.py         # Centralized baseline
│   └── run_individual_clients.py       # Solo model training
├── autorun/            # Automated validation system
│   ├── pair_sampler.py                 # Database pair sampling
│   ├── data_preprocessor.py            # Automated data preparation
│   ├── gpu_scheduler.py                # Multi-GPU task scheduling
│   └── fedavg.py                       # Enhanced FedAvg trainer
├── summary/            # Experiment aggregation and visualization
│   ├── run_and_generate_horizontal_table.py    # Horizontal FL tables
│   ├── run_and_generate_vertical_table.py      # Vertical FL tables
│   ├── plot_auto_horizontal.py                 # FL performance plots
│   └── Various plotting scripts for analysis
├── script/             # Utility scripts for data processing
│   ├── train_bge_softmax.py            # BGE model training
│   ├── embed_full.sh                   # Embedding generation
│   └── untrained_version/              # Baseline matcher
└── utils/              # Helper functions and utilities
    ├── wikidbs.py                      # Core data structures
    └── schema_formatter.py             # Schema processing
```

## Quick Start

### Basic Graph Analysis

```bash
# Run graph analysis components
cd src && export PYTHONPATH=. && python -u analysis/analysis_graph_components.py

# Generate plots and summaries
cd src && export PYTHONPATH=. && python -u summary/plot_datasets_info.py
cd src && export PYTHONPATH=. && python -u summary/plot_graph.py
```

### Simple Federated Learning Experiment

```bash
# Train FedAvg on a database pair
python src/demo/train_fedavg.py --seed 0 --databases 02799 79665

# Compare with individual client training
python src/demo/run_individual_clients.py -m nn --seed 0

# Run centralized baseline
python src/demo/centralized_training.py
```

## Federated Learning Experiments

### Individual Algorithm Training

Run individual FL algorithms with seeded runs for reproducibility:

```bash
# Horizontal Federated Learning
python src/demo/train_fedavg.py --seed 0 --databases 02799 79665
python src/demo/train_fedprox.py --seed 1 --databases 02799 79665  
python src/demo/train_scaffold.py --seed 2 --databases 02799 79665
python src/demo/train_fedov.py --seed 3 --databases 02799 79665

# Vertical Federated Learning
python src/demo/train_splitnn.py --seed 4

# Individual client training (Solo models)
python src/demo/run_individual_clients.py -m nn --seed 0

# Centralized baseline training
python src/demo/centralized_training.py
```

### Comprehensive Experiment Generation

Generate LaTeX tables with automatic missing experiment detection:

```bash
# Generate horizontal FL comparison tables
python src/summary/run_and_generate_horizontal_table.py --show-std --num-seeds 5

# Generate vertical FL comparison tables  
python src/summary/run_and_generate_vertical_table.py --show-std --num-seeds 5

# Generate ML model comparison tables
python src/summary/run_and_generate_ml_model_tables.py --runs 5
```

### File Naming Convention

All FL experiment results use predictable naming for reproducibility:
- Format: `{algorithm}_{database_ids}_seed{N}.json`
- Example: `fedavg_02799_79665_seed0.json`
- Enables `--skip-runs` functionality in batch execution scripts

## Graph Analysis Pipeline

### Data Flow

1. **Schema Loading**: Database schemas are loaded using `utils/wikidbs.py` classes
2. **Graph Construction**: `analysis/build_raw_graph.py` creates graph structures
3. **Embedding Generation**: BGE models compute node and edge embeddings
4. **Similarity Analysis**: GPU-accelerated operations perform matching
5. **Visualization**: Results are exported and visualized

### Key Analysis Components

```bash
# Build graph from database schemas
cd src && export PYTHONPATH=. && python -u analysis/build_raw_graph.py

# Compute join sizes (GPU-accelerated)
cd src && export PYTHONPATH=. && python -u analysis/compute_all_join_size_fast.py
```

## Automated Validation System

The automated system provides a comprehensive pipeline for large-scale federated learning validation experiments.

### Basic Usage

```bash
# Run with default parameters (200 pairs, similarity 0.98-1.0)
./run_automated_fl_validation.sh
```

### Custom Parameters

```bash
# Custom similarity range and sample size
./run_automated_fl_validation.sh \
    --min-similarity 0.95 \
    --max-similarity 0.99 \
    --sample-size 100 \
    --num-gpus 4 \
    --max-concurrent 2
```

### Resume/Skip Steps

```bash
# Resume from last successful step
./run_automated_fl_validation.sh --resume

# Skip steps if data already exists
./run_automated_fl_validation.sh --skip-sampling --skip-preprocessing
```

### System Components

1. **Pair Sampler**: Filters database pairs by similarity threshold and data quality
2. **Data Preprocessor**: Joins tables, selects labels, cleans data, creates train/test splits
3. **GPU Scheduler**: Manages parallel execution across multiple GPUs with load balancing
4. **Enhanced Trainers**: Parameterized versions supporting FedAvg, Solo, and Combined training

### Output Structure

```
out/autorun/
├── sampled_pairs.json              # Sampled database pairs
├── logs/                           # All execution logs
└── results/                        # Training results with comprehensive metrics

data/auto/
└── {db_id1}_{db_id2}/             # Individual pair data
    ├── config.json                 # Pair configuration
    └── train/test CSV files        # Processed datasets
```

## Training and Model Building

### BGE Model Training

```bash
# Train BGE softmax model
cd src && export PYTHONPATH=. && python -u script/train_bge_softmax.py

# Generate embeddings
./src/script/embed_full.sh
```

### Baseline Matcher

```bash
# Run untrained baseline matcher
cd src/script/untrained_version && ./run_all.sh
```

## Analysis and Visualization

The system generates comprehensive visualizations and analysis results:

### Available Figures

- **Graph Analysis**: Component distributions, community detection, degree distributions
- **FL Performance**: Accuracy, precision, recall, F1 score distributions  
- **FL Comparisons**: Delta improvements between federated and solo training
- **Database Clustering**: Similarity-based database relationships

### Generate Visualizations

```bash
# Plot FL performance distributions
cd src && export PYTHONPATH=. && python -u summary/plot_auto_horizontal.py

# Generate comprehensive analysis plots
cd src && export PYTHONPATH=. && python -u summary/plot_datasets_info.py
```

## Architecture Overview

### Federated Learning Pipeline

1. **Data Preparation**: `prepare_horizontal_data.py` splits databases into client data
2. **Training Phase**: FL algorithms train on distributed data with consistent seeding
3. **Evaluation Phase**: Models tested on held-out test sets with unified label encoding
4. **Aggregation Phase**: Results from multiple seed runs aggregated (mean ± std)
5. **Table Generation**: LaTeX tables generated with automatic missing experiment detection

### GPU Dependencies

The project heavily relies on RAPIDS cuML ecosystem:
- cuDF, cuML, cuGraph for GPU-accelerated data processing
- Requires CUDA 12.1+ compatible GPU
- DGL with CUDA support for graph neural networks

## Results and Figures

Generated figures are available in the `fig/` directory:

- `fl_performance_*_distribution.{pdf,png}`: FL algorithm performance distributions
- `fl_delta_*_distribution.{pdf,png}`: Improvement over solo training
- `community_*.png`: Graph community analysis
- `database_cluster.png`: Database similarity clustering

LaTeX tables are generated in the `results/tables/` directory with comprehensive performance comparisons.

## Requirements

### System Requirements
- Linux/Unix environment  
- Python >= 3.12
- CUDA 12.1+ (for GPU acceleration)
- Sufficient disk space for processed data and results

### Data Requirements
- WikiDB database dump in `data/` directory
- Graph outputs saved to `out/graphs/` directory
- Figures generated in `fig/` directory

### Key Development Guidelines

- Run any Python script from `src/` directory with `export PYTHONPATH=.`
- Every execution should be done from project root
- Many scripts expect data in `data/` directory with specific structure

## Citation

If you use WikiDBGraph in your research, please cite:

```bibtex
@article{wu2025wikidbgraph,
  title={WikiDBGraph: Large-Scale Database Graph of Wikidata for Collaborative Learning},
  author={Wu, Zhaomin and Wang, Ziyang and He, Bingsheng},
  journal={arXiv preprint arXiv:2505.16635},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.