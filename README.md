# WikiDBs - Database Similarity Graph and Federated Learning

A graph-based database analysis system for WikiData databases that performs similarity analysis, schema matching, and federated learning experiments.

## Project Overview

WikiDBs constructs a similarity graph over databases extracted from WikiData, then uses this graph for federated learning validation experiments. The pipeline consists of:

1. **Schema Embedding**: Convert database schemas to vector embeddings using BGE-M3
2. **Similarity Graph**: Compute pairwise similarity and build a graph structure
3. **Federated Learning**: Train and evaluate FL algorithms on similar database pairs

## Quick Start

```bash
# Install dependencies
pip install -e .

# Set environment for running scripts
export PYTHONPATH=src
```

## Source Directory Structure

```
src/
├── preprocess/          # Data preprocessing and embedding pipeline
│   ├── GitTables/       # GitTables dataset support
│   └── summary/         # Ablation study analysis
├── model/               # Neural network models (FL algorithms, embedders)
├── demo/                # Training scripts for FL experiments
├── autorun/             # Automated FL validation system
├── analysis/            # Graph analysis and visualization
├── baseline/            # Baseline methods (SANTOS, string matching)
├── summary/             # Results aggregation and plotting
├── train/               # Model training utilities
├── test/                # Test scripts
└── utils/               # Shared utilities
```

---

## Preprocessing Pipeline (`src/preprocess/`)

Unified, extensible pipeline for data processing from raw databases to embeddings and graphs.

### Core Modules

| Module | Description |
|--------|-------------|
| `config.py` | Centralized hyperparameter configuration |
| `schema_serializer.py` | Serialize database schemas with ablation modes |
| `triplet_generator.py` | Generate contrastive learning triplets |
| `embedding_generator.py` | Batch embedding generation using BGE-M3 |
| `similarity_computer.py` | All-pairs cosine similarity computation |
| `faiss_similarity_computer.py` | GPU-accelerated similarity with FAISS |
| `edge_filter.py` | Filter edges by similarity threshold |
| `graph_builder.py` | Build DGL graphs from filtered edges |
| `trainer.py` | Contrastive model training |
| `evaluator.py` | Model evaluation with ROC/AUC metrics |

### Schema Serialization Modes

The `SchemaSerializer` supports three modes for ablation studies:

```python
from preprocess import SchemaSerializer

# Full mode (default) - schema + sample values
serializer = SchemaSerializer(mode="full", sample_size=3)

# Schema only - table and column names only
serializer = SchemaSerializer(mode="schema_only")

# Data only - sample values only
serializer = SchemaSerializer(mode="data_only", sample_size=3)
```

### Configuration

All hyperparameters are centralized in `PreprocessConfig`:

```python
from preprocess import PreprocessConfig

config = PreprocessConfig(
    serialization_mode="full",    # "schema_only", "data_only", "full"
    sample_size=3,                # representative values per column
    num_negatives=6,              # negatives per triplet
    similarity_threshold=0.6713   # edge filtering threshold
)
```

### End-to-End Pipeline

```bash
# Run complete preprocessing pipeline
./src/preprocess/run_preprocess.sh --mode full --sample-size 3 --num-negatives 6
```

### Ablation Scripts

| Script | Description |
|--------|-------------|
| `run_ablation_encoder_model.sh` | Compare different encoder models |
| `run_ablation_sample_size.sh` | Vary sample size per column |
| `run_ablation_num_negatives.sh` | Vary number of negative samples |
| `run_ablation_serialization_mode.sh` | Compare schema/data/full modes |
| `run_ablation_threshold.sh` | Analyze similarity threshold effects |

### GitTables Support (`src/preprocess/GitTables/`)

Self-supervised preprocessing pipeline for the GitTables dataset:

| Module | Description |
|--------|-------------|
| `table_extractor.py` | Extract tables from GitTables parquet files |
| `synthetic_splitter.py` | Generate synthetic partitions (vertical/horizontal) |
| `table_serializer.py` | Serialize tables to text format |
| `triplet_generator.py` | Generate self-supervised triplets |
| `trainer_gittables.py` | Train on GitTables triplets |
| `evaluator_gittables.py` | Evaluate on GitTables |
| `run_gittables_preprocess.sh` | Complete GitTables pipeline |

---

## Models (`src/model/`)

| Module | Description |
|--------|-------------|
| `BGEEmbedder.py` | BGE-M3 embedding model with training and inference |
| `WKDataset.py` | WikiDB dataset loader |
| `FedAvg.py` | Federated Averaging implementation |
| `FedProx.py` | FedProx with proximal term regularization |
| `SCAFFOLD.py` | SCAFFOLD with variance reduction |
| `FedOV.py` | One-shot vertical federated learning |
| `FedGNN.py` | Graph Neural Network for FL |
| `FedGTA.py` | Graph-based FL with attention |
| `SplitNN.py` | Split learning for vertical FL |
| `column_encoder.py` | Column-level encoding model |

---

## Demo/Training Scripts (`src/demo/`)

Training scripts for federated learning experiments:

| Script | Type | Description |
|--------|------|-------------|
| `train_fedavg.py` | Horizontal FL | Federated Averaging |
| `train_fedprox.py` | Horizontal FL | FedProx algorithm |
| `train_scaffold.py` | Horizontal FL | SCAFFOLD algorithm |
| `train_fedov.py` | One-shot FL | One-shot vertical FL |
| `train_splitnn.py` | Vertical FL | Split neural network |
| `train_fedgnn.py` | Graph FL | Graph neural network FL |
| `train_fedgta.py` | Graph FL | Graph FL with attention |
| `train_fedtree.py` | Tree FL | Tree-based FL |

### Supporting Modules

| Module | Description |
|--------|-------------|
| `prepare_horizontal_data.py` | Prepare data for horizontal FL |
| `prepare_vertical_data.py` | Prepare data for vertical FL |
| `centralized_training.py` | Centralized baseline training |
| `run_centralized_training.py` | Run centralized experiments |
| `run_individual_clients.py` | Run individual client training |

### Usage

```bash
# Single FL experiment
python src/demo/train_fedavg.py --seed 0 --databases 02799 79665

# With custom parameters
python src/demo/train_fedprox.py --mu 0.01 --global-rounds 20 --local-epochs 5
```

---

## Automated FL Validation (`src/autorun/`)

Comprehensive pipeline for automated federated learning validation experiments.

| Module | Description |
|--------|-------------|
| `pair_sampler.py` | Sample database pairs by similarity |
| `data_preprocessor.py` | Prepare FL data from database pairs |
| `semantic_data_preprocessor.py` | Semantic column matching preprocessor |
| `gpu_scheduler.py` | Multi-GPU task scheduling |
| `fedavg.py` / `fedprox.py` / `scaffold.py` | Parameterized FL trainers |
| `solo.py` | Individual client training baseline |
| `fedov.py` | One-shot FL trainer |

### Usage

```bash
# Run automated validation
./run_automated_fl_validation.sh \
    --min-similarity 0.98 \
    --sample-size 200 \
    --num-gpus 4
```

See `src/autorun/README.md` for detailed documentation.

---

## Analysis (`src/analysis/`)

Graph analysis and visualization tools:

| Module | Description |
|--------|-------------|
| `CommunityDetection.py` | Community detection algorithms |
| `EdgeProperties.py` | Edge property analysis |
| `NodeProperties.py` | Node property analysis |
| `NodeSemantic.py` | Semantic node analysis |
| `NodeStatistical.py` | Statistical node analysis |
| `WikiDBSubgraph.py` | Subgraph extraction and analysis |
| `visualize_graph.py` | Graph visualization |
| `estimate_dirichlet_alpha.py` | Data heterogeneity estimation |
| `find_sparse_subgraph_and_task.py` | Find suitable FL subgraphs |
| `collect_similar_database_pairs.py` | Collect similar database pairs |
| `compute_all_join_size.py` | Compute join sizes between databases |

---

## Baseline Methods (`src/baseline/`)

### SANTOS (`src/baseline/santos/`)

SANTOS knowledge base synthesis and evaluation:

| Module | Description |
|--------|-------------|
| `synthesize_kb.py` | Synthesize SANTOS knowledge base |
| `score_pairs.py` | Score database pairs with SANTOS |
| `evaluate_auc.py` | Evaluate AUC metrics |
| `run_santos_evaluation.sh` | Run SANTOS evaluation pipeline |

### String Matching (`src/baseline/string_match/`)

Simple string matching baseline:

| Module | Description |
|--------|-------------|
| `string_match_evaluator.py` | String-based similarity evaluation |
| `run_string_match_baseline.sh` | Run string matching baseline |

---

## Summary & Visualization (`src/summary/`)

Results aggregation, table generation, and plotting:

### Table Generation

| Script | Description |
|--------|-------------|
| `run_and_generate_horizontal_table.py` | Generate horizontal FL results tables |
| `run_and_generate_vertical_table.py` | Generate vertical FL results tables |
| `run_and_generate_ml_model_tables.py` | Generate ML model comparison tables |
| `generate_federated_learning_tables.py` | Generate FL summary tables |
| `print_auto_horizontal.py` | Print automated horizontal results |
| `print_gittables_metrics.py` | Print GitTables evaluation metrics |
| `print_predictions_metrics.py` | Print prediction metrics |

### Plotting

| Script | Description |
|--------|-------------|
| `plot_auto_horizontal.py` | Plot automated horizontal FL results |
| `plot_raw_vs_semantic.py` | Compare raw vs semantic column matching |
| `plot_size_vs_gain.py` | Dataset size vs FL gain analysis |
| `plot_component_distribution.py` | Graph component distribution |
| `plot_graph.py` | Graph visualization |
| `plot_sim_distribution.py` | Similarity distribution plots |
| `plot_test_results.py` | Test results visualization |

---

## Training Utilities (`src/train/`)

| Script | Description |
|--------|-------------|
| `train_bge_softmax.py` | Train BGE-M3 with softmax loss |
| `split_dataset.py` | Split datasets for training |
| `build_graph.py` | Build graph from embeddings |
| `embed_full.sh` | Full embedding generation |

---

## Utilities (`src/utils/`)

| Module | Description |
|--------|-------------|
| `schema_formatter.py` | Format database schemas to text |
| `wikidbs.py` | WikiDB helper functions |
| `load_from_uci.py` | Load UCI datasets |
| `llm_judger.py` | LLM-based evaluation |
| `print_schema.py` | Print schema information |

---

## Requirements

- Python 3.10+
- PyTorch with CUDA support
- DGL (Deep Graph Library)
- sentence-transformers
- transformers
- FAISS (optional, for GPU-accelerated similarity)

## License

[Add license information]
