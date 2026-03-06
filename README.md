# WikiDBGraph

**WikiDBGraph: A Data Management Benchmark Suite for Collaborative Learning over Database Silos**

> Zhaomin Wu\*, Ziyang Wang\*, Bingsheng He — National University of Singapore
> (\* equal contribution)

WikiDBGraph constructs a similarity graph over real-world relational databases extracted from Wikidata, and uses the graph to benchmark **collaborative learning (CL)** algorithms across database silos. The pipeline covers schema embedding, contrastive fine-tuning, similarity graph construction, and automated federated learning (FL) validation.

📄 [Technical Report](WikiDBGraph_Technical_Report.pdf)

---

## Repository Structure

```
WikiDBGraph/
├── src/
│   ├── preprocess/          # Schema embedding, graph construction, contrastive training
│   │   ├── GitTables/       # Self-supervised extension to GitTables
│   │   └── summary/         # Ablation study plots and tables
│   ├── model/               # FL algorithm implementations
│   ├── demo/                # Single-experiment training scripts
│   ├── autorun/             # Automated large-scale FL validation pipeline
│   ├── analysis/            # Graph analysis and visualization
│   ├── summary/             # Results aggregation and plotting
│   ├── train/               # Embedding training utilities
│   ├── utils/               # Shared helpers
│   └── test/                # Test scripts and notebooks
├── baseline/                # External baseline methods (git submodules)
│   ├── santos/
│   ├── deepjoin/
│   ├── starmie/
│   ├── FedGTA/
│   └── SFL-Structural-Federated-Learning/
├── data/                    # Data directory (not tracked)
│   ├── unzip/               # WikiDB raw database dump
│   ├── graph/               # Precomputed similarity graph
│   ├── schema/              # Schema embeddings
│   ├── auto/                # Preprocessed FL pair data
│   ├── auto_semantic/       # Semantically matched FL pair data
│   └── GitTables/           # GitTables dataset
├── fig/                     # Figures
├── tables/                  # LaTeX result tables
├── out/                     # Experiment outputs and logs
├── run_automated_fl_validation.sh
├── run_semantic_auto_fl_validation.sh
├── pyproject.toml
└── WikiDBGraph_Technical_Report.pdf
```

---

## Installation

```bash
pip install -e .
export PYTHONPATH=src
```

**Requirements**: Python 3.12+, PyTorch 2.4 with CUDA, DGL, sentence-transformers, FAISS (optional, for GPU-accelerated similarity).

---

## Pipeline Overview

```
Raw WikiDB databases
        │
        ▼
1. Schema Serialization     src/preprocess/schema_serializer.py
        │
        ▼
2. Contrastive Embedding    src/preprocess/embedding_generator.py  +  trainer.py
        │
        ▼
3. Similarity Graph         src/preprocess/similarity_computer.py  →  graph_builder.py
        │
        ▼
4. FL Validation            src/autorun/  (automated, multi-GPU)
        │
        ▼
5. Analysis & Reporting     src/analysis/  +  src/summary/
```

---

## Preprocessing (`src/preprocess/`)

| Module | Description |
|--------|-------------|
| `config.py` | Centralized hyperparameter configuration |
| `schema_serializer.py` | Serialize database schemas; supports `full`, `schema_only`, `data_only` modes |
| `triplet_generator.py` | Generate contrastive learning triplets |
| `negative_pool_generator.py` | Hard-negative pool construction |
| `embedding_generator.py` | Batch embedding generation with BGE-M3 |
| `similarity_computer.py` | All-pairs cosine similarity |
| `faiss_similarity_computer.py` | GPU-accelerated similarity with FAISS |
| `edge_filter.py` | Filter edges by similarity threshold |
| `graph_builder.py` | Build DGL graph from filtered edges |
| `trainer.py` | Contrastive fine-tuning (InfoNCE loss) |
| `evaluator.py` | ROC/AUC evaluation |

### Configuration

```python
from preprocess import PreprocessConfig

config = PreprocessConfig(
    serialization_mode="full",    # "schema_only" | "data_only" | "full"
    sample_size=3,                # representative values per column
    num_negatives=6,              # negatives per triplet
    similarity_threshold=0.6713   # edge filtering threshold
)
```

### Schema Serialization Modes

```python
from preprocess import SchemaSerializer

serializer = SchemaSerializer(mode="full", sample_size=3)   # schema + sample values
serializer = SchemaSerializer(mode="schema_only")            # table/column names only
serializer = SchemaSerializer(mode="data_only", sample_size=3)  # sample values only
```

### Run the Full Preprocessing Pipeline

```bash
./src/preprocess/run_preprocess.sh --mode full --sample-size 3 --num-negatives 6
```

### Ablation Scripts

| Script | Description |
|--------|-------------|
| `run_ablation_encoder_model.sh` | Compare encoder models |
| `run_ablation_sample_size.sh` | Vary per-column sample size |
| `run_ablation_num_negatives.sh` | Vary number of negatives |
| `run_ablation_serialization_mode.sh` | Compare serialization modes |
| `run_ablation_threshold.sh` | Threshold sensitivity analysis |

### GitTables Extension (`src/preprocess/GitTables/`)

Self-supervised extension to heterogeneous web tables using synthetic partitioning:

| Module | Description |
|--------|-------------|
| `table_extractor.py` | Extract tables from GitTables parquet files |
| `gittables_dataset.py` | Dataset loader |
| `synthetic_splitter.py` | Vertical / horizontal partition generation |
| `table_serializer.py` | Table-to-text serialization |
| `triplet_generator.py` | Self-supervised triplet generation |
| `trainer_gittables.py` | Fine-tune on GitTables triplets |
| `evaluator_gittables.py` | Evaluate on GitTables |

```bash
./src/preprocess/GitTables/run_gittables_preprocess.sh
```

---

## Models (`src/model/`)

| Module | Description |
|--------|-------------|
| `col_embedding_model.py` | BGE-M3-based column embedding model |
| `column_encoder.py` | Column-level encoder |
| `WKDataset.py` | WikiDB dataset loader |
| `FedAvg.py` | Federated Averaging |
| `FedProx.py` | FedProx (proximal regularization) |
| `SCAFFOLD.py` | SCAFFOLD (variance reduction) |
| `FedOV.py` | One-shot vertical FL |
| `SplitNN.py` | Split neural network |
| `cal_sim.py` | Similarity calculation utilities |

---

## Single-Experiment Training (`src/demo/`)

| Script | FL Type | Description |
|--------|---------|-------------|
| `train_fedavg.py` | Horizontal | Federated Averaging |
| `train_fedprox.py` | Horizontal | FedProx |
| `train_scaffold.py` | Horizontal | SCAFFOLD |
| `train_fedov.py` | One-shot | One-shot vertical FL |
| `train_splitnn.py` | Vertical | Split neural network |
| `train_fedtree.py` | Tree | Tree-based FL |

```bash
# Run a single horizontal FL experiment
python src/demo/train_fedavg.py --seed 0 --databases 02799 79665

# Prepare data manually
python src/demo/prepare_horizontal_data.py
python src/demo/prepare_vertical_data.py
```

---

## Automated FL Validation (`src/autorun/`)

Large-scale, multi-GPU pipeline that samples database pairs by graph similarity, preprocesses them, and runs FL experiments in parallel.

| Module | Description |
|--------|-------------|
| `pair_sampler.py` | Sample pairs by similarity range |
| `data_preprocessor.py` | Auto-join, label selection, train/test split |
| `fedavg.py` / `fedprox.py` / `scaffold.py` | Parameterized FL trainers |
| `fedov.py` | One-shot FL trainer |
| `solo.py` | Individual client baseline |

```bash
# Default run (200 pairs, similarity 0.98–1.0, 4 GPUs)
./run_automated_fl_validation.sh

# Custom parameters
./run_automated_fl_validation.sh \
    --min-similarity 0.95 \
    --max-similarity 0.99 \
    --sample-size 100 \
    --num-gpus 4 \
    --max-concurrent 2

# Semantic column matching variant
./run_semantic_auto_fl_validation.sh

# Resume from last successful step
./run_automated_fl_validation.sh --resume
```

### Output Layout

```
out/autorun/
├── sampled_pairs.json
├── logs/
│   ├── sampling.log
│   ├── preprocessing.log
│   └── {pair_id}_{task}_gpu{id}.log
└── results/
    ├── {pair_id}_fedavg_results.json
    └── execution_report.json

data/auto/
└── {db_id1}_{db_id2}/
    ├── config.json
    ├── {db_id1}_train.csv
    ├── {db_id1}_test.csv
    ├── {db_id2}_train.csv
    └── {db_id2}_test.csv
```

---

## Graph Analysis (`src/analysis/`)

| Module | Description |
|--------|-------------|
| `CommunityDetection.py` | Community detection (Louvain, etc.) |
| `EdgeProperties.py` | Edge-level statistics |
| `NodeProperties.py` | Node-level statistics |
| `NodeSemantic.py` | Semantic node analysis |
| `NodeStatistical.py` | Statistical node analysis |
| `collect_similar_database_pairs.py` | Collect high-similarity pairs |
| `compute_all_join_size.py` / `compute_all_join_size_fast.py` | Estimate AllJoinSize |
| `build_raw_graph.py` | Construct graph from raw similarities |
| `analysis_graph_components.py` | Connected component analysis |
| `estimate_dirichlet_alpha.py` | Data heterogeneity estimation |

---

## External Baselines (`baseline/`)

Git submodules for external comparison methods:

| Submodule | Method |
|-----------|--------|
| `santos/` | SANTOS table union search |
| `deepjoin/` | DeepJoin semantic column matching |
| `starmie/` | Starmie table discovery |
| `FedGTA/` | FedGTA graph-topology-aware FL |
| `SFL-Structural-Federated-Learning/` | Structural FL |

Refer to each submodule's own documentation for setup and usage.

---

## Results & Visualization (`src/summary/`)

| Script | Description |
|--------|-------------|
| `run_and_generate_horizontal_table.py` | Horizontal FL result tables |
| `run_and_generate_vertical_table.py` | Vertical FL result tables |
| `run_and_generate_ml_model_tables.py` | ML model comparison tables |
| `generate_federated_learning_tables.py` | FL summary tables |
| `print_auto_horizontal.py` | Print automated horizontal results |
| `plot_auto_horizontal.py` | Plot automated horizontal results |
| `plot_sim_distribution.py` | Similarity distribution |
| `plot_component_distribution.py` | Graph component distribution |
| `plot_size_vs_gain.py` | Dataset size vs. FL gain |
| `plot_feature_skew.py` | Feature skew analysis |
| `plot_matched_ratio.py` | Column match ratio |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Citation

If you use WikiDBGraph in your research, please cite:

```bibtex
@inproceedings{wu2026wikidbgraph,
  title     = {{WikiDBGraph}: A Data Management Benchmark Suite for Collaborative Learning over Database Silos},
  author    = {Wu, Zhaomin and Wang, Ziyang and He, Bingsheng},
  booktitle = {Proceedings of the IEEE International Conference on Data Engineering (ICDE)},
  year      = {2026},
  organization = {IEEE}
}
```
