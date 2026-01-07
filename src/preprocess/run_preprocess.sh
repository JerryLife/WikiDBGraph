#!/bin/bash
#
# Preprocessing Pipeline Script
#
# This script runs the complete preprocessing pipeline from data/unzip to data/graph.
# All parameters match the existing implementation for identical results.
#
# Usage: ./run_preprocess.sh [options]
#

set -e

# Default paths
BASE_DIR=$(pwd)
SCHEMA_DIR="${BASE_DIR}/data/schema"
CSV_DIR="${BASE_DIR}/data/unzip"
QID_PAIRS="${BASE_DIR}/data/qid_pairs.csv"  # Shared across all runs
NEGATIVE_POOL_SOURCE="${BASE_DIR}/data/negative_candidates.csv"  # Source file
MODEL_OUTPUT_DIR="${BASE_DIR}/out/col_matcher_bge-m3_lr1e-05"

# Default hyperparameters (matching existing implementation)
SERIALIZATION_MODE="full"
SAMPLE_SIZE=3
NUM_NEGATIVES=6
SIMILARITY_THRESHOLD=0.6713
SEED=0

# Training parameters
LR=1e-05
EPOCHS=10
BATCH_SIZE=32
GPU_ID="0"  # GPU device ID (e.g., "0", "1", "0,1" for multi-GPU)

# Flags
SKIP_TRIPLETS=0
SKIP_TRAINING=0
SKIP_EMBEDDINGS=0
SKIP_TESTING=0
SKIP_SIMILARITY=1
SKIP_FILTERING=1
SKIP_GRAPH=1
AUTO_OUTPUT_DIR=1  # Auto-generate output dir from params

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

show_help() {
    cat << EOF
Preprocessing Pipeline Script

Usage: $0 [OPTIONS]

PIPELINE STEPS:
    1. Generate triplets for contrastive learning
    2. Train BGE-M3 embedding model
    3. Generate embeddings for all databases
    4. Compute all-pairs similarity
    5. Filter edges by threshold
    6. Build DGL graph

OPTIONS:
    --mode MODE             Serialization mode: schema_only|data_only|full (default: full)
    --sample-size N         Sample values per column (default: 3)
    --num-negatives N       Negatives per triplet (default: 6)
    --threshold FLOAT       Similarity threshold (default: 0.6713)
    --seed N                Random seed (default: 0)
    --lr FLOAT              Learning rate (default: 1e-05)
    --epochs N              Training epochs (default: 10)
    --batch-size N          Training batch size (default: 32)
    --gpu ID                GPU device ID to use (default: 0)
    --output-dir DIR        Output directory (default: data/graph)
    --model-path PATH       Use pretrained model instead of training
    --skip-triplets         Skip triplet generation
    --skip-training         Skip model training
    --skip-embeddings       Skip embedding generation
    --skip-testing          Skip model testing/evaluation
    --skip-similarity       Skip similarity computation
    --skip-filtering        Skip edge filtering
    --skip-graph            Skip graph building
    --help                  Show this help message

EXAMPLES:
    # Run full pipeline with defaults
    $0

    # Run with custom threshold
    $0 --threshold 0.94

    # Run ablation with schema-only mode
    $0 --mode schema_only --output-dir out/ablation/schema_only

    # Skip training (use existing model)
    $0 --skip-training --model-path out/model/best
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) SERIALIZATION_MODE="$2"; shift 2 ;;
        --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
        --num-negatives) NUM_NEGATIVES="$2"; shift 2 ;;
        --threshold) SIMILARITY_THRESHOLD="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; AUTO_OUTPUT_DIR=0; shift 2 ;;
        --model-path) MODEL_PATH="$2"; SKIP_TRAINING=1; shift 2 ;;
        --skip-triplets) SKIP_TRIPLETS=1; shift ;;
        --skip-training) SKIP_TRAINING=1; shift ;;
        --skip-embeddings) SKIP_EMBEDDINGS=1; shift ;;
        --skip-testing) SKIP_TESTING=1; shift ;;
        --skip-similarity) SKIP_SIMILARITY=1; shift ;;
        --skip-filtering) SKIP_FILTERING=1; shift ;;
        --skip-graph) SKIP_GRAPH=1; shift ;;
        --help) show_help; exit 0 ;;
        *) log_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Validate environment
if [[ ! -d "$SCHEMA_DIR" ]]; then
    log_error "Schema directory not found: $SCHEMA_DIR"
    exit 1
fi

if [[ ! -d "$CSV_DIR" ]]; then
    log_error "CSV directory not found: $CSV_DIR"
    exit 1
fi

# Auto-generate output directory name from parameters if not specified
if [[ $AUTO_OUTPUT_DIR -eq 1 ]]; then
    # Format: graph_mode{mode}_ss{sample_size}_neg{num_negatives}
    OUTPUT_DIR="${BASE_DIR}/out/graph_${SERIALIZATION_MODE}_ss${SAMPLE_SIZE}_neg${NUM_NEGATIVES}"
fi

# All intermediate files stored within output directory (keeps data/ clean)
TRIPLETS_DIR="${OUTPUT_DIR}/triplets"
NEGATIVE_POOL="${OUTPUT_DIR}/negative_candidates.csv"

# Create output directories
mkdir -p "$OUTPUT_DIR" "$TRIPLETS_DIR" "${OUTPUT_DIR}/logs"

# Set Python path early (needed for negative pool generation)
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH}"

# Handle negative pool: copy from source OR generate from schema
if [[ ! -f "$NEGATIVE_POOL" ]]; then
    if [[ -f "$NEGATIVE_POOL_SOURCE" ]]; then
        log "Copying negative pool from source to output directory..."
        cp "$NEGATIVE_POOL_SOURCE" "$NEGATIVE_POOL"
    else
        log "Step 0: Generating negative pool from schema directory..."
        python -m preprocess.negative_pool_generator \
            --schema-dir "$SCHEMA_DIR" \
            --qid-pairs "$QID_PAIRS" \
            --output "$NEGATIVE_POOL" \
            2>&1 | tee "${OUTPUT_DIR}/logs/negative_pool_generation.log"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Negative pool generation failed!"
            exit 1
        fi
        log_success "Negative pool generated"
    fi
fi

log "=========================================="
log "PREPROCESSING PIPELINE"
log "=========================================="
log "Configuration:"
log "  Serialization mode: $SERIALIZATION_MODE"
log "  Sample size: $SAMPLE_SIZE"
log "  Num negatives: $NUM_NEGATIVES"
log "  Similarity threshold: $SIMILARITY_THRESHOLD"
log "  Seed: $SEED"
log "  Output directory: $OUTPUT_DIR"
log "  Triplets directory: $TRIPLETS_DIR"
log "  Negative pool: $NEGATIVE_POOL"
log "=========================================="

# Step 1: Generate triplets
TRIPLETS_TRAIN="${TRIPLETS_DIR}/triplets_train.jsonl"
TRIPLETS_VAL="${TRIPLETS_DIR}/triplets_val.jsonl"
if [[ $SKIP_TRIPLETS -eq 0 ]]; then
    if [[ -f "$TRIPLETS_TRAIN" ]] && [[ -f "$TRIPLETS_VAL" ]]; then
        log "Step 1: Triplets already exist, skipping: $TRIPLETS_DIR"
    elif [[ -f "$QID_PAIRS" ]] && [[ -f "$NEGATIVE_POOL" ]]; then
        log "Step 1: Generating triplets..."
        python -m preprocess.triplet_generator \
            --qid-pairs "$QID_PAIRS" \
            --negative-pool "$NEGATIVE_POOL" \
            --output-dir "$TRIPLETS_DIR" \
            --num-negatives "$NUM_NEGATIVES" \
            --seed "$SEED" \
            2>&1 | tee "${OUTPUT_DIR}/logs/triplet_generation.log"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Triplet generation failed!"
            exit 1
        fi
        log_success "Triplet generation complete"
    else
        log_error "QID pairs ($QID_PAIRS) or negative pool ($NEGATIVE_POOL) not found!"
        exit 1
    fi
else
    log "Skipping triplet generation (--skip-triplets)"
fi

# Step 2: Train embedding model
MODEL_SAVE_DIR="${OUTPUT_DIR}/weights/finetuned_bge_m3_softmax_lr${LR}"
if [[ $SKIP_TRAINING -eq 0 ]]; then
    if [[ -d "${MODEL_SAVE_DIR}/best" ]]; then
        log "Step 2: Model already exists, skipping training: ${MODEL_SAVE_DIR}/best"
        MODEL_PATH="${MODEL_SAVE_DIR}/best"
    else
        log "Step 2: Training BGE-M3 embedding model..."
        python -m preprocess.trainer \
            --train-triplets "${TRIPLETS_DIR}/triplets_train.jsonl" \
            --val-triplets "${TRIPLETS_DIR}/triplets_val.jsonl" \
            --schema-dir "$SCHEMA_DIR" \
            --csv-dir "$CSV_DIR" \
            --output-dir "$MODEL_SAVE_DIR" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --gpu "$GPU_ID" \
            2>&1 | tee "${OUTPUT_DIR}/logs/training.log"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Training failed!"
            exit 1
        fi
        MODEL_PATH="${MODEL_SAVE_DIR}/best"
        log_success "Training complete"
    fi
else
    log "Skipping model training (--skip-training)"
    if [[ -z "$MODEL_PATH" ]]; then
        CANDIDATE_PATH="${MODEL_SAVE_DIR}/best"
        if [[ -d "$CANDIDATE_PATH" ]]; then
            MODEL_PATH="$CANDIDATE_PATH"
            log "Using existing finetuned model: $MODEL_PATH"
        else
            LEGACY_PATH="${MODEL_OUTPUT_DIR}/weights/finetuned_bge_m3_softmax_lr${LR}/best"
            if [[ -d "$LEGACY_PATH" ]]; then
                MODEL_PATH="$LEGACY_PATH"
                log "Using existing finetuned model (legacy location): $MODEL_PATH"
            else
                MODEL_PATH=""
                log_warning "No finetuned model found, using pretrained BAAI/bge-m3"
            fi
        fi
    fi
fi

log "Using model: ${MODEL_PATH:-BAAI/bge-m3 (pretrained)}"

# Step 3: Generate embeddings
EMBEDDINGS_FILE="${OUTPUT_DIR}/database_embeddings.pt"
if [[ $SKIP_EMBEDDINGS -eq 0 ]]; then
    if [[ -f "$EMBEDDINGS_FILE" ]]; then
        log "Step 3: Embeddings already exist, skipping: $EMBEDDINGS_FILE"
    else
        log "Step 3: Generating embeddings (mode: $SERIALIZATION_MODE)..."
        MODEL_ARG=""
        if [[ -n "$MODEL_PATH" ]]; then
            MODEL_ARG="--model-path $MODEL_PATH"
        fi
        python -m preprocess.embedding_generator \
            --schema-dir "$SCHEMA_DIR" \
            --csv-dir "$CSV_DIR" \
            --output "$EMBEDDINGS_FILE" \
            $MODEL_ARG \
            --mode "$SERIALIZATION_MODE" \
            --batch-size 32 \
            --gpu "$GPU_ID" \
            2>&1 | tee "${OUTPUT_DIR}/logs/embedding_generation.log"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Embedding generation failed!"
            exit 1
        fi
        log_success "Embedding generation complete"
    fi
else
    log "Skipping embedding generation (--skip-embeddings)"
fi

# Step 3.5: Test on test triplets
TEST_TRIPLETS="${TRIPLETS_DIR}/triplets_test.jsonl"
# Fallback to seed0 if main test file doesn't exist (backward compatibility)
if [[ ! -f "$TEST_TRIPLETS" ]] && [[ -f "${TRIPLETS_DIR}/triplets_test_seed0.jsonl" ]]; then
    TEST_TRIPLETS="${TRIPLETS_DIR}/triplets_test_seed0.jsonl"
fi
TEST_RESULTS_DIR="${OUTPUT_DIR}/test_results"
if [[ $SKIP_TESTING -eq 0 ]]; then
    if [[ -f "${TEST_RESULTS_DIR}/summary.txt" ]]; then
        log "Step 3.5: Test results already exist, skipping: ${TEST_RESULTS_DIR}/summary.txt"
    elif [[ -f "$EMBEDDINGS_FILE" ]] && [[ -f "$TEST_TRIPLETS" ]]; then
        log "Step 3.5: Running evaluation on test triplets (seeds 0-4)..."
        python -m preprocess.evaluator \
            --embedding-path "$EMBEDDINGS_FILE" \
            --test-triplets "$TEST_TRIPLETS" \
            --output-dir "$TEST_RESULTS_DIR" \
            --seeds 0 1 2 3 4 \
            --gpu "$GPU_ID" \
            2>&1 | tee "${OUTPUT_DIR}/logs/evaluation.log"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Evaluation failed!"
            exit 1
        fi
        log_success "Evaluation complete"
    else
        log_warning "Skipping evaluation: embeddings or test triplets not found"
    fi
else
    log "Skipping testing/evaluation (--skip-testing)"
fi

# Step 3.6: Generate embeddings and evaluate with ORIGINAL pretrained model (for comparison)
# Use a shared folder so all ablation studies reuse the same baseline
ORIGINAL_OUTPUT_DIR="${BASE_DIR}/out/original_bge-m3"
ORIGINAL_EMBEDDINGS_FILE="${ORIGINAL_OUTPUT_DIR}/database_embeddings.pt"
ORIGINAL_TEST_RESULTS_DIR="${ORIGINAL_OUTPUT_DIR}/test_results"
if [[ $SKIP_TESTING -eq 0 ]] && [[ -n "$MODEL_PATH" ]]; then
    mkdir -p "${ORIGINAL_OUTPUT_DIR}/logs"
    
    # Generate embeddings with original pretrained model (no model-path = use pretrained)
    if [[ -f "$ORIGINAL_EMBEDDINGS_FILE" ]]; then
        log "Step 3.6a: Original model embeddings already exist, skipping"
    else
        log "Step 3.6a: Generating embeddings with ORIGINAL pretrained model..."
        python -m preprocess.embedding_generator \
            --schema-dir "$SCHEMA_DIR" \
            --csv-dir "$CSV_DIR" \
            --output "$ORIGINAL_EMBEDDINGS_FILE" \
            --mode "$SERIALIZATION_MODE" \
            --batch-size 32 \
            --gpu "$GPU_ID" \
            2>&1 | tee "${ORIGINAL_OUTPUT_DIR}/logs/embedding_generation.log"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Original model embedding generation failed!"
            exit 1
        fi
        log_success "Original model embedding generation complete"
    fi
    
    # Evaluate original model
    if [[ -f "${ORIGINAL_TEST_RESULTS_DIR}/summary.txt" ]]; then
        log "Step 3.6b: Original model test results already exist, skipping"
    elif [[ -f "$ORIGINAL_EMBEDDINGS_FILE" ]] && [[ -f "$TEST_TRIPLETS" ]]; then
        log "Step 3.6b: Running evaluation on original model (seeds 0-4)..."
        python -m preprocess.evaluator \
            --embedding-path "$ORIGINAL_EMBEDDINGS_FILE" \
            --test-triplets "$TEST_TRIPLETS" \
            --output-dir "$ORIGINAL_TEST_RESULTS_DIR" \
            --seeds 0 1 2 3 4 \
            --gpu "$GPU_ID" \
            2>&1 | tee "${ORIGINAL_OUTPUT_DIR}/logs/evaluation.log"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Original model evaluation failed!"
            exit 1
        fi
        log_success "Original model evaluation complete"
    fi
fi

# Step 4: Compute all-pairs similarity
PREDICTIONS_FILE="${OUTPUT_DIR}/all_exhaustive_predictions.pt"
if [[ $SKIP_SIMILARITY -eq 0 ]]; then
    if [[ -f "$PREDICTIONS_FILE" ]]; then
        log "Step 4: Predictions already exist, skipping: $PREDICTIONS_FILE"
    else
        log "Step 4: Computing all-pairs similarity..."
        python -m preprocess.similarity_computer \
            --embeddings "$EMBEDDINGS_FILE" \
            --output "$PREDICTIONS_FILE" \
            --threshold "$SIMILARITY_THRESHOLD" \
            --batch-size 256 \
            --gpu "$GPU_ID" \
            2>&1 | tee "${OUTPUT_DIR}/logs/similarity_computation.log"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Similarity computation failed!"
            exit 1
        fi
        log_success "Similarity computation complete"
    fi
else
    log "Skipping similarity computation (--skip-similarity)"
fi

# Step 5: Filter edges
EDGES_FILE="${OUTPUT_DIR}/filtered_edges_threshold_${SIMILARITY_THRESHOLD}.csv"
if [[ $SKIP_FILTERING -eq 0 ]]; then
    if [[ -f "$EDGES_FILE" ]]; then
        log "Step 5: Filtered edges already exist, skipping: $EDGES_FILE"
    else
        log "Step 5: Filtering edges (threshold: $SIMILARITY_THRESHOLD)..."
        python -m preprocess.edge_filter \
            --predictions "$PREDICTIONS_FILE" \
            --output "$EDGES_FILE" \
            --threshold "$SIMILARITY_THRESHOLD" \
            2>&1 | tee "${OUTPUT_DIR}/logs/edge_filtering.log"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Edge filtering failed!"
            exit 1
        fi
        log_success "Edge filtering complete"
    fi
else
    log "Skipping edge filtering (--skip-filtering)"
fi

# Step 6: Build graph
GRAPH_FILE="${OUTPUT_DIR}/graph_raw_${SIMILARITY_THRESHOLD}.dgl"
if [[ $SKIP_GRAPH -eq 0 ]]; then
    if [[ -f "$GRAPH_FILE" ]]; then
        log "Step 6: Graph already exists, skipping: $GRAPH_FILE"
    else
        log "Step 6: Building DGL graph..."
        python -m preprocess.graph_builder \
            --edges "$EDGES_FILE" \
            --embeddings "$EMBEDDINGS_FILE" \
            --output "$GRAPH_FILE" \
            2>&1 | tee "${OUTPUT_DIR}/logs/graph_building.log"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Graph building failed!"
            exit 1
        fi
        log_success "Graph building complete"
    fi
else
    log "Skipping graph building (--skip-graph)"
fi

# Summary
log "=========================================="
log "PIPELINE COMPLETE"
log "=========================================="
log "Outputs:"
log "  Triplets: $TRIPLETS_DIR"
log "  Embeddings: $EMBEDDINGS_FILE"
log "  Test results (fine-tuned): $TEST_RESULTS_DIR"
if [[ -n "$MODEL_PATH" ]]; then
    log "  Test results (original): $ORIGINAL_TEST_RESULTS_DIR"
fi
log "  Predictions: $PREDICTIONS_FILE"
log "  Filtered edges: $EDGES_FILE"
log "  DGL Graph: $GRAPH_FILE"
log "  Logs: ${OUTPUT_DIR}/logs/"
log "=========================================="
