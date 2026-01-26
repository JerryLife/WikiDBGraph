#!/bin/bash
#
# GitTables Preprocessing Pipeline Script
#
# Self-supervised preprocessing using synthetic partitioning for
# Collaborative Learning scenarios.
#
# Usage: ./run_gittables_preprocess.sh [options]
#

set -e

# Default paths
BASE_DIR=$(pwd)
GITTABLES_DIR="${BASE_DIR}/data/GitTables"
OUTPUT_BASE="${BASE_DIR}/out/gittables"

# Default hyperparameters
SERIALIZATION_MODE="full"
SAMPLE_SIZE=3
NUM_NEGATIVES=2
SPLIT_TYPE="both"  # vertical, horizontal, or both
MIN_COLUMNS=6
MIN_ROWS=50
SEED=42

# Training parameters
LR=1e-05
EPOCHS=10
BATCH_SIZE=32
GPU_ID="0"

# Flags
SKIP_TRIPLETS=0
SKIP_TRAINING=0
SKIP_EMBEDDINGS=0
SKIP_TESTING=0
MAX_TABLES_PER_TOPIC=""  # Empty = no limit
TOPICS=""  # Empty = all topics

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
GitTables Preprocessing Pipeline Script

Self-supervised preprocessing using synthetic partitioning for
Collaborative Learning (Vertical/Horizontal FL).

Usage: $0 [OPTIONS]

PIPELINE STEPS:
    1. Generate triplets from synthetic table splits
    2. Train BGE-M3 embedding model
    3. Generate embeddings for all table splits
    4. Evaluate on test triplets

OPTIONS:
    --split-type TYPE       Split type: vertical|horizontal|both (default: both)
    --mode MODE             Serialization mode: schema_only|data_only|full (default: full)
    --sample-size N         Sample values per column (default: 3)
    --num-negatives N       Negatives per triplet (default: 2)
    --min-columns N         Min columns for vertical split (default: 6)
    --min-rows N            Min rows for horizontal split (default: 50)
    --seed N                Random seed (default: 42)
    --lr FLOAT              Learning rate (default: 1e-05)
    --epochs N              Training epochs (default: 10)
    --batch-size N          Training batch size (default: 32)
    --gpu ID                GPU device ID (default: 0)
    --output-dir DIR        Output directory (default: auto-generated)
    --max-tables N          Max tables per topic (for testing)
    --topics TOPIC...       Specific topics to process
    --skip-triplets         Skip triplet generation
    --skip-training         Skip model training
    --skip-embeddings       Skip embedding generation
    --skip-testing          Skip model testing
    --help                  Show this help message

EXAMPLES:
    # Run full pipeline with defaults
    $0

    # Run on 2 topics with limited tables (for testing)
    $0 --max-tables 100 --topics crime_rate_tables_licensed growth_rate_tables_licensed

    # Run vertical splits only
    $0 --split-type vertical

    # Skip training (use existing model)
    $0 --skip-training --model-path out/gittables/weights/best
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --split-type) SPLIT_TYPE="$2"; shift 2 ;;
        --mode) SERIALIZATION_MODE="$2"; shift 2 ;;
        --sample-size) SAMPLE_SIZE="$2"; shift 2 ;;
        --num-negatives) NUM_NEGATIVES="$2"; shift 2 ;;
        --min-columns) MIN_COLUMNS="$2"; shift 2 ;;
        --min-rows) MIN_ROWS="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --lr) LR="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --gpu) GPU_ID="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; SKIP_TRAINING=1; shift 2 ;;
        --max-tables) MAX_TABLES_PER_TOPIC="$2"; shift 2 ;;
        --topics) 
            shift
            TOPICS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                TOPICS="$TOPICS $1"
                shift
            done
            ;;
        --skip-triplets) SKIP_TRIPLETS=1; shift ;;
        --skip-training) SKIP_TRAINING=1; shift ;;
        --skip-embeddings) SKIP_EMBEDDINGS=1; shift ;;
        --skip-testing) SKIP_TESTING=1; shift ;;
        --help) show_help; exit 0 ;;
        *) log_error "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Auto-generate output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="${OUTPUT_BASE}/${SPLIT_TYPE}_${SERIALIZATION_MODE}_ss${SAMPLE_SIZE}_neg${NUM_NEGATIVES}"
fi

TRIPLETS_DIR="${OUTPUT_DIR}/triplets"

# Create directories
mkdir -p "$OUTPUT_DIR" "$TRIPLETS_DIR" "${OUTPUT_DIR}/logs"

# Set Python path
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH}"

log "=========================================="
log "GITTABLES PREPROCESSING PIPELINE"
log "=========================================="
log "Configuration:"
log "  Split type: $SPLIT_TYPE"
log "  Serialization mode: $SERIALIZATION_MODE"
log "  Sample size: $SAMPLE_SIZE"
log "  Num negatives: $NUM_NEGATIVES"
log "  Min columns: $MIN_COLUMNS"
log "  Min rows: $MIN_ROWS"
log "  Seed: $SEED"
log "  Output directory: $OUTPUT_DIR"
log "=========================================="

# Step 1: Generate triplets
TRIPLETS_TRAIN="${TRIPLETS_DIR}/triplets_train.jsonl"
TRIPLETS_VAL="${TRIPLETS_DIR}/triplets_val.jsonl"
if [[ $SKIP_TRIPLETS -eq 0 ]]; then
    if [[ -f "$TRIPLETS_TRAIN" ]] && [[ -f "$TRIPLETS_VAL" ]]; then
        log "Step 1: Triplets already exist, skipping: $TRIPLETS_DIR"
    else
        log "Step 1: Generating triplets from GitTables..."
        
        TOPICS_ARG=""
        if [[ -n "$TOPICS" ]]; then
            TOPICS_ARG="--topics $TOPICS"
        fi
        
        MAX_TABLES_ARG=""
        if [[ -n "$MAX_TABLES_PER_TOPIC" ]]; then
            MAX_TABLES_ARG="--max-tables-per-topic $MAX_TABLES_PER_TOPIC"
        fi
        
        python -m preprocess.GitTables.triplet_generator \
            --gittables-dir "$GITTABLES_DIR" \
            --output-dir "$TRIPLETS_DIR" \
            --split-type "$SPLIT_TYPE" \
            --mode "$SERIALIZATION_MODE" \
            --sample-size "$SAMPLE_SIZE" \
            --num-negatives "$NUM_NEGATIVES" \
            --min-columns "$MIN_COLUMNS" \
            --min-rows "$MIN_ROWS" \
            --seed "$SEED" \
            $TOPICS_ARG \
            $MAX_TABLES_ARG \
            2>&1 | tee "${OUTPUT_DIR}/logs/triplet_generation.log"
        
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Triplet generation failed!"
            exit 1
        fi
        log_success "Triplet generation complete"
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
        
        # Use the GitTables-specific trainer for pre-serialized triplets
        python -m preprocess.GitTables.trainer_gittables \
            --train-triplets "${TRIPLETS_DIR}/triplets_train.jsonl" \
            --val-triplets "${TRIPLETS_DIR}/triplets_val.jsonl" \
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
            log "Using existing model: $MODEL_PATH"
        else
            MODEL_PATH=""
            log_warning "No model found, will use pretrained BAAI/bge-m3"
        fi
    fi
fi

log "Using model: ${MODEL_PATH:-BAAI/bge-m3 (pretrained)}"

# Step 3a: Test with PRETRAINED model (baseline)
TEST_TRIPLETS="${TRIPLETS_DIR}/triplets_test.jsonl"
PRETRAINED_RESULTS_DIR="${OUTPUT_DIR}/test_results_pretrained"
if [[ $SKIP_TESTING -eq 0 ]]; then
    if [[ -f "${PRETRAINED_RESULTS_DIR}/summary.txt" ]]; then
        log "Step 3a: Pretrained baseline results already exist, skipping: ${PRETRAINED_RESULTS_DIR}/summary.txt"
    elif [[ -f "$TEST_TRIPLETS" ]]; then
        log "Step 3a: Running PRETRAINED baseline evaluation on test triplets..."
        
        # Use the GitTables-specific evaluator with pretrained BAAI/bge-m3 (no --model-path)
        python -m preprocess.GitTables.evaluator_gittables \
            --test-triplets "$TEST_TRIPLETS" \
            --output-dir "$PRETRAINED_RESULTS_DIR" \
            --seeds 0 1 2 3 4 \
            --gpu "$GPU_ID" \
            2>&1 | tee "${OUTPUT_DIR}/logs/evaluation_pretrained.log"
        
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Pretrained evaluation failed!"
            exit 1
        fi
        log_success "Pretrained baseline evaluation complete"
    else
        log_warning "Skipping pretrained evaluation: test triplets not found"
    fi
else
    log "Skipping pretrained testing (--skip-testing)"
fi

# Step 3b: Test with FINE-TUNED model
FINETUNED_RESULTS_DIR="${OUTPUT_DIR}/test_results_finetuned"
if [[ $SKIP_TESTING -eq 0 ]]; then
    if [[ -f "${FINETUNED_RESULTS_DIR}/summary.txt" ]]; then
        log "Step 3b: Fine-tuned results already exist, skipping: ${FINETUNED_RESULTS_DIR}/summary.txt"
    elif [[ -f "$TEST_TRIPLETS" ]] && [[ -n "$MODEL_PATH" ]] && [[ -d "$MODEL_PATH" ]]; then
        log "Step 3b: Running FINE-TUNED model evaluation on test triplets..."
        log "  Using model: $MODEL_PATH"
        
        # Use the GitTables-specific evaluator with fine-tuned model
        python -m preprocess.GitTables.evaluator_gittables \
            --test-triplets "$TEST_TRIPLETS" \
            --output-dir "$FINETUNED_RESULTS_DIR" \
            --seeds 0 1 2 3 4 \
            --gpu "$GPU_ID" \
            --model-path "$MODEL_PATH" \
            2>&1 | tee "${OUTPUT_DIR}/logs/evaluation_finetuned.log"
        
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Fine-tuned evaluation failed!"
            exit 1
        fi
        log_success "Fine-tuned model evaluation complete"
    elif [[ -z "$MODEL_PATH" ]] || [[ ! -d "$MODEL_PATH" ]]; then
        log_warning "Skipping fine-tuned evaluation: no fine-tuned model found at ${MODEL_PATH:-N/A}"
    else
        log_warning "Skipping fine-tuned evaluation: test triplets not found"
    fi
else
    log "Skipping fine-tuned testing (--skip-testing)"
fi

# Summary
log "=========================================="
log "PIPELINE COMPLETE"
log "=========================================="
log "Outputs:"
log "  Triplets: $TRIPLETS_DIR"
log "  Model: ${MODEL_PATH:-N/A}"
log "  Pretrained test results: $PRETRAINED_RESULTS_DIR"
log "  Fine-tuned test results: $FINETUNED_RESULTS_DIR"
log "  Logs: ${OUTPUT_DIR}/logs/"
log "=========================================="
