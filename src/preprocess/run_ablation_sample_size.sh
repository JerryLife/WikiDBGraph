#!/bin/bash
#
# Ablation Study: Sample Size
#
# Compares different sample sizes (number of representative values per column).
# Each configuration gets its own triplets, model, and embeddings in its output directory.
#
# Usage: ./run_ablation_sample_size.sh [--gpu ID] [--skip-training]
#

set -e

BASE_DIR=$(pwd)

# Default parameters
GPU_ID="0"
SKIP_TRAINING=""
LR="1e-05"
EPOCHS=10
BATCH_SIZE=32
SEED=0

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU_ID="$2"; shift 2 ;;
        --skip-training) SKIP_TRAINING="--skip-training"; shift ;;
        --lr) LR="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH}"

# Sample sizes to test
SAMPLE_SIZES=(1 3 5 10)

log "=========================================="
log "ABLATION STUDY: Sample Size"
log "=========================================="
log "Sample sizes: ${SAMPLE_SIZES[*]}"
log "GPU: $GPU_ID"
log "Training: ${SKIP_TRAINING:-enabled}"
log "Output format: out/graph_full_ss{size}_neg6/"
log "=========================================="

mkdir -p out

# Step 0: Evaluate original pretrained model as baseline (once)
# Use a shared folder so all ablation studies reuse the same baseline
ORIGINAL_OUTPUT_DIR="${BASE_DIR}/out/original_bge-m3"
ORIGINAL_EMBEDDINGS="${ORIGINAL_OUTPUT_DIR}/database_embeddings.pt"
ORIGINAL_TEST_RESULTS="${ORIGINAL_OUTPUT_DIR}/test_results"
if [[ ! -f "${ORIGINAL_TEST_RESULTS}/summary.txt" ]]; then
    log "Step 0: Evaluating ORIGINAL pretrained model as baseline..."
    mkdir -p "${ORIGINAL_OUTPUT_DIR}/logs"
    
    # Generate embeddings with pretrained model (no model-path)
    if [[ ! -f "$ORIGINAL_EMBEDDINGS" ]]; then
        python -m preprocess.embedding_generator \
            --schema-dir "data/schema" \
            --csv-dir "data/unzip" \
            --output "$ORIGINAL_EMBEDDINGS" \
            --mode "full" \
            --batch-size 32 \
            --gpu "$GPU_ID" \
            2>&1 | tee "${ORIGINAL_OUTPUT_DIR}/logs/embedding_generation.log"
    fi
    
    # Use test triplets from first ablation config
    TEST_TRIPLETS="${BASE_DIR}/out/graph_full_ss${SAMPLE_SIZES[0]}_neg6/triplets/triplets_test.jsonl"
    if [[ ! -f "$TEST_TRIPLETS" ]]; then
        TEST_TRIPLETS="${BASE_DIR}/out/graph_full_ss${SAMPLE_SIZES[0]}_neg6/triplets/triplets_test_seed0.jsonl"
    fi
    
    if [[ -f "$ORIGINAL_EMBEDDINGS" ]] && [[ -f "$TEST_TRIPLETS" ]]; then
        python -m preprocess.evaluator \
            --embedding-path "$ORIGINAL_EMBEDDINGS" \
            --test-triplets "$TEST_TRIPLETS" \
            --output-dir "$ORIGINAL_TEST_RESULTS" \
            --seeds 0 1 2 3 4 \
            --gpu "$GPU_ID" \
            2>&1 | tee "${ORIGINAL_OUTPUT_DIR}/logs/evaluation.log"
        log_success "Original model baseline evaluation complete"
    else
        log_warning "Skipping original baseline: test triplets not found yet"
    fi
else
    log "Step 0: Original model baseline already exists, skipping"
fi

for size in "${SAMPLE_SIZES[@]}"; do
    OUTPUT_DIR="${BASE_DIR}/out/graph_full_ss${size}_neg6"
    EMBEDDINGS_FILE="${OUTPUT_DIR}/database_embeddings.pt"
    TEST_RESULTS_FILE="${OUTPUT_DIR}/test_results/summary.txt"
    MODEL_DIR="${OUTPUT_DIR}/weights/finetuned_bge_m3_softmax_lr${LR}/best"
    
    # Auto-detect: skip if embeddings AND test results already exist (pipeline fully completed)
    if [[ -f "$EMBEDDINGS_FILE" ]] && [[ -f "$TEST_RESULTS_FILE" ]]; then
        log "⏭️  Skipping sample_size=$size (complete: embeddings + test_results exist)"
        continue
    fi
    
    # If embeddings exist but no test results, just run evaluation
    if [[ -f "$EMBEDDINGS_FILE" ]] && [[ ! -f "$TEST_RESULTS_FILE" ]]; then
        log "🔄 Embeddings exist for sample_size=$size but no test results, running evaluation only..."
        TEST_TRIPLETS_SS="${OUTPUT_DIR}/triplets/triplets_test.jsonl"
        if [[ ! -f "$TEST_TRIPLETS_SS" ]]; then
            TEST_TRIPLETS_SS="${OUTPUT_DIR}/triplets/triplets_test_seed0.jsonl"
        fi
        mkdir -p "${OUTPUT_DIR}/test_results"
        python -m preprocess.evaluator \
            --embedding-path "$EMBEDDINGS_FILE" \
            --test-triplets "$TEST_TRIPLETS_SS" \
            --output-dir "${OUTPUT_DIR}/test_results" \
            --seeds 0 1 2 3 4 \
            --gpu "$GPU_ID" \
            2>&1 | tee -a "out/ablation_sample_size_${size}.log"
        log_success "Evaluation completed for sample_size=$size"
        continue
    fi

    # Auto-detect: if model exists but no embeddings, skip training
    SKIP_FLAG="$SKIP_TRAINING"
    if [[ -d "$MODEL_DIR" ]]; then
        log "🔄 Model exists for sample_size=$size, skipping training and continuing from embeddings..."
        SKIP_FLAG="--skip-training --skip-triplets"
    fi
    
    log "Running sample_size: $size"
    
    # Run full pipeline with training for each sample size
    # Each gets its own triplets, model, and embeddings
    ./src/preprocess/run_preprocess.sh \
        --mode "full" \
        --sample-size "$size" \
        --num-negatives 6 \
        --gpu "$GPU_ID" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --seed "$SEED" \
        $SKIP_FLAG \
        2>&1 | tee "out/ablation_sample_size_${size}.log"
    
    log_success "Completed sample_size: $size"
done

log "=========================================="
log "Generating comparison plot..."
log "=========================================="

mkdir -p fig

# Generate professional comparison plot
python -m preprocess.summary.plot_ablation_sample_size \
    --results-dir "out" \
    --sizes 0 ${SAMPLE_SIZES[*]} \
    --output "fig/ablation_sample_size.png"

log_success "Ablation study complete!"
log "Results in: out/graph_full_ss*_neg6/"
log "Plot in: fig/ablation_sample_size.png"
