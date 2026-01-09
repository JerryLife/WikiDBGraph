#!/bin/bash
#
# Ablation Study: Number of Negatives
#
# Compares different numbers of negative samples per triplet.
# Each configuration gets its own triplets, model, and embeddings in its output directory.
#
# Usage: ./run_ablation_num_negatives.sh [--gpu ID] [--skip-training]
#

set -e

BASE_DIR=$(pwd)

# Default parameters
GPU_ID="0"
SKIP_TRAINING=""
LR="1e-05"
EPOCHS=10
BATCH_SIZE=16
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

# Number of negatives to test
NUM_NEGATIVES=(2 4 6 10 15)

log "=========================================="
log "ABLATION STUDY: Number of Negatives"
log "=========================================="
log "Num negatives: ${NUM_NEGATIVES[*]}"
log "GPU: $GPU_ID"
log "Training: ${SKIP_TRAINING:-enabled}"
log "Output format: out/graph_full_ss3_neg{num}/"
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
    
    # Use any test triplets from first ablation config
    TEST_TRIPLETS="${BASE_DIR}/out/graph_full_ss3_neg${NUM_NEGATIVES[0]}/triplets/triplets_test.jsonl"
    if [[ ! -f "$TEST_TRIPLETS" ]]; then
        TEST_TRIPLETS="${BASE_DIR}/out/graph_full_ss3_neg${NUM_NEGATIVES[0]}/triplets/triplets_test_seed0.jsonl"
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

for num in "${NUM_NEGATIVES[@]}"; do
    OUTPUT_DIR="${BASE_DIR}/out/graph_full_ss3_neg${num}"
    EMBEDDINGS_FILE="${OUTPUT_DIR}/database_embeddings.pt"
    MODEL_DIR="${OUTPUT_DIR}/weights/finetuned_bge_m3_softmax_lr${LR}/best"
    
    # Auto-detect: skip if embeddings AND test results already exist (pipeline fully completed)
    TEST_RESULTS_FILE="${OUTPUT_DIR}/test_results/summary.txt"
    if [[ -f "$EMBEDDINGS_FILE" ]] && [[ -f "$TEST_RESULTS_FILE" ]]; then
        log "⏭️  Skipping num_negatives=$num (complete: embeddings + test_results exist)"
        continue
    fi
    
    # If embeddings exist but no test results, just run evaluation
    if [[ -f "$EMBEDDINGS_FILE" ]] && [[ ! -f "$TEST_RESULTS_FILE" ]]; then
        log "🔄 Embeddings exist for num_negatives=$num but no test results, running evaluation only..."
        TEST_TRIPLETS="${OUTPUT_DIR}/triplets/triplets_test.jsonl"
        if [[ ! -f "$TEST_TRIPLETS" ]]; then
            TEST_TRIPLETS="${OUTPUT_DIR}/triplets/triplets_test_seed0.jsonl"
        fi
        mkdir -p "${OUTPUT_DIR}/test_results"
        python -m preprocess.evaluator \
            --embedding-path "$EMBEDDINGS_FILE" \
            --test-triplets "$TEST_TRIPLETS" \
            --output-dir "${OUTPUT_DIR}/test_results" \
            --seeds 0 1 2 3 4 \
            --gpu "$GPU_ID" \
            2>&1 | tee -a "out/ablation_num_negatives_${num}.log"
        log_success "Evaluation completed for num_negatives=$num"
        continue
    fi
    
    # Auto-detect: if model exists but no embeddings, skip training and continue from embeddings
    SKIP_FLAG="$SKIP_TRAINING"
    if [[ -d "$MODEL_DIR" ]]; then
        log "🔄 Model exists for num_negatives=$num, skipping training and continuing from embeddings..."
        SKIP_FLAG="--skip-training --skip-triplets"
    fi
    
    log "Running num_negatives: $num"
    
    # Run full pipeline with training for each num_negatives value
    # Each gets its own triplets, model, and embeddings
    ./src/preprocess/run_preprocess.sh \
        --mode "full" \
        --sample-size 3 \
        --num-negatives "$num" \
        --gpu "$GPU_ID" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --seed "$SEED" \
        $SKIP_FLAG \
        2>&1 | tee "out/ablation_num_negatives_${num}.log"
    
    log_success "Completed num_negatives: $num"
done

log "=========================================="
log "Generating comparison plot..."
log "=========================================="

mkdir -p fig

# Generate comparison plot
python -m preprocess.summary.plot_ablation_num_negatives \
    --results-dir "out" \
    --nums ${NUM_NEGATIVES[*]} \
    --output "fig/ablation_num_negatives.png"

log_success "Ablation study complete!"
log "Results in: out/graph_full_ss3_neg*/"
log "Plot in: fig/ablation_num_negatives.png"
