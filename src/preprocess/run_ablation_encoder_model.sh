#!/bin/bash
#
# Ablation Study: Encoder Model
#
# Compares different encoder models: BGE-M3 vs sentence-transformers/all-mpnet-base-v2
# Each model gets its own embeddings and evaluation in separate output directories.
#
# Usage: ./run_ablation_encoder_model.sh [--gpu ID] [--skip-training]
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
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}
log_success() {
    echo -e "${GREEN}[$(date '+%Y-%m-%d %H:%M:%S')] ✅ $1${NC}"
}
log_error() {
    echo -e "${RED}[$(date '+%Y-%m-%d %H:%M:%S')] ❌ $1${NC}"
}
log_warning() {
    echo -e "${YELLOW}[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  $1${NC}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU_ID="$2"; shift 2 ;;
        --skip-training) SKIP_TRAINING="--skip-training"; shift ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# Models to test: (model_type, display_name, folder_suffix)
declare -A MODELS
MODELS["bge-m3"]="bge-m3"
MODELS["mpnet"]="all-mpnet-base-v2"

log "=========================================="
log "ABLATION STUDY: Encoder Model"
log "=========================================="
log "Models: ${!MODELS[*]}"
log "GPU: $GPU_ID"
log "Training: ${SKIP_TRAINING:-enabled}"
log "=========================================="

mkdir -p out

# Store test triplets path (created once, shared by all models)
TRIPLETS_DIR="${BASE_DIR}/out/graph_full_ss3_neg6/triplets"
TEST_TRIPLETS="${TRIPLETS_DIR}/triplets_test.jsonl"
if [[ ! -f "$TEST_TRIPLETS" ]]; then
    TEST_TRIPLETS="${TRIPLETS_DIR}/triplets_test_seed0.jsonl"
fi

# Ensure test triplets exist (need to run default config first if not)
if [[ ! -f "$TEST_TRIPLETS" ]]; then
    log_warning "Test triplets not found. Running default configuration first..."
    ./src/preprocess/run_preprocess.sh \
        --mode full \
        --sample-size 3 \
        --num-negatives 6 \
        --gpu "$GPU_ID" \
        --skip-similarity
    
    # Update triplets path
    TEST_TRIPLETS="${TRIPLETS_DIR}/triplets_test.jsonl"
    if [[ ! -f "$TEST_TRIPLETS" ]]; then
        TEST_TRIPLETS="${TRIPLETS_DIR}/triplets_test_seed0.jsonl"
    fi
fi

for model_type in "${!MODELS[@]}"; do
    model_name="${MODELS[$model_type]}"
    
    # Output directory includes model name if not default bge-m3
    if [[ "$model_type" == "bge-m3" ]]; then
        OUTPUT_DIR="${BASE_DIR}/out/graph_full_ss3_neg6"
        ORIGINAL_DIR="${BASE_DIR}/out/original_bge-m3"
    else
        OUTPUT_DIR="${BASE_DIR}/out/graph_full_ss3_neg6_${model_type}"
        ORIGINAL_DIR="${BASE_DIR}/out/original_${model_type}"
    fi
    
    EMBEDDINGS_FILE="${OUTPUT_DIR}/database_embeddings.pt"
    TEST_RESULTS_DIR="${OUTPUT_DIR}/test_results"
    
    # Skip if test results already exist
    if [[ -f "${TEST_RESULTS_DIR}/summary.txt" ]]; then
        log "⏭️  Skipping model=${model_type} (test results already exist)"
        continue
    fi
    
    log "==========================================" 
    log "Running model: $model_type ($model_name)"
    log "Output: $OUTPUT_DIR"
    log "=========================================="
    
    mkdir -p "${OUTPUT_DIR}/logs"
    
    # Step 1: Generate embeddings with pretrained model (for original baseline)
    ORIGINAL_EMBEDDINGS="${ORIGINAL_DIR}/database_embeddings.pt"
    ORIGINAL_TEST_RESULTS="${ORIGINAL_DIR}/test_results"
    
    if [[ ! -f "${ORIGINAL_TEST_RESULTS}/summary.txt" ]]; then
        mkdir -p "${ORIGINAL_DIR}/logs"
        
        if [[ ! -f "$ORIGINAL_EMBEDDINGS" ]]; then
            log "Step 1a: Generating embeddings with original ${model_type} model..."
            python -m preprocess.embedding_generator \
                --schema-dir "data/schema" \
                --csv-dir "data/unzip" \
                --output "$ORIGINAL_EMBEDDINGS" \
                --model-type "$model_type" \
                --mode "full" \
                --batch-size 32 \
                --gpu "$GPU_ID" \
                2>&1 | tee "${ORIGINAL_DIR}/logs/embedding_generation.log"
        fi
        
        if [[ -f "$ORIGINAL_EMBEDDINGS" ]] && [[ -f "$TEST_TRIPLETS" ]]; then
            log "Step 1b: Evaluating original ${model_type} model..."
            python -m preprocess.evaluator \
                --embedding-path "$ORIGINAL_EMBEDDINGS" \
                --test-triplets "$TEST_TRIPLETS" \
                --output-dir "$ORIGINAL_TEST_RESULTS" \
                --seeds 0 1 2 3 4 \
                --gpu "$GPU_ID" \
                2>&1 | tee "${ORIGINAL_DIR}/logs/evaluation.log"
            log_success "Original ${model_type} model evaluation complete"
        fi
    else
        log "Step 1: Original ${model_type} baseline already exists, skipping"
    fi 
    
    # Step 2: Train fine-tuned model (for all models)
    TRAIN_TRIPLETS="${TRIPLETS_DIR}/triplets_train.jsonl"
    VAL_TRIPLETS="${TRIPLETS_DIR}/triplets_val.jsonl"
    MODEL_OUTPUT_DIR="${OUTPUT_DIR}/weights/finetuned_${model_type}_softmax_lr${LR}"
    FINETUNED_MODEL="${MODEL_OUTPUT_DIR}/best"
    
    if [[ -z "$SKIP_TRAINING" ]] && [[ ! -d "$FINETUNED_MODEL" ]]; then
        log "Step 2: Training fine-tuned ${model_type} model..."
        mkdir -p "$MODEL_OUTPUT_DIR"
        python -m preprocess.trainer \
            --train-triplets "$TRAIN_TRIPLETS" \
            --val-triplets "$VAL_TRIPLETS" \
            --schema-dir "data/schema" \
            --csv-dir "data/unzip" \
            --output-dir "$MODEL_OUTPUT_DIR" \
            --model-type "$model_type" \
            --lr "$LR" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --gpu "$GPU_ID" \
            2>&1 | tee "${OUTPUT_DIR}/logs/training_${model_type}.log"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Training failed for ${model_type}!"
            exit 1
        fi
        log_success "Training complete for ${model_type}"
    else
        log "Step 2: Fine-tuned ${model_type} model already exists or training skipped"
    fi
    
    # Step 3: Generate embeddings with fine-tuned model
    if [[ ! -f "$EMBEDDINGS_FILE" ]] && [[ -d "$FINETUNED_MODEL" ]]; then
        log "Step 3: Generating embeddings with fine-tuned ${model_type}..."
        python -m preprocess.embedding_generator \
            --schema-dir "data/schema" \
            --csv-dir "data/unzip" \
            --output "$EMBEDDINGS_FILE" \
            --model-type "$model_type" \
            --model-path "$FINETUNED_MODEL" \
            --mode "full" \
            --batch-size 32 \
            --gpu "$GPU_ID" \
            2>&1 | tee "${OUTPUT_DIR}/logs/embedding_generation.log"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Embedding generation failed for ${model_type}!"
            exit 1
        fi
        log_success "Embeddings generated for ${model_type}"
    fi
    
    # Step 4: Evaluate fine-tuned model
    if [[ ! -f "${TEST_RESULTS_DIR}/summary.txt" ]] && [[ -f "$EMBEDDINGS_FILE" ]]; then
        log "Step 4: Evaluating fine-tuned ${model_type}..."
        python -m preprocess.evaluator \
            --embedding-path "$EMBEDDINGS_FILE" \
            --test-triplets "$TEST_TRIPLETS" \
            --output-dir "$TEST_RESULTS_DIR" \
            --seeds 0 1 2 3 4 \
            --gpu "$GPU_ID" \
            2>&1 | tee "${OUTPUT_DIR}/logs/evaluation.log"
        if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
            log_error "Evaluation failed for ${model_type}!"
            exit 1
        fi
        log_success "Evaluation complete for ${model_type}"
    fi
    
    log_success "Model ${model_type} complete!"
    
done

log "=========================================="
log "Generating comparison table..."
log "=========================================="

mkdir -p fig

# Generate comparison table
python -m preprocess.summary.plot_ablation_encoder_model \
    --results-dir "out" \
    --output "fig/ablation_encoder_model.tex"

log_success "Ablation study complete!"
log "Results in: out/original_*/test_results/"
log "Table in: fig/ablation_encoder_model.tex"
