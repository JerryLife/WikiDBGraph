#!/bin/bash
#
# Ablation Study: Serialization Mode
#
# Compares schema_only, data_only, and full serialization modes.
# Each mode gets its own triplets, model, and embeddings in its output directory.
#
# Usage: ./run_ablation_serialization_mode.sh [--gpu ID] [--skip-training]
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

# Modes to test
MODES=("schema_only" "data_only")

log "=========================================="
log "ABLATION STUDY: Serialization Mode"
log "=========================================="
log "Modes: ${MODES[*]}"
log "GPU: $GPU_ID"
log "Training: ${SKIP_TRAINING:-enabled}"
log "Output format: out/graph_{mode}_ss3_neg6/"
log "=========================================="

mkdir -p out

for mode in "${MODES[@]}"; do
    log "Running mode: $mode"
    
    # Run full pipeline with training for each serialization mode
    # Each gets its own triplets, model, and embeddings
    ./src/preprocess/run_preprocess.sh \
        --mode "$mode" \
        --sample-size 3 \
        --num-negatives 6 \
        --gpu "$GPU_ID" \
        --lr "$LR" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --seed "$SEED" \
        $SKIP_TRAINING \
        2>&1 | tee "out/ablation_serialization_${mode}.log"
    
    log_success "Completed mode: $mode"
done

log "=========================================="
log "Generating comparison table..."
log "=========================================="

mkdir -p fig

# Generate comparison table
python -m preprocess.summary.plot_ablation_serialization_mode \
    --results-dir "out" \
    --output "fig/ablation_serialization_mode.tex"

log_success "Ablation study complete!"
log "Results in: out/graph_*_ss3_neg6/"
log "Table in: fig/ablation_serialization_mode.tex"
