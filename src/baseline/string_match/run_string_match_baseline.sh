#!/bin/bash
#
# String Match Baseline Evaluation Script
#
# Evaluates naive string matching baseline (Jaccard similarity on column names)
# on the same test triplets as the embedding-based methods.
#
# Usage: ./run_string_match_baseline.sh
#

set -e

# Configuration
BASE_DIR=$(pwd)
SCHEMA_DIR="${BASE_DIR}/data/schema"
OUTPUT_DIR="${BASE_DIR}/out/baseline_string_match"

# Use the same triplets as main pipeline
TRIPLETS_SRC="${BASE_DIR}/out/graph_full_ss3_neg6/triplets"

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

# Validate environment
if [[ ! -d "$SCHEMA_DIR" ]]; then
    log_error "Schema directory not found: $SCHEMA_DIR"
    exit 1
fi

# Find test triplets
TEST_TRIPLETS=""
if [[ -f "${TRIPLETS_SRC}/triplets_test.jsonl" ]]; then
    TEST_TRIPLETS="${TRIPLETS_SRC}/triplets_test.jsonl"
elif [[ -f "${TRIPLETS_SRC}/triplets_test_seed0.jsonl" ]]; then
    TEST_TRIPLETS="${TRIPLETS_SRC}/triplets_test_seed0.jsonl"
else
    log_error "Test triplets not found in: $TRIPLETS_SRC"
    log_error "Please run run_preprocess.sh first to generate triplets."
    exit 1
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}/test_results" "${OUTPUT_DIR}/logs"

# Set Python path
export PYTHONPATH="${BASE_DIR}/src:${PYTHONPATH}"

log "=========================================="
log "STRING MATCH BASELINE EVALUATION"
log "=========================================="
log "Configuration:"
log "  Schema dir: $SCHEMA_DIR"
log "  Test triplets: $TEST_TRIPLETS"
log "  Output dir: $OUTPUT_DIR"
log "=========================================="

# Run string match evaluator
if [[ -f "${OUTPUT_DIR}/test_results/summary.txt" ]]; then
    log "Results already exist, skipping: ${OUTPUT_DIR}/test_results/summary.txt"
else
    log "Running string match evaluation..."
    python3 -m baseline.string_match.string_match_evaluator \
        --schema-dir "$SCHEMA_DIR" \
        --test-triplets "$TEST_TRIPLETS" \
        --output-dir "${OUTPUT_DIR}/test_results" \
        --seeds 0 1 2 3 4 \
        2>&1 | tee "${OUTPUT_DIR}/logs/evaluation.log"
    
    if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
        log_error "String match evaluation failed!"
        exit 1
    fi
    log_success "String match evaluation complete"
fi

# Summary
log "=========================================="
log "BASELINE EVALUATION COMPLETE"
log "=========================================="
log "Results: ${OUTPUT_DIR}/test_results/summary.txt"
log "Logs: ${OUTPUT_DIR}/logs/"
log "=========================================="

# Print summary
if [[ -f "${OUTPUT_DIR}/test_results/summary.txt" ]]; then
    echo ""
    cat "${OUTPUT_DIR}/test_results/summary.txt"
fi
