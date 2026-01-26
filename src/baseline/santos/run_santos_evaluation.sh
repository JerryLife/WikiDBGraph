#!/bin/bash
# run_santos_evaluation.sh: Run the SANTOS baseline evaluation
#
# This script runs the complete SANTOS baseline pipeline and evaluates
# using the same methodology as the preprocess evaluator:
# - 1:1 balanced positive:negative ratio
# - 5 seeds for robustness
# - Mean ± Std for AUC, Precision, Recall, F1
#
# Caching: Steps are skipped if their output files exist. Use --force to rebuild.

set -e

# Parse arguments
FORCE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Default Params
DATA_DIR="data/unzip" # Overwrite in config or via python args
TRIPLETS_DIR="out/graph_full_ss3_neg6/triplets"  # Directory containing triplet files
OUTPUT_DIR="out/santos_eval"
KB_INDEX_DIR="out/santos/index"  # Where synthesize_kb.py saves pickle files
LIMIT="" # Empty for all
SEEDS="0 1 2 3 4"  # Match preprocess evaluator
NUM_WORKERS=64  # Number of parallel workers for KB synthesis
USE_PROCESS=""  # Set to "--use-process" to use multiprocessing instead of threading
CHUNK_SIZE=100  # Files per chunk for parallel Pass 2

# Export for Python scripts
export SANTOS_NUM_WORKERS="$NUM_WORKERS"

mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "SANTOS Evaluation Pipeline"
echo "=========================================="
echo "Configuration:"
echo "  Triplets Dir: $TRIPLETS_DIR"
echo "  Output: $OUTPUT_DIR"
echo "  Seeds: $SEEDS"
echo "  Workers: $NUM_WORKERS"
echo "  Chunk Size: $CHUNK_SIZE"
echo "  Mode: ${USE_PROCESS:-Threading (default)}"
echo "  Force rebuild: $FORCE"
echo "=========================================="

# Collect all triplet files for filtering and scoring
TRIPLET_FILES=()
for seed in $SEEDS; do
    TRIPLET_FILE="$TRIPLETS_DIR/triplets_test_seed${seed}.jsonl"
    if [ -f "$TRIPLET_FILE" ]; then
        TRIPLET_FILES+=("$TRIPLET_FILE")
    fi
done
echo "Found ${#TRIPLET_FILES[@]} triplet file(s) for seeds: $SEEDS"

# 1. Synthesize KB (skip if all pickle files exist)
KB_FILES=(
    "$KB_INDEX_DIR/synth_type_kb.pkl"
    "$KB_INDEX_DIR/synth_type_lookup.pkl"
    "$KB_INDEX_DIR/synth_relation_kb.pkl"
    "$KB_INDEX_DIR/synth_relation_lookup.pkl"
    "$KB_INDEX_DIR/synth_relation_inverted_index.pkl"
)

KB_CACHED=true
for f in "${KB_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        KB_CACHED=false
        break
    fi
done

if [ "$FORCE" = true ] || [ "$KB_CACHED" = false ]; then
    echo "[1/3] Synthesizing Knowledge Base (filtered to evaluation DBs)..."
    python -m src.baseline.santos.synthesize_kb $USE_PROCESS --chunk-size $CHUNK_SIZE \
        --filter-triplets "${TRIPLET_FILES[@]}"
else
    echo "[1/3] [SKIP] Knowledge Base cached (all 5 pickle files exist)"
fi

# 2. Score Pairs for each seed
SCORES_FILE="$OUTPUT_DIR/scores.csv"
if [ "$FORCE" = true ] || [ ! -f "$SCORES_FILE" ]; then
    echo "[2/3] Scoring Test Pairs for all seeds..."
    LIMIT_ARG=""
    if [ -n "$LIMIT" ]; then
        LIMIT_ARG="--limit $LIMIT"
    fi
    
    # Clear existing scores file
    rm -f "$SCORES_FILE"
    
    # Score each seed and concatenate results
    FIRST_SEED=true
    for seed in $SEEDS; do
        TRIPLET_FILE="$TRIPLETS_DIR/triplets_test_seed${seed}.jsonl"
        TEMP_SCORES="$OUTPUT_DIR/scores_seed${seed}.csv"
        
        echo "  Scoring seed $seed..."
        python -m src.baseline.santos.score_pairs \
            --triplets "$TRIPLET_FILE" \
            --output "$TEMP_SCORES" \
            $LIMIT_ARG
        
        # Concatenate: include header only for first seed
        if [ "$FIRST_SEED" = true ]; then
            cat "$TEMP_SCORES" > "$SCORES_FILE"
            FIRST_SEED=false
        else
            tail -n +2 "$TEMP_SCORES" >> "$SCORES_FILE"
        fi
        rm -f "$TEMP_SCORES"
    done
    echo "  Combined scores: $SCORES_FILE"
else
    echo "[2/3] [SKIP] Scores cached ($SCORES_FILE exists)"
fi

# 3. Evaluate (1:1 balanced with seeds, matching preprocess evaluator)
# Always re-run evaluation (fast step, ensures fresh metrics)
echo "[3/3] Calculating Metrics (1:1 balanced, multi-seed)..."
# Use the first triplet file for evaluation (structure is similar across seeds)
python -m src.baseline.santos.evaluate_auc \
    --scores "$SCORES_FILE" \
    --triplets "${TRIPLET_FILES[0]}" \
    --output-dir "$OUTPUT_DIR" \
    --seeds $SEEDS \
    --plot "$OUTPUT_DIR/roc_curve.png"

echo "=========================================="
echo "Done. Results in $OUTPUT_DIR"
echo "=========================================="
cat "$OUTPUT_DIR/summary.txt"
