#!/bin/bash
#PBS -P CFP01-CF-065
#PBS -N col_matcher
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=96:00:00
#PBS -o /hpctmp/e1351271/wkdbs/logs/tmp_col_matcher.out
#PBS -e /hpctmp/e1351271/wkdbs/logs/tmp_col_matcher.err
#PBS -j n
cd /hpctmp/e1351271/wkdbs
module load singularity
image="/app1/common/singularity-img/hopper/cuda/cuda_12.1.1-cudnn8-devel-ubuntu22.04.sif"


timestamp=$(date +"%m%d_%H%M")
# LEVEL="column"
LEVEL="database"
# LEVEL="database-wkpid"
# DATASET="qid"
DATASET="all"
# DATASET="neg_sample"
# METHOD="cosine"
# METHOD="cross_encoder_idf"
# METHOD="cosine_db_idf"
# METHOD="cosine_test_idf"
# METHOD="bge-m3"
METHOD="bge-m3_lr1e-5_ft"
# METHOD="bge-large-en-v1.5"
# METHOD="bge-m3_idf"
SEED="42"

MODEL_PATH="/hpctmp/e1351271/wkdbs/out/col_matcher_bge-m3_database/weights/finetuned_bge_m3_softmax_lr1e-05/best"

LOG_FILE="logs/col_matcher_${DATASET}_${METHOD}_${LEVEL}_${SEED}_${PBS_JOBID}_${timestamp}.out"
ERR_FILE="logs/col_matcher_${DATASET}_${METHOD}_${LEVEL}_${SEED}_${PBS_JOBID}_${timestamp}.err"

singularity exec --nv "$image" bash <<EOT > "$LOG_FILE" 2> "$ERR_FILE"
set -e
cd /hpctmp/e1351271/wkdbs/src
source /hpctmp/e1351271/virtualenvs/wkdbs/bin/activate
echo "=== Job $DATASET $METHOD $LEVEL $SEED started at $(date) ==="
echo "---------------------------------"

export PYTHONPATH=.
python -u script/matcher.py --dataset $DATASET --method $METHOD --seed $SEED --level $LEVEL --model-path $MODEL_PATH

echo "=== Job $DATASET $METHOD $LEVEL $SEED finished at $(date) ==="
EOT

echo "Done: DATASET=$DATASET, METHOD=$METHOD, LEVEL=$LEVEL, SEED=$SEED"