#!/bin/bash
#PBS -P CFP01-CF-065
#PBS -N train_bge_softmax
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=96:00:00
#PBS -o /hpctmp/e1351271/wkdbs/logs/tmp_train_bge_softmax.out
#PBS -e /hpctmp/e1351271/wkdbs/logs/tmp_train_bge_softmax.err
#PBS -j n

cd /hpctmp/e1351271/wkdbs
module load singularity
image="/app1/common/singularity-img/hopper/cuda/cuda_12.1.1-cudnn8-devel-ubuntu22.04.sif"

timestamp=$(date +"%m%d_%H%M")
LOG_FILE="logs/train_bge_softmax_${PBS_JOBID}_${timestamp}.out"
ERR_FILE="logs/train_bge_softmax_${PBS_JOBID}_${timestamp}.err"

singularity exec --nv $image bash << EOT > $LOG_FILE 2> $ERR_FILE
set -e
cd /hpctmp/e1351271/wkdbs/src
source /hpctmp/e1351271/virtualenvs/wkdbs/bin/activate
echo "=== Job train_bge_softmax started at $(date) ==="
echo "---------------------------------"

export PYTHONPATH=.
python -u script/train_bge_softmax.py

echo "=== Job train_bge_softmax finished at $(date) ==="
echo "---------------------------------"
EOT

echo "Done: train_bge_softmax, $(date)"