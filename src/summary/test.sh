#!/bin/bash
#PBS -P CFP01-CF-065
#PBS -N plot_datasets_info
#PBS -q auto
#PBS -l select=1:ngpus=1
#PBS -l walltime=96:00:00
#PBS -o /hpctmp/e1351271/wkdbs/logs/tmp_plot_datasets_info.out
#PBS -e /hpctmp/e1351271/wkdbs/logs/tmp_plot_datasets_info.err
#PBS -j n
cd /hpctmp/e1351271/wkdbs
# module load singularity
# image="/app1/common/singularity-img/hopper/cuda/cuda_12.1.1-cudnn8-devel-ubuntu22.04.sif"



timestamp=$(date +"%m%d_%H%M")
LOG_FILE="logs/plot_${PBS_JOBID}_${timestamp}.out"
ERR_FILE="logs/plot_${PBS_JOBID}_${timestamp}.err"

# singularity exec "$image" bash <<EOT > "$LOG_FILE" 2> "$ERR_FILE"
# set -e
cd /hpctmp/e1351271/wkdbs/src
{
  source ~/.bashrc
  conda activate torch2
  echo "=== Job plot started at $(date) ==="
  export PYTHONPATH=.
  python -u summary/plot_datasets_info.py
  echo "=== Job plot finished at $(date) ==="
} > "$LOG_FILE" 2> "$ERR_FILE"