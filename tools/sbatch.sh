#!/bin/bash

#---------------------------------------------------------------------
# SLURM OPTIONS (LPS or BELUGA)
#---------------------------------------------------------------------
#SBATCH --time=01-00:00         #time limit (DD-HH:MM)
#SBATCH --job-name=root2h5
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0
#---------------------------------------------------------------------

export SBATCH_VAR=$SLURM_ARRAY_TASK_ID

. root2h5.sh

mkdir -p log_files
mv *_${SBATCH_VAR}.out log_files 2>/dev/null

# COMMAND LINES
#sbatch -w atlas16 --array=0-10%1 sbatch.sh
