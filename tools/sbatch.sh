#!/bin/bash

#---------------------------------------------------------------------
# SLURM OPTIONS (LPS or BELUGA)
#---------------------------------------------------------------------
#SBATCH --time=00-08:00         #time limit (DD-HH:MM)
#SBATCH --mem=256G              #memory per node
#SBATCH --job-name=root2h5
#SBATCH --output=%x_%A_%a.out
#SBATCH --array=0
#---------------------------------------------------------------------

export SBATCH_VAR=$SLURM_ARRAY_TASK_ID

. root2h5.sh

mkdir -p log_files
mv *_${SBATCH_VAR}.out log_files 2>/dev/null

# COMMAND LINES FOR SIGNAL     SAMPLES: sbatch -w atlas16 sbatch.sh
# COMMAND LINES FOR BACKGROUND SAMPLES: sbatch -w atlas16 --array=0-9%1 sbatch.sh
