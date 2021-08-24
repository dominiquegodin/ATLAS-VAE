#!/bin/bash

#---------------------------------------------------------------------
# SLURM OPTIONS (LPS or BELUGA)
#---------------------------------------------------------------------
#SBATCH --account=def-arguinj
#SBATCH --time=00-06:00         #time limit (DD-HH:MM)
#SBATCH --nodes=1               #number of nodes
##SBATCH --mem=128G              #memory per node (Beluga only)
#SBATCH --cpus-per-task=4       #number of CPU threads per node
#SBATCH --gres=gpu:1            #number of GPU(s) per node
#SBATCH --job-name=vae
#SBATCH --output=%x_%A.out
##SBATCH --array=0
#---------------------------------------------------------------------

##export SBATCH_VAR=$SLURM_ARRAY_TASK_ID

if   [[ -d "/nvme1" ]]
then
    PATHS=/lcg,/opt,/nvme1
else
    PATHS=/lcg,/opt
fi

SIF=/opt/tmp/godin/sing_images/tf-2.3.3-gpu-jupyter_sing-2.6.sif
singularity shell --nv --bind $PATHS $SIF vae.sh #$SBATCH_VAR

mkdir -p log_files
mv *.out log_files 2>/dev/null
