#!/bin/bash

#---------------------------------------------------------------------
# SLURM OPTIONS (LPS or BELUGA)
#---------------------------------------------------------------------
#SBATCH --account=def-arguinj
#SBATCH --time=00-06:00         #time limit (DD-HH:MM)
#SBATCH --gres=gpu:1            #number of GPU(s) per node
#SBATCH --job-name=vae
#SBATCH --output=%x_%A.out
#---------------------------------------------------------------------

if   [[ -d "/nvme1" ]]
then
    PATHS=/lcg,/opt,/nvme1
else
    PATHS=/lcg,/opt
fi

SIF=/opt/tmp/godin/sing_images/tf-2.1.0-gpu-py3_sing-2.6.sif
singularity shell --nv --bind $PATHS $SIF vae.sh

mkdir -p log_files
mv *.out log_files 2>/dev/null
