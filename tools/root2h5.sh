source setup.sh

# COMMAND LINES FOR SIGNAL SAMPLES: sbatch -w atlas16 sbatch.sh
python root2h5.py --sample_type=topo-ttbar
python root2h5.py --sample_type=UFO-ttbar
python root2h5.py --sample_type=BSM

# COMMAND LINES FOR BACKGROUND SAMPLES: sbatch -w atlas16 --array=0-9%1 sbatch.sh
#if [ $SBATCH_VAR -le 9 ]; then python root2h5.py --sample_type=topo-dijet --tag $SBATCH_VAR; fi
#if [ $SBATCH_VAR -le 9 ]; then python root2h5.py --sample_type=UFO-dijet --tag $SBATCH_VAR; fi
#if [ $SBATCH_VAR -ge 9 ]; then python root2h5.py --merging=ON; fi

