source setup.sh
if [ $SBATCH_VAR -le 10 ]; then python root2h5.py --tag $SBATCH_VAR; fi
if [ $SBATCH_VAR -ge 10 ]; then python root2h5.py --merging=ON     ; fi