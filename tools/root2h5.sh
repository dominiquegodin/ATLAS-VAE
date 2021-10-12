source setup.sh


# SIGNAL SAMPLES
# COMMAND LINE: sbatch -w atlas16 sbatch.sh
#python root2h5.py --sample_type=topo-ttbar
#python root2h5.py --sample_type=UFO-ttbar
#python root2h5.py --sample_type=BSM


# BACKGROUND SAMPLES
# COMMAND LINE: sbatch -w atlas16 --array=0-9%1 sbatch.sh
sample_type=UFO-dijet
#sample_type=topo-dijet
if [ $sample_type == UFO-dijet ] && ([ $SBATCH_VAR -eq 1 ] || [ $SBATCH_VAR -eq 2 ])
then library=np; else library=np; fi
if [ $SBATCH_VAR -le 9 ]
then python root2h5.py --sample_type=$sample_type --library=$library --tag $SBATCH_VAR; fi
if [ $SBATCH_VAR -ge 9 ]
then python root2h5.py --sample_type=$sample_type --merging=ON; fi

