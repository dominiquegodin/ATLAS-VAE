# ROOT SETUP
export PATH=/usr/local/bin:$PATH
export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
source /cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase/user/atlasLocalSetup.sh
source $ATLAS_LOCAL_ROOT_BASE/packageSetups/localSetup.sh root


# NUMPY SETUP
export PATH="/lcg/storage15/software64/anaconda3-atlas16/bin:$PATH"
export PYTHONHOME=/lcg/storage15/software64/anaconda3-atlas16
export PYTHONPATH=/lcg/storage15/software64/anaconda3-atlas16/lib/python3.8/site-packages:$PYTHONPATH