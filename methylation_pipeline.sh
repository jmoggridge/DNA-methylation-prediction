#!/bin/bash

#SBATCH --time=0-02:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=16000 # requested memory (in MB)
#SBATCH --mail-type=END

### or: 
# srun --pty --account "def-nricker" -t 0-03:00:00 --mem=20G /bin/bash

SOURCEDIR=~/scratch/DNA_methylation
# load python and scipy
module load python 3.7.0
module load scipy-stack

# create virtualenv and update pip
virtualenv --no-download ~/$SOURCEDIR
source ~/$SOURCEDIR/bin/activate
pip install --no-index --upgrade pip

# install packages
pip install scikit-learn --no-index
pip install pandas --no-index
pip install matplotlib --no-index
pip install seaborn --no-index
pip install xgboost --no-index
#
pip install pickle --no-index

# this script generates features and creates new csv files for X and y
python ./cpg_preprocessing.py

# this script does the model selection
python ./cpg_model_selection.py

deactivate


# use this to activate the virtual environment; will need to 
source ~/$SOURCEDIR/bin/activate