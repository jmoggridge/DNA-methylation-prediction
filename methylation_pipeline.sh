#!/bin/bash

#SBATCH --time=0-03:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=24000 # requested memory (in MB)
#SBATCH --mail-type=END

### or: 
# srun --pty --account "def-nricker" -t 0-03:00:00 --nodes=2 --mem=20G /bin/bash

# First, create the virtual environment for running cpg_* scripts

# setup modules in virtual environment in the project directory
# SOURCEDIR=~/scratch/DNA_methylation

# load python and scipy
module load python/3.7.0
module load scipy-stack

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# install packages (happens everytime login node is used)
pip install --no-index --upgrade pip
pip install --no-index scikit-learn
pip install --no-index pandas 
pip install --no-index matplotlib 
pip install --no-index seaborn 
pip install --no-index xgboost 
pip install pickle-mixin

### Run these scripts in sequentially, separate batches
# this script generates features and creates new csv files for X and y
# python ./cpg_1_preprocessing.py

# # this script does the model selection
# python ./cpg_2_model_selection.py 


deactivate



