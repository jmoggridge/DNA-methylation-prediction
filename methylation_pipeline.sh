#!/bin/bash

#SBATCH --time=0-02:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=20000 # requested memory (in MB)

### or: 
# srun --pty --account "def-nricker" -t 0-03:00:00 --nodes=2 --mem=20G /bin/bash

# First, create the virtual environment for running cpg_* scripts

# setup modules in virtual environment in the project directory
# SOURCEDIR=~/scratch/DNA_methylation

echo "Loading modules and creating environment"
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
# to solve the "DataFrame object has no attribute 'to_numpy()" error
pip3 install --upgrade pandas 

### Run these scripts in sequentially, separate batches
# this script generates features and creates new csv files for X and y
# python ./cpg_1_preprocessing.py

echo "Starting python script ./cpg_2a_logistic_reg.py"

# # these scripts do the modelling
python ./cpg_2a_Logistic_reg.py

echo "Finished script"


deactivate

echo "Deactivated environmnet"
