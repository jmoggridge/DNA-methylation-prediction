#!/bin/bash

#SBATCH --time=0-02:00:00 ## days-hours:minutes:seconds
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4 # number of threads
#SBATCH --mem=20000 # requested memory (in MB)

### or: 
srun --pty --account "def-nricker" -t 0-02:00:00 --nodes=1 --ntasks-per-node=8 --mem=20G /bin/bash

# First, create the virtual environment for running cpg_* scripts

# setup modules in virtual environment in the project directory
# SOURCEDIR=~/scratch/DNA_methylation

echo "Loading modules and creating environment"
# load python and scipy
module load python #/3.7.0 doesn't work anymore?
module load scipy-stack

virtualenv --no-download ~/scratch/DNA_methylation/env
source ~/scratch/DNA_methylation/env/bin/activate

# install packages (seem to need to do this each time compute node is used)
pip install --no-index --upgrade pip
pip install --no-index numpy
pip install --no-index pandas 
pip install --no-index scikit-learn
pip install pickle-mixin

# to solve the "DataFrame object has no attribute 'to_numpy()" error
pip install pandas 
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




# pip install --no-index matplotlib 
# pip install --no-index seaborn 
# pip install --no-index xgboost 


# cp jmoggrid@cedar.computecanada:/scratch/jmoggrid/DNA_methylation/results/* ./results/  

## reply from computecanada help - don't load the numpy module. just do packages with pip...