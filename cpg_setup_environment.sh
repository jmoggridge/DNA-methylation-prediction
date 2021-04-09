# srun --pty --account "def-nricker" -t 0-02:00:00 --nodes=1 --ntasks-per-node=8 --mem=20G /bin/bash

virtualenv --no-download ~/scratch/DNA_methylation/env
source ~/scratch/DNA_methylation/env/bin/activate

# sci-kit learn stopped working on April 5th. CC says install these manually.
module load python
pip install pytz
pip install python-dateutil
pip install scipy
pip install numpy
pip install matplotlib
pip install scikit-learn
pip install pickle-mixin

