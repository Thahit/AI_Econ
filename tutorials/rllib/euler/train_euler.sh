#!/bin/bash

#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --mem-per-cpu=4G
#SBATCH --job-name=FORL
#SBATCH --output=log/debug.out
#SBATCH --error=log/debug.err


export SETUPTOOLS_USE_DISTUTILS=stdlib
export LD_LIBRARY_PATH=/lib64:$LD_LIBRARY_PATH

module load gcc/6.3.0 python/3.6.6 eth_proxy 
source env/bin/activate

cd ..
python training_script.py --run-dir runs/phase1 --note euler_test
