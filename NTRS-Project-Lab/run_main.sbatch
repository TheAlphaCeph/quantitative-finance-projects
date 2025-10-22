#!/bin/bash
#SBATCH --job-name=run_rf
#SBATCH --output=run_rf.out
#SBATCH --error=run_rf.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --mem=16GB
#SBATCH --account=pi-chard

# 1. Load the Python module
module load python/miniforge-24.1.2

# 2. Initialize conda by sourcing its initialization script
source /software/python-miniforge-24.1.2-el8-x86_64/etc/profile.d/conda.sh

# 3. Activate the "msfm" conda environment
conda activate msfm

# 4. Go to the project directory 
cd /home/victorqian/NTRS_models

# 5. Run script
python run_rf.py