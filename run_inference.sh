#!/bin/bash
#SBATCH --job-name=sign-baseline                # Job name
#SBATCH --partition=gengpu                   # GPU partition
#SBATCH --nodes=1                            # 1 node
#SBATCH --ntasks-per-node=1                  # 1 task per node
#SBATCH --cpus-per-task=4                    # 4 CPU cores
#SBATCH --mem=24GB                           # 24GB CPU RAM
#SBATCH --time=36:00:00                      # Max time
#SBATCH --gres=gpu:1                         # Request 1 GPU
#SBATCH -e results/%x_%j.e                   # Error log
#SBATCH -o results/%x_%j.o                   # Output log
#SBATCH --output results/%x_%j.out           # Print log (stdout)

# Activate environment
source /opt/flight/etc/setup.sh
flight env activate gridware
module load libs/nvidia-cuda/11.2.0/bin
module load gnu

# Activate your venv
source venv/bin/activate

export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export http_proxy=http://hpc-proxy00.city.ac.uk:3128

export WANDB_MODE=offline

# Show Python info
which python
python --version

python evaluate_saved_models.py
