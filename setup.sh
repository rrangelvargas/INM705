#!/bin/bash

# Standard Gridware activation (always needed)
source /opt/flight/etc/setup.sh
flight env activate gridware
module add gnu

# OPTIONAL: Only run this once to create your venv, then comment it out
python3 -m venv venv

# Activate your virtual environment
source venv/bin/activate

# Show Python info to confirm venv is active
which python
python --version

# Install dependencies (proxy + GPU-aware PyTorch)
pip install --proxy http://hpc-proxy00.city.ac.uk:3128 -r requirements.txt
pip install --proxy http://hpc-proxy00.city.ac.uk:3128 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
