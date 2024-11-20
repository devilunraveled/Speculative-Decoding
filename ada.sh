#!/bin/bash
#SBATCH -n 10
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2048
#SBATCH --time=4-00:00:00
#SBATCH --nodelist=gnode073
#SBATCH --mail-type=END

## load the necessary modules
module load u18/python/3.11.2
echo "Python version: $(python --version)"

## load the lataest CUDA version.
module load u18/cuda/12.1
echo "CUDA version: $(nvcc --version)"

module load u18/gcc/9.4.0
echo "GCC version: $(gcc --version)"

## remove venv if it exists
# rm -rf env

## create the virtual environment
# python3 -m venv env

## Create and acticate venv to run the code in.
source env/bin/activate
echo "Virtual environment activated: $(which python)"

## Upgrade pip to the latest version.
# pip install --upgrade pip

## install the libraries.
# pip install -r requirements.txt

# export CUDA_VISIBLE_DEVICES=0,1,2,3
# export CUDA_LAUNCH_BLOCKING=1

echo "Running on GPUs: $CUDA_VISIBLE_DEVICES"
## Running the training.
python pipeline.py bllsum speculative
