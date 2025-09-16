#!/bin/sh

#SBATCH --time=0:15:00
#SBATCH --account=pi-amurugan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --ntasks-per-node=1 # num cores to drive each gpu
#SBATCH --cpus-per-task=1   # set this to the desired number of threads

#SBATCH --output=cluster_test.out
#SBATCH --error=cluster_test.err

module load python/3.11.9
module load tensorflow
module load cudnn
python cluster_test.py