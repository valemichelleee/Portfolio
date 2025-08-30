#!/bin/bash -l
#SBATCH --job-name=yolov8x-train
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load python/3.12-conda # Load the Python environment
conda activate base      # now works just like under salloc

echo "[$(date)] Starting YOLOv8x training on $(hostname)"

# Run the multi-GPU training
python train_api.py

echo "[$(date)] Finished training"
