#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu1
##
#SBATCH --job-name=swh
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err
##
#SBATCH --gres=gpu:rtx3090:4

hostname
date

module add CUDA/11.2.2
module add ANACONDA/2020.11

python /home1/wonhyung64/Github/YOLO/main.py --data-dir /home1/wonhyung64/data
python /home1/wonhyung64/Github/YOLO/main.py --name voc/2012 --data-dir /home1/wonhyung64/data
python /home1/wonhyung64/Github/YOLO/main.py --name coco/2017 --data-dir /home1/wonhyung64/data
