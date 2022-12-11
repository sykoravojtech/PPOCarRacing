#!/bin/bash
#SBATCH --ntasks 8
#SBATCH --nodes 4

#SBATCH --partition amd
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 1
#SBATCH --mem-per-cpu 32G
#SBATCH --job-name car
#SBATCH --output slurm-lr-constant-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out


srun --ntasks=8 -l --multi-prog ./commands.conf