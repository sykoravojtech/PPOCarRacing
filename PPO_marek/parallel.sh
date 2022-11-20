#!/bin/bash
#SBATCH --ntasks 8
#SBATCH --nodes 8

#SBATCH --partition amd
#SBATCH --ntasks-per-node 1 
#SBATCH --cpus-per-task 2
#SBATCH --mem-per-cpu 8G
#SBATCH --job-name car
#SBATCH --output slurm-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out


srun --ntasks=8 -l --multi-prog ./commands.conf