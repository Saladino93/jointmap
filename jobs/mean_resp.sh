#!/bin/sh
#SBATCH --job-name=mean_resp
#SBATCH --time=01:00:00
#SBATCH --partition=shared-cpu 
#SBATCH --output=/home/users/d/darwish/scratch/slurms/mean_resp_%J.out
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=30
#SBATCH --nodes=16

module load foss/2020b CFITSIO/4.0.0 GSL Autotools

conda activate lenscarf

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HOME=/home/users/d/darwish/
export SCRATCH=/home/users/d/darwish/scratch

srun ~/.conda/envs/lenscarf/bin/python ../scripts/get_calc_resp_lru.py