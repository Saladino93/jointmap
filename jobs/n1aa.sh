#!/bin/sh
#SBATCH --job-name=check_response
#SBATCH --time=00:15:00
#SBATCH --partition=debug-cpu 
#SBATCH --output=/home/users/d/darwish/scratch/slurms/lenscarf_check_response_%J.out
#SBATCH --cpus-per-task=2
#SBATCH --ntasks-per-node=60
#SBATCH --nodes=1


module load foss/2020b CFITSIO/4.0.0 GSL Autotools

conda activate lenscarf

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HOME=/home/users/d/darwish/
export SCRATCH=/home/users/d/darwish/scratch

#srun ~/.conda/envs/lenscarf/bin/python ../scripts/derotation_n1.py
#srun ~/.conda/envs/lenscarf/bin/python ../scripts/noise_biases_check.py
#srun ~/.conda/envs/lenscarf/bin/python ../scripts/noise_biases.py
#srun ~/.conda/envs/lenscarf/bin/python ../scripts/quickn1.py
srun ~/.conda/envs/lenscarf/bin/python ../scripts/n1s.py