#!/bin/sh -l
#SBATCH --job-name=analyse
#SBATCH --time=00:40:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --nodes=16
#SBATCH --output=/scratch/snx3000/odarwish/slurms/analyse-%J.out
#SBATCH --account=sm80
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=24
export OMP_PLACES=threads
export OMP_PROC_BIND=false

srun ~/.conda/envs/lenscarf/bin/python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/spt3g.yaml