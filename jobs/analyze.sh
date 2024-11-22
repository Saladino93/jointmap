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


srun python ../scripts/analyse_parallel.py --qe_key p_p --cmbversion phi_alpha --version phi_omega_alpha_joint --imin 0 --imax 63 --itmax 12 --lmax_qlm 5120 --selected a p o
srun python ../scripts/analyse_parallel.py --qe_key p_p --cmbversion phi_alpha --version phi_omega_joint --imin 0 --imax 63 --itmax 15 --lmax_qlm 5120 --selected p o
srun python ../scripts/analyse_parallel.py --qe_key p_p --cmbversion alpha --version alpha_joint --imin 0 --imax 31 --itmax 15 --lmax_qlm 5120 --selected a