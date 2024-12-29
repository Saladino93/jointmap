#!/bin/sh -l
#SBATCH --job-name=itbh
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=sm80
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=24
export OMP_PLACES=threads
export OMP_PROC_BIND=false

srun python ../scripts/param_joint.py -k p_p -itmax 20 -imin 99 -imax 99 -v test_randomizing_alpha_new_low_ell -joint_module -cmb_version test_randomizing_alpha_new -no_curl -no_tau -selected p o -tol 6 -beam 1. -nlev_t 1. -lmax_ivf 4000 -mmax_ivf 4000 -lmin_elm 30 -lmin_blm 30 -lmax_unl 4000 -mmax_unl 4000 -lmax_qlm 5120
