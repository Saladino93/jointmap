#!/bin/sh -l
#SBATCH --job-name=itbh
#SBATCH --time=00:10:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --nodes=8
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=lp44
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=12
export OMP_PLACES=threads
export OMP_PROC_BIND=false

#srun python ../scripts/param_joint.py -k p_p -itmax 0 -imin 0 -imax 31 -v test_3_low_lmax -joint_module -cmb_version test_3 -no_curl -no_tau -no_birefringence -selected p o -tol 6 -beam 1. -nlev_t 1. -lmax_ivf 3000 -mmax_ivf 3000 -lmin_elm 30 -lmin_blm 200 -lmax_unl 4000 -mmax_unl 4000 -lmax_qlm 5120
#srun python ../scripts/param_joint.py -k p_p -itmax 0 -imin 0 -imax 31 -v test_3_low_lmax_randomized -joint_module -cmb_version test_3 -no_curl -no_tau -no_birefringence -selected p o -tol 6 -beam 1. -nlev_t 1. -lmax_ivf 3000 -mmax_ivf 3000 -lmin_elm 30 -lmin_blm 200 -lmax_unl 4000 -mmax_unl 4000 -lmax_qlm 5120 -randomize


srun python ../scripts/analyse_parallel.py --qe_key p_p --cmbversion test_3 --version test_3_low_lmax --imin 0 --imax 31 --itmax 0 --lmax_qlm 5120 --selected p o
srun python ../scripts/analyse_parallel.py --qe_key p_p --cmbversion test_3 --version test_3_low_lmax_randomized --imin 0 --imax 31 --itmax 0 --lmax_qlm 5120 --selected p o