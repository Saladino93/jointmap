#!/bin/sh -l
#SBATCH --job-name=itbh
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --nodes=32
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=sm80
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=24
export OMP_PLACES=threads
export OMP_PROC_BIND=false

#srun python ../scripts/param_joint.py -k p_p -itmax 10 -imin 64 -imax 127 -v phi_omega_alpha_joint -joint_module -cmb_version phi_alpha -no_tau -selected a p o -tol 7 -beam 1. -nlev_t 1. -lmax_ivf 4000 -mmax_ivf 4000 -lmin_elm 30 -lmin_blm 30 -lmax_unl 4000 -mmax_unl 4000
#srun python ../scripts/param_joint.py -k p_p -itmax 10 -imin 64 -imax 127 -v phi_omega_alpha_joint_randomized -joint_module -cmb_version phi_alpha -no_tau -selected a p o -tol 7 -beam 1. -nlev_t 1. -lmax_ivf 4000 -mmax_ivf 4000 -lmin_elm 30 -lmin_blm 30 -lmax_unl 4000 -mmax_unl 4000 -randomize
#srun python ../scripts/analyse_parallel.py --qe_key p_p --cmbversion phi_alpha --version phi_omega_alpha_joint --imin 0 --imax 127 --itmax 10 --lmax_qlm 5120 --selected a p o
#srun python ../scripts/analyse_parallel.py --qe_key p_p --cmbversion phi_alpha --version phi_omega_alpha_joint_randomized --imin 0 --imax 127 --itmax 10 --lmax_qlm 5120 --selected a p o
#srun python ./param_joint.py -k p_p -itmax 10 -imin 0 -imax 0 -v phi_tau_joint_prova -joint_module -cmb_version phi_tau_joint_prova -no_birefringence -selected f p o -tol 7 -beam 1. -nlev_t 1. -lmax_ivf 4000 -mmax_ivf 4000 -lmin_elm 30 -lmin_blm 30 -lmax_unl 4000 -mmax_unl 4000
#srun python ../scripts/param_joint.py -k p_p -itmax 30 -imin 99 -imax 99 -v test_randomizing_new -joint_module -cmb_version test_randomizing_new -no_curl -no_tau -no_birefringence -selected p o -tol 6 -beam 1. -nlev_t 1. -lmax_ivf 4000 -mmax_ivf 4000 -lmin_elm 30 -lmin_blm 200 -lmax_unl 4000 -mmax_unl 4000 -lmax_qlm 5120
