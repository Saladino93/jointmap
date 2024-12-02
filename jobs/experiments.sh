#!/bin/sh -l
#SBATCH --job-name=itbh
#SBATCH --time=02:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
#SBATCH --constraint=gpu
#SBATCH --nodes=2
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=lp44
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=12
export OMP_PLACES=threads
export OMP_PROC_BIND=false

srun python ../scripts/param_joint.py -k p_p -itmax 15 -imin 0 -imax 0 -v experiment_tau_only_special_estimator -joint_module -cmb_version experiment_tau_only -no_birefringence -no_lensing -no_curl -selected f -tol 6 -beam 1. -nlev_t 0.1 -lmax_ivf 5000 -mmax_ivf 5000 -lmin_elm 30 -lmin_blm 200 -lmax_unl 5000 -mmax_unl 5000  -lmax_qlm 5120


srun python ../scripts/param_joint.py -k p_p -itmax 15 -imin 0 -imax 0 -v experiment_tau_only_special -joint_module -cmb_version experiment_tau_only -no_birefringence -no_lensing -no_curl -selected f p o -tol 6 -beam 1. -nlev_t 0.1 -lmax_ivf 5000 -mmax_ivf 5000 -lmin_elm 30 -lmin_blm 200 -lmax_unl 5000 -mmax_unl 5000  -lmax_qlm 5120
srun python ../scripts/param_joint.py -k p_p -itmax 15 -imin 0 -imax 3 -v experiment_tau_lensing -joint_module -cmb_version experiment_tau_lensing -no_birefringence -no_curl -selected f p o -tol 6 -beam 1. -nlev_t 1. -lmax_ivf 4000 -mmax_ivf 4000 -lmin_elm 30 -lmin_blm 200 -lmax_unl 4000 -mmax_unl 4000  -lmax_qlm 5120
srun python ../scripts/param_joint.py -k p_p -itmax 15 -imin 0 -imax 3 -v experiment_tau_lensing_curl -joint_module -cmb_version experiment_tau_lensing_curl -no_birefringence -selected f p o -tol 6 -beam 1. -nlev_t 1. -lmax_ivf 4000 -mmax_ivf 4000 -lmin_elm 30 -lmin_blm 200 -lmax_unl 4000 -mmax_unl 4000  -lmax_qlm 5120


#srun python ../scripts/analyse_parallel.py --qe_key p_p --cmbversion phi --version phi_omega_joint --imin 0 --imax 63 --itmax 3 --lmax_qlm 5120 --selected p o
#srun python ../scripts/analyse_parallel.py --qe_key p_p --cmbversion phi --version phi_omega_joint_randomized --imin 0 --imax 63 --itmax 3 --lmax_qlm 5120 --selected p o