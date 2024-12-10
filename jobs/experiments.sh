#!/bin/sh -l
#SBATCH --job-name=itbh
#SBATCH --time=04:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=lp44
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=24
export OMP_PLACES=threads
export OMP_PROC_BIND=false

#srun python ../scripts/param_joint.py -k p_p -itmax 15 -imin 0 -imax 0 -v experiment_tau_1 -joint_module -cmb_version experiment_tau_only -no_lensing -no_birefringence -no_curl -selected f -tol 6 -beam 1. -nlev_t 0.35 -lmax_ivf 5000 -mmax_ivf 5000 -lmin_elm 30 -lmin_blm 200 -lmax_unl 5000 -mmax_unl 5000  -lmax_qlm 5120
srun python ../scripts/param_joint.py -k p_p -itmax 15 -imin 0 -imax 0 -v experiment_tau_lensing_2 -joint_module -cmb_version experiment_tau_lensing -no_birefringence -no_curl -selected f p o -tol 6 -beam 1. -nlev_t 0.35 -lmax_ivf 5000 -mmax_ivf 5000 -lmin_elm 30 -lmin_blm 200 -lmax_unl 5000 -mmax_unl 5000  -lmax_qlm 5120



python ./param_joint.py -k p_p -itmax 15 -imin 0 -imax 0 -v experiment_tau_1 -joint_module -cmb_version experiment_tau_only -no_lensing -no_birefringence -no_curl -selected f -tol 6 -beam 1. -nlev_t 0.35 -lmax_ivf 5000 -mmax_ivf 5000 -lmin_elm 30 -lmin_blm 200 -lmax_unl 5000 -mmax_unl 5000  -lmax_qlm 5120