#!/bin/sh -l
#SBATCH --job-name=itbh
#SBATCH --time=00:30:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --nodes=8
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=lp44
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=24
export OMP_PLACES=threads
export OMP_PROC_BIND=false

#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/config_full_alpha_disabled_lensing_cmb_s4.yaml
#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/config_full_alpha_lensing_cmb_s4.yaml
#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/config_full_no_alpha_no_curl_lensing_cmb_s4.yaml
#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/config_full_tau_disabled_lensing_cmb_s4.yaml
#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/config_full_alpha_no_curl_lensing_cmb_s4.yaml


#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/config_full.yaml
srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/config_full_cmb_s4.yaml
srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/config_full_a_d_disabled_cmb_s4.yaml