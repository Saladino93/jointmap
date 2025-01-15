#!/bin/sh -l
#SBATCH --job-name=analyze
#SBATCH --time=00:20:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --nodes=16
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=lp44
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=24
export OMP_PLACES=threads
export OMP_PROC_BIND=false

#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_so_a_disabled.yaml
#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_a_disabled.yaml
#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official.yaml
#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_multiple.yaml
#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_multiple_disabled.yaml
#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_deep_multiple.yaml
#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_a_disabled_scale_dependent.yaml
#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_lensing_only_a_disabled.yaml


#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_lensing_only.yaml
#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_a_disabled_check_factor_2.yaml
#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_check_factor_2.yaml
#srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_so_a_disabled_check_factor_2.yaml

srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_multiple_disabled.yaml
srun python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_multiple.yaml
