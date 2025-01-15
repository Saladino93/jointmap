#!/bin/sh -l
#SBATCH --job-name=official_a_disabled
#SBATCH --time=01:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --constraint=gpu
#SBATCH --nodes=64
#SBATCH --output=/scratch/snx3000/odarwish/slurms/lenscarf-slurm-%J.out
#SBATCH --account=lp44
#SBATCH --partition=normal

module load daint-gpu
module load cray-python/3.9.4.1
source /users/odarwish/bin/lenscarf/bin/activate

export OMP_NUM_THREADS=24
export OMP_PLACES=threads
export OMP_PROC_BIND=false

#srun python ../scripts/param_joint.py -c ../scripts/configs/official_a_disabled.yaml
#srun python ../scripts/param_joint.py -c ../scripts/configs/official_a_disabled_scale_dependent.yaml
srun python ../scripts/param_joint.py -c ../scripts/configs/official_lensing_only.yaml