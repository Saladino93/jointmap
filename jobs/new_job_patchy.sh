#!/bin/sh
#SBATCH --job-name=check_response
#SBATCH --time=01:00:00
#SBATCH --partition=shared-cpu
#SBATCH --output=/home/users/d/darwish/scratch/slurms/lenscarf_check_response_%J.out
#SBATCH --cpus-per-task=30
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=2


module load foss/2020b CFITSIO/4.0.0 GSL Autotools

conda activate lenscarf

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HOME=/home/users/d/darwish/
export SCRATCH=/home/users/d/darwish/scratch


srun ~/.conda/envs/lenscarf/bin/python ../scripts/param_joint.py -c ../scripts/configs/official_spt3g_patchy.yaml
#srun ~/.conda/envs/lenscarf/bin/python ../scripts/param_joint.py -c ../scripts/configs/official_new.yaml
#srun ~/.conda/envs/lenscarf/bin/python ../scripts/param_joint.py -c ../scripts/configs/official_multiple_len_resp.yaml
#srun ~/.conda/envs/lenscarf/bin/python ../scripts/param_joint.py -c ../scripts/configs/official_prova_no_alpha.yaml