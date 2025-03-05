#!/bin/sh
#SBATCH --job-name=check_response
#SBATCH --time=00:30:00
#SBATCH --partition=shared-cpu 
#SBATCH --output=/home/users/d/darwish/scratch/slurms/lenscarf_check_response_%J.out
#SBATCH --cpus-per-task=30
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1


module load foss/2020b CFITSIO/4.0.0 GSL Autotools

conda activate lenscarf

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HOME=/home/users/d/darwish/
export SCRATCH=/home/users/d/darwish/scratch


#srun ~/.conda/envs/lenscarf/bin/python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_new.yaml
#srun ~/.conda/envs/lenscarf/bin/python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_prova_no_alpha.yaml
#srun ~/.conda/envs/lenscarf/bin/python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_multiple_len_resp.yaml
#srun ~/.conda/envs/lenscarf/bin/python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_multiple.yaml
#srun ~/.conda/envs/lenscarf/bin/python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_spt3g.yaml
#srun ~/.conda/envs/lenscarf/bin/python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_spt3g_patchy.yaml
#srun ~/.conda/envs/lenscarf/bin/python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_spt3g.yaml
#srun ~/.conda/envs/lenscarf/bin/python ../scripts/complete-analysis-parallel.py -c ../scripts/configs/official_a_mf.yaml

srun ~/.conda/envs/lenscarf/bin/python ../scripts/complete-analysis_standard.py -c ../scripts/configs/official_new_a.yaml
#srun ~/.conda/envs/lenscarf/bin/python ../scripts/complete-analysis_standard.py -c ../scripts/configs/official_a_mf.yaml
