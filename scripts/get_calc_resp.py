from mpi4py import MPI
import numpy as np
import healpy as hp
import os
from delensalot.core.iterator import statics

def get_total_pairs(N):
    """Calculate total number of unique pairs"""
    return (N * (N-1)) // 2

def get_pair_from_index(k, N):
    """Convert a linear index k into the corresponding (i,j) pair"""
    # Find row (i) and position in row (j)
    i = N - 2 - int(np.sqrt(-8*k + 4*N*(N-1)-7)/2.0 - 0.5)
    j = k + i + 1 - N*(N-1)//2 + (N-i)*((N-i)-1)//2
    return i, j

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

NN = 128
mean = 0
counter = 0

# Calculate total number of pairs
total_pairs = get_total_pairs(NN)

# Distribute pairs across ranks
pairs_per_rank = total_pairs // size + (1 if rank < total_pairs % size else 0)
start_idx = rank * (total_pairs // size) + min(rank, total_pairs % size)
end_idx = start_idx + pairs_per_rank

almfile = "alm0_norm.npy"
base_dir = "/home/users/d/darwish/scratch/JOINTRECONSTRUCTION/apo_new_v2_version_apo_new_official_check_factor_2_v2_recs/"

# Process assigned pairs
for k in range(start_idx, end_idx):
    i, j = get_pair_from_index(k, NN)
    
    # Construct paths and load data
    directory_i = f"{base_dir}p_p_sim{i:04}apo_new_official_check_factor_2_v2/"
    directory_j = f"{base_dir}p_p_sim{j:04}apo_new_official_check_factor_2_v2/"
    
    alm_i = np.load(directory_i + almfile)
    alm_j = np.load(directory_j + almfile)
    
    mean += hp.alm2cl(alm_i, alm_j)
    counter += 1

# Gather results from all ranks
total_mean = comm.reduce(mean, op=MPI.SUM, root=0)
total_counter = comm.reduce(counter, op=MPI.SUM, root=0)

if rank == 0:
    final_mean = total_mean / total_counter
    np.savetxt(f"/home/users/d/darwish/scratch/final_mean_{NN}.txt", final_mean)


itrs = [0, 15]

means = {i: 0 for i in itrs}
counters = {i: 0 for i in itrs}

for k in range(start_idx, end_idx):
    i, j = get_pair_from_index(k, NN)
    
    # Construct paths and load data
    directory_i = f"{base_dir}p_p_sim{i:04}apo_new_official_check_factor_2_v2/"
    directory_j = f"{base_dir}p_p_sim{j:04}apo_new_official_check_factor_2_v2/"
    
    rec = statics.rec()
    plms_i = rec.load_plms(directory_i, itrs)#could also have a limited dictionary as a cache
    plms_j = rec.load_plms(directory_j, itrs)

    for idx, itr in enumerate(itrs):
        means[itr] += hp.alm2cl(np.split(plms_i[idx], 3)[0], np.split(plms_j[idx], 3)[0])
        counters[itr] += 1

for itr in itrs:
    total_mean = comm.reduce(means[itr], op=MPI.SUM, root=0)
    total_counter = comm.reduce(counters[itr], op=MPI.SUM, root=0)

    if rank == 0:
        final_mean = total_mean / total_counter
        np.savetxt(f"/home/users/d/darwish/scratch/final_mean_{NN}_{itr}.txt", final_mean)




