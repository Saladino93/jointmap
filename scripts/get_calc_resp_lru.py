from mpi4py import MPI
import numpy as np
import healpy as hp
import os
from delensalot.core.iterator import statics
from collections import OrderedDict


def get_total_pairs(N):
    """Calculate total number of unique pairs"""
    return (N * (N-1)) // 2

def get_pair_from_index(k, N):
    """Convert a linear index k into the corresponding (i,j) pair"""
    # Find row (i) and position in row (j)
    i = N - 2 - int(np.sqrt(-8*k + 4*N*(N-1)-7)/2.0 - 0.5)
    j = k + i + 1 - N*(N-1)//2 + (N-i)*((N-i)-1)//2
    return i, j

class LRUCache:
    """Least Recently Used (LRU) cache with a fixed size"""
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        # Move the accessed item to the end (most recently used)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            # Move the updated item to the end
            self.cache.move_to_end(key)
        else:
            # Remove least recently used item if cache is full
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

def process_pairs(start_idx, end_idx, NN, base_dir, cache_size=2):
    """Process pairs with caching for plms data"""
    # Initialize caches
    plms_cache = LRUCache(cache_size)
    rec = statics.rec()
    itrs = [0, 15]
    
    means = {i: 0 for i in itrs}
    counters = {i: 0 for i in itrs}

    for k in range(start_idx, end_idx):
        i, j = get_pair_from_index(k, NN)
        
        # Try to get plms_i from cache
        directory_i = f"{base_dir}p_p_sim{i:04}apo_new_official_check_factor_2_v2/"
        plms_i = plms_cache.get(i)
        if plms_i is None:
            plms_i = rec.load_plms(directory_i, itrs)
            plms_cache.put(i, plms_i)
        
        # Try to get plms_j from cache
        directory_j = f"{base_dir}p_p_sim{j:04}apo_new_official_check_factor_2_v2/"
        plms_j = plms_cache.get(j)
        if plms_j is None:
            plms_j = rec.load_plms(directory_j, itrs)
            plms_cache.put(j, plms_j)

        # Process the pairs
        for idx, itr in enumerate(itrs):
            means[itr] += hp.alm2cl(np.split(plms_i[idx], 3)[0], 
                                  np.split(plms_j[idx], 3)[0])
            counters[itr] += 1
            
    return means, counters

# Main code
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

NN = 128
base_dir = "/home/users/d/darwish/scratch/JOINTRECONSTRUCTION/apo_new_v2_version_apo_new_official_check_factor_2_v2_recs/"

# First part remains the same...
# [Your existing code for the first part]

# Second part with caching
total_pairs = get_total_pairs(NN)
pairs_per_rank = total_pairs // size + (1 if rank < total_pairs % size else 0)
start_idx = rank * (total_pairs // size) + min(rank, total_pairs % size)
end_idx = start_idx + pairs_per_rank

# Process pairs with caching
means, counters = process_pairs(start_idx, end_idx, NN, base_dir)

# Reduce and save results
itrs = [0, 15]
for itr in itrs:
    total_mean = comm.reduce(means[itr], op=MPI.SUM, root=0)
    total_counter = comm.reduce(counters[itr], op=MPI.SUM, root=0)

    if rank == 0:
        final_mean = total_mean / total_counter
        np.savetxt(f"/home/users/d/darwish/scratch/final_mean_{NN}_{itr}.txt", final_mean)