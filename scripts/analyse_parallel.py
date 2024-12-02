from mpi4py import MPI
import healpy as hp
import numpy as np
from plancklens import utils
from os.path import join as opj
from iterativefg import utils as itu
from delensalot.core.iterator import statics
import argparse

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Argument parsing
parser = argparse.ArgumentParser(description='Analyse the output of the joint iterative.')
parser.add_argument('--qe_key', type=str, default="p_p", help='Key of the QE')
parser.add_argument('--cmbversion', type=str, default='', help='CMB version of the output')
parser.add_argument('--version', type=str, default='', help='Version of the output')
parser.add_argument('--imin', type=int, default = 0, help='Minimum sim')
parser.add_argument('--imax', type=int, default = 3, help='Maximum sim')
parser.add_argument('--itmax', type=int, default = 3, help='Maximum iteration')
parser.add_argument('--lmax_qlm', type=int, default = 5120)
parser.add_argument('--std_rec', action='store_true')
parser.add_argument('--lmin_binning', type=int, default = 20)
parser.add_argument('--lmax_binning', type=int, default = 4000)
parser.add_argument('--deltal_binning', type=int, default = 40)
parser.add_argument('--selected', dest='selected', nargs='+',  default = "a", help="List of selected estimators, separated by spaces.")
parser.add_argument('--unlensed', action='store_true')

args = parser.parse_args()
qe_key = args.qe_key
version = args.version
cmbversion = args.cmbversion
imin = args.imin
imax = args.imax
itmax = args.itmax
lmax_qlm = args.lmax_qlm
std_rec = args.std_rec
lmin_binning = args.lmin_binning
lmax_binning = args.lmax_binning
deltal_binning = args.deltal_binning
selected = args.selected
unlensed = args.unlensed

print("Selected estimators", selected)

bin_edges = np.arange(lmin_binning, lmax_binning, deltal_binning)
el = (bin_edges[1:] + bin_edges[:-1]) / 2
bin = lambda x: itu.bin_theory(x, bin_edges)[1]

folder_ = "JOINTRECONSTRUCTION"
directory = f"/users/odarwish/scratch/{folder_}/"
saving_directory = "/users/odarwish/scratch/joint_map_outputs/"

nome = {"p": "plm", "f": "tau_lm", "a": "alpha_lm", "o": "alpha_lm" if "a" in selected else "olm"}
nome = {"p": "plm", "f": "tau_lm", "a": "alpha_lm", "o": "alpha_lm" if "a" in selected else "plm"}


size_mappa = hp.Alm.getsize(lmax_qlm)

def get_input(directory, cmbversion, idx, lmax_qlm):
    if unlensed:
        return [np.zeros(size_mappa, dtype=complex) for s in selected]
    return [utils.alm_copy(hp.read_alm(directory+f"/{cmbversion}/simswalpha/sim_{idx:04}_{nome[s]}.fits"), lmax = lmax_qlm) for s in selected]

def get_reconstruction_and_input(version="", qe_key="ptt_bh_s", idx=0, cmbversion=""):
    return np.concatenate([np.load(directory + f"{cmbversion}_version_{version}_recs/{qe_key}_sim{idx:04}{version}/{s}lm0_norm.npy") for s in selected]), get_input(directory, cmbversion, idx, lmax_qlm)

def get_reconstruction_and_input_it(version="", qe_key="ptt_bh_s", idx=0, iters = [0, 1, 2, 3, 4], cmbversion = ""):
    return statics.rec.load_plms(directory + f"{cmbversion}_version_{version}_recs/{qe_key}_sim{idx:04}{version}/", iters), get_input(directory, cmbversion, idx, lmax_qlm)

# Split simulations across MPI processes
all_sims = np.arange(imin, imax)
local_sims = np.array_split(all_sims, size)[rank]

local_total_qe = []
local_total_rand_qe = []
local_total_qe_cross = []
local_total_rand_qe_cross = []

inputs = []

nfields = len(selected)

# Process each simulation in the local subset
for idx in local_sims:
    print(f"Rank {rank} processing sim {idx}")
    recs_qe = get_reconstruction_and_input(version, qe_key, idx=idx, cmbversion=cmbversion)
    input_fields = recs_qe[1]
    input = [hp.alm2cl(x) for x in input_fields]
    inputs.append(input)

    local_total_qe.append(np.concatenate([hp.alm2cl(x) for x in np.split(recs_qe[0], nfields)]))
    #local_total_rand_qe.append(hp.alm2cl(recs_qe[casi[1]][0]))
    local_total_qe_cross.append(np.concatenate([hp.alm2cl(x, xi) for x, xi in zip(np.split(recs_qe[0], nfields), input_fields)]))
    #local_total_rand_qe_cross.append(hp.alm2cl(recs_qe[casi[1]][0], input))

# Gather results to root process
total_qe = comm.gather(local_total_qe, root=0)
#total_rand_qe = comm.gather(local_total_rand_qe, root=0)
total_qe_cross = comm.gather(local_total_qe_cross, root=0)
#total_rand_qe_cross = comm.gather(local_total_rand_qe_cross, root=0)

total_inputs = comm.gather(inputs, root=0)

# Process iterative results if required
if itmax >= 0:
    iters = np.arange(itmax + 1)
    local_total_qe_it = []
    #local_total_rand_qe_it = []
    local_total_qe_it_cross = []
    #local_total_rand_qe_it_cross = []

    for idx in local_sims:
        print(f"Rank {rank} processing iterative sim {idx}")
        recs_it = get_reconstruction_and_input_it(version, qe_key, idx=idx, iters=iters, cmbversion=cmbversion)
        temp = []
        #temp_rand = []
        temp_cross = []
        #temp_rand_cross = []

        for i in iters:
            if std_rec:
                phi = recs_it[0][i]
                inputs = recs_it[1]
                temp.append(np.concatenate([hp.alm2cl(x) for x in np.split(phi, nfields)]))
                temp_cross.append(np.concatenate([hp.alm2cl(x, input) for x, input in zip(np.split(phi, nfields), inputs)]))
            else:
                phi = recs_it[0][i]
                inputs = recs_it[1]
                temp.append(np.concatenate([hp.alm2cl(x) for x in np.split(phi, nfields)]))
                temp_cross.append(np.concatenate([hp.alm2cl(x, input) for x, input in zip(np.split(phi, nfields), inputs)]))

        local_total_qe_it.append(temp)
        local_total_qe_it_cross.append(temp_cross)

    total_qe_it = comm.gather(local_total_qe_it, root=0)
    total_qe_it_cross = comm.gather(local_total_qe_it_cross, root=0)

    if rank == 0:
        total_qe_it = np.concatenate(total_qe_it)
        total_qe_it_cross = np.concatenate(total_qe_it_cross)
        
        np.save(saving_directory + f"total_qe_it_{qe_key}_{version}_{cmbversion}_{imin}_{imax}_{itmax}", total_qe_it)
        np.save(saving_directory + f"total_qe_it_cross_{qe_key}_{version}_{cmbversion}_{imin}_{imax}_{itmax}", total_qe_it_cross)

# Save outputs only on the root process
if rank == 0:
    total_qe = np.concatenate(total_qe)
    total_qe_cross = np.concatenate(total_qe_cross)
    total_inputs = np.concatenate(total_inputs)

    np.save(saving_directory + f"input_{version}_{cmbversion}_{imin}_{imax}_{itmax}", total_inputs)
    np.save(saving_directory + f"total_qe_{qe_key}_{version}_{cmbversion}_{imin}_{imax}_{itmax}", total_qe)
    np.save(saving_directory + f"total_qe_cross_{qe_key}_{version}_{cmbversion}_{imin}_{imax}_{itmax}", total_qe_cross)