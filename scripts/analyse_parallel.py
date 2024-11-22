from mpi4py import MPI
import healpy as hp
import numpy as np
from plancklens import utils, shts
from os.path import join as opj
from iterativefg import utils as itu
import statics
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
parser.add_argument('-selected', dest='selected', nargs='+',  default = "a", help="List of selected estimators, separated by spaces.")


parser.add_argument('--fgcase', type=str, default='sommamasked', help='Foreground case')
parser.add_argument('--fgcase_B', type=str, default='', help='Foreground case')
parser.add_argument('--foreground', type=str, default='', help='Specific foreground to focus on. "" for total.')
parser.add_argument('--fgversion', type=str, default='webskysoILC', help='Simulations suite.')

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

bin_edges = np.arange(lmin_binning, lmax_binning, deltal_binning)
el = (bin_edges[1:] + bin_edges[:-1]) / 2
bin = lambda x: itu.bin_theory(x, bin_edges)[1]

directory = "/users/odarwish/scratch/"
saving_directory = "/users/odarwish/scratch/joint_map_outputs/"

suffix = lambda fgcase, fgcase_B, fgversion, settingsversion: f'S4{fgcase}{fgcase_B}{fgversion}{settingsversion}'
casi = [suffix(s, s_B, fgversion, version) for s, s_B in zip(fgcases, fgcases_B)]

def get_reconstruction_and_input(caso="S4", qe_key="ptt_bh_s", idx=0, cmbversion=""):
    return np.load(directory + f"lenscarfrecs/{caso}/{qe_key}_sim{idx:04}plancklens/phi_plm_it000_norm.npy"), utils.alm_copy(hp.read_alm(directory+f"cmbs/{cmbversion}/sim_{idx:04}_plm.fits"), lmax=lmax_qlm)

def get_reconstruction_and_input_it(caso="S4", qe_key="ptt_bh_s", idx=0, iters=[0, 1, 2, 3, 4]):
    return statics.rec.load_plms(directory + f"lenscarfrecs/{caso}/{qe_key}_sim{idx:04}plancklens/", iters), utils.alm_copy(hp.read_alm(directory+f"cmbs/{cmbversion}/sim_{idx:04}_plm.fits"), lmax=lmax_qlm)

# Split simulations across MPI processes
all_sims = np.arange(imin, imax)
local_sims = np.array_split(all_sims, size)[rank]

local_total_qe = []
local_total_rand_qe = []
local_total_qe_cross = []
local_total_rand_qe_cross = []

# Process each simulation in the local subset
for idx in local_sims:
    print(f"Rank {rank} processing sim {idx}")
    recs_qe = {c: get_reconstruction_and_input(c, qe_key, idx=idx, cmbversion=cmbversion) for c in casi}
    input = recs_qe[casi[0]][1]
    input = utils.alm_copy(input, lmax=lmax_qlm)
    ii = hp.alm2cl(input)
    iibin = bin(ii)

    local_total_qe.append(hp.alm2cl(recs_qe[casi[0]][0]))
    local_total_rand_qe.append(hp.alm2cl(recs_qe[casi[1]][0]))
    local_total_qe_cross.append(hp.alm2cl(recs_qe[casi[0]][0], input))
    local_total_rand_qe_cross.append(hp.alm2cl(recs_qe[casi[1]][0], input))

# Gather results to root process
total_qe = comm.gather(local_total_qe, root=0)
total_rand_qe = comm.gather(local_total_rand_qe, root=0)
total_qe_cross = comm.gather(local_total_qe_cross, root=0)
total_rand_qe_cross = comm.gather(local_total_rand_qe_cross, root=0)

# Process iterative results if required
if itmax >= 0:
    iters = np.arange(itmax + 1)
    local_total_qe_it = []
    local_total_rand_qe_it = []
    local_total_qe_it_cross = []
    local_total_rand_qe_it_cross = []

    for idx in local_sims:
        print(f"Rank {rank} processing iterative sim {idx}")
        recs_it = {c: get_reconstruction_and_input_it(c, qe_key, idx=idx, iters=iters) for c in casi}
        temp = []
        temp_rand = []
        temp_cross = []
        temp_rand_cross = []

        for i in iters:
            if std_rec:
                phi = recs_it[casi[0]][0][i]
                temp.append(hp.alm2cl(phi))
                temp_cross.append(hp.alm2cl(phi, input))
                phi = recs_it[casi[1]][0][i]
                temp_rand.append(hp.alm2cl(phi))
                temp_rand_cross.append(hp.alm2cl(phi, input))
            else:
                phi, noise = np.split(recs_it[casi[0]][0][i], 2)
                temp.append(hp.alm2cl(phi))
                temp_cross.append(hp.alm2cl(phi, input))
                phi, noise = np.split(recs_it[casi[1]][0][i], 2)
                temp_rand.append(hp.alm2cl(phi))
                temp_rand_cross.append(hp.alm2cl(phi, input))

        local_total_qe_it.append(temp)
        local_total_rand_qe_it.append(temp_rand)
        local_total_qe_it_cross.append(temp_cross)
        local_total_rand_qe_it_cross.append(temp_rand_cross)

    total_qe_it = comm.gather(local_total_qe_it, root=0)
    total_rand_qe_it = comm.gather(local_total_rand_qe_it, root=0)
    total_qe_it_cross = comm.gather(local_total_qe_it_cross, root=0)
    total_rand_qe_it_cross = comm.gather(local_total_rand_qe_it_cross, root=0)

    if rank == 0:
        total_qe_it = np.concatenate(total_qe_it)
        total_rand_qe_it = np.concatenate(total_rand_qe_it)
        total_qe_it_cross = np.concatenate(total_qe_it_cross)
        total_rand_qe_it_cross = np.concatenate(total_rand_qe_it_cross)
        
        np.save(saving_directory + f"total_qe_it_{qe_key}_{fg_case_out_name}_{fgversion}_{version}_{cmbversion}_{imin}_{imax}_{itmax}", total_qe_it)
        np.save(saving_directory + f"total_rand_qe_it_{qe_key}_{fg_case_out_name}_{fgversion}_{version}_{cmbversion}_{imin}_{imax}_{itmax}", total_rand_qe_it)
        np.save(saving_directory + f"total_qe_it_cross_{qe_key}_{fg_case_out_name}_{fgversion}_{version}_{cmbversion}_{imin}_{imax}_{itmax}", total_qe_it_cross)
        np.save(saving_directory + f"total_rand_qe_it_cross_{qe_key}_{fg_case_out_name}_{fgversion}_{version}_{cmbversion}_{imin}_{imax}_{itmax}", total_rand_qe_it_cross)

# Save outputs only on the root process
if rank == 0:
    total_qe = np.concatenate(total_qe)
    total_rand_qe = np.concatenate(total_rand_qe)
    total_qe_cross = np.concatenate(total_qe_cross)
    total_rand_qe_cross = np.concatenate(total_rand_qe_cross)

    np.savetxt(saving_directory + f"input_{fg_case_out_name}_{fgversion}_{version}_{cmbversion}_{imin}_{imax}_{itmax}.txt", ii)
    np.save(saving_directory + f"total_qe_{qe_key}_{fg_case_out_name}_{fgversion}_{version}_{cmbversion}_{imin}_{imax}_{itmax}", total_qe)
    np.save(saving_directory + f"total_rand_qe_{qe_key}_{fg_case_out_name}_{fgversion}_{version}_{cmbversion}_{imin}_{imax}_{itmax}", total_rand_qe)
    np.save(saving_directory + f"total_qe_cross_{qe_key}_{fg_case_out_name}_{fgversion}_{version}_{cmbversion}_{imin}_{imax}_{itmax}", total_qe_cross)
    np.save(saving_directory + f"total_rand_qe_cross_{qe_key}_{fg_case_out_name}_{fgversion}_{version}_{cmbversion}_{imin}_{imax}_{itmax}", total_rand_qe_cross)
