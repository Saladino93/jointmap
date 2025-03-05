from mpi4py import MPI
import healpy as hp
import numpy as np
from plancklens import utils
import yaml
import os
from os.path import join as opj
from iterativefg import utils as itu
from delensalot.core.iterator import statics
import argparse
from types import SimpleNamespace

############################################################################################################################

import lenspyx
from plancklens import shts

def _get_dlm(dlm):
    dclm = np.zeros_like(dlm)
    lmax_dlm = hp.Alm.getlmax(dlm.size, -1)
    mmax_dlm = lmax_dlm
    # potentials to deflection
    p2d = np.sqrt(np.arange(lmax_dlm + 1, dtype=float) * np.arange(1, lmax_dlm + 2, dtype=float))
    #p2d[:self.lmin_dlm] = 0
    hp.almxfl(dlm, p2d, mmax_dlm, inplace=True)
    hp.almxfl(dclm, p2d, mmax_dlm, inplace=True)
    return dlm, dclm, lmax_dlm, mmax_dlm

def lensed_alpha(alm0_input, plm0_input):
    dlm, dclm, lmax_dlm, mmax_dlm = _get_dlm(plm0_input)
    lmax_map = hp.Alm.getlmax(alm0_input.size)
    nside_lens = 2048
    a0_len = lenspyx.alm2lenmap(
        alm0_input, [dlm, None], geometry=('healpix', {'nside': nside_lens}),
        epsilon=1e-8, verbose=0)
    alm0_len = shts.map2alm(a0_len, lmax = lmax_map)
    return alm0_len


############################################################################################################################

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Define command line arguments with defaults
parser = argparse.ArgumentParser(description='Analyse the output of the joint iterative.')
parser.add_argument("-c", "--config", type=str, help="Path to configuration file", default=None)
parser.add_argument('-k', dest='qe_key', type=str, default='p_p', help='rec. type')
parser.add_argument('-imin', dest='imin', type=int, default=0, help='minimal sim index')
parser.add_argument('-imax', dest='imax', type=int, default=0, help='maximal sim index')
parser.add_argument('-itmax', dest='itmax', type=int, default=3, help='maximal iter index')
parser.add_argument('-v', dest='v', type=str, default='', help='iterator version')
parser.add_argument('-cmb_version', type=str, default="")
parser.add_argument('-lmax_qlm', dest='lmax_qlm', type=int, default=5120, help='lmax_qlm')
parser.add_argument('-selected', dest='selected', nargs='+', default=["a"], help="List of selected estimators, separated by spaces")
parser.add_argument('--std_rec', action='store_true', help='standard reconstruction')
parser.add_argument('--unlensed', action='store_true', help='use unlensed spectra')

args = parser.parse_args()

# First get command line arguments
cmd_args = vars(args)

# Load configuration file if provided and prioritize its values
if args.config:
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    # Merge with command line args, giving priority to config file
    merged_args = {**cmd_args, **config}  # Config values override command line args
else:
    merged_args = cmd_args
    
args = SimpleNamespace(**merged_args)

print(args)

# Process selected estimators
def process_strings(strings):
    """Process selected estimators to handle disabled operators"""
    return [s[0] if len(s) == 2 else s for s in strings], [len(s) == 2 for s in strings]

selected, disabled = process_strings(args.selected)
disabled_dict = dict(zip(selected, disabled))

if rank == 0:
    print("Selected estimators (disabled operator?):", disabled_dict)
    print(f"Processing simulations {args.imin} to {args.imax}")
    print(f"Max iteration: {args.itmax}")

# Binning parameters
binning = getattr(args, 'binning', {
    'lmin': 20,
    'lmax': 4000,
    'delta': 40
})
lmin_binning = binning.get('lmin', 20)
lmax_binning = binning.get('lmax', 4000)
deltal_binning = binning.get('delta', 40)

bin_edges = np.arange(lmin_binning, lmax_binning, deltal_binning)
el = (bin_edges[1:] + bin_edges[:-1]) / 2
bin = lambda x: itu.bin_theory(x, bin_edges)[1]

# Set up paths
scratch = os.getenv("SCRATCH", "/scratch")
folder_ = "JOINTRECONSTRUCTION"
directory = os.path.join(scratch, folder_)
saving_directory = os.path.join(scratch, "joint_map_outputs")

if rank == 0:
    os.makedirs(saving_directory, exist_ok=True)

# Field name mapping
nome = {
    "p": "plm",
    "f": "tau_lm",
    "a": "alpha_lm",
    "o": "olm" #"alpha_lm" if "a" in selected else "olm"
}

nome_down = {
    "p": "tau_lm",
    "f": "plm",
    "a": "olm",
    "o": "alpha_lm"
}

size_mappa = hp.Alm.getsize(args.lmax_qlm)


def read_input(path, lmax_qlm):
    if not os.path.exists(path):
        return np.zeros(hp.Alm.getsize(lmax_qlm), dtype=complex)
    else:
        return utils.alm_copy(hp.read_alm(path), lmax = lmax_qlm)



def get_input(directory, cmbversion, idx, lmax_qlm):
    """Load input alms for each selected field"""
    if args.unlensed:
        return [np.zeros(size_mappa, dtype=complex) for s in selected]
    return [read_input(os.path.join(directory, cmbversion, "simswalpha", f"sim_{idx:04}_{nome[s]}.fits"), lmax_qlm) for s in selected]

def get_input_down(directory, cmbversion, idx, lmax_qlm):
    """Load input alms for each selected field"""
    return [read_input(os.path.join(directory, cmbversion, "simswalpha", f"sim_{idx:04}_{nome_down[s]}.fits"), lmax_qlm) for s in selected]

def get_input_lensed(directory, cmbversion, idx, lmax_qlm, key):
    if key == "a":
        qp = nome["p"]
        plm0_input = read_input(os.path.join(directory, cmbversion, "simswalpha", f"sim_{idx:04}_{qp}.fits"), lmax_qlm)
        qa = nome["a"]
        alm0_input = read_input(os.path.join(directory, cmbversion, "simswalpha", f"sim_{idx:04}_{qa}.fits"), lmax_qlm)
        alm0_input_lensed = lensed_alpha(alm0_input, plm0_input)
        return utils.alm_copy(alm0_input_lensed, lmax = lmax_qlm)
    else:
        return np.zeros(hp.Alm.getsize(lmax_qlm), dtype=np.complex128)

def get_reconstruction_and_input(version, qe_key, idx, cmbversion):
    """Load reconstructed alms and input alms"""
    rec_path = os.path.join(directory, f"{cmbversion}_version_{version}_recs", 
                           f"{qe_key}_sim{idx:04}{version}")
    recs = [np.load(os.path.join(rec_path, f"{s}lm0_norm.npy")) for s in selected]
    return np.concatenate(recs), get_input(directory, cmbversion, idx, args.lmax_qlm), get_input_down(directory, cmbversion, idx, args.lmax_qlm)

def get_reconstruction_and_input_it(version, qe_key, idx, iters, cmbversion):
    """Load iterative reconstruction alms and input alms"""
    rec_path = os.path.join(directory, f"{cmbversion}_version_{version}_recs",
                           f"{qe_key}_sim{idx:04}{version}")
    return statics.rec.load_plms(rec_path, iters), get_input(directory, cmbversion, idx, args.lmax_qlm), get_input_down(directory, cmbversion, idx, args.lmax_qlm)

# Split simulations across MPI processes
all_sims = np.arange(args.imin, args.imax)
local_sims = np.array_split(all_sims, size)[rank]

local_total_qe = []
local_total_qe_cross = []
inputs = []
inputs_down = []
inputs_lensed = []
nfields = len(selected)

# Process each simulation in the local subset
for idx in local_sims:
    if rank == 0:
        print(f"Processing sim {idx}")
    recs_qe = get_reconstruction_and_input(args.v, args.qe_key, idx=idx, cmbversion=args.cmb_version)
    input_fields = recs_qe[1]
    input = [hp.alm2cl(x) for x in input_fields]
    inputs.append(input)

    input_down = [hp.alm2cl(x) for x in recs_qe[2]]
    inputs_down.append(input_down)

    input_lensed = [hp.alm2cl(get_input_lensed(directory, args.cmb_version, idx, args.lmax_qlm, s)) for s in selected]
    inputs_lensed.append(input_lensed)
    
    local_total_qe.append(np.concatenate([hp.alm2cl(x) for x in np.split(recs_qe[0], nfields)]))
    local_total_qe_cross.append(np.concatenate([hp.alm2cl(x, xi) for x, xi in zip(np.split(recs_qe[0], nfields), input_fields)]))

# Gather results to root process
total_qe = comm.gather(local_total_qe, root=0)
total_qe_cross = comm.gather(local_total_qe_cross, root=0)
total_inputs = comm.gather(inputs, root=0)
total_inputs_down = comm.gather(inputs_down, root=0)
total_inputs_lensed = comm.gather(inputs_lensed, root=0)

# Process iterative results if required
if args.itmax >= 0:
    iters = np.arange(args.itmax + 1)
    local_total_qe_it = []
    local_total_qe_it_cross = []
    local_total_qe_it_cross_down = []
    local_total_qe_it_lensed = []

    for idx in local_sims:
        if rank == 0:
            print(f"Processing iterative sim {idx}")
        recs_it = get_reconstruction_and_input_it(args.v, args.qe_key, idx=idx, 
                                                iters=iters, cmbversion=args.cmb_version)
        temp = []
        temp_cross = []
        temp_cross_down = []
        temp_cross_lensed = []


        inputs = recs_it[1]
        inputs_down = recs_it[2]
        inputs_lensed = [get_input_lensed(directory, args.cmb_version, idx, args.lmax_qlm, s) for s in selected]

        for i in iters:
            phi = recs_it[0][i]
            temp.append(np.concatenate([hp.alm2cl(x) for x in np.split(phi, nfields)]))
            temp_cross.append(np.concatenate([hp.alm2cl(x, input) for x, input in zip(np.split(phi, nfields), inputs)]))
            temp_cross_down.append(np.concatenate([hp.alm2cl(x, input) for x, input in zip(np.split(phi, nfields), inputs_down)]))
            temp_cross_lensed.append(np.concatenate([hp.alm2cl(x, input) for x, input in zip(np.split(phi, nfields), inputs_lensed)]))

        local_total_qe_it.append(temp)
        local_total_qe_it_cross.append(temp_cross)
        local_total_qe_it_cross_down.append(temp_cross_down)
        local_total_qe_it_lensed.append(temp_cross_lensed)

    total_qe_it = comm.gather(local_total_qe_it, root=0)
    total_qe_it_cross = comm.gather(local_total_qe_it_cross, root=0)
    total_qe_it_cross_down = comm.gather(local_total_qe_it_cross_down, root=0)
    total_qe_it_lensed = comm.gather(local_total_qe_it_lensed, root=0)

    if rank == 0:
        total_qe_it = np.concatenate(total_qe_it)
        total_qe_it_cross = np.concatenate(total_qe_it_cross)
        total_qe_it_cross_down = np.concatenate(total_qe_it_cross_down)
        total_qe_it_lensed = np.concatenate(total_qe_it_lensed)
        
        # Save iterative results
        np.save(opj(saving_directory, 
                f"total_qe_it_{args.qe_key}_{args.v}_{args.cmb_version}_{args.imin}_{args.imax}_{args.itmax}"), 
                total_qe_it)
        np.save(opj(saving_directory, 
                f"total_qe_it_cross_{args.qe_key}_{args.v}_{args.cmb_version}_{args.imin}_{args.imax}_{args.itmax}"), 
                total_qe_it_cross)
        np.save(opj(saving_directory,
                f"total_qe_it_cross_down_{args.qe_key}_{args.v}_{args.cmb_version}_{args.imin}_{args.imax}_{args.itmax}"),
                total_qe_it_cross_down)

        np.save(opj(saving_directory,
                f"total_qe_it_cross_lensed_{args.qe_key}_{args.v}_{args.cmb_version}_{args.imin}_{args.imax}_{args.itmax}"),
                total_qe_it_lensed)

# Save outputs only on the root process
if rank == 0:
    total_qe = np.concatenate(total_qe)
    total_qe_cross = np.concatenate(total_qe_cross)
    total_inputs = np.concatenate(total_inputs)
    total_inputs_down = np.concatenate(total_inputs_down)
    total_inputs_lensed = np.concatenate(total_inputs_lensed)

    np.save(opj(saving_directory, f"input_{args.v}_{args.cmb_version}_{args.imin}_{args.imax}_{args.itmax}"), 
            total_inputs)
    np.save(opj(saving_directory, f"input_down_{args.v}_{args.cmb_version}_{args.imin}_{args.imax}_{args.itmax}"),
            total_inputs_down)
    np.save(opj(saving_directory, f"input_lensed_{args.v}_{args.cmb_version}_{args.imin}_{args.imax}_{args.itmax}"),
            total_inputs_lensed)
    np.save(opj(saving_directory, f"total_qe_{args.qe_key}_{args.v}_{args.cmb_version}_{args.imin}_{args.imax}_{args.itmax}"), 
            total_qe)
    np.save(opj(saving_directory, f"total_qe_cross_{args.qe_key}_{args.v}_{args.cmb_version}_{args.imin}_{args.imax}_{args.itmax}"), 
            total_qe_cross)
