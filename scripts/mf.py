from delensalot.biases import grads_mf
from delensalot.core.iterator.cs_iterator_operator import iterator_cstmf as iterator_cstmf

from plancklens.sims import phas, maps

import os
from os.path import join as opj


cmb_version = "meanfield"

suffix = cmb_version # descriptor to distinguish this parfile from others...
folder_ = "JOINTRECONSTRUCTION"
TEMP =  opj(os.environ['SCRATCH'], folder_, suffix)
DATDIRwalpha = opj(os.environ['SCRATCH'],folder_, suffix, 'simswalpha')

nside = 2048
dlmax = 1024
lmax_unl_generation = 5000 #lmax for saving without CMBs

fields_of_interest = 3*["T"]

libPHASCMB_mf = phas.lib_phas(os.path.join(DATDIRwalpha, 'phas_cmb_mf'), 3, lmax_unl_generation + dlmax)


iters = [1]
simidxs = [0]
key = "p"

for itr in iters:
    result = grads_mf.get_graddet_sim_mf_trick(iterator, itr, simidxs, 
                             key, libPHASCMB_mf, zerolensing = False, recache = False)