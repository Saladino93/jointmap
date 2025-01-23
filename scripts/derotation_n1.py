"""
Get noise biases for 
"""

from os.path import join as opj
import os
from plancklens import utils
from plancklens.helpers import mpi
import numpy as np
import healpy as hp
from plancklens.n1 import n1
from plancklens import nhl, n0s, qresp

def camb_clfile_gradient(fname, lmax=None):
    """CAMB spectra (lenspotentialCls, lensedCls or tensCls types) returned as a dict of numpy arrays.
    Args:
        fname (str): path to CAMB output file
        lmax (int, optional): outputs cls truncated at this multipole.
    """
    cols = np.loadtxt(fname).transpose()
    ell = np.int_(cols[0])
    if lmax is None: lmax = ell[-1]
    assert ell[-1] >= lmax, (ell[-1], lmax)
    cls = {k : np.zeros(lmax + 1, dtype=float) for k in ['tt', 'ee', 'bb', 'te']}
    w = ell * (ell + 1) / (2. * np.pi)  # weights in output file
    idc = np.where(ell <= lmax) if lmax is not None else np.arange(len(ell), dtype=int)
    for i, k in enumerate(['tt', 'ee', 'bb', 'te']):
        cls[k][ell[idc]] = cols[i + 1][idc] / w[idc]
    return cls

cls_path = opj(os.environ['HOME'], 'jointmap', 'data')
#cls_path = opj("/Users/omard/Downloads/", 'giulio')
cls_unl = utils.camb_clfile(opj(cls_path, 'lensedCMB_dmn1_lenspotentialCls.dat'))
cls_len = utils.camb_clfile(opj(cls_path, 'lensedCMB_dmn1_lensedCls.dat'))
cls_grad = camb_clfile_gradient(opj(cls_path, 'lensedCMB_dmn1_lensedgradCls.dat'))
cls_rot = np.loadtxt(opj(cls_path, 'new_lensedCMB_dmn1_field_rotation_power.dat')).T[1]

SO_case = False
nlev_t = 1. if not SO_case else 6.
nlev_p = nlev_t*np.sqrt(2)
beam_fwhm = 1. if SO_case else 1.4
cls_unl_fid = cls_unl
lmin_cmb = 30
lmin_blm, lmin_elm, lmin_tlm = lmin_cmb, lmin_cmb, lmin_cmb
lmax_cmb = 4000
itermax = 1
ret_curl = True

lmax_qlm = 5120

lt, le, lb = (np.arange(lmax_cmb + 1) >= lmin_tlm), (np.arange(lmax_cmb + 1) >= lmin_elm), (np.arange(lmax_cmb + 1) >= lmin_blm)

transf = hp.gauss_beam(beam_fwhm / 180 / 60 * np.pi, lmax=lmax_cmb)


ns = 1
ACB = 7
ell = np.arange(0, len(cls_unl["tt"])+1)
cls_alpha = 10**(-ACB)*2*np.pi/(ell*(ell+1))**ns
cls_alpha[0] = 0

cls_alpha_residual = np.loadtxt(os.environ['HOME']+"/jointmap/scripts/cls_alpha_residual.txt")
cls_alpha_residual = np.nan_to_num(cls_alpha_residual)
cls_unl["aa"] = cls_alpha_residual[:cls_alpha.size]

qe_key = "a_p"

dir = os.environ['HOME']+"/jointmap/scripts/s4data/"

source = "a"

it = 10
fal, dat_delcls, cls_w, cls_f = np.load(f"{dir}fal_{it}.npy", allow_pickle=True).take(0), np.load(f"{dir}dat_delcls_{it}.npy", allow_pickle=True).take(0), np.load(f"{dir}cls_w_{it}.npy", allow_pickle=True).take(0), np.load(f"{dir}cls_f_{it}.npy", allow_pickle=True).take(0)
cls_ivfs_arr = utils.cls_dot([fal, dat_delcls, fal])
cls_ivfs = dict()
for i, a in enumerate(['t', 'e', 'b']):
    for j, b in enumerate(['t', 'e', 'b'][i:]):
        if np.any(cls_ivfs_arr[i, j + i]):
            cls_ivfs[a + b] = cls_ivfs_arr[i, j + i]
n_gg = nhl.get_nhl(qe_key, qe_key, cls_w, cls_ivfs, lmax_cmb, lmax_cmb, lmax_out=lmax_qlm)[0]
r_gg_true = qresp.get_response(qe_key, lmax_cmb, source, cls_w, cls_f, fal, lmax_qlm=lmax_qlm)[0]
N0_unbiased = n_gg * utils.cli(r_gg_true ** 2)  # N0 of QE estimator after rescaling by Rfid / Rtrue to make it unbiased

print("Read", f"{dir}fid_delcls_10.npy")
fid_delcls_10 = np.load(f"{dir}fid_delcls_10.npy", allow_pickle=True)

lib_dir = os.environ['HOME']+"/jointmap/scripts/n1s_aa_itr_10_rho2/"
n1lib = n1.library_n1(lib_dir, cls_w['tt'], cls_w['te'], cls_w['ee'], lmaxphi=2500, dL=10, lps=None)
#n1_ap = n1lib.get_n1('a_p', 'p', fid_delcls_10[-1]['pp'], fal["ee"]*1e-30, fal["ee"], fal["bb"], lmax_qlm)*utils.cli(r_gg_true ** 2)
n1_aa = n1lib.get_n1('a_p', 'a', cls_unl['aa'], fal["ee"]*1e-30, fal["ee"], fal["bb"], lmax_qlm)*utils.cli(r_gg_true ** 2)

if mpi.rank == 0:
    np.savetxt("n1_aa_derot.txt", n1_aa)