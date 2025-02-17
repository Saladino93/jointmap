import numpy as np
from plancklens.n1 import n1
from plancklens import utils
from plancklens.helpers import mpi
import healpy as hp

from os.path import join as opj
import os
from plancklens import utils


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


##also, get some theory calculation for comparison
from plancklens import nhl, n0s, qresp

qe_key = "p_p"

caso_s4 = "" #cmb-s4
caso_so = "_so"
caso_spt = "_spt"

caso = caso_spt

SO_name = "so"
SPT_name = "spt"
S4_name = ""
names = {caso_s4:S4_name, caso_so:SO_name, caso_spt:SPT_name}

name = names[caso]
dir = f"{name}data/"
print(dir)

ns = 1
ACB = 7
ell = np.arange(0, len(cls_unl["tt"])+1)
cls_alpha = 10**(-ACB)*2*np.pi/(ell*(ell+1))**ns
cls_alpha[0] = 0

#cls_unl["aa"] = cls_alpha

nlev_tdict = {caso_s4: 1., caso_so: 6., caso_spt: 1.6}
beam_fwhm_dict = {caso_s4: 1., caso_so: 1.4, caso_spt: 1.4}

nlev_t = nlev_tdict[caso]
nlev_p = nlev_t*np.sqrt(2)
beam_fwhm = beam_fwhm_dict[caso]
cls_unl_fid = cls_unl
lmin_cmb = 30
lmin_blm, lmin_elm, lmin_tlm = lmin_cmb, lmin_cmb, lmin_cmb
lmax_cmb = 4000
itermax = 10
ret_curl = True

lmax_qlm = 5120

cls_unl_fid = cls_unl
lmin_cmb = 30
lmin_blm, lmin_elm, lmin_tlm = lmin_cmb, lmin_cmb, lmin_cmb
lmax_cmb = 4000
ret_curl = True

lmax_qlm = 5120

lt, le, lb = (np.arange(lmax_cmb + 1) >= lmin_tlm), (np.arange(lmax_cmb + 1) >= lmin_elm), (np.arange(lmax_cmb + 1) >= lmin_blm)

transf = hp.gauss_beam(beam_fwhm / 180 / 60 * np.pi, lmax=lmax_cmb)
ftl = utils.cli(cls_len['tt'][:lmax_cmb + 1] + (nlev_t / 60. / 180. * np.pi) ** 2 / transf ** 2)*lt# isotropic approximation to the  T filtering, must match that applied to the data
fel = utils.cli(cls_len['ee'][:lmax_cmb + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf ** 2)*le # isotropic approximation to the E filtering
fbl = utils.cli(cls_len['bb'][:lmax_cmb + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf ** 2)*lb # isotropic approximation to the P filtering
fals = {'tt':ftl, 'ee':fel, 'bb':fbl}
dat_cls = {'tt':(cls_len['tt'][:lmax_cmb + 1] + (nlev_t / 60. / 180. * np.pi) ** 2 / transf ** 2)*lt,
            'ee': (cls_len['ee'][:lmax_cmb + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf ** 2)*le,
               'bb': (cls_len['bb'][:lmax_cmb + 1] + (nlev_p / 60. / 180. * np.pi) ** 2 / transf ** 2)*lb}

cls_ivfs_arr = utils.cls_dot([fals, dat_cls, fals])

cls_ivfs = dict()
for i, a in enumerate(['t', 'e', 'b']):
    for j, b in enumerate(['t', 'e', 'b'][i:]):
        if np.any(cls_ivfs_arr[i, j + i]):
            cls_ivfs[a + b] = cls_ivfs_arr[i, j + i]

qe_key_A = f"p_p"
qe_key_B = f"f_p"
source_A = "p"
source_B = "f"

lib_dir = './n1s_AB_v1'

n1lib = n1.library_n1(lib_dir, cls_len['tt'], cls_len['te'], cls_len['ee'], lmaxphi=4500, dL=10, lps=None)
n1_AB_no_resp = n1lib.get_n1(qe_key_A, source_A, cls_unl['pp'], ftl, fel, fbl, lmax_qlm, kB = qe_key_B)#*utils.cli(r_ggs["a"] ** 2)#note, should be 'aa' not 'pp'!!


if mpi.rank == 0:

    r_gg_fid_A, r_cc_fid_A = qresp.get_response(qe_key_A, lmax_cmb, source_A, cls_len, cls_grad, fals, lmax_qlm = lmax_qlm)[0:2]
    r_gg_fid_B, r_cc_fid_B = qresp.get_response(qe_key_B, lmax_cmb, source_B, cls_len, cls_grad, fals, lmax_qlm = lmax_qlm)[0:2]
    np.savetxt("n1_cross_f_p.txt", n1_AB_no_resp*utils.cli(r_gg_fid_A*r_gg_fid_B))