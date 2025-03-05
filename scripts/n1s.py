from os.path import join as opj
import os
from plancklens import utils
from plancklens.helpers import mpi
import numpy as np
import healpy as hp

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
ls = np.arange(cls_rot.size)
from plancklens.utils import cli
factor = cli(ls*(ls+1)/2)
cls_unl["oo"] = cls_rot*factor**2.


ns = 1
ACB = 7
ell = np.arange(0, len(cls_unl["tt"])+1)
cls_alpha = 10**(-ACB)*2*np.pi/(ell*(ell+1))**ns
cls_alpha[0] = 0


##also, get some theory calculation for comparison
from plancklens import nhl, n0s, qresp

qe_key = "p_p"

nlev_t = 1.
nlev_p = nlev_t*np.sqrt(2)
beam_fwhm = 1.
cls_unl_fid = cls_unl
lmin_cmb = 30
lmin_blm, lmin_elm, lmin_tlm = lmin_cmb, lmin_cmb, lmin_cmb
lmax_cmb = 4000
itermax = 1
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


from plancklens.n1 import n1

lib_dir = './n1s_ranks_v2'
n1lib = n1.library_n1(lib_dir, cls_len['tt'], cls_len['te'], cls_len['ee'], lmaxphi=4500, dL=10, lps=None)
#n1_aa = n1lib.get_n1('a_p', 'a', cls_alpha, ftl, fel, fbl, lmax_qlm)
#n1_ap = n1lib.get_n1('a_p', 'p', cls_unl['pp'], ftl, fel, fbl, lmax_qlm)
#n1_pa = n1lib.get_n1('p_p', 'a', cls_alpha, ftl, fel, fbl, lmax_qlm)
n1_op = n1lib.get_n1('x_p', 'p', cls_unl["pp"], ftl, fel, fbl, lmax_qlm)
n1_oa = n1lib.get_n1('x_p', 'a', cls_alpha, ftl, fel, fbl, lmax_qlm)
n1_oo = n1lib.get_n1('x_p', 'x', cls_unl["oo"], ftl, fel, fbl, lmax_qlm)
r_gg_fid, r_cc_fid = qresp.get_response(qe_key, lmax_cmb, "p", cls_len, cls_grad, fals, lmax_qlm = lmax_qlm)[0:2]
n1_op *= utils.cli(r_cc_fid**2)
n1_oa *= utils.cli(r_cc_fid**2)
n1_oo *= utils.cli(r_cc_fid**2)

if mpi.rank == 0:
    np.savetxt("n1_op_QE.txt", n1_op)
    np.savetxt("n1_oa_QE.txt", n1_oa)
    np.savetxt("n1_oo_QE.txt", n1_oo)


dir = opj(os.environ['HOME'], 'jointmap', 'scripts', "s4data/") 
it = 10
fal, dat_delcls, cls_w, cls_f = np.load(f"{dir}fal_{it}.npy", allow_pickle=True).take(0), np.load(f"{dir}dat_delcls_{it}.npy", allow_pickle=True).take(0), np.load(f"{dir}cls_w_{it}.npy", allow_pickle=True).take(0), np.load(f"{dir}cls_f_{it}.npy", allow_pickle=True).take(0)
cls_ivfs_arr = utils.cls_dot([fal, dat_delcls, fal])
cls_ivfs = dict()

fid_delcls_10 = np.load(f"{dir}fid_delcls_{it}.npy", allow_pickle=True)

for i, a in enumerate(['t', 'e', 'b']):
    for j, b in enumerate(['t', 'e', 'b'][i:]):
        if np.any(cls_ivfs_arr[i, j + i]):
            cls_ivfs[a + b] = cls_ivfs_arr[i, j + i]

lib_dir = './n1s_ranks_v2_it'
#n1lib = n1.library_n1(lib_dir, cls_len['tt'], cls_len['te'], cls_len['ee'], lmaxphi=4500, dL=10, lps=None)
#n1_aa = n1lib.get_n1('a_p', 'a', cls_alpha, ftl, fel, fbl, lmax_qlm)
#n1_ap = n1lib.get_n1('a_p', 'p', cls_unl['pp'], ftl, fel, fbl, lmax_qlm)
#n1_pa = n1lib.get_n1('p_p', 'a', cls_alpha, ftl, fel, fbl, lmax_qlm)

cls_alpha_residual = np.loadtxt(opj(os.environ['HOME'], 'jointmap', 'scripts')+"/cls_alpha_residual.txt")

n1lib = n1.library_n1(lib_dir, cls_w['tt'], cls_w['te'], cls_w['ee'], lmaxphi=4500, dL=10, lps=None)
n1_op = n1lib.get_n1('x_p', 'p', fid_delcls_10[-1]['pp'], fal["ee"], fal["ee"], fal["bb"], lmax_qlm)
n1_oa = n1lib.get_n1('x_p', 'a', cls_alpha_residual, fal["ee"], fal["ee"], fal["bb"], lmax_qlm)

roo = qresp.get_response(qe_key, lmax_cmb, "p", cls_w, cls_f, fal, lmax_qlm=lmax_qlm)[1]
n1_op *= utils.cli(roo ** 2)
n1_oa *= utils.cli(roo ** 2)

rho_sqd_phi = cls_unl["oo"][:lmax_qlm + 1] * utils.cli(cls_unl["oo"][:lmax_qlm + 1] + utils.cli(roo[:lmax_qlm + 1])) #should be ok even if I ignore some N1
cls_oo = cls_unl["oo"][:lmax_qlm + 1]
cls_oo *= (1. - rho_sqd_phi)  # The true residual lensing spec.
n1_oo = n1lib.get_n1('x_p', 'x', cls_oo, fal["ee"], fal["ee"], fal["bb"], lmax_qlm)
n1_oo *= utils.cli(roo ** 2)


if mpi.rank == 0:
    np.savetxt("n1_op_itr_10.txt", n1_op)
    np.savetxt("n1_oa_itr_10.txt", n1_oa)
    np.savetxt("n1_oo_itr_10.txt", n1_oo)


