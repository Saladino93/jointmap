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


caso_s4 = "" #cmb-s4
caso_so = "_so"
caso_spt = "_spt"

caso = caso_spt

SO_name = "so"
SPT_name = "spt"
S4_name = ""
names = {caso_s4:S4_name, caso_so:SO_name, caso_spt:SPT_name}

name = names[caso]

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


########

dir = f"{name}data/"
print("Directory", dir)

if mpi.rank == 0:
    if not os.path.exists(dir):
        os.makedirs(dir)

if False:
    from delensalot.biases import iterbiasesN0N1

    qe_key = 'p_p'

    cachedir = f"n0n1_cachedir_{name}_stuff_new"
    itbias = iterbiasesN0N1.iterbiases(nlev_t, nlev_p, beam_fwhm, (lmin_cmb, lmin_cmb, lmin_cmb), lmax_cmb,
                                        lmax_qlm, cls_unl_fid, None, cachedir, verbose=True)
    N0s_iter = {}
    N1s_iter = {}
    rggs_iter = {}

    for itermax in [0, 1, 10]:
        Ns, fid_delcls, dat_cls = itbias.get_n0n1(qe_key, itermax, None, None, version = "wN1")
        print("Save info")
        np.save(f"{dir}fid_delcls_{itermax}.npy", fid_delcls)
        N0s, N1s, rgg, _ = Ns
        N0s_iter[itermax] = N0s
        N1s_iter[itermax] = N1s
        rggs_iter[itermax] = rgg
        if mpi.rank == 0:
            np.savetxt(f"{dir}/N0s_{itermax}.txt", N0s)
            np.savetxt(f"{dir}/N1s_{itermax}.txt", N1s)

########




nggs = {}
r_ggs = {}
for source in ["p", "a"]:
    qe_key = f"{source}_p"

    GG_N0, CC_N0 = nhl.get_nhl(qe_key, qe_key, cls_len, cls_ivfs, lmax_cmb, lmax_cmb, lmax_out = lmax_qlm)[0:2]

    r_gg_fid, r_cc_fid = qresp.get_response(qe_key, lmax_cmb, source, cls_len, cls_grad, fals, lmax_qlm = lmax_qlm)[0:2]

    n_gg = GG_N0 * utils.cli(r_gg_fid ** 2)

    nggs[source] = n_gg
    r_ggs[source] = r_gg_fid



out_dir = f"noise_biases{caso}/"
if mpi.rank == 0:
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    np.savetxt(out_dir+"ngg_a_QE.txt", nggs["a"])

if True:
    lib_dir = f'./n1s_aa{caso}'
    n1lib = n1.library_n1(lib_dir, cls_len['tt'], cls_len['te'], cls_len['ee'], lmaxphi=4000, dL=10, lps=None)
    n1_ap = n1lib.get_n1('a_p', 'p', cls_unl['pp'], ftl, fel, fbl, lmax_qlm)*utils.cli(r_ggs["a"] ** 2)
    if mpi.rank == 0:
        np.savetxt(out_dir+"n1_ap_QE.txt", n1_ap)
    #n1_aa = n1lib.get_n1('a_p', 'a', cls_unl['aa'], ftl, fel, fbl, lmax_qlm)*utils.cli(r_ggs["a"] ** 2)#note, should be 'aa' not 'pp'!!
    #if mpi.rank == 0:
    #    np.savetxt(out_dir+"n1_aa_QE.txt", n1_aa)


#### TAKE CARE OF ITERATED QUANTITIES

qe_key = "a_p"

it = 1
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
if mpi.rank == 0:
    np.savetxt(out_dir+f"ngg_a_itr_{it}.txt", N0_unbiased)

print("Read", f"{dir}fid_delcls_{it}.npy")
fid_delcls_10 = np.load(f"{dir}fid_delcls_{it}.npy", allow_pickle=True)

lib_dir = f'./n1s_aa_itr_{it}{caso}_check_new'
n1lib = n1.library_n1(lib_dir, cls_w['tt'], cls_w['te'], cls_w['ee'], lmaxphi=4000, dL=10, lps=None)
n1_ap = n1lib.get_n1('a_p', 'p', fid_delcls_10[-1]['pp'], fal["ee"], fal["ee"], fal["bb"], lmax_qlm)*utils.cli(r_gg_true ** 2)
if mpi.rank == 0:
    np.savetxt(out_dir+f"n1_ap_itr_{it}.txt", n1_ap)

n1_aa = n1lib.get_n1('a_p', 'a', cls_unl['aa'], fal["ee"], fal["ee"], fal["bb"], lmax_qlm)*utils.cli(r_gg_true ** 2)
if mpi.rank == 0:
    np.savetxt(out_dir+f"n1_aa_itr_{it}.txt", n1_aa)

#cls_alpha_residual = np.loadtxt("cls_alpha_residual.txt")
#n1_aa_residual = n1lib.get_n1('a_p', 'a', cls_alpha_residual, fal["ee"], fal["ee"], fal["bb"], lmax_qlm)*utils.cli(r_gg_true ** 2)
#if mpi.rank == 0:
#    np.savetxt(out_dir+"n1_aa_residual_itr_10.txt", n1_aa_residual)


## Now, take care of input lensed

if caso == "":
    directorycmb = "/users/odarwish/scratch/JOINTRECONSTRUCTION/alpha_phi_cmb_new_rot/simswalpha/"
    alm0_input = hp.read_alm(directorycmb+"sim_0000_alpha_lm.fits")
    plm0_input = hp.read_alm(directorycmb+"sim_0000_plm.fits")

    def _get_dlm(idx):
        dlm = plm0_input.copy()
        dclm = np.zeros_like(plm0_input)
        lmax_dlm = hp.Alm.getlmax(dlm.size, -1)
        mmax_dlm = lmax_dlm
        # potentials to deflection
        p2d = np.sqrt(np.arange(lmax_dlm + 1, dtype=float) * np.arange(1, lmax_dlm + 2, dtype=float))
        #p2d[:self.lmin_dlm] = 0
        hp.almxfl(dlm, p2d, mmax_dlm, inplace=True)
        hp.almxfl(dclm, p2d, mmax_dlm, inplace=True)
        return dlm, dclm, lmax_dlm, mmax_dlm



    dlm, dclm, lmax_dlm, mmax_dlm = _get_dlm(0)

    import lenspyx
    from plancklens import shts
    lmax_map = hp.Alm.getlmax(alm0_input.size)
    nside_lens = 2048
    a0_len = lenspyx.alm2lenmap(
        alm0_input, [dlm, None], geometry=('healpix', {'nside': nside_lens}),
        epsilon=1e-8, verbose=0)
    alm0_len = shts.map2alm(a0_len, lmax = lmax_map)

    directory = "/users/odarwish/scratch/JOINTRECONSTRUCTION/alpha_phi_cmb_new_rot_version_alpha_phi_cmb_new_rot_test_jan_4_recs/p_p_sim0000alpha_phi_cmb_new_rot_test_jan_4/"
    alm0_norm = np.load(directory+"alm0_norm.npy")


    alm0_len_ = utils.alm_copy(alm0_len, lmax = 5000)
    alm0_input_ = utils.alm_copy(alm0_input, lmax = 5000)

    claa = hp.alm2cl(alm0_input_)
    claa_len = hp.alm2cl(alm0_len_)

    claa_rec = hp.alm2cl(alm0_input_, alm0_norm)
    claa_len_rec = hp.alm2cl(alm0_len_, alm0_norm)

    np.savetxt(out_dir+"claa_len_rec.txt", claa_len_rec)
    np.savetxt(out_dir+"claa_len.txt", claa_len)
    np.savetxt(out_dir+"claa_rec.txt", claa_rec)
    np.savetxt(out_dir+"claa.txt", claa)