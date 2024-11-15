"""
Gives the QE applied on some map.

ASYMMETRIC ESTIMATOR.
"""

import os
from os.path import join as opj
import numpy as np
import healpy as hp

import plancklens

from iterativefg import utils as u

from plancklens import utils, qresp, qest, qecl, nhl
from plancklens.qcinv import cd_solve
from plancklens.sims import maps, phas
from plancklens.filt import filt_simple, filt_util, filt_cinv

from lenspyx.lensing import get_geom

from lenspyx.remapping.deflection_029 import deflection

from delensalot.core.iterator import cs_iterator as iterator_base, steps
from delensalot.utility.utils_steps import nrstep
from delensalot.utils import cli
from delensalot.utility import utils_sims
from delensalot.core.helper import utils_scarf
from delensalot.utility.utils_hp import gauss_beam, almxfl, alm_copy, Alm
from delensalot.core.opfilt.QE_opfilt_aniso_t import alm_filter_ninv as alm_filter_QE
from delensalot.core.opfilt.QE_opfilt_iso_t import alm_filter_nlev as alm_filter_QE_iso
from delensalot.core.opfilt.MAP_opfilt_iso_t import alm_filter_nlev_wl as alm_filter_iso

from delensalot.core.iterator import cs_iterator_multi_bh as iterator_base_bh
from delensalot.core.opfilt.MAP_bh_opfilt_aniso_t import alm_filter_ninv_wl as alm_filter_bh

from delensalot.core.iterator.cs_iterator_asymm import iterator_cstmf as iterator_cstmf_asymm 
from delensalot.core.opfilt.MAP_opfilt_iso_t import alm_filter_nlev_wl as alm_filter_iso_MAP


from itfgs.sims.sims_postborn import sims_postborn, SehgalSim

from plancklens.helpers import mpi

import argparse

parser = argparse.ArgumentParser(description='Create a fake point source map and a mask for it.')
parser.add_argument("--fgcase_A", type=str, help="Foreground case A.", default="") #which case of the suits of simulations, e.g. masked, non masked, etc...
parser.add_argument("--fgcase_B", type=str, help="Foreground case B.", default="") 
parser.add_argument("--fgversion", type=str, help="Foreground version.", default="") #which suits of simulations you are going to use, e.g. Websky
parser.add_argument("--settingsversion", type=str, help="Settings version.", default="")
parser.add_argument("--cmbversion", type=str, help="CMB version.", default="")
parser.add_argument("--beam", type=float, help="Beam FWHM.", default=1.)
parser.add_argument("--lmax_ivf", type=int, help="Maximum l for CMB lensing reconstruction.", default=4000)
parser.add_argument("--lmin_ivf", type=int, help="Minimum l for QE.", default=10)
parser.add_argument("--lmax_qlm", type=int, help="Maximum l for QE.", default=5120)
parser.add_argument("--lmax_unl", type=int, help="Maximum l for unlensed CMB reconstruction.", default=5120)
parser.add_argument("--lmax_cmb", type=int, help="Maximum l for generated unlensed CMB (add 1024 to this).", default=4096)
parser.add_argument("--version", type=str, help="General version.", default="")
parser.add_argument("--mean_field", action="store_true", help="Use mean field.")
parser.add_argument("--fg_effective", action="store_true", help="Foreground effective noise.")
parser.add_argument("--foreground", type=str, help="Foreground.", default="somma")
parser.add_argument("--qe_key", type = str, help = "Key for the QE.", default = "ptt")
parser.add_argument("--itmax", type = int, help = "Maximum number of iterations.", default = 1)
parser.add_argument("--zero_noise", action="store_true", help="Zero noise.")
parser.add_argument("--no_cmb", action="store_true", help="Zero CMB.")
parser.add_argument("--imin", type=int, help="Minimum simulation index.", default=0)
parser.add_argument("--imax", type=int, help="Maximum simulation index.", default=0)
parser.add_argument("--generate_phi", action="store_true", help="Generate phi.")
parser.add_argument("--real_space", action="store_true", help="Real space.")
parser.add_argument("--stdrec", action="store_true", help="Standard reconstruction.")
parser.add_argument("--directory", type = str, help = "Directory of maps.", default = "/users/odarwish/scratch/ITERATIVEFG/webskycmbskySO/ILC/ilcmaps/")
parser.add_argument("--lmin_fgs", type=int, help="Below this, set foregrounds to zero (done to make CG step work better)", default = 1000)
parser.add_argument("--cg_tol", type = int, help = "Tolerance for CG.", default = 8)

args = parser.parse_args()
fgcase_A = args.fgcase_A
fgcase_B = args.fgcase_B
fgversion = args.fgversion
version = args.version
beam = args.beam
lmax_ivf = args.lmax_ivf
lmin_ivf = args.lmin_ivf
lmax_qlm = args.lmax_qlm
lmax_unl = args.lmax_unl
lmax_cmb = args.lmax_cmb
settingsversion = args.settingsversion #this is if I change some lmax, mmax, beam, nlev_p, etc.
mean_field = args.mean_field
fg_effective = args.fg_effective
cmbversion = args.cmbversion
qe_key = args.qe_key
itmax = args.itmax
zero_noise = args.zero_noise
nocmb = args.no_cmb
foreground = args.foreground
generate_phi = args.generate_phi
real_space = args.real_space
stdrec = args.stdrec
directory = args.directory
lmin_fgs = args.lmin_fgs
cg_tol = args.cg_tol

iters_to_be_done = range(0, itmax+1)

if mean_field:
    print("Using mean field.")

if fgversion == "webskysoILC":

    print("Using websky foregrounds.")

    #directorywebsky = "/users/odarwish/scratch/ITERATIVEFG/webskycmbskySO/ILC/ilcmaps/"

    """def get_extra_tlm(fgcase, fgversion):
        
        version = "" if fgversion == "" else f"_{fgversion}"

        foregrounds = [foreground]

        phases = np.load(opj(directory, f"phases.npy")) if "rand" in fgcase else 1

        if fgcase == "sommamasked":
            return np.sum([hp.read_alm(opj(directory, f"{foreground}_masked_ilc_alm.fits")) for foreground in foregrounds], axis = 0)
        elif fgcase == "somma":
            return np.sum([hp.read_alm(opj(directory, f"{foreground}_ilc_alm.fits")) for foreground in foregrounds], axis = 0)
        elif fgcase == "randsommamasked":
            return np.sum([hp.read_alm(opj(directory, f"rand_{foreground}_masked_ilc_alm.fits")) for foreground in foregrounds], axis = 0)
        elif fgcase == "randsomma":
            return np.sum([hp.read_alm(opj(directory, f"rand_{foreground}_ilc_alm.fits")) for foreground in foregrounds], axis = 0)
        elif fgcase == "somma_depr_tsz_masked":
            return np.sum([hp.read_alm(opj(directory, f"{foreground}_masked_ilc_depr_tsz_alm.fits")) for foreground in foregrounds], axis = 0)
        elif fgcase == "somma_depr_cib_masked":
            return np.sum([hp.read_alm(opj(directory, f"{foreground}_masked_ilc_depr_cib_alm.fits")) for foreground in foregrounds], axis = 0)
        elif fgcase == "somma_depr_cib_tsz_masked":
            return np.sum([hp.read_alm(opj(directory, f"{foreground}_masked_ilc_depr_cib_tsz_alm.fits")) for foreground in foregrounds], axis = 0)
        elif fgcase == "somma_depr_tsz":
            return np.sum([hp.read_alm(opj(directory, f"{foreground}_ilc_depr_tsz_alm.fits")) for foreground in foregrounds], axis = 0)
        elif fgcase == "randsomma_depr_tsz_masked":
            return np.sum([hp.read_alm(opj(directory, f"rand_{foreground}_masked_ilc_depr_tsz_alm.fits")) for foreground in foregrounds], axis = 0)
        elif fgcase == "randsomma_depr_tsz":
            return np.sum([hp.read_alm(opj(directory, f"rand_{foreground}_ilc_depr_tsz_alm.fits")) for foreground in foregrounds], axis = 0)
        elif fgcase == "randsomma_depr_cib_masked":
            return np.sum([hp.read_alm(opj(directory, f"rand_{foreground}_masked_ilc_depr_cib_alm.fits")) for foreground in foregrounds], axis = 0)
        elif fgcase == "randsomma_depr_cib":
            return np.sum([hp.read_alm(opj(directory, f"rand_{foreground}_ilc_depr_cib_alm.fits")) for foreground in foregrounds], axis = 0)
        elif fgcase == "randsomma_depr_cib_tsz_masked":
            return np.sum([hp.read_alm(opj(directory, f"rand_{foreground}_masked_ilc_depr_cib_tsz_alm.fits")) for foreground in foregrounds], axis = 0)
        elif fgcase == "randsomma_depr_cib_tsz":
            return np.sum([hp.read_alm(opj(directory, f"rand_{foreground}_ilc_depr_cib_tsz_alm.fits")) for foreground in foregrounds], axis = 0)

        elif fgcase == "":
            return None
        else:
            raise ValueError("Invalid foreground case.")"""

    def get_extra_tlm(fgcase, fgversion):

        # Append version suffix if provided
        version = f"_{fgversion}" if fgversion else ""


        filter_fg = np.ones(8000)
        ls = np.arange(filter_fg.size)
        filter_fg *= (ls > lmin_fgs)

        # Dictionary mapping non-rand fgcase to corresponding filename patterns
        file_patterns = {
            "sommamasked": "{foreground}_masked_ilc_alm.fits",
            "somma": "{foreground}_ilc_alm.fits",
            "somma_depr_tsz_masked": "{foreground}_masked_ilc_depr_tsz_alm.fits",
            "somma_depr_cib_masked": "{foreground}_masked_ilc_depr_cib_alm.fits",
            "somma_depr_cib_tsz_masked": "{foreground}_masked_ilc_depr_cib_tsz_alm.fits",
            "somma_depr_tsz": "{foreground}_ilc_depr_tsz_alm.fits",
            "somma_depr_cib_masked": "{foreground}_masked_ilc_depr_cib_alm.fits",
            "somma_depr_cib": "{foreground}_ilc_depr_cib_alm.fits",
            "somma_depr_cib_tsz_masked": "{foreground}_masked_ilc_depr_cib_tsz_alm.fits",
            "somma_depr_cib_tsz": "{foreground}_ilc_depr_cib_tsz_alm.fits",
        }

        # Load phases if "rand" is in fgcase, else default to 1
        phases = np.load(opj(directory, "phases.npy")) if "rand" in fgcase else 1
        if "rand" in fgcase:
            print("APPLYING PHASES", phases)

        # Determine if we're in a "rand" case
        is_rand = "rand" in fgcase

        # Remove "rand" prefix from fgcase if present to get the non-rand equivalent
        base_fgcase = fgcase.replace("rand", "") if is_rand else fgcase

        # Check for empty fgcase
        if fgcase == "":
            return None

        # Get filename pattern or raise error if fgcase is invalid
        if base_fgcase in file_patterns:
            # Generate paths using the non-rand pattern
            filenames = [opj(directory, file_patterns[base_fgcase].format(foreground=foreground))]
            
            # Sum up the ALMs from specified files
            alms = np.sum([hp.read_alm(filename) for filename in filenames], axis=0)

            alms = hp.almxfl(alms, filter_fg)
            
            # Multiply by phases if it's a "rand" case
            return alms * phases if is_rand else alms
        else:
            raise ValueError("Invalid foreground case.")
            

    SimsShegalDict = {}
    SimsShegalDict['kappa'] = lambda idx: f"/scratch/snx3000/odarwish/SKYSIMS/WEBSKYSIMS/kap_alm.fits"  #opj(directorywebsky, f'kap_alm.fits')

else:
    raise ValueError("Invalid foreground version.")

class Extra(object):

    def __init__(self, name, extra_tlm):
        self.name = name
        self.extra_tlm = extra_tlm

    def get_extra_tlm(self, idx):
        return self.extra_tlm
    
    def get_name(self):
        return self.name
    
    def __call__(self, idx):
        return self.get_extra_tlm(idx)



get_Extra = lambda fgcase: Extra(fgcase+foreground+fgversion, get_extra_tlm(fgcase, fgversion)) if fgcase != "" else None

extra_tlm_A = get_Extra(fgcase_A)
extra_tlm_B = get_Extra(fgcase_B)


##############################################################################################################

def camb_clfile_gradient(fname, lmax=None, nocorr = False):
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
        cls[k][ell[idc]] = cols[i + 1][idc] / (w[idc] if not nocorr else 1)
    return cls

##############################################################################################################

## HERE THE MAIN CODE STARTS ##

suffix = f'S4{fgcase_A}_{fgcase_B}{fgversion}{settingsversion}' # descriptor to distinguish this parfile from others...

directory = "/users/odarwish/scratch/ITERATIVEFG/"

SIMDIR = opj(directory, 'lensing', 'cmbs', cmbversion)  # This is where the postborn are (or will be saved)
lib_dir_CMB = opj(directory, 'lensing', 'cmbs', 'cmbphas')
TEMP =  opj(directory, 'lensing', 'lenscarfrecs', suffix)


# Fiducial CMB spectra for QE and iterative reconstructions
cls_path = "../data/websky/"
cls_unl = utils.camb_clfile(opj(cls_path, 'websky_lenspotentialCls.dat'))
cls_len = utils.camb_clfile(opj(cls_path, 'websky_lensedCls.dat'))
cls_grad = camb_clfile_gradient(opj(cls_path, 'websky_lensedgradCls.dat'))

"""
cls_path = "../data/giulio/"
cls_unl = utils.camb_clfile(opj(cls_path, 'lensedCMB_dmn1_lenspotentialCls.dat'))
cls_len = utils.camb_clfile(opj(cls_path, 'lensedCMB_dmn1_lensedCls.dat'))
cls_grad = camb_clfile_gradient(opj(cls_path, 'lensedCMB_dmn1_lensedgradCls.dat'))
"""

lmax_ivf, mmax_ivf = (lmax_ivf, lmax_ivf)

lmin_tlm, lmin_elm, lmin_blm = (lmin_ivf, lmin_ivf, lmin_ivf) # The fiducial transfer functions are set to zero below these lmins
# for delensing useful to cut much more B. It can also help since the cg inversion does not have to reconstruct those.

lmax_cmb = lmax_cmb
lmax_qlm, mmax_qlm = (lmax_qlm, lmax_qlm) # Lensing map is reconstructed down to this lmax and mmax
# NB: the QEs from plancklens does not support mmax != lmax, but the MAP pipeline does
lmax_unl, mmax_unl = (lmax_unl, lmax_unl) # Delensed CMB is reconstructed down to this lmax and mmax

zbounds     = (-1.,1.)

#----------------- pixelization and geometry info for the input maps and the MAP pipeline and for lensing operations
nside = 2048
    
geominfo = ('healpix', {'nside': nside})
lenjob_geometry = get_geom(geominfo)
geominfo_defl = ('thingauss', {'lmax': 4200 + 300, 'smax': 2})
lenjob_geometry_defl = get_geom(geominfo_defl)

Lmin = 2 # The reconstruction of all lensing multipoles below that will not be attempted
step_val = np.arange(0, lmax_qlm+1, 1)
step_val = 0.5 #*(step_val >= Lmin) #((step_val>20) & (step_val < 2000))*0.2 
#stepper = nrstep(val = step_val) # handler of the size steps in the MAP BFGS iterative search
stepper = steps.nrstep(lmax_qlm, mmax_qlm, val=0.5)

# Multigrid chain descriptor
chain_descrs = lambda lmax_sol, cg_tol : [[0, ["diag_cl"], lmax_sol, nside, np.inf, cg_tol, cd_solve.tr_cg, cd_solve.cache_mem()]]

#chain_descrs = lambda lmax_sol, cg_tol : [[1, ["diag_cl"], 1024, 512, 3, 0.0, cd_solve.tr_cg, cd_solve.cache_mem()],
#                                          #[1, ["split(stage(2),  512, diag_cl)"], 1024, 512, 3, 0.0, cd_solve.tr_cg, cd_solve.cache_mem()], 
#                                          [0, ["split(stage(1), 1024, diag_cl)"], lmax_sol, nside, np.inf, cg_tol, cd_solve.tr_cg, cd_solve.cache_mem()]]

libdir_iterators = lambda qe_key, simidx, version: opj(TEMP,'%s_sim%04d'%(qe_key, simidx) + version)
#------------------

# Fiducial model of the transfer function
transf_tlm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_tlm)
transf_elm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_elm)
transf_blm   =  gauss_beam(beam/180 / 60 * np.pi, lmax=lmax_ivf) * (np.arange(lmax_ivf + 1) >= lmin_blm)
transf_d = {'t':transf_tlm, 'e':transf_elm, 'b':transf_blm}
transf_d_A = {'t':transf_tlm, 'e':transf_elm, 'b':transf_blm}
transf_d_B = {'t':transf_tlm, 'e':transf_elm, 'b':transf_blm}


#beam convolved noises, with common beam of 1.4 arcmin
noise_ilc_matrix = np.loadtxt("../data/ILC/SO_noise_ilc.txt").T #ell noise_ilc noise_ilc_depr_tsz noise_ilc_depr_cib noise_ilc_depr_joint
ell_noise = noise_ilc_matrix[0]
noise_ilc = noise_ilc_matrix[1]
noise_ilc_depr_tsz = noise_ilc_matrix[2]
noise_ilc_depr_cib = noise_ilc_matrix[3]
noise_ilc_depr_joint = noise_ilc_matrix[4]

#adding the zero ell values
ell_noise = np.append(np.array([0, 1]), ell_noise)
noise_ilc = np.append(np.array([0, 0]), noise_ilc)
noise_ilc_depr_tsz = np.append(np.array([0, 0]), noise_ilc_depr_tsz)
noise_ilc_depr_cib = np.append(np.array([0, 0]), noise_ilc_depr_cib)
noise_ilc_depr_joint = np.append(np.array([0, 0]), noise_ilc_depr_joint)


def get_noise(fgcase):
    if "sommamasked" in fgcase:
        noise_ilc_out = noise_ilc
    elif "somma_depr_tsz_masked" in fgcase:
        noise_ilc_out = noise_ilc_depr_tsz
    elif "somma_depr_cib_masked" in fgcase:
        noise_ilc_out = noise_ilc_depr_cib
    elif "somma_depr_cib_tsz_masked" in fgcase:
        noise_ilc_out = noise_ilc_depr_joint
    return noise_ilc_out

noise_ilc_A = get_noise(fgcase_A)
noise_ilc_B = get_noise(fgcase_B)

cross_noise = np.loadtxt("../data/ILC/SO_noise_ilc_cross_ilc_depr_tsz_cib.txt")
cross_noise = np.append(np.array([0, 0]), cross_noise)

#nlev_t = 6.
#noise_ilc = (nlev_t / 180 / 60 * np.pi) ** 2 * np.ones_like(transf_tlm)

selection = (np.arange(noise_ilc_A.size) >= lmin_tlm)
Nx = lmin_fgs #just a quick hack for now to avoid doing multigridding for faster cg convergence
noise_ilc_A[:Nx] = noise_ilc_A[Nx]
#noise_ilc_A *= selection

noise_ilc_B[:Nx] = noise_ilc_B[Nx]
#noise_ilc_B *= selection

cross_noise[:Nx] = cross_noise[Nx]
#just a quick hack for now

#basically, you want the foreground power of the actual map, e.g. masked or unmasked case
print("Fg cases are", fgcase_A, fgcase_B)


if fgcase_A != "":
    def get_quick_fg(extra_tlm, fgcase, fgcase2 = None):

        ps_noise = extra_tlm(0) if fgcase != "" else 0

        if fgcase2 != None:
            ps_noise2 = extra_tlm(0) if fgcase2 != "" else 0
        else:
            ps_noise2 = ps_noise

        cls = hp.alm2cl(ps_noise, ps_noise2) #cross-foregrounds power spectrum

        bin_edges = np.arange(1, 6100, 10)
        elbin, clsbinned = u.bin_theory(cls, bin_edges)

        print("Getting smoothed foreground power spectrum.")
        import scipy.signal
        fgs = np.exp(scipy.signal.savgol_filter(np.log(cls), 20, 3)) #can also just bin the raw spectrum, then interpolate
        fgs = np.nan_to_num(fgs, posinf = 0, neginf = 0, nan = 0)
        ls = np.arange(transf_tlm.size)
        fgs = np.interp(ls, np.arange(fgs.size), fgs)

        #fgs = np.exp(scipy.signal.savgol_filter(np.log(clsbinned), 20, 5)) #can also just bin the raw spectrum, then interpolate
        #fgs = np.interp(ell, elbin, fgs)

        fgs[:2] = 0

        if fgcase != "":
            selection = (np.arange(fgs.size) >= lmin_tlm)
            fgs *= selection
            fgs[:Nx] = fgs[Nx]

        return fgs
    

    fgs_A = get_quick_fg(extra_tlm_A, fgcase_A)
    fgs_B = get_quick_fg(extra_tlm_B, fgcase_B)
    fgs_AB = get_quick_fg(extra_tlm_A, fgcase_A, fgcase_B)
    fgs_BA = get_quick_fg(extra_tlm_B, fgcase_B, fgcase_A)

print("Foreground power spectrum A is", fgs_A)
print("Foreground power spectrum B is", fgs_B)

fgs_A = np.nan_to_num(fgs_A)
fgs_B = np.nan_to_num(fgs_B)
fgs_AB = np.nan_to_num(fgs_AB)
fgs_BA = np.nan_to_num(fgs_BA)

# Moving average function
def moving_average(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')


def get_fit(noise_ilc_spectrum):

    from scipy.optimize import curve_fit

    factor = (180 * 60 / np.pi) ** -2

    def fitfunc(x, a):
        return a**2*factor

    lmin_fg_fit = lmax_ivf-10
    fgs_fit = noise_ilc_spectrum[lmin_fg_fit:]
    ells = np.arange(lmin_fg_fit, noise_ilc_spectrum.size)
    popt, pcov = curve_fit(fitfunc, ells, fgs_fit)    
    fit = fitfunc(ells, *popt)
    noise_ilc_spectrum = fit#white noise spectrum after beam deconvolution

    return noise_ilc_spectrum


def get_effective_noise(noise_ilc, fgs):

    noise_ilc_avg = noise_ilc #moving_average(noise_ilc, 200)
    tfm = lambda x: x #get_fit(x) #x
    noise_ilc_spectrum = tfm(noise_ilc_avg[:lmax_ivf + 1] * cli(transf_tlm ** 2) + fgs)


    effective_noise = np.sqrt((noise_ilc_spectrum)*(180 * 60 / np.pi) ** 2*transf_tlm**2.)

    #np.savetxt("effective_noise.txt", effective_noise**2*(180 * 60 / np.pi) ** -2)
    #np.savetxt("cls_tt.txt", cls_len['tt'][:lmax_ivf + 1])

    # Isotropic approximation to the filtering (used eg for response calculations)
    ftl =  cli(cls_len['tt'][:lmax_ivf + 1] + noise_ilc_spectrum) * (transf_tlm > 0)
    fel =  cli(cls_len['ee'][:lmax_ivf + 1] + noise_ilc_spectrum*0) * (transf_elm > 0)
    fbl =  cli(cls_len['bb'][:lmax_ivf + 1] + noise_ilc_spectrum*0) * (transf_blm > 0)

    # Same using unlensed spectra (used for unlensed response used to initiate the MAP curvature matrix)
    ftl_unl =  cli(cls_unl['tt'][:lmax_ivf + 1] + noise_ilc_spectrum) * (transf_tlm > 0)
    fel_unl =  cli(cls_unl['ee'][:lmax_ivf + 1] + noise_ilc_spectrum*0.) * (transf_elm > 0)
    fbl_unl =  cli(cls_unl['bb'][:lmax_ivf + 1] + noise_ilc_spectrum*0.) * (transf_blm > 0)

    return effective_noise, ftl, fel, fbl, ftl_unl, fel_unl, fbl_unl


effective_noise_A, ftl_A, fel_A, fbl_A, ftl_unl_A, fel_unl_A, fbl_unl_A = get_effective_noise(noise_ilc_A, fgs_A)
effective_noise_B, ftl_B, fel_B, fbl_B, ftl_unl_B, fel_unl_B, fbl_unl_B = get_effective_noise(noise_ilc_B, fgs_B)
effective_noise_AB, ftl_AB, fel_AB, fbl_AB, ftl_unl_AB, fel_unl_AB, fbl_unl_AB = get_effective_noise(cross_noise, fgs_AB)
effective_noise_BA, ftl_BA, fel_BA, fbl_BA, ftl_unl_BA, fel_unl_BA, fbl_unl_BA = get_effective_noise(cross_noise, fgs_BA)

# -------------------------
# ---- Input simulation libraries. Here we use the NERSC FFP10 CMBs with homogeneous noise and consistent transfer function
#       We define explictly the phase library such that we can use the same phases for for other purposes in the future as well if needed
#       I am putting here the phases in the home directory such that they dont get NERSC auto-purged
pix_phas = phas.pix_lib_phas(opj(os.environ['HOME'], 'pixphas_nside%s'%nside), 3, (hp.nside2npix(nside),)) # T, Q, and U noise phases
#       actual data transfer function for the sim generation:
#transf_dat =  gauss_beam(beam / 180 / 60 * np.pi, lmax=4096) # (taking here full sims cmb's which are given to 4096)
#sims_cmb_len = sims_postborn(SIMDIR, lmax_cmb, cls_unl, extra_tlm = extra_tlm, generate_phi = False, nocmb = nocmb)

dlmax = 1024
Nfields = 3 if not generate_phi else 4
libPHASCMB = phas.lib_phas(os.path.join(lib_dir_CMB, 'phas'), Nfields, lmax_cmb + dlmax)
#sims_cmb_len = SehgalSim(sims = SimsShegalDict, lib_dir = SIMDIR, lmax_cmb = lmax_cmb, cls_unl = cls_unl, dlmax = dlmax, lmin_dlm = 2, lib_pha = libPHASCMB, extra_tlm = extra_tlm)
sims_cmb_len_A = SehgalSim(SimsShegalDict, lib_dir = SIMDIR, lmax_cmb = lmax_cmb, cls_unl = cls_unl, extra_tlm = extra_tlm_A, generate_phi = generate_phi, nocmb = nocmb, lib_pha = libPHASCMB)
sims_cmb_len_B = SehgalSim(SimsShegalDict, lib_dir = SIMDIR, lmax_cmb = lmax_cmb, cls_unl = cls_unl, extra_tlm = extra_tlm_B, generate_phi = generate_phi, nocmb = nocmb, lib_pha = libPHASCMB)


#NOISE NEEDS TO BE CORRELATED


Nf = 2

"""spectrum_t_A = {}
spectrum_t_A["tt"] = noise_ilc_A
spectrum_t_B = {}
spectrum_t_B["tt"] = noise_ilc_B"""

rmat = np.zeros((noise_ilc_A.size, Nf, Nf), dtype=float)
rmat[:, 0, 0] = noise_ilc_A
rmat[:, 1, 1] = noise_ilc_B
rmat[:, 0, 1] = cross_noise
rmat[:, 1, 0] = rmat[:, 0, 1]

for ell in range(4, noise_ilc_A.size):
    #print("KKKK", ell)
    t, v = np.linalg.eigh(rmat[ell, :, :])
    assert np.all(t >= 0.), (ell, t, rmat[ell, :, :])  # Matrix not positive semidefinite
    rmat[ell, :, :] = np.dot(v, np.dot(np.diag(np.sqrt(t)), v.T))

### Noise for A and B NEEDS TO BE CORRELATED!!
noise_phas = phas.lib_phas(os.path.join(lib_dir_CMB, 'noise_phas'), 2, lmax_cmb + dlmax)

sims_A      = maps.cmb_maps_nlev(sims_cmb_len_A, transf_d["t"], None, None, nside, pix_lib_phas=pix_phas, zero_noise = zero_noise, noise_phas = noise_phas, rmat = rmat, noise_index = 0)
sims_B      = maps.cmb_maps_nlev(sims_cmb_len_B, transf_d["t"], None, None, nside, pix_lib_phas=pix_phas, zero_noise = zero_noise, noise_phas = noise_phas, rmat = rmat, noise_index = 1)


# Makes the simulation library consistent with the zbounds
sims_A_MAP  = utils_sims.ztrunc_sims(sims_A, nside, [zbounds])
sims_B_MAP  = utils_sims.ztrunc_sims(sims_B, nside, [zbounds])
# -------------------------

ivfs_A   = filt_simple.library_fullsky_sepTP(opj(TEMP, 'ivfs_A'), sims_A, nside, transf_d_A["t"], cls_len, ftl_A, fel_A, fbl_A, cache=True)
ivfs_B   = filt_simple.library_fullsky_sepTP(opj(TEMP, 'ivfs_B'), sims_B, nside, transf_d_B["t"], cls_len, ftl_B, fel_B, fbl_B, cache=True)

# ---- QE libraries from plancklens to calculate unnormalized QE (qlms) and their spectra (qcls)
mc_sims_bias = np.arange(60, dtype=int)
mc_sims_var  = np.arange(60, 300, dtype=int)

#fal = {}
#fal["tt"] = ftl
#fal["ee"] = fel
#fal["bb"] = fbl

#resplib = qresp.resp_lib_simple(opj(TEMP, 'qlms_dd'), lmax_ivf, cls_weight = cls_len, cls_cmb = cls_grad, fal = fal, lmax_qlm = lmax_qlm, transf = transf_tlm)
qlms_dd_AB = qest.library_sepTP(opj(TEMP, 'qlms_dd_AB'), ivfs_A, ivfs_B,  cls_len['te'], nside, lmax_qlm=lmax_qlm) #, resplib = resplib)
qlms_dd_AA = qest.library_sepTP(opj(TEMP, 'qlms_dd_AA'), ivfs_A, ivfs_A,  cls_len['te'], nside, lmax_qlm=lmax_qlm) #, resplib = resplib)
qlms_dd_BB = qest.library_sepTP(opj(TEMP, 'qlms_dd_BB'), ivfs_B, ivfs_B,  cls_len['te'], nside, lmax_qlm=lmax_qlm)
qlms_dd_BA = qest.library_sepTP(opj(TEMP, 'qlms_dd_BA'), ivfs_B, ivfs_A,  cls_len['te'], nside, lmax_qlm=lmax_qlm)

#qcls_dd = qecl.library(opj(TEMP, 'qcls_dd'), qlms_dd, qlms_dd, mc_sims_bias)

# -------------------------
# This following block is only necessary if a full, Planck-like QE lensing power spectrum analysis is desired
# This uses 'ds' and 'ss' QE's, crossing data with sims and sims with other sims.

# This remaps idx -> idx + 1 by blocks of 60 up to 300. This is used to remap the sim indices for the 'MCN0' debiasing term in the QE spectrum
ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
                                   np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
ds_dict = { k : -1 for k in range(300)} # This remap all sim. indices to the data maps to build QEs with always the data in one leg

#ivfs_d = filt_util.library_shuffle(ivfs, ds_dict)
#ivfs_s = filt_util.library_shuffle(ivfs, ss_dict)


#qlms_ds = qest.library_sepTP(opj(TEMP, 'qlms_ds'), ivfs, ivfs_d, cls_len['te'], nside, lmax_qlm=lmax_qlm)
#qlms_ss = qest.library_sepTP(opj(TEMP, 'qlms_ss'), ivfs, ivfs_s, cls_len['te'], nside, lmax_qlm=lmax_qlm)

#qcls_ds = qecl.library(opj(TEMP, 'qcls_ds'), qlms_ds, qlms_ds, np.array([]))  # for QE RDN0 calculations
#qcls_ss = qecl.library(opj(TEMP, 'qcls_ss'), qlms_ss, qlms_ss, np.array([]))  # for QE RDN0 / MCN0 calculations

# -------------------------

#ivfs_d_aniso = filt_util.library_shuffle(ivfs_aniso, ds_dict)
#qlms_dd_aniso = qest.library_sepTP(os.path.join(TEMP_aniso, 'qlms_dd_aniso'), ivfs_aniso, ivfs_aniso,   cls_len['te'], nside, lmax_qlm = lmax_qlm)


ninv_geom = utils_scarf.Geom.get_healpix_geometry(nside, zbounds = zbounds)

# --------

from plancklens.helpers import mpi

sim_indices = np.arange(args.imin, args.imax + 1)

for simidx in sim_indices[mpi.rank::mpi.size]:

    k = qe_key
    libdir_iterator_lambda = lambda version: libdir_iterators(k, simidx, version)
    lib_plancklens = libdir_iterator_lambda(f"plancklens{version}")
    #lib_plancklens_aniso = libdir_iterator_lambda("plancklens_aniso")
    #lib_delensalot_qe = libdir_iterator_lambda("delensalot_qe")
    #lib_delensalot_qe_iso = libdir_iterator_lambda("delensalot_qe_iso")

    libs = [lib_plancklens] #, lib_plancklens_aniso, lib_delensalot_qe, lib_delensalot_qe_iso]
    libsnames = ["plancklens"] #, "plancklens_aniso", "delensalot_qe", "delensalot_qe_iso"]

    libs_direcs = {key: lib for lib, key in zip(libs, libsnames)}

    for libdir_iterator in libs:
        if not os.path.exists(libdir_iterator):
            os.makedirs(libdir_iterator)

    phi_name = "phi_plm_it000.npy"
    phi_name_norm = "phi_plm_it000_norm.npy"

    path_plm0s = {key: opj(lib, phi_name) for lib, key in zip(libs, libsnames)}
    path_plm0s_norm = {key: opj(lib, phi_name_norm) for lib, key in zip(libs, libsnames)}

    print("Working on QE key", k)

    for key, p in path_plm0s.items():
        print(key, p)
        #if not os.path.exists(p):
        if True:
            if key == "plancklens":

                tr = int(os.environ.get('OMP_NUM_THREADS', 24))           

                print("Getting real data maps.")


                sht_job = utils_scarf.scarfjob()
                #sht_job.set_geometry(ninvjob_geometry)
                sht_job.set_geometry(lenjob_geometry)
                sht_job.set_triangular_alm_info(lmax_ivf, mmax_ivf)
                sht_job.set_nthreads(tr)

                mf_resp = np.zeros(lmax_qlm + 1, dtype=float)

                mf_sims = np.unique(np.array([]))
                mf0 = 0 #qlms_dd.get_sim_qlm_mf(k, mf_sims)

                wflm0 = None

                lensing_qe = "ptt"
                lensing_source = "p"

                Rpp_AA_unl = qresp.get_response(lensing_qe, lmax_ivf, lensing_source, cls_unl, cls_unl, {'e': fel_unl_A, 'b': fbl_unl_A, 't':ftl_unl_A}, lmax_qlm=lmax_qlm)[0]
                Rpp_BB_unl = qresp.get_response(lensing_qe, lmax_ivf, lensing_source, cls_unl, cls_unl, {'e': fel_unl_B, 'b': fbl_unl_B, 't':ftl_unl_B}, lmax_qlm=lmax_qlm)[0]
                Rpp_AB_unl = qresp.get_response(lensing_qe, lmax_ivf, lensing_source, cls_unl, cls_unl, {'e': fel_unl_A, 'b': fbl_unl_A, 't':ftl_unl_A}, {'e': fel_unl_B, 'b': fbl_unl_A, 't':ftl_unl_B}, lmax_qlm=lmax_qlm)[0]
                Rpp_BA_unl = qresp.get_response(lensing_qe, lmax_ivf, lensing_source, cls_unl, cls_unl, {'e': fel_unl_B, 'b': fbl_unl_B, 't':ftl_unl_B}, {'e': fel_unl_A, 'b': fbl_unl_B, 't':ftl_unl_A}, lmax_qlm=lmax_qlm)[0]
                
                Rpp_AA = qresp.get_response(lensing_qe, lmax_ivf, lensing_source, cls_len, cls_grad,  {'e': fel_A, 'b': fbl_A, 't':ftl_A}, lmax_qlm=lmax_qlm)[0]
                Rpp_BB = qresp.get_response(lensing_qe, lmax_ivf, lensing_source, cls_len, cls_grad,  {'e': fel_B, 'b': fbl_B, 't':ftl_B}, lmax_qlm=lmax_qlm)[0]

                Rpp_AB = qresp.get_response(lensing_qe, lmax_ivf, lensing_source, cls_len, cls_grad,  {'e': fel_A, 'b': fbl_A, 't':ftl_A}, {'e': fel_B, 'b': fbl_B, 't':ftl_B}, lmax_qlm=lmax_qlm)[0]
                Rpp_BA = qresp.get_response(lensing_qe, lmax_ivf, lensing_source, cls_len, cls_grad,  {'e': fel_B, 'b': fbl_B, 't':ftl_B}, {'e': fel_A, 'b': fbl_A, 't':ftl_A}, lmax_qlm=lmax_qlm)[0]

                #Rpp_AB = qresp.get_response(lensing_qe, lmax_ivf, lensing_source, cls_len, cls_grad,  {'e': fel_A, 'b': fbl_A, 't':ftl_A}, {'e': fel_B, 'b': fbl_B, 't':ftl_B}, lmax_qlm=lmax_qlm)[0]
                #Rpp_BA = qresp.get_response(lensing_qe, lmax_ivf, lensing_source, cls_len, cls_grad,  {'e': fel_B, 'b': fbl_B, 't':ftl_B}, {'e': fel_A, 'b': fbl_A, 't':ftl_A}, lmax_qlm=lmax_qlm)[0]

                cls_ivfs_A = {'e': fel_A, 'b': fbl_A, 'tt':ftl_A}
                cls_ivfs_B = {'e': fel_B, 'b': fbl_B, 'tt':ftl_B}
                cls_ivfs_AB = {'e': fel_AB, 'b': fbl_AB, 'tt':ftl_AB}
                cls_ivfs_BA = {'e': fel_BA, 'b': fbl_BA, 'tt':ftl_BA}

                #np.savetxt("ftls.txt", np.c_[ftl_A, ftl_B, ftl_AB])

                #NGA = nhl.get_nhl(k, k, cls_len, cls_ivfs_A, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm)[0]
                #NGB = nhl.get_nhl(k, k, cls_len, cls_ivfs_B, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm)[0]
                #NG_AB = nhl.get_nhl(k, k, cls_len, cls_ivfs_A, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm, cls_ivfs_bb=cls_ivfs_B, cls_ivfs_ab=cls_ivfs_AB, cls_ivfs_ba=cls_ivfs_BA)[0]
                #NG_BA = nhl.get_nhl(k, k, cls_len, cls_ivfs_B, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm, cls_ivfs_bb=cls_ivfs_A, cls_ivfs_ab=cls_ivfs_BA, cls_ivfs_ba=cls_ivfs_AB)[0]


                #NGA = nhl.get_nhl(k, k, cls_len, cls_ivfs_A, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm, cls_weights2 = cls_len, cls_ivfs_bb=cls_ivfs_B, cls_ivfs_ab=cls_ivfs_A, cls_ivfs_ba=cls_ivfs_B)[0]
                #NGB = nhl.get_nhl(k, k, cls_len, cls_ivfs_B, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm, cls_ivfs_bb=cls_ivfs_A, cls_ivfs_ab=cls_ivfs_BA, cls_ivfs_ba=cls_ivfs_AB)[0]
                #NG_AB = nhl.get_nhl(k, k, cls_len, cls_ivfs_AB, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm, cls_ivfs_bb=cls_ivfs_BA, cls_ivfs_ab=cls_ivfs_A, cls_ivfs_ba=cls_ivfs_B)[0]
                #NG_BA = nhl.get_nhl(k, k, cls_len, cls_ivfs_AB, lmax_ivf, lmax_ivf, lmax_out=lmax_qlm, cls_ivfs_bb=cls_ivfs_BA, cls_ivfs_ab=cls_ivfs_B, cls_ivfs_ba=cls_ivfs_A)[0]

                path_to_tempura = "/users/odarwish/tempura/"
                import sys
                sys.path.append(path_to_tempura)
                import pytempura as cs


                #GET ILC-DEPR ILC , T\nabla T
                #lensing weights
                wx0  = ftl_A
                wx1  = wx0.copy()
                wxy0 = cls_len['tt'][:lmax_ivf + 1]*ftl_B
                wxy1 = wxy0.copy()

                #total spectra
                a0a1 = cli(ftl_A)
                b0b1 = cli(ftl_B)
                a0b1 = cls_len['tt'][:lmax_ivf + 1]+cross_noise[:lmax_ivf + 1] #note, should be total cross!!!
                a1b0 = cls_len['tt'][:lmax_ivf + 1]+cross_noise[:lmax_ivf + 1] #note, should be total cross!!!

                NGA, Nc0 = cs.noise_spec.qtt_asym('lens', lmax_qlm, lmin_ivf, lmax_ivf, wx0, wxy0, wx1, wxy1, a0a1, b0b1, a0b1, a1b0)
            
                #GET DEPR ILC-ILC , T\nabla T
                #lensing weights
                wx0  = ftl_B
                wx1  = wx0.copy()
                wxy0 = cls_len['tt'][:lmax_ivf + 1]*ftl_A
                wxy1 = wxy0.copy()

                #total spectra
                a0a1 = cli(ftl_B)
                b0b1 = cli(ftl_A)
                a0b1 = cls_len['tt'][:lmax_ivf + 1]+cross_noise[:lmax_ivf + 1] #note, should be total cross!!!
                a1b0 = cls_len['tt'][:lmax_ivf + 1]+cross_noise[:lmax_ivf + 1] #note, should be total cross!!!

                NGB, Bc0 = cs.noise_spec.qtt_asym('lens', lmax_qlm, lmin_ivf, lmax_ivf,wx0,wxy0,wx1,wxy1,a0a1,b0b1,a0b1,a1b0)

                #Here, do cross-noise covariance
                #lensing weights
                wx0  = ftl_A
                wx1  = ftl_B
                wxy0 = cls_len['tt'][:lmax_ivf + 1]*ftl_B
                wxy1 = cls_len['tt'][:lmax_ivf + 1]*ftl_A

                #total spectra
                a0b1 = cli(ftl_A) #ilc
                b0a1 = cli(ftl_B) #D-ilc
                a0a1 = cls_len['tt'][:lmax_ivf + 1]+cross_noise[:lmax_ivf + 1] #note, should be total cross!!!
                b0b1 = cls_len['tt'][:lmax_ivf + 1]+cross_noise[:lmax_ivf + 1] #note, should be total cross!!!

                NG_AB, Xc0 = cs.noise_spec.qtt_asym('lens', lmax_qlm, lmin_ivf, lmax_ivf,wx0,wxy0,wx1,wxy1,a0a1,b0b1,a0b1,b0a1)
                NG_BA = NG_AB

                NGA = utils.cli(Rpp_AB ** 2) * NGA
                NGB = utils.cli(Rpp_BA ** 2) * NGB
                NG_AB = utils.cli(Rpp_AB*Rpp_BA) * NG_AB
                NG_BA = utils.cli(Rpp_AB*Rpp_BA) * NG_BA

                np.savetxt(f"NGS.txt", np.c_[NGA, NGB, NG_AB, NG_BA])
                np.savetxt(f"RS.txt", np.c_[Rpp_AB, Rpp_BA, Rpp_AA, Rpp_BB])

                noise_matrix = np.array([[NGA, NG_AB], [NG_BA, NGB]])
                noise_matrix = np.moveaxis(noise_matrix, -1, 0)
                non_zero = noise_matrix[:, 0, 0] != 0
                inv_noise_matrix = np.zeros_like(noise_matrix)
                inv_noise_matrix[non_zero] = np.linalg.inv(noise_matrix[non_zero]) #three axis: L, a, b
                np.save("inv_noise_matrix.npy", inv_noise_matrix)
                
                minimum_variance_weights = np.sum(inv_noise_matrix, axis = 2)/np.apply_over_axes(np.sum, inv_noise_matrix, [1, 2])[:, 0, 0][:, np.newaxis]
                minimum_variance_weights = np.nan_to_num(minimum_variance_weights, posinf=0, neginf=0, nan=0.0).T

                source_of_aniso = lensing_source

                print("Building unnormalised QE with plancklens.")
                plm0_AA  = qlms_dd_AA.get_sim_qlm(k, int(simidx), asymm = False)
                plm0_BB  = qlms_dd_BB.get_sim_qlm(k, int(simidx), asymm = False)

                plm0_AB  = qlms_dd_AB.get_sim_qlm(k, int(simidx), asymm = True)  #Unormalised quadratic estimate
                plm0_BA  = qlms_dd_BA.get_sim_qlm(k, int(simidx), asymm = True)  #Unormalised quadratic estimate

                np.save("AB_unnorm.npy", plm0_AB)
                np.save("BA_unnorm.npy", plm0_BA)

                plm0_AB_norm = hp.almxfl(plm0_AB, utils.cli(Rpp_AB))
                plm0_BA_norm = hp.almxfl(plm0_BA, utils.cli(Rpp_BA))
        
                plm0_AA_norm = hp.almxfl(plm0_AA, utils.cli(Rpp_AA))
                plm0_BB_norm = hp.almxfl(plm0_BB, utils.cli(Rpp_BB))
                
                plm0norm = almxfl(plm0_AB_norm, minimum_variance_weights[0], mmax_qlm, False)+almxfl(plm0_BA_norm, minimum_variance_weights[1], mmax_qlm, False)
                #if not os.path.exists(path_plm0s_norm[key]):
                np.save(path_plm0s_norm[key], plm0norm)
                np.save("AB.npy", plm0_AB_norm)
                np.save("BA.npy", plm0_BA_norm)
                np.save("weights.npy", minimum_variance_weights)

                np.save(libs_direcs[key]+"/plm0_AA_it000_norm.npy", plm0_AA_norm)
                np.save(libs_direcs[key]+"/plm0_AA_it000_unnorm.npy", plm0_AA)

                np.save(libs_direcs[key]+"/plm0_BB_it000_norm.npy", plm0_BB_norm)
                np.save(libs_direcs[key]+"/plm0_BB_it000_unnorm.npy", plm0_BB)



                cpp = np.copy(cls_unl['pp'][:lmax_qlm + 1])
                cpp[:Lmin] *= 0.
                WFpp_AB = cpp * utils.cli(cpp + NGA)
                WFpp_BA = cpp * utils.cli(cpp + NGB)
                WFpp_AA = cpp * utils.cli(cpp + cli(Rpp_AA))

                #almxfl(plm0_AB_norm, WFpp_AB*(cpp > 0), mmax_qlm, True)
                #almxfl(plm0_BA_norm, WFpp_BA*(cpp > 0), mmax_qlm, True)

                wA, wB = minimum_variance_weights
                get_mv_noise = lambda nA, nB, nAB: wA**2*nA + wB**2*nB + 2*wA*wB*nAB

                Nmv = get_mv_noise(NGA, NGB, NG_AB)
                WFpp_mv = cpp * utils.cli(cpp + Nmv)
                plm0 = almxfl(plm0norm, WFpp_mv*(cpp > 0), mmax_qlm, False)
                plm0 = utils.alm_copy(plm0, lmax_qlm)
                np.savetxt("WFpp.txt", np.c_[WFpp_mv, WFpp_AB, WFpp_BA, WFpp_AA])

                #print("minimum_variance_weights shaaape", minimum_variance_weights.shape)
                #plm0 = almxfl(plm0_AB_norm, minimum_variance_weights[0], mmax_qlm, False)+almxfl(plm0_BA_norm, minimum_variance_weights[1], mmax_qlm, False)

                ffi = deflection(lenjob_geometry_defl, np.zeros_like(plm0), mmax_qlm, numthreads = tr, epsilon = 1e-7)


                filtr_A = alm_filter_iso(effective_noise_A, ffi, transf_tlm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf))
                filtr_B = alm_filter_iso(effective_noise_B, ffi, transf_tlm, (lmax_unl, mmax_unl), (lmax_ivf, mmax_ivf))
                        
                updating_key = "p"
                

                filtrs = [filtr_A, filtr_B]

                k_geom = filtr_A.ffi.geom

                datmaps_real_A = sims_A_MAP.get_sim_tmap(int(simidx))
                datmaps_real_B = sims_B_MAP.get_sim_tmap(int(simidx))
                datmaps = [sht_job.map2alm(datmaps_real) for datmaps_real in [datmaps_real_A, datmaps_real_B]]
                #np.save("../notebooks/datmaps.npy", datmaps)

                Rpp_unl = cli(Nmv)
                Rpp_unl_A = cli(NGA)
                Rpp_unl_B = cli(NGB)

                Rpp_unl = Rpp_AB+Rpp_BA
                
                iterator = iterator_cstmf_asymm(libs_direcs[key], updating_key, (lmax_qlm, mmax_qlm), datmaps,
                            plm0, plm0*0., Rpp_unl, cpp, cls_unl, filtrs, k_geom, chain_descrs(lmax_unl, cg_tol), stepper
                            , wflm0=wflm0, asymm_weights = minimum_variance_weights)


                tol_iter   = lambda it : 10 ** (- cg_tol)
                soltn_cond = lambda it: True

                for iter in iters_to_be_done:
                    iterator.chain_descr  = chain_descrs(lmax_unl, tol_iter(iter))
                    iterator.soltn_cond   = soltn_cond(iter)
                    iterator.iterate(iter, 'p')
                




            





