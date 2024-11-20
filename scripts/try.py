import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os

from delensalot.core import cachers

from lenspyx.remapping import utils_geom as utils_scarf
from delensalot.core.secondaries import secondaries



cmbversion = "cmblensing_only"
scratch = os.environ["SCRATCH"]
scratch = f"{scratch}/{cmbversion}/"

dir = f"{scratch}/simswalpha/"

#alpha_both = hp.read_alm(dir+"sim_0000_alpha_lm.fits").astype(np.complex128)
phi_both = hp.read_alm(dir+"sim_0000_plm.fits").astype(np.complex128)
elm_both = hp.read_alm(dir+"sim_0000_elm.fits").astype(np.complex128)
blm_both = hp.read_alm(dir+"sim_0000_blm.fits").astype(np.complex128)

lmax = hp.Alm.getlmax(phi_both.size)
mmax = lmax

nside = 2048
ninv_geom = utils_scarf.Geom.get_healpix_geometry(nside)
ninv_geom = utils_scarf.Geom.get_thingauss_geometry(lmax + 100, 2)

alpha_map = ninv_geom.synthesis(phi_both, spin = 0, lmax = lmax, mmax = mmax, nthreads = 128).squeeze()

R = secondaries.Rotation(name = "r", lmax = lmax, mmax = mmax, sht_tr = 64)
R.set_field(alpha_map)
eblm = np.array([elm_both, blm_both])

RotationOp = secondaries.Rotation(name = "r", lmax = lmax, mmax = mmax, sht_tr = 32)
RotationOp.set_field(alpha_map)
Operator = secondaries.Operators([RotationOp])

elm_2d = eblm.reshape((1, eblm.size))

Operator(eblm = eblm, lmax_in = lmax, spin = 2, lmax_out = lmax, mmax_out = mmax,
                                 backwards=True, gclm_out=elm_2d, out_sht_mode='GRAD_ONLY', q_pbgeom = ninv_geom)


print("Done")