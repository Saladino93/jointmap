"""
Will plot Figure for alpha, to show the power spectra, reconstructed power spectra, as well as the effect of lensing on alpha.
"""


import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join as opj

from plancklens import utils

from iterativefg import utils as itu



color_palette = [
    "#377eb8",  # Blue
    "#ff7f00",  # Orange
    "#4daf4a",  # Green
    "#f781bf",  # Pink
    "#a65628",  # Brown
    "#984ea3",  # Purple
    "#999999",  # Grey
    "#e41a1c",  # Red
    "#dede00"   # Yellow
]


bin_edges = np.arange(2, 1000, 20)
bin_edges = np.append(bin_edges, np.arange(1000, 4000, 50))

cmbversion = "alpha_phi_cmb_new_rot"
version = "alpha_phi_cmb_new_rot_test_jan_4"

#plt.title(r"$\alpha$")

alpha = hp.read_alm(f"/scratch/snx3000/odarwish/JOINTRECONSTRUCTION/{cmbversion}/simswalpha/sim_0012_alpha_lm.fits")
alpha = utils.alm_copy(alpha, 5000)
input = hp.alm2cl(alpha)

dir = f"/scratch/snx3000/odarwish/JOINTRECONSTRUCTION/{cmbversion}_version_{version}_recs/p_p_sim0012{version}/"
plm0 = np.load(dir + "alm0_norm.npy")
plm0_12 = np.load(dir + "alm0_11_norm.npy")

# Perform calculations

out_dir = "noise_biases/"
n_gg = np.loadtxt(out_dir+"ngg_a_QE.txt") #GG_N0 * utils.cli(r_gg_fid ** 2)
n1_ap = np.loadtxt(out_dir+"n1_ap_QE.txt")
n1_aa = np.loadtxt(out_dir+"n1_aa_QE.txt")


cltot = hp.alm2cl(plm0)
plt.loglog(cltot, label=r"$C_L^{\hat{\alpha}\hat{\alpha}}$", color=color_palette[0])
N0_rand = hp.alm2cl(plm0_12)
plt.loglog(N0_rand, label=r"$N_0^{\mathrm{rand}}$", color=color_palette[1])
plt.loglog(input, label=r"$ C_L^{\alpha\alpha}$", color=color_palette[2])
plt.plot(n_gg, label=r"$N_0^{\mathrm{th}}$", color=color_palette[3])
#plt.plot(cltot - N0_rand, label=r"$C_L^{\hat{\alpha}\hat{\alpha}}$ - $N_0^{\mathrm{rand}}$", color=color_palette[4], alpha = 0.4)

# Calculate difference and bin
difference = cltot - N0_rand - input[:5001]
el, x = itu.bin_theory(difference, bin_edges)
plt.plot(el, x, label=r"$C_L^{\hat{\alpha}\hat{\alpha}} - N_0^{\mathrm{rand}} - C_L^{\alpha\alpha}$", color=color_palette[5])

#plt.plot(el, el*0+4.01177468e-09, color = "black")

plt.loglog(n1_aa, color = "brown", label = r"$N_1^{\alpha\alpha}$")
plt.loglog(n1_ap, color = "cyan", ls = "--", label = r"$N_1^{\alpha\phi}$")

reconstruction_cross = np.loadtxt(out_dir+"claa_rec.txt")
el, x = itu.bin_theory(reconstruction_cross, bin_edges)
plt.plot(el, x, label=r"$C_L^{\hat{\alpha}\alpha}$", color=color_palette[6])


# Add labels and legend
plt.ylabel("$C_L$", fontsize=18)
plt.xlabel("$L$", fontsize=18)
plt.legend(ncols = 2, fontsize = 12)

plt.ylim(ymax = 1e-7)
plt.xlim(5, 2500)
plt.savefig("/users/odarwish/JointCMBiterative/figures/examples/N0_QE_alpha_subtraction.pdf", dpi=300)

# Display the plot
plt.show()
plt.close()



signal = input
ls = np.arange(0, len(signal), 1)
fsky = 0.4
Nmodes = (2*(ls+1))*fsky
errorbar = np.sqrt(2*(signal+N0_rand)**2)/np.sqrt(Nmodes)
#plt.errorbar(ls, signal, yerr = errorbar, fmt = "o", color = "red")
#plt.fill_between(ls, signal-errorbar, signal+errorbar, alpha = 0.2, color = "red")

plt.loglog(input, label=r"$C_L^{\alpha\alpha}$", color=color_palette[2])

reconstruction_cross = np.loadtxt(out_dir+"claa_rec.txt")
el, x = itu.bin_theory(reconstruction_cross, bin_edges)
plt.plot(el, x, label=r"$C_L^{\hat{\alpha}\alpha}$", color=color_palette[6])

reconstruction_cross = np.loadtxt(out_dir+"claa_len_rec.txt")
el, x = itu.bin_theory(reconstruction_cross, bin_edges)
plt.plot(el, x, label=r"$C_L^{\hat{\alpha}\tilde{\alpha}}$", color=color_palette[5])


plt.ylabel("$C_L$", fontsize=18)
plt.xlabel("$L$", fontsize=18)
plt.legend(ncols = 2, fontsize = 12)

plt.ylim(ymax = 1e-7)
plt.xlim(5, 2500)
plt.savefig("/users/odarwish/JointCMBiterative/figures/examples/QE_alpha_cross.pdf", dpi=300)