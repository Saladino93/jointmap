import healpy as hp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from iterativefg import utils as itu
import utils_data as ud

# Set up the output directories
out_dir = "/users/odarwish/JointCMBiterative/figures/examples/"
out_dir = ud.out_dir

# Set global plotting parameters for better visibility
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['font.size'] = 14

# Color blind friendly palette
color_palette = [
    "#0077BB",  # Blue
    "#EE7733",  # Orange
    "#009988",  # Teal
    "#CC3311",  # Red
    "#33BBEE",  # Cyan
    "#EE3377",  # Magenta
    "#BBBBBB",  # Grey
    "#000000"   # Black
]

# Create figure with GridSpec
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.3)

# Top panel (main plot)
ax1 = fig.add_subplot(gs[0])

# Make borders thicker
for spine in ax1.spines.values():
    spine.set_linewidth(1.5)

# Set up bin edges
bin_edges = np.arange(2, 1000, 20)
bin_edges = np.append(bin_edges, np.arange(1000, 4000, 50))

# Load data
directory = "../data/plots/plot_alpha_data/"
input = np.loadtxt(directory+"input.txt")

dircuves = "noise_biases/"

qty = ud.get_noise_curves()
ngg = qty["ngg0"]
ngg_10 = qty["nggmax"]
n1_ap = qty["n1ap0"]
n1_aa = qty["n1aa0"]

# Plot in top panel with distinct colors and line styles
cltot = np.loadtxt(directory+"cltot.txt")
ax1.loglog(cltot, label=r"$C_L^{\hat{\alpha}\hat{\alpha}}$", 
           color=color_palette[0], linestyle='-')

N0_rand = np.loadtxt(directory+"N0_rand.txt")
ax1.loglog(N0_rand, label=r"$N_0^{\mathrm{rand}}$", 
           color=color_palette[1], linestyle='--')

ax1.loglog(input, label=r"$ C_L^{\alpha\alpha}$", 
           color=color_palette[7], linestyle='-')

ax1.plot(ngg, label=r"$N_0^{\mathrm{th}}$", 
         color=color_palette[2], linestyle=':')

# Calculate and plot difference
difference = cltot - N0_rand - input[:5001]
el, x = ud.bin_theory(difference, bin_edges)
ax1.plot(el, x, label=r"$C_L^{\hat{\alpha}\hat{\alpha}} - N_0^{\mathrm{rand}} - C_L^{\alpha\alpha}$", 
         color=color_palette[3], linestyle='-.')

ax1.loglog(n1_aa, color=color_palette[4], linestyle='--', 
           label=r"$N_1^{\alpha\alpha}$")
ax1.loglog(n1_ap, color=color_palette[5], linestyle=':', 
           label=r"$N_1^{\alpha\phi}$")

# Plot reconstruction cross-correlation
reconstruction_cross = qty["claa_rec"]
el, x = itu.bin_theory(reconstruction_cross, bin_edges)
ax1.plot(el, x, label=r"$C_L^{\hat{\alpha}\alpha}$", 
         color=color_palette[6], linestyle='-')

# Customize top panel
ax1.set_ylabel("$C_L$", fontsize=18)
ax1.set_xlabel("$L$", fontsize=18)
ax1.legend(ncols=2, fontsize=12, frameon=True, 
          facecolor='white', edgecolor='black')
ax1.set_ylim(ymax=1e-7)
ax1.set_xlim(5, 1000)

# Add grid for better readability
ax1.grid(True, linestyle='--', alpha=0.7, color='#CCCCCC')

# Increase tick label sizes
ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(axis='both', which='minor', labelsize=12)

# Bottom panel
ax2 = fig.add_subplot(gs[1])

# Make borders thicker
for spine in ax2.spines.values():
    spine.set_linewidth(1.5)

# Set up bin edges for bottom panel
bin_edges = np.arange(2, 1000, 10)
bin_edges = np.append(bin_edges, np.arange(1000, 4000, 50))

# Calculate and plot ratios
difference = cltot - N0_rand - input[:5001]
el, x = itu.bin_theory(difference, bin_edges)
el, input_binned = itu.bin_theory(input[:5001], bin_edges)
ax2.plot(el, x/input_binned, 
         label=r"$C_L^{\hat{\alpha}\hat{\alpha}} - N_0^{\mathrm{rand}} - C_L^{\alpha\alpha}$", 
         color=color_palette[3], linestyle='-.')

# Calculate error bars
signal = input
ls = np.arange(0, len(signal), 1)
fsky = 0.4
Nmodes = (2*(ls+1))*fsky
errorbar = np.sqrt(2*(signal+N0_rand)**2)/np.sqrt(Nmodes)

ax2.plot(el, input_binned/input_binned, 
         label=r"$C_L^{\alpha\alpha}$", 
         color=color_palette[7], linestyle='-')

el, x = itu.bin_theory(n1_aa, bin_edges)
ax2.plot(el, x/input_binned, 
         color=color_palette[4], linestyle='--', 
         label=r"$N_1^{\alpha\alpha}$")

# Customize bottom panel
ax2.set_ylabel(r"$C_L/C_L^{\alpha\alpha}$", fontsize=18)
ax2.set_xlabel(r"$L$", fontsize=18)
ax2.set_xlim(5, 1000)
ax2.grid(True, linestyle='--', alpha=0.7, color='#CCCCCC')

# Increase tick label sizes for bottom panel
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='minor', labelsize=12)

ax2.set_xscale("log")
ax2.yaxis.set_minor_locator(plt.LinearLocator(numticks=10))
ax2.set_ylim(-0.1, 0.9)

# Save the figure
plt.savefig(f"{out_dir}combined_alpha_plots.pdf", dpi=300, bbox_inches='tight')
plt.close()