#Plotting Wiener filter
import utils_data as ud
from jointmap import plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", nargs='+', default="official_check_factor_2") #config_paths = ["official_multiple", "official_check_factor_2", "official_so_a_disabled", "official_multiple_disabled"]
parser.add_argument("--key", type=str, default="a") #a cosmic birefringence, p lensing potential, o lensing curl, f patchy
args = parser.parse_args()
config = args.config
sindex = args.key

plot.set_default_params()

colors = plot.COLOR_BLIND_FRIENDLY["main"]

#hardcoded...

config_paths = config
config_paths = [f"configs/{config}.yaml" for config in config_paths]


for iconfig, config_path in enumerate(config_paths):
    selected, autoits, crossits, auto_input_its, crossits_down, auto_input_down_its, crossits_lensed, itmax, name = ud.process_config_npy(config_path)
    savingname = name if iconfig == 0 else savingname
    index = selected.index(sindex)

    ymin, ymax = (0, 1)

    autos = autoits.mean(axis = 0)
    crosses = crossits.mean(axis = 0)
    crosses_down = crossits_down.mean(axis = 0)
    std_crosses = crossits.std(axis = 0) / np.sqrt(autoits.shape[0])
    crosses_lensed = crossits_lensed.mean(axis = 0)
    std_crosses_lensed = crossits_lensed.std(axis = 0) / np.sqrt(autoits.shape[0])

    inputs = auto_input_its.mean(axis = 0)
    inputsdown = auto_input_down_its.mean(axis = 0)

    autos_split = np.split(autos, len(selected), axis = -1)
    cross_split = np.split(crosses, len(selected), axis = -1)
    cross_down_split = np.split(crosses_down, len(selected), axis = -1)
    cross_lensed_split = np.split(crosses_lensed, len(selected), axis = -1)
    std_cross_split = np.split(std_crosses, len(selected), axis = -1)
    std_cross_lensed_split = np.split(std_crosses_lensed, len(selected), axis = -1)

    itmax = 15

    caso = {0: "QE", itmax: "MAP"}

    lines = []

    if iconfig == 0:
        # Create a figure with subplots
        nrows = 2
        ncols = len(selected)
        fig, axes = plt.subplots(nrows=nrows, ncols = ncols, figsize = (6*ncols, 6), squeeze = False, sharex = True)
        plt.subplots_adjust(hspace=0.3)

    for i, k in enumerate(selected):
        original_idx = i
        for s in range(nrows):
            ax = axes[s, i]
            for j, itr in enumerate([0, itmax]):
                if s == 0:
                    el, cross = ud.cross_corr_coeff_from_cl(autos_split[original_idx][itr], inputs[original_idx], cross_split[original_idx][itr], plot=False, bin = False)
                else:
                    el, cross = ud.cross_corr_coeff_from_cl(autos_split[original_idx][itr], inputsdown[original_idx], cross_down_split[original_idx][itr], plot=False, bin = False)

                if iconfig == 0:
                    ax.plot(el, cross, label = caso[itr] if ((i == 0) and (iconfig == 0)) else None, color = colors[2+j])            
                elif iconfig == 1: #derotation case
                    if (s == 1) and (i == len(selected)-1) and (itr > 0):
                        ax.plot(el, cross, label = "MAP no derotation", color = colors[4+j+s])
                        ax.legend(fontsize = 14)
                elif iconfig == 2: #mean-field subtracted
                    if (k == "f") and (itr > 0):
                        ax.plot(el, cross, label = "MAP MF sub." if s == 0 else None, color = colors[0], alpha = 0.4, zorder = 0)
                        if s == 0:
                            ax.legend()

                        rhoBH_contamination, rhoBH_input = np.loadtxt(ud.directory_files+"rhoBHs.txt").T
                        if s == 0:
                            ax.plot(el, rhoBH_input, color = colors[1], label = "BH" if s == 0 else None, ls = "--", zorder = 0, alpha = 0.4)
                            ax.legend(ncol = 2, fontsize = 14)
                        else:
                            ax.plot(el, rhoBH_contamination, color = colors[1], ls = "--", alpha = 0.4)

            ax.set_xlim(2, 1000)
            ax.set_xscale("log")
            #ax.set_ylim(ymin, ymax)
            ax.tick_params(axis='both', which='major', labelsize=18)
            
            if i == 0:
                ax.set_ylabel(r"$\rho_L$", fontsize=20)
                if s == 0:
                    ax.legend(ncol = 2, fontsize = 14)
            
            if s == 1:
                ax.set_xlabel(r"$L$", fontsize=20)
                ax.set_title(f"{ud.nomeLatexXdown[k]}", fontsize=18)
            else:
                ax.set_title(f"{ud.nomeLatexX[k]}", fontsize=18)

#fig.tight_layout()
plt.savefig(f"{ud.out_dir}xcorr_{ud.nome[sindex]}_{savingname}.pdf")
#plt.savefig(f"prova.pdf")

