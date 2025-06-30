#Plotting Wiener filter
import utils_data as ud
from jointmap import plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", nargs='+', default="spt3g") #config_paths = ["official_multiple", "official_check_factor_2", "official_so_a_disabled", "official_multiple_disabled"]
parser.add_argument("--key", type=str, default="a") #a cosmic birefringence, p lensing potential, o lensing curl, f patchy
args = parser.parse_args()
config = args.config
sindex = args.key

plot.set_default_params()

colors = plot.COLOR_BLIND_FRIENDLY["main"]
colors_2 = plot.COLOR_BLIND_FRIENDLY["wong"]

#hardcoded...

config_paths = config
config_paths = [f"configs/{config}.yaml" for config in config_paths]


for iconfig, config_path in enumerate(config_paths):
    selected, autoits, crossits, auto_input_its, crossits_down, auto_input_down_its, crossits_lensed, itmax, name, crossits_down_2, crossits_down_3 = ud.process_config_npy(config_path)
    savingname = name if iconfig == 0 else savingname
    index = selected.index(sindex)

    ymin, ymax = (0, 1)

    autos = autoits.mean(axis = 0)
    crosses = crossits.mean(axis = 0)
    crosses_down = crossits_down.mean(axis = 0)
    if type(crossits_down_2) == np.ndarray:
        crossits_down_2 = crossits_down_2.mean(axis = 0)
        crossits_down_3 = crossits_down_3.mean(axis = 0)
    else:
        crossits_down_2 = 0
        crossits_down_3 = 0

    std_crosses = crossits.std(axis = 0) / np.sqrt(autoits.shape[0])
    crosses_lensed = crossits_lensed.mean(axis = 0)
    std_crosses_lensed = crossits_lensed.std(axis = 0) / np.sqrt(autoits.shape[0])

    inputs = auto_input_its.mean(axis = 0)
    inputsdown = auto_input_down_its.mean(axis = 0)

    autos_split = np.split(autos, len(selected), axis = -1)
    cross_split = np.split(crosses, len(selected), axis = -1)
    cross_down_split = np.split(crosses_down, len(selected), axis = -1)
    if type(crossits_down_2) == np.ndarray:
        cross_down_2_split = np.split(crossits_down_2, len(selected), axis = -1)
        cross_down_3_split = np.split(crossits_down_3, len(selected), axis = -1)
    else:
        cross_down_2_split = 0
        cross_down_3_split = 0
    crosses_downs = [cross_down_split, cross_down_2_split, cross_down_3_split]
    cross_lensed_split = np.split(crosses_lensed, len(selected), axis = -1)
    std_cross_split = np.split(std_crosses, len(selected), axis = -1)
    std_cross_lensed_split = np.split(std_crosses_lensed, len(selected), axis = -1)

    itmax = itmax

    caso = {0: "QE", itmax: "MAP no MF sub."}
    caso_mf = {0: "QE", itmax: "MAP"}

    lines = []

    if iconfig == 0:
        # Create a figure with subplots
        nrows = 4
        ncols = len(selected)
        fig, axes = plt.subplots(nrows=nrows, ncols = ncols, figsize = (6*ncols, 10), squeeze = False, sharex = True)
        plt.subplots_adjust(hspace=0.4)

    for i, k in enumerate(selected):
        original_idx = i

        for s in range(nrows):
            ax = axes[s, i]
            for j, itr in enumerate([0, itmax]):
                if s == 0:
                    el, cross = ud.cross_corr_coeff_from_cl(autos_split[original_idx][itr], inputs[original_idx], cross_split[original_idx][itr], plot=False, bin = False)
                    if (k == "a") and (s == 0):
                            np.savetxt(f"crossa_iconfig_{iconfig}_itr_{itr}.txt", np.c_[el, cross])
                else:
                    if iconfig < 2:
                        el, cross = ud.cross_corr_coeff_from_cl(autos_split[original_idx][itr], inputs[ud.get_down_index(selected, k, s)], cross_down_split[original_idx][itr], plot=False, bin = False)
                        if (k == "o") and (s == 1):
                            np.savetxt(f"crosso_iconfig_{iconfig}_itr_{itr}.txt", np.c_[el, cross])
                    elif iconfig == 2:
                        el, cross = ud.cross_corr_coeff_from_cl(autos_split[original_idx][itr], inputs[ud.get_down_index(selected, k, s)], crosses_downs[s-1][original_idx][itr], plot=False, bin = False)
                if s <= 1:
                    if (iconfig == 0) and (itr > 0):
                        ax.plot(el, cross, label = caso[itr] if ((i == 0) and (iconfig == 0)) else None, color = colors[2+j], ls = "--", zorder = 1)            
                    elif iconfig == 1: #no derotation case
                        if (s == 1) and (i == len(selected)-1) and (itr > 0):
                            ax.plot(el, cross, label = "MAP delensing only", color = colors[4+j+s])
                            ax.legend(fontsize = 14)
                if iconfig == 2: #mean-field subtracted
                    if (itr >= 0):
                        make_label = (((i == 0) and (iconfig == 2)))
                        ax.plot(el, cross, label = caso_mf[itr] if make_label else None, color = colors[1] if itr != 0 else colors[0], ls = "-")#, zorder = 1)
                        if (s == 0) and make_label:
                            ax.legend(ncol = 3)

                        if k == "f":
                            rhoBH_contamination, rhoBH_input = np.loadtxt(ud.directory_files+"rhoBHs.txt").T
                            colore = colors_2[2]
                            alpha = 0.6
                            if (s == 0) and (itr == 0):
                                ax.plot(el, rhoBH_input, color = colore, label = "BH" if s == 0 else None, ls = "--", zorder = 1, alpha = alpha)
                                ax.legend(ncol = 1, fontsize = 14)
                            elif (s == 1):
                                ax.plot(el, rhoBH_contamination, color = colore, ls = "--", alpha = alpha)

            """if s == 0:
                if k == "a":
                    
                    rho_QE, rho_MAP = ud.get_rhos()
                    ax.plot(rho_QE, lw = 2, color = colors_2[3])
                    ax.plot(rho_MAP, lw = 2, color = colors_2[2])

                    #rho_QE, rho_MAP = ud.get_rhos(n1 = False)
                    #ax.plot(rho_QE, lw = 2, color = colors_2[7], ls = ":")
                    #ax.plot(rho_MAP, lw = 2, color = colors_2[6], ls = ":")
            """

            ax.set_xlim(2, 3000)
            ax.set_xscale("log")
            #ax.set_ylim(ymin, ymax)
            ax.tick_params(axis='both', which='major', labelsize=18)
            
            if i == 0:
                ax.set_ylabel(r"$\rho_L$", fontsize=25)
                if s == 0:
                    ax.legend(ncol = 1, fontsize = 14)
            
            if s >= 1:
                ax.set_title(f"{ud.generate_hat_bracket(k, selected[ud.get_down_index(selected, k, s)])}", fontsize=22)
            else:
                ax.set_title(f"{ud.nomeLatexX[k]}", fontsize=22)
            if s == nrows-1:
                ax.set_xlabel(r"$L$", fontsize=25)

fig.align_ylabels(axes[:, 0])

#fig.tight_layout()
plt.savefig(f"{ud.out_dir}xcorr_{ud.nome[sindex]}_{savingname}.pdf")
#plt.savefig(f"prova.pdf")
