#Plotting Wiener filter
import utils_data as ud
from jointmap import plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="official_check_factor_2")
parser.add_argument("--index", type=int, default=0) #0 cosmic birefringence, 1 lensing potential, 2 lensing curl
args = parser.parse_args()
config = args.config
index = args.index

plot.set_default_params()

colors = plot.COLOR_BLIND_FRIENDLY["main"]

#hardcoded...

config_paths = [config]
config_paths = [f"configs/{config}.yaml" for config in config_paths]


for i, config_path in enumerate(config_paths):
       selected, autoits, crossits, auto_input_its, _, _, crossits_lensed, itmax, name = ud.process_config_npy(config_path)

       qty = ud.get_noise_curves(name)

       cls_alpha = qty["claath"]
       ngg = qty["ngg0"]
       ngg_10 = qty["nggmax"]

       WFth = cls_alpha/(cls_alpha+ngg)
       WFth10 = cls_alpha/(cls_alpha+ngg_10)

       WFths = [WFth, WFth10]

       [np.savetxt(f"{ud.out_dir_wiener_data}WF_itr_{i}.txt", W) for i, W in enumerate(WFths)]

       autos = autoits.mean(axis = 0)
       crosses = crossits.mean(axis = 0)
       std_crosses = crossits.std(axis = 0) / np.sqrt(autoits.shape[0])
       crosses_lensed = crossits_lensed.mean(axis = 0)
       std_crosses_lensed = crossits_lensed.std(axis = 0) / np.sqrt(autoits.shape[0])

       inputs = auto_input_its.mean(axis = 0)

       autos_split = np.split(autos, len(selected), axis = -1)
       cross_split = np.split(crosses, len(selected), axis = -1)
       cross_lensed_split = np.split(crosses_lensed, len(selected), axis = -1)
       std_cross_split = np.split(std_crosses, len(selected), axis = -1)
       std_cross_lensed_split = np.split(std_crosses_lensed, len(selected), axis = -1)

       caso = {0: "QE", itmax: "itr"}

       for j, itr in enumerate([0, itmax]):
              WF = cross_split[index][itr]/inputs[index]
              WF_lensed = cross_lensed_split[index][itr]/inputs[index]

              std_WF = std_cross_split[index][itr]/inputs[index]
              std_WF_lensed = std_cross_lensed_split[index][itr]/inputs[index]

              np.savetxt(f"{ud.out_dir_wiener_data}WF_{ud.nome[index]}_{config}_{itr}.txt", np.c_[WF, WF_lensed])

              el, WF = ud.decorator(WF)
              plt.loglog(el, WF, ls = "--", color=colors[2*j], 
                     label=r"$C_L^{\hat{\alpha}\alpha}$")
              el, std_WF = ud.decorator(std_WF)
              plt.fill_between(el, WF-std_WF, WF+std_WF, color=colors[2*j], alpha=0.2)

              el, WF_lensed = ud.decorator(WF_lensed)
              plt.loglog(el, WF_lensed, ls=":", color=colors[2*j], 
                     label=r"$C_L^{\hat{\alpha}\widetilde{\alpha}}$")
              
              plt.loglog(WFths[j], color=colors[2*j], 
                     label=r"$W_F^{\mathrm{th}}$")

       legend_elements = [
              Line2D([0], [0], color='black', linestyle='--', label=r'$C_L^{\hat{\alpha}\alpha}$'),
              Line2D([0], [0], color='black', linestyle=':', label=r'$C_L^{\hat{\alpha}\widetilde{\alpha}}$'),
              Line2D([0], [0], color='black', linestyle='-', label=r'$W_F^{\mathrm{th}}$')
                            ]

       # Add the legend with black lines
       legend = plt.legend(handles=legend_elements, ncol=3, fontsize = 14)

       #plt.yscale("linear")

       plt.tick_params(axis='both', which='major', labelsize=18)

       plt.xlim(1, 2000)
       plt.ylim(1e-5)

       plt.xlabel(r"$L$", fontsize = 18)
       plt.ylabel(r"$W_L$", fontsize = 18)
       plt.savefig(f"{ud.out_dir}WF_{ud.nome[index]}_{config}.pdf")
