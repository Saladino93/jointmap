#Plotting Wiener filter
import utils_data as ud
from jointmap import plot
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import argparse


def SNR(claa, noise, Lmax, fsky = 1.):
    ells = np.arange(len(noise))
    nmodes = (2*ells + 1)*fsky
    total = claa+noise
    num = claa**2*nmodes
    den = 2*total**2
    result = num/den
    selection = (ells <= Lmax) & (ells >= 1)
    return np.sqrt(np.sum(result[selection]))

def SNRbire(A, noise, Lmax, fsky = 1.):
    ells = np.arange(len(noise))
    nmodes = (2*ells + 1)*fsky
    factor = (ells)*(ells+1)/2/np.pi
    total = A+noise*factor
    num = A**2*nmodes
    den = 2*total**2
    result = num/den
    selection = (ells <= Lmax) & (ells >= 1)
    return np.sqrt(np.sum(result[selection]))
    

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default="official_check_factor_2") #config_paths = ["official_check_factor_2", "official_so_a_disabled"]
parser.add_argument("--key", type=str, default="a") #a cosmic birefringence, p lensing potential, o lensing curl, f patchy
args = parser.parse_args()
config = args.config
sindex = args.key

plot.set_default_params()

colors = plot.COLOR_BLIND_FRIENDLY["main"]

#hardcoded...

config_paths = [config]
config_paths = [f"configs/{config}.yaml" for config in config_paths]


for i, config_path in enumerate(config_paths):
    selected, autoits, crossits, auto_input_its, _, _, crossits_lensed, itmax, name = ud.process_config_npy(config_path)
    index = selected.index(sindex)

    qty = ud.get_noise_curves(case = name)
    ymin, ymax = (1e-11, 1e-7) if ("s4" in name) or ("spt" in name) else (5e-10, 1e-6)

    #to be generalized
    cls_alpha = qty["claath"]
    ngg = qty["ngg0"]
    ngg_10 = qty["nggmax"]
    n1ap = qty["n1ap0"]
    n1ap_10 = qty["n1apmax"]

    WFth = cls_alpha/(cls_alpha+ngg)
    WFth10 = cls_alpha/(cls_alpha+ngg_10)

    N0s = {0: ngg, itmax: ngg_10}
    N1s = {0: n1ap, itmax: n1ap_10}
    WFths = {0: WFth, itmax: WFth10}

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

    caso = {0: "QE", itmax: "MAP", 1: "Itr 1"}

    lines = []

    SNRS = {}
    SNRSA = {}

    for j, itr in enumerate([0, itmax]):
        WF = cross_split[index][itr]/inputs[index]
        WF_lensed = cross_lensed_split[index][itr]/inputs[index]
        WFth = WFths[itr]

        xx = autos_split[i][itr]*WF_lensed**-2.-inputs[i]
        line1 = plt.plot(N0s[itr], label=caso[itr], ls="--", color=colors[j], lw=3)[0]
        plt.loglog(N1s[itr], color=colors[j], lw=2, ls=":")
        plt.plot(xx, color=colors[j], ls="-", alpha=0.4)
        lines.append(line1)

        noise = N0s[itr][:5001]+N1s[itr][:5001]
        A = 1e-7
        SNRS[itr] = np.array([SNR(cls_alpha, noise, 100), SNRbire(A, noise, 100)])    

        if j == 1 and 'spt' in name:
            elb, resb, stds = qty["resb"]
            resb = (resb)
            #plt.errorbar(elb, resb, stds, ls = "--", color = colors[j], lw = 2, marker = "o", markersize = 3)
            plt.plot(qty["n1ap1"], color = colors[j+4], lw = 2, ls = ":")
            line1 = plt.plot(qty["ngg1"], ls="--", color=colors[j+4])[0]
            lines.append(line1)


    print("SNRS", SNRS[itmax]/SNRS[0])

    # Create the style legend (first legend)
    legend_elements = [
        Line2D([0], [0], color='black', linestyle='-', label=r'$C_L^{\hat{\alpha}\hat{\alpha}}-C_L^{\alpha\alpha}$'),
        Line2D([0], [0], color='black', linestyle='--', label=r'$N^0_L$'),
        Line2D([0], [0], color='black', linestyle=':', linewidth = 2, label=r'$N^{1,\alpha\phi}_L$')
                        ]
    
    legend1 = plt.legend(handles = legend_elements, 
                        loc = 'upper left', ncol = 3)

    # Add it to the plot
    plt.gca().add_artist(legend1)

    lista = [0, itmax] if ('spt' not in name) else [0, itmax, 1]
    ncol = len(lista) if ('spt' not in name) else 2

    # Create the case legend (second legend)
    plt.legend([lines[j] for j, _ in enumerate(lista)], 
            [caso[itr] for itr in lista], 
            loc='upper right', ncol = ncol)

    plt.tick_params(axis='both', which='major', labelsize=18)

    plt.xlim(2, 2000)

    plt.ylim(ymin, ymax)

    plt.xlabel(r"$L$", fontsize = 18)
    plt.ylabel(r"$N_L$", fontsize = 18)
    plt.savefig(f"{ud.out_dir}noise_{ud.nome[sindex]}_{name}.pdf")
