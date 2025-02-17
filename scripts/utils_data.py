import numpy as np
import yaml
import scipy
import healpy as hp
from plancklens import utils
import os


directory_files = "/Volumes/OmarFiles/joint_map_outputs/"
out_dir = "../plots/"
out_dir = "/Users/omard/Documents/papers/JointCMBiterative/figures/examples/"
if not os.path.exists(out_dir):
    print(f"Directory {out_dir} does not exist.")
    #halt program
    exit()

out_data = "../data/"
out_dir_wiener_data = f"{out_data}/plots/wiener_curves/"
dirs = [out_dir, out_data, out_dir_wiener_data]
for dir in dirs:
    os.makedirs(dir, exist_ok=True)

nome = {"a": "alpha", "p": "phi", "o": "omega", "f": "tau"}
nomeLatex = {"a": r"$\widehat{\alpha}$", "p": r"$\widehat{\phi}$", "o": r"$\widehat{\omega}$", "f": r"$\widehat{\tau}$"}
nomeLatexX = {"a": r"$\widehat{\alpha}[\alpha]$", "p": r"$\widehat{\phi}[\phi]$", "o": r"$\widehat{\omega}[\omega]$", "f": r"$\widehat{\tau}[\tau]$"}
nomeLatexXdown = {"a": r"$\widehat{\alpha}[\omega]$", "p": r"$\widehat{\phi}[\tau]$", "o": r"$\widehat{\omega}[\alpha]$", "f": r"$\widehat{\tau}[\phi]$"}


def get_noise_curves(case = "s4"):

    ell = np.arange(0, 5001)
    ACB = 7
    ns = 1.
    cls_alpha = 10**(-ACB)*2*np.pi/(ell*(ell+1))**(ns)
    cls_alpha[0] = 0

    data_dir = f"../data/plots/noise_curves/noise_biases_{case}/"

    nggs = np.loadtxt(f"{data_dir}ngg_a_QE.txt")[:5001]
    ngg_10 = np.loadtxt(f"{data_dir}ngg_a_itr_10.txt")[:5001]

    n1_ap_10 = np.loadtxt(f"{data_dir}/n1_ap_itr_10.txt")

    n1_ap_QE = np.loadtxt(f"{data_dir}/n1_ap_QE.txt")

    clas = {}
    if case == "spt":
        elb, resb, stds = np.loadtxt(f"{data_dir}res_rand_spt_3g.txt").T
        clas["resb"] = [elb, resb, stds]
        n1_ap_1 = np.loadtxt(f"{data_dir}/n1_ap_itr_1.txt")
        ngg_1 = np.loadtxt(f"{data_dir}/ngg_a_itr_1.txt")[:5001]
        clas["n1ap1"] = n1_ap_1
        clas["ngg1"] = ngg_1
        n1_aa_10 = 0
        n1_aa_QE = 0
    else:
        n1_aa_10 = np.loadtxt(f"{data_dir}/n1_aa_itr_10.txt")
        n1_aa_QE = np.loadtxt(f"{data_dir}/n1_aa_QE.txt")

    if case == "s4":
        claa = np.loadtxt(f"{data_dir}/claa.txt")
        claa_len = np.loadtxt(f"{data_dir}/claa_len.txt")

        claa_rec = np.loadtxt(f"{data_dir}/claa_rec.txt")
        claa_len_rec = np.loadtxt(f"{data_dir}/claa_len_rec.txt")
        clas = {"claa": claa, "claa_rec": claa_rec, "claa_len": claa_len, "claa_len_rec": claa_len_rec}

    return {"claath": cls_alpha, "ngg0": nggs, "nggmax": ngg_10, "n1ap0": n1_ap_QE, "n1apmax": n1_ap_10, "n1aa0": n1_aa_QE, "n1aamax": n1_aa_10, **clas}

def bin_theory(cl, bin_edges):
    l = np.arange(len(cl))
    sums = scipy.stats.binned_statistic(l, l, statistic="sum", bins=bin_edges)
    cl = scipy.stats.binned_statistic(l, l*cl, statistic="sum", bins=bin_edges)
    cl = cl[0] / sums[0]
    el = (bin_edges[1:] + bin_edges[:-1])/2
    return el, cl


bin_edges = np.concatenate([np.arange(1, 100, 1), np.arange(100, 800, 100), np.arange(800, 4000, 600)])
def decorator(x, edges = bin_edges):
    return bin_theory(x, edges)


def cross_corr_coeff(a, b, base=0, color=None, plot=True, ax=None, label=None):
    ls = np.arange(len(hp.alm2cl(a)))
    decorator = lambda x: bin_theory(x, bin_edges)
    el, x = decorator(hp.alm2cl(a, b))
    el, aa = decorator(hp.alm2cl(a, a))
    el, bb = decorator(hp.alm2cl(b, b))
    xcorr = np.sqrt(x**2 / (aa * bb))
    if plot and ax is not None:
        ax.plot(el, xcorr - base, color=color, label=label)
    return el, xcorr


def cross_corr_coeff_from_cl(a, b, x, base=0, color=None, plot=True, ax=None, label=None, bin = False):
    ls = np.arange(len(a))
    decorator = lambda x: bin_theory(x, bin_edges) if bin else (ls, x)
    el, x = decorator(x)
    el, aa = decorator(a)
    el, bb = decorator(b)
    xcorr = x / np.sqrt(aa * bb)
    #xcorr = np.sqrt(x**2 / (aa * bb))
    if plot and ax is not None:
        ax.plot(el, xcorr - base, color=color, label=label)
    return el, xcorr

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    

def load_inputs(config, selected, scratch, cmbversion, lmax_rec, simidx = 0):
    #input_names = {"p": f"sim_{simidx:04}_plm", "o": f"sim_{simidx:04}_olm", "f": f"sim_{simidx:04}_tau_lm", "a": f"sim_{simidx:04}_alpha_lm"}
    input_names = {"p": f"sim_{simidx:04}_plm", "o": f"sim_{simidx:04}_alpha_lm", "f": f"sim_{simidx:04}_tau_lm", "a": f"sim_{simidx:04}_alpha_lm"}
    inputs = {}
    print("Inputs in", f"{scratch}/{cmbversion}")
    for k in selected:
        input = hp.read_alm(f"{scratch}/{cmbversion}/simswalpha/{input_names[k]}.fits")
        inputs[k] = utils.alm_copy(input, lmax=lmax_rec)
    return inputs


# Function to process a configuration and prepare data for plotting
def process_config(config_path, itrs, subset_selected, simidx = 0):
    config = load_config(config_path)
    scratch = os.getenv("SCRATCH") + "/JOINTRECONSTRUCTION/"
    cmbversion = config["cmb_version"]
    version = config["v"]
    imin = config["imin"]
    simidx = imin
    qe_key = config["k"]
    its_folder = f"{scratch}/{cmbversion}_version_{version}_recs/{qe_key}_sim{simidx:04}{version}/"
    print("Reading from", its_folder)
    recs = statics.rec()
    plms = recs.load_plms(its_folder, itrs=itrs)

    Nselected = len(config["selected"])
    lmax_rec = hp.Alm.getlmax(np.split(plms[0], Nselected)[0].shape[0])
    selected = list(map(lambda s: s[0] if len(s) == 2 else s, config["selected"]))
    subset_selected = [k for k in subset_selected if k in selected]
    inputs = load_inputs(config, selected, scratch, cmbversion, lmax_rec, simidx)
    title = config.get("title", f"Config: {os.path.basename(config_path)}")
    return selected, subset_selected, plms, inputs, lmax_rec, title

def process_config_npy(config_path, directory = directory_files):

    config = load_config(config_path)
    scratch = directory
    cmbversion = config["cmb_version"]
    version = config["v"]
    imin, imax = config["imin"], config["imax"]
    itmax = config["itmax"]
    qe_key = config["k"]
    name = config["name"]
    
    its_file = f"{scratch}/total_qe_it_{qe_key}_{version}_{cmbversion}_{imin}_{imax}_{itmax}.npy"
    autoits = np.load(its_file)

    its_file = f"{scratch}/total_qe_it_cross_{qe_key}_{version}_{cmbversion}_{imin}_{imax}_{itmax}.npy"
    crossits = np.load(its_file)

    its_file = f"{scratch}/total_qe_it_cross_lensed_{qe_key}_{version}_{cmbversion}_{imin}_{imax}_{itmax}.npy"
    print(its_file)
    if os.path.exists(its_file):
        crossits_lensed = np.load(its_file)
    else:
        crossits_lensed = 0

    its_file = f"{scratch}/total_qe_it_cross_down_{qe_key}_{version}_{cmbversion}_{imin}_{imax}_{itmax}.npy"
    if os.path.exists(its_file):
        crossits_down = np.load(its_file)
    else:
        crossits_down = 0

    its_file = f"{scratch}/input_{version}_{cmbversion}_{imin}_{imax}_{itmax}.npy"
    auto_input_its = np.load(its_file)

    its_file = f"{scratch}/input_down_{version}_{cmbversion}_{imin}_{imax}_{itmax}.npy"
    if os.path.exists(its_file):
        auto_input_down_its = np.load(its_file)
    else:
        auto_input_down_its = 0


    selected = list(map(lambda s: s[0] if len(s) == 2 else s, config["selected"]))
    
    return selected, autoits, crossits, auto_input_its, crossits_down, auto_input_down_its, crossits_lensed, itmax, name

