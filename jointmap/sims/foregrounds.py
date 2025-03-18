import numpy as np
import pathlib
import healpy as hp
from os.path import join as opj
from iterativefg.cmbsky import foregrounds_utils as fgu
from plancklens import utils



class Extra(object):

    def __init__(self, name, extra_tlm: dict[str, np.ndarray]):
        self.name = name
        self.extra_tlm = extra_tlm

    def get_extra_tlm(self, idx, field = ""):
        return self.extra_tlm[field](idx)
    
    def get_name(self):
        return self.name
    
    def __call__(self, idx, field = ""):
        return self.get_extra_tlm(idx, field)
    


def get_caso(c):
    if c == "sum":
        return "total"
    elif c == "radio":
        return "radiops"
    else:
        return c
        
def get_foregrounds_getter(config, case = "", depr_cases = [], method = "ilc", extra = "_masked", randomize = False, include_phi = False, gaussian_phi = False):
    """
    Get the foregrounds getter for the given case.
    If include_phi is True, the phi field is added to the foregrounds.

    Parameters
    ----------
    config : dict
        The config dictionary.
    case : str
        The case to get the foregrounds for.
    depr_cases : list
        The deprecated cases to include. e.g. ["tsz"] , ["tsz", "cib"]
    method : str
        The method to use to get the foregrounds. e.g. "ilc"
    extra : str
        The extra to use to get the foregrounds. e.g. "_masked"
    randomize : bool
        Whether to randomize the foregrounds.
    include_phi : bool
        Whether to include the phi field in the foregrounds.
    """

    fg_power = 0
    noise_power = 0
    
    if (include_phi) or (case != ""):

        fgs = {}

        if include_phi:
            """kappa_lm = hp.read_alm(opj("/capstor/scratch/cscs/odarwish/agora/cmbkappa/", f"kappa_raytraced_2048_alm.fits"))
            lmax = hp.Alm.getlmax(len(kappa_lm))
            ls = np.arange(lmax+1)
            factor_to_phi = (ls*(ls+1))/2
            factor_to_phi = factor_to_phi**-1.
            factor_to_phi[0] = 0.
            phi_lm = hp.almxfl(kappa_lm, factor_to_phi)"""

            maindir = "/capstor/scratch/cscs/odarwish/agora/cmb/phi/"
            fgs["plm"] = lambda idx: hp.read_alm(opj(maindir, f"agora_phiG_phi1_seed{idx}.alm")) if gaussian_phi else hp.read_alm(opj(maindir, f"agora_phiNG_phi1_seed1.alm"))

        if case != "":
            outpath = pathlib.Path(config["path"])
            ilcpath = config["foregrounds"]["ilcpath"]
            name = config["name"]
            outdir = outpath/name
            ilcpath = outdir/ilcpath
            #joint depr_cases with _ underscore
            depr_cases = "_".join(depr_cases)
            dodepr = "_depr_" if len(depr_cases) > 0 else ""
            nome = opj(ilcpath, f"{case}{extra}{method}{dodepr}{depr_cases}_alm.fits")
            fgs[""] = lambda idx: fgu.randomizing_fg(hp.read_alm(nome)) if randomize else hp.read_alm(nome)


            processed_noise = config["foregrounds"]["processed_noise"]
            processed_noise = np.load(processed_noise, allow_pickle=True).item()
            ilcdict = processed_noise[f"ilc{dodepr}{depr_cases}"]
            fg_power = ilcdict["power_fg_only"][get_caso(case)]
            noise_power = ilcdict["noise_total"]

        
        extra_tlm = Extra(case, fgs)


    else:
        extra_tlm = None

    return extra_tlm, fg_power, noise_power