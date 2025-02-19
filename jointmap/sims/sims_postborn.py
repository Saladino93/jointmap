"""Simulation module including GF non-linear kappa maps, although you can include also any kappa map, including gaussian ones.

You just have to give the path for the map.



# From GF:
# map0_kappa_ecp262_dmn2_lmax8000_first.fits: Born approximation + Non linear effects
# map0_kappa_ecp262_dmn2_lmax8000.fits : full post-Born+ Non linear effect
        
"""

from plancklens import utils
from jointmap.sims import sims_cmbs

import os
#from delensalot.utility import utils_hp
import healpy as hp
import numpy as np


class Alm:
    """alm arrays useful statics. Directly from healpy but excluding keywords


    """
    @staticmethod
    def getsize(lmax:int, mmax:int):
        """Number of entries in alm array with lmax and mmax parameters

        Parameters
        ----------
        lmax : int
          The maximum multipole l, defines the alm layout
        mmax : int
          The maximum quantum number m, defines the alm layout

        Returns
        -------
        nalm : int
            The size of a alm array with these lmax, mmax parameters

        """
        return ((mmax+1) * (mmax+2)) // 2 + (mmax+1) * (lmax-mmax)

    @staticmethod
    def getidx(lmax:int, l:int or np.ndarray, m:int or np.ndarray):
        """Returns index corresponding to (l,m) in an array describing alm up to lmax.

        In HEALPix C++ and healpy, :math:`a_{lm}` coefficients are stored ordered by
        :math:`m`. I.e. if :math:`\ell_{max}` is 16, the first 16 elements are
        :math:`m=0, \ell=0-16`, then the following 15 elements are :math:`m=1, \ell=1-16`,
        then :math:`m=2, \ell=2-16` and so on until the last element, the 153th, is
        :math:`m=16, \ell=16`.

        Parameters
        ----------
        lmax : int
          The maximum l, defines the alm layout
        l : int
          The l for which to get the index
        m : int
          The m for which to get the index

        Returns
        -------
        idx : int
          The index corresponding to (l,m)
        """
        return m * (2 * lmax + 1 - m) // 2 + l

    @staticmethod
    def getlmax(s:int, mmax:int or None):
        """Returns the lmax corresponding to a given healpy array size.

        Parameters
        ----------
        s : int
          Size of the array
        mmax : int
          The maximum m, defines the alm layout

        Returns
        -------
        lmax : int
          The maximum l of the array, or -1 if it is not a valid size.
        """
        if mmax is not None and mmax >= 0:
            x = (2 * s + mmax ** 2 - mmax - 2) / (2 * mmax + 2)
        else:
            x = (-3 + np.sqrt(1 + 8 * s)) / 2
        if x != np.floor(x):
            return -1
        else:
            return int(x)

def almxfl(alm:np.ndarray, fl:np.ndarray, mmax:int or None, inplace:bool):
    """Multiply alm by a function of l.

    Parameters
    ----------
    alm : array
      The alm to multiply
    fl : array
      The function (at l=0..fl.size-1) by which alm must be multiplied.
    mmax : None or int
      The maximum m defining the alm layout. Default: lmax.
    inplace : bool
      If True, modify the given alm, otherwise make a copy before multiplying.

    Returns
    -------
    alm : array
      The modified alm, either a new array or a reference to input alm,
      if inplace is True.

    """
    lmax = Alm.getlmax(alm.size, mmax)
    if mmax is None or mmax < 0:
        mmax = lmax
    assert fl.size > lmax, (fl.size, lmax)
    if inplace:
        for m in range(mmax + 1):
            b = m * (2 * lmax + 1 - m) // 2 + m
            alm[b:b + lmax - m + 1] *= fl[m:lmax+1]
        return
    else:
        ret = np.copy(alm)
        for m in range(mmax + 1):
            b = m * (2 * lmax + 1 - m) // 2 + m
            ret[b:b + lmax - m + 1] *= fl[m:lmax+1]
        return ret

class sims_postborn(sims_cmbs.sims_cmb_len):
    """Simulations of CMBs each having the same GF postborn + non linear effect deflection kappa
        Args:
            lib_dir: the phases of the CMB maps and the lensed CMBs will be stored there
            lmax_cmb: cmb maps are generated down to this max multipole
            cls_unl: dictionary of unlensed CMB spectra
            dlmax, nside_lens, facres, nbands: lenspyx lensing module parameters
            wcurl: include field rotation map in the lensing deflection (default to False for historical reasons)


        This just redefines the sims_cmbs.sims_cmb_len method to feed the nonlinear kmap
    """
    def __init__(self, lib_dir, lmax_cmb, cls_unl:dict, wcurl = False,
                 dlmax=1024, lmin_dlm = 2, nside_lens=4096, facres=0, nbands=8, cache_plm=True, lib_pha = None, extra_tlm = None, epsilon = 1e-7,
                 generate_phi = False, nocmb = False, zerolensing = False):

        lmax_plm = lmax_cmb + dlmax
        mmax_plm = lmax_plm

        self.nocmb = nocmb
        self.zerolensing = zerolensing

        self.generate_phi = generate_phi

        cmb_cls = {}
        for k in cls_unl.keys():
            if generate_phi:
              cmb_cls[k] = np.copy(cls_unl[k][:lmax_cmb + dlmax + 1])
            else:
              if ('p' not in k) and ('o' not in k):
                cmb_cls[k] = np.copy(cls_unl[k][:lmax_cmb + dlmax + 1])

        self.wcurl = wcurl
        self.lmax_plm = lmax_plm
        self.mmax_plm = mmax_plm
        self.cache_plm = cache_plm
        super(sims_postborn, self).__init__(lib_dir,  lmax_cmb, cmb_cls,
                                            dlmax=dlmax, nside_lens=nside_lens, facres=facres, nbands=nbands, lmin_dlm = lmin_dlm, lib_pha = lib_pha, extra_tlm = extra_tlm, epsilon = epsilon, nocmb = nocmb, zerolensing = zerolensing)

    def get_sim_kappa(self, idx: int):
        pass

    def get_sim_omega(self, idx: int):
        pass

    def get_sim_plm(self, idx: int):

        fn = os.path.join(self.lib_dir, f'plm_in_{idx}_lmax{self.lmax_plm}.fits')

        if self.generate_phi:
            if not os.path.exists(fn):
              print("Generating plm", flush = True)
              plm = super(sims_postborn, self).get_sim_plm(idx)
              hp.write_alm(fn, plm)
            return hp.read_alm(fn)

        try:

          if not os.path.exists(fn):
              p2k = 0.5 * np.arange(self.lmax_plm + 1) * np.arange(1, self.lmax_plm + 2, dtype=float)
              #plm = utils_hp.almxfl(hp.map2alm(hp.read_map(self.path, dtype=float), lmax=self.lmax_plm), utils.cli(p2k), self.mmax_plm, False)
              #plm = utils_hp.almxfl(hp.map2alm(self.get_sim_kappa(idx), lmax = self.lmax_plm), utils.cli(p2k), self.mmax_plm, False)
              plm = almxfl(utils.alm_copy(self.get_sim_kappa_alm(idx), lmax = self.lmax_plm), utils.cli(p2k), self.mmax_plm, False)
              if self.cache_plm:
                  hp.write_alm(fn, plm)
              return plm
          print('Reading saved CMB lensing potential sim')
          return hp.read_alm(fn)
        
        except Exception as e:
          print(e)
          print("Generating plm")
          return super(sims_postborn, self).get_sim_plm(idx)

    def get_sim_olm(self, idx):
        fn = os.path.join(self.lib_dir, f'olm_in_{idx}_lmax{self.lmax_plm}.fits')
        
        if (not self.wcurl):
            return np.zeros_like(self.get_sim_plm(idx))

        if not os.path.exists(fn):
            p2k = 0.5 * np.arange(self.lmax_plm + 1) * np.arange(1, self.lmax_plm + 2, dtype=float)
            plm = almxfl(utils.alm_copy(self.get_sim_omega_alm(idx), lmax = self.lmax_plm), utils.cli(p2k), self.mmax_plm, False)
            if self.cache_plm:
                hp.write_alm(fn, plm)
            return plm
        print('Reading saved CMB lensing curl sim')
        return hp.read_alm(fn)
    


class sims_postborn_test(sims_cmbs.sims_cmb_len):
    """Simulations of CMBs each having the same GF postborn + non linear effect deflection kappa
        Args:
            lib_dir: the phases of the CMB maps and the lensed CMBs will be stored there
            lmax_cmb: cmb maps are generated down to this max multipole
            cls_unl: dictionary of unlensed CMB spectra
            dlmax, nside_lens, facres, nbands: lenspyx lensing module parameters
            wcurl: include field rotation map in the lensing deflection (default to False for historical reasons)


        This just redefines the sims_cmbs.sims_cmb_len method to feed the nonlinear kmap
    """
    def __init__(self, lib_dir, lmax_cmb, cls_unl:dict, wcurl = False,
                 dlmax=1024, lmin_dlm = 2, nside_lens=4096, facres=0, nbands=8, cache_plm=True, lib_pha = None, extra_tlm = None, epsilon = 1e-7):

        lmax_plm = lmax_cmb + dlmax
        mmax_plm = lmax_plm

        cmb_cls = {}
        for k in cls_unl.keys():
            cmb_cls[k] = np.copy(cls_unl[k][:lmax_cmb + dlmax + 1])

        self.wcurl = wcurl
        self.lmax_plm = lmax_plm
        self.mmax_plm = mmax_plm
        self.cache_plm = cache_plm
        super(sims_postborn_test, self).__init__(lib_dir,  lmax_cmb, cmb_cls,
                                            dlmax=dlmax, nside_lens=nside_lens, facres=facres, nbands=nbands, lmin_dlm = lmin_dlm, lib_pha = lib_pha, extra_tlm = extra_tlm, epsilon = epsilon)

    def get_sim_kappa(self, idx: int):
        pass

    def get_sim_omega(self, idx: int):
        pass


class SehgalSim(sims_postborn):

    kappakey = 'kappa'

    def __init__(self, sims: dict, **kwargs):
        super().__init__(**kwargs)
        self.sims = sims
    
    def get_sim_kappa(self, idx, verbose: bool = True):
        if verbose:
            print('Getting special kappa!')
        nome = self.sims[self.kappakey](idx)
        return hp.read_map(nome)
    
    def get_sim_kappa_alm(self, idx, verbose: bool = True):
        if verbose:
            print('Getting special kappa alm!')
        nome = self.sims[self.kappakey](idx)
        return hp.read_alm(nome)