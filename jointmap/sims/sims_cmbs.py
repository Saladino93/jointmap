"""Generic cmb-only sims module
"""
import numpy as np, healpy as hp
import os
from plancklens.helpers import mpi
from plancklens.sims import phas, maps
from plancklens import utils
import pickle as pk
from lenspyx.remapping import deflection
from lenspyx.remapping.utils_geom import Geom 
from lenspyx import cachers
from jointmap.sims import cmbs

verbose = False

def _get_fields(cls):
    if verbose: print(cls.keys())
    fields = ['p', 't', 'e', 'b', 'o']
    ret = ['p', 't', 'e', 'b', 'o']
    for _f in fields:
        if not ((_f + _f) in cls.keys()): ret.remove(_f)
    for _k in cls.keys():
        for _f in _k:
            if _f not in ret: ret.append(_f)
    return ret

class sims_cmb_unlensed(object):
    """Unlensed CMB skies simulation library.

        Note:
            These sims do not contain aberration or modulation

        Args:
            lib_dir: lensed cmb alms will be cached there
            lmax: lensed cmbs are produced up to lmax
            cls_unl(dict): unlensed cmbs power spectra
            lib_pha(optional): random phases library for the unlensed maps (see *plancklens.sims.phas*)
            offsets_plm: offset lensing plm simulation index (useful e.g. for MCN1), tuple with block_size and offsets
            offsets_cmbunl: offset unlensed cmb (useful e.g. for MCN1), tuple with block_size and offsets
            dlmax(defaults to 1024): unlensed cmbs are produced up to lmax + dlmax, for accurate lensing at lmax
            nside_lens(defaults to 4096): healpy resolution at which the lensed maps are produced
            facres(defaults to 0): sets the interpolation resolution in lenspyx
            nbands(defaults to 16): number of band-splits in *lenspyx.alm2lenmap(_spin)*
            verbose(defaults to True): lenspyx timing info printout

    """
    def __init__(self, lib_dir, lmax, cls_unl, lib_pha=None, offsets_plm=None, offsets_cmbunl=None,
                 dlmax=1024, nside_lens=4096, facres=0, nbands=8, verbose=True):
        if not os.path.exists(lib_dir) and mpi.rank == 0:
            os.makedirs(lib_dir)
        mpi.barrier()
        fields = _get_fields(cls_unl)

        if lib_pha is None and mpi.rank == 0:
            lib_pha = phas.lib_phas(os.path.join(lib_dir, 'phas'), len(fields), lmax + dlmax)
        elif lib_pha is not None:
            print('Using specified lib_pha!')
        #else:  # Check that the lib_alms are compatible :
        #    assert lib_pha.lmax == lmax + dlmax
        mpi.barrier()


        self.lmax = lmax
        self.dlmax = dlmax
        self.lmax_unl = lmax + dlmax
        # lenspyx parameters:
        self.nside_lens = nside_lens
        self.nbands = nbands
        self.facres = facres

        self.unlcmbs = cmbs.sims_cmb_unl(cls_unl, lib_pha)
        self.lib_dir = lib_dir
        self.fields = _get_fields(cls_unl)

        self.offset_plm = offsets_plm if offsets_plm is not None else (1, 0)
        self.offset_cmb = offsets_cmbunl if offsets_cmbunl is not None else (1, 0)

        fn_hash = os.path.join(lib_dir, 'sim_hash.pk')
        if mpi.rank == 0 and not os.path.exists(fn_hash) :
            pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
        mpi.barrier()
        utils.hash_check(self.hashdict(), pk.load(open(fn_hash, 'rb')))
        
        self.verbose=verbose

    @staticmethod
    def offset_index(idx, block_size, offset):
        """Offset index by amount 'offset' cyclically within blocks of size block_size

        """
        return (idx // block_size) * block_size + (idx % block_size + offset) % block_size

    def hashdict(self):
        return {'unl_cmbs': self.unlcmbs.hashdict(),'lmax':self.lmax,
                'nside_lens':self.nside_lens}

    def _is_full(self):
        return self.unlcmbs.lib_pha.is_full()

    def get_sim_alm(self, idx, field):
        if field == 't':
            return self.get_sim_tlm(idx)
        elif field == 'e':
            return self.get_sim_elm(idx)
        elif field == 'b':
            return self.get_sim_blm(idx)
        else :
            assert 0,(field,self.fields)

    def get_sim_tlm(self, idx):
        fname = os.path.join(self.lib_dir, 'unl_sim_%04d_tlm.fits' % idx)
        if not os.path.exists(fname):
            tlm= self.unlcmbs.get_sim_tlm(self.offset_index(idx, self.offset_cmb[0], self.offset_cmb[1]))
            hp.write_alm(fname, hp.map2alm(tlm, lmax=self.lmax, iter=0))
        return hp.read_alm(fname)

    def get_sim_elm(self, idx):
        fname = os.path.join(self.lib_dir, 'sim_%04d_elm.fits' % idx)
        if not os.path.exists(fname):
            self._cache_eblm(idx)
        return hp.read_alm(fname)

    def get_sim_blm(self, idx):
        fname = os.path.join(self.lib_dir, 'sim_%04d_blm.fits' % idx)
        if not os.path.exists(fname):
            self._cache_eblm(idx)
        return hp.read_alm(fname)


class sims_cmb_len(object):
    """Lensed CMB skies simulation library.
        Note:
            To produce the lensed CMB, the package lenspyx is mandatory
        Note:
            These sims do not contain aberration or modulation
        Args:
            lib_dir: lensed cmb alms will be cached there
            lmax: lensed cmbs are produced up to lmax
            cls_unl(dict): unlensed cmbs power spectra
            lib_pha(optional): random phases library for the unlensed maps (see *plancklens.sims.phas*)
            offsets_plm: offset lensing plm simulation index (useful e.g. for MCN1), tuple with block_size and offsets
            offsets_cmbunl: offset unlensed cmb (useful e.g. for MCN1), tuple with block_size and offsets
            dlmax(defaults to 1024): unlensed cmbs are produced up to lmax + dlmax, for accurate lensing at lmax
            nside_lens(defaults to 4096): healpy resolution at which the lensed maps are produced
            facres(defaults to 0): sets the interpolation resolution in lenspyx
            nbands(defaults to 16): number of band-splits in *lenspyx.alm2lenmap(_spin)*
            verbose(defaults to True): lenspyx timing info printout
    """
    def __init__(self, lib_dir, lmax, cls_unl, lib_pha=None, offsets_plm=None, offsets_cmbunl=None,
                 dlmax=1024, nside_lens=4096, facres=0, nbands=8, verbose=True, lmin_dlm = 2, extra_tlm = None, epsilon = 1e-10, 
                 nocmb = False, zerolensing = False, zerobirefringence = True, zerocurl = True, zerotau = True, 
                 cases = ["p"], randomize_function = lambda x, idx: x, get_aniso_index = None):
        if not os.path.exists(lib_dir) and mpi.rank == 0:
            os.makedirs(lib_dir)
        mpi.barrier()
        fields = _get_fields(cls_unl)
        
        if lib_pha is None:
            lib_pha = phas.lib_phas(os.path.join(lib_dir, 'phas'), len(fields), lmax + dlmax)
        else:  # Check that the lib_alms are compatible :
            assert lib_pha.lmax == lmax + dlmax
        mpi.barrier()

        self.lmin_dlm = lmin_dlm

        self.lmax = lmax
        self.dlmax = dlmax
        self.lmax_unl = lmax + dlmax
        self.dlmax_gl = 1024+dlmax
        # lenspyx parameters:
        self.nside_lens = nside_lens
        self.nbands = nbands
        self.facres = facres

        self.nocmb = nocmb
        self.cases = cases

        print("CMB chain", cases)
        
        self.zerolensing = ("p" not in cases)
        self.zerobirefringence = ("a" not in cases)
        self.zerocurl = ("o" not in cases)
        self.zerotau = ("f" not in cases)

        self.randomize_function = randomize_function

        self.get_aniso_index = get_aniso_index

        self.unlcmbs = cmbs.sims_cmb_unl(cls_unl, lib_pha)
        self.lib_dir = lib_dir
        self.fields = _get_fields(cls_unl)

        self.offset_plm = offsets_plm if offsets_plm is not None else (1, 0)
        self.offset_cmb = offsets_cmbunl if offsets_cmbunl is not None else (1, 0)

        self.epsilon = epsilon

        fn_hash = os.path.join(lib_dir, 'sim_hash.pk')
        if mpi.rank == 0 and not os.path.exists(fn_hash) :
            pk.dump(self.hashdict(), open(fn_hash, 'wb'), protocol=2)
        mpi.barrier()
        utils.hash_check(self.hashdict(), pk.load(open(fn_hash, 'rb')))
        try:
            import lenspyx
        except ImportError:
            print("Could not import lenspyx module")
            lenspyx = None
        self.lens_module = lenspyx
        self.verbose=verbose

        #function to get some extra tlm
        self.extra_tlm = extra_tlm

    @staticmethod
    def offset_index(idx, block_size, offset):
        """Offset index by amount 'offset' cyclically within blocks of size block_size
        """
        return (idx // block_size) * block_size + (idx % block_size + offset) % block_size

    def hashdict(self):
        return {'unl_cmbs': self.unlcmbs.hashdict(),'lmax':self.lmax,
                'offset_plm':self.offset_plm, 'offset_cmb':self.offset_cmb,
                'nside_lens':self.nside_lens, 'facres':self.facres, 'lmin_dlm':self.lmin_dlm}

    def _is_full(self):
        return self.unlcmbs.lib_pha.is_full()

    def get_sim_alm(self, idx, field):
        if field == 't':
            return self.get_sim_tlm(idx)
        elif field == 'e':
            return self.get_sim_elm(idx)
        elif field == 'b':
            return self.get_sim_blm(idx)
        elif field == 'p':
            return self.get_sim_plm(idx)
        elif field == 'o':
            return self.get_sim_olm(idx)
        else :
            assert 0,(field,self.fields)

    def get_sim_plm(self, idx):
        index = self.offset_index(idx, self.offset_plm[0], self.offset_plm[1])
        try:
            pfname = os.path.join(self.lib_dir, 'sim_%04d_plm.fits' % index)
            plm = hp.read_alm(pfname)*(1-self.zerolensing)
            print("Getting plm sim from cache", flush = True)
            return plm
        except:
            result = self.unlcmbs.get_sim_plm(index)*(1-self.zerolensing)
            hp.write_alm(pfname, result)
            return result
        
    
    def get_sim_tau_lm(self, idx):
        index = self.offset_index(idx, self.offset_plm[0], self.offset_plm[1])
        pfname = os.path.join(self.lib_dir, 'sim_%04d_tau_lm.fits' % index)
        try:
            return hp.read_alm(pfname)
        except:
            if 'f' in self.fields:
                print("Getting tau sim from unlcmbs")
                result = self.unlcmbs.get_sim_tau_lm(index)*(1-self.zerotau)
                hp.write_alm(pfname, result)
                return result
            else:
                return np.zeros_like(self.get_sim_plm(idx))
        

    def get_sim_alpha_lm(self, idx):
        index = self.offset_index(idx, self.offset_plm[0], self.offset_plm[1])
        pfname = os.path.join(self.lib_dir, 'sim_%04d_alpha_lm.fits' % index)
        if os.path.exists(pfname):
            return hp.read_alm(pfname)
        try:
            return hp.read_alm(pfname)
        except:
            if 'a' in self.fields:
                print("Getting alpha sim from unlcmbs")
                result = self.unlcmbs.get_sim_alpha_lm(index)*(1-self.zerobirefringence)
                hp.write_alm(pfname, result)
                return result
            else:
                return np.zeros_like(self.get_sim_plm(idx))

    def get_sim_olm(self, idx):
        index = self.offset_index(idx, self.offset_plm[0], self.offset_plm[1])
        pfname = os.path.join(self.lib_dir, 'sim_%04d_olm.fits' % index)
        try:
            return hp.read_alm(pfname)
        except:
            if ('o' in self.fields) and (not self.zerocurl):
                print("Getting olm sim from unlcmbs")
                result = self.unlcmbs.get_sim_olm(idx)*(1-self.zerocurl)
                hp.write_alm(pfname, result)
                return result
            else:
                print("No curl mode.")
                return np.zeros_like(self.get_sim_plm(idx))

    def _get_dlm(self, idx):
        dlm = self.get_sim_plm(idx)
        dclm = self.get_sim_olm(idx) # curl mode
        lmax_dlm = hp.Alm.getlmax(dlm.size, -1)
        mmax_dlm = lmax_dlm
        # potentials to deflection
        p2d = np.sqrt(np.arange(lmax_dlm + 1, dtype=float) * np.arange(1, lmax_dlm + 2, dtype=float))
        p2d[:self.lmin_dlm] = 0
        hp.almxfl(dlm, p2d, mmax_dlm, inplace=True)
        hp.almxfl(dclm, p2d, mmax_dlm, inplace=True)
        return dlm, dclm, lmax_dlm, mmax_dlm
    

    @staticmethod
    def rotate_polarization(Q, U, angle):
        c, s = np.cos(2*angle), np.sin(2*angle)
        #c, s = ne.evaluate('cos(2 * angle)'), ne.evaluate('sin(2 * angle)')
        #Qrot = ne.evaluate('Q * c - U * s')
        #Urot = ne.evaluate('Q * s + U * c')

        #Qrot = Q * c + U * s
        #Urot = - Q * s + U * c
        Qrot = Q * c - U * s
        Urot = Q * s + U * c

        #QU = Q + 1j * U
        #QU *= np.exp(1j * 2*angle)
        #QU *= ne.evaluate('exp(1j * 2*angle)')
        #Qrot, Urot = np.real(QU), np.imag(QU)
        return Qrot, Urot
    

    @staticmethod
    def patchy_tau(Q, U, tau):
        exp = np.exp(-tau)
        Q *= exp
        U *= exp
        return Q, U


    def _get_f(self, idx):
        dlm, dclm, lmax_dlm, mmax_dlm = self._get_dlm(idx)
        lenjob_geometry = Geom.get_thingauss_geometry(self.lmax_unl + self.dlmax_gl, 2)
        f = deflection(lenjob_geometry, dlm, mmax_dlm, cacher=cachers.cacher_mem(safe=False), #dclm=dclm,
                       epsilon=self.epsilon)
        return f


    def apply(self, case, elm, blm, idx, get_index = lambda x: x):
        """
        Apply different transformations to the E and B mode polarization fields.
        This function implements a clean pipeline of transformations without conditional flags.
        
        Parameters:
        -----------
        case : str
            The type of transformation to apply:
            - "p" for lensing
            - "a" for birefringence (alpha rotation)
            - "f" for patchy tau effects
        elm : array
            E-mode spherical harmonic coefficients
        blm : array
            B-mode spherical harmonic coefficients
        idx : int
            Index of the simulation
        get_index: function
            Function to get the index of the simulation. It can be a fixed for example.
            
        Returns:
        --------
        elm, blm : tuple
            Transformed E and B mode coefficients
        """
        if case == "a":
            # Birefringence case: rotate polarization
            alpha_lm = self.get_sim_alpha_lm(get_index(idx, case))
            nside_rotation = self.nside_lens
            alpha = hp.alm2map(alpha_lm, nside=nside_rotation)
            lmax_map = hp.Alm.getlmax(elm.size)
            Q, U = hp.alm2map_spin([elm, blm], spin=2, nside=nside_rotation, lmax=lmax_map)
            Q, U = self.rotate_polarization(Q, U, alpha)
            elm, blm = hp.map2alm_spin([Q, U], 2, lmax=lmax_map)
            del Q, U
        elif case == "f":
            # Patchy tau case: apply patchy tau ampl
            tau_lm = self.get_sim_tau_lm(get_index(idx, case))
            tau = hp.alm2map(tau_lm, nside=self.nside_lens)
            lmax_map = hp.Alm.getlmax(elm.size)
            Q, U = hp.alm2map_spin([elm, blm], spin=2, nside=self.nside_lens, lmax=lmax_map)
            Q, U = self.patchy_tau(Q, U, tau)
            elm, blm = hp.map2alm_spin([Q, U], 2, lmax=lmax_map)
            del Q, U
        elif case == "p":
            # Lensing case: apply lensing transformation
            dlm, dclm, _, _ = self._get_dlm(get_index(idx, case))
            lmax_map = hp.Alm.getlmax(elm.size)
            Qlen, Ulen = self.lens_module.alm2lenmap_spin(
                [elm, blm], [dlm, dclm], 2,
                geometry=('healpix', {'nside': self.nside_lens}),
                epsilon=self.epsilon, verbose=0
            )
            elm, blm = hp.map2alm_spin([Qlen, Ulen], 2, lmax=lmax_map)
            del Qlen, Ulen
        elif case == "o":
            return elm, blm #already done with "p", though should put o only case..., perhaps, _get_dlm should return only a field based on cases
                
        return elm, blm
    
    def _cache_eblm(self, idx):
        elm = self.unlcmbs.get_sim_elm(self.offset_index(idx, self.offset_cmb[0], self.offset_cmb[1]))
        blm = None if 'b' not in self.fields else self.unlcmbs.get_sim_blm(self.offset_index(idx, self.offset_cmb[0], self.offset_cmb[1]))

        print("Building CMB", self.cases)
        for case in self.cases:
            print("Applying %s" % case)
            elm, blm = self.apply(case, elm, blm, idx, get_index = self.get_aniso_index)

        elm = utils.alm_copy(elm, self.lmax)
        blm = utils.alm_copy(blm, self.lmax)

        hp.write_alm(os.path.join(self.lib_dir, 'sim_%04d_elm.fits' % idx), elm)
        del elm
        hp.write_alm(os.path.join(self.lib_dir, 'sim_%04d_blm.fits' % idx), blm)

    def get_sim_tlm(self, idx, fixed_index = None):
        
        fname = os.path.join(self.lib_dir, 'sim_%04d_tlm.fits' % idx)

        if not os.path.exists(fname):
            
            tlm = self.unlcmbs.get_sim_tlm(self.offset_index(idx if fixed_index is None else fixed_index, self.offset_cmb[0], self.offset_cmb[1]))

            dlm, dclm, _, _ = self._get_dlm(idx)

            #assert 'o' not in self.fields, 'not implemented'

            #lmaxd = hp.Alm.getlmax(dlm.size)
            #p2d = np.sqrt(np.arange(lmaxd + 1, dtype=float) * np.arange(1, lmaxd + 2))
            #p2d[:self.lmin_dlm] = 0

            #hp.almxfl(dlm, p2d, inplace=True)
            #hp.almxfl(dclm, p2d, inplace=True)

            Tlen = self.lens_module.alm2lenmap(tlm, [dlm, dclm], geometry = ('healpix', {'nside': self.nside_lens}), epsilon = self.epsilon, verbose = 0, pol = False)

            hp.write_alm(fname, hp.map2alm(Tlen, lmax=self.lmax, iter=0))

            #if (not os.path.exists(pfname)):
            #    hp.write_alm(pfname, plm)
            #    hp.write_alm(ofname, olm)


        if (self.extra_tlm is not None):
            extrafname = os.path.join(self.lib_dir, f'sim_{idx:04}_{self.extra_tlm.get_name()}lm.fits')
            if (not os.path.exists(extrafname)):
                extra_tlm = utils.alm_copy(self.extra_tlm(idx), lmax=self.lmax)
                hp.write_alm(extrafname, extra_tlm)

        total = hp.read_alm(fname)*(1-self.nocmb) #if to account or not for CMB contribution
        #print("CMB contribution is", total)

        if self.extra_tlm is not None:
            #print('NOTE: adding extra tlm', flush = True)
            total += hp.read_alm(extrafname)

        return self.randomize_function(total, idx)

    def get_sim_elm(self, idx):
        fname = os.path.join(self.lib_dir, 'sim_%04d_elm.fits' % idx)
        if not os.path.exists(fname):
            self._cache_eblm(idx)
        return self.randomize_function(hp.read_alm(fname), idx)

    def get_sim_blm(self, idx):
        fname = os.path.join(self.lib_dir, 'sim_%04d_blm.fits' % idx)
        if not os.path.exists(fname):
            self._cache_eblm(idx)
        return self.randomize_function(hp.read_alm(fname), idx)



class cmb_maps_nlev_sehgal(maps.cmb_maps_nlev):
    def __init__(self, fixed_noise_index: int = None, zero_noise: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.fixed_noise_index = fixed_noise_index
        self.zero_noise = zero_noise
        
    def get_sim_tnoise(self, idx):
        """Returns noise temperature map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map

        """
        idx = self.fixed_noise_index if self.fixed_noise_index is not None else idx
        print(f'Noise index is {idx}')
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        if self.zero_noise:
            print('Setting noise sim to zero!')
        return (1-self.zero_noise)*self.nlev_t / vamin * self.pix_lib_phas.get_sim(idx, idf = 0)
    
    def get_sim_qnoise(self, idx):
        """Returns noise Q-polarization map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map

        """
        idx = self.fixed_noise_index if self.fixed_noise_index is not None else idx
        print(f'Noise index is {idx}')
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        if self.zero_noise:
            print('Setting pol noise sim to zero!')
        return (1-self.zero_noise)*self.nlev_p / vamin * self.pix_lib_phas.get_sim(idx, idf=1)

    def get_sim_unoise(self, idx):
        """Returns noise U-polarization map for a simulation

            Args:
                idx: simulation index

            Returns:
                healpy map

        """
        idx = self.fixed_noise_index if self.fixed_noise_index is not None else idx
        print(f'Noise index is {idx}')
        vamin = np.sqrt(hp.nside2pixarea(self.nside, degrees=True)) * 60
        if self.zero_noise:
            print('Setting pol noise sim to zero!')
        return (1-self.zero_noise)*self.nlev_p / vamin * self.pix_lib_phas.get_sim(idx, idf=2)
