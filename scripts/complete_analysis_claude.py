from mpi4py import MPI
import healpy as hp
import numpy as np
from plancklens import utils
import yaml
import os
from os.path import join as opj
from iterativefg import utils as itu
from delensalot.core.iterator import statics
import argparse
from types import SimpleNamespace
from typing import Dict, List, Tuple, Optional
import lenspyx
from plancklens import shts
from dataclasses import dataclass

@dataclass
class SimulationResult:
    """Container for all correlation results from a simulation"""
    auto_spectra: np.ndarray
    cross_spectra: List[np.ndarray]  # List of cross spectra for each field rotation
    lensed_spectra: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, np.ndarray]:
        """Convert results to dictionary format for saving"""
        result = {
            'auto': self.auto_spectra,
            'cross': self.cross_spectra[0],  # Original cross correlation
            'cross_down': self.cross_spectra[1],
            'cross_down_2': self.cross_spectra[2],
            'cross_down_3': self.cross_spectra[3]
        }
        if self.lensed_spectra is not None:
            result['cross_lensed'] = self.lensed_spectra
        return result

class FieldMapping:
    """Handles field mappings while preserving original rotation logic"""
    def __init__(self):
        # Original mappings preserved
        self.nome = {
            "p": "plm",
            "f": "tau_lm",
            "a": "alpha_lm",
            "o": "olm"
        }
        
        # Keep exact original rotations
        self.rotations = [
            {  # Original mapping
                "p": "plm",
                "f": "tau_lm",
                "a": "alpha_lm",
                "o": "olm"
            },
            {  # First rotation (nome_down)
                "p": "tau_lm",
                "f": "plm",
                "a": "olm",
                "o": "alpha_lm"
            },
            {  # Second rotation (nome_down_2)
                "o": "tau_lm",
                "a": "plm",
                "f": "olm",
                "p": "alpha_lm"
            },
            {  # Third rotation (nome_down_3)
                "a": "tau_lm",
                "o": "plm",
                "p": "olm",
                "f": "alpha_lm"
            }
        ]
    
    def get_mapping(self, rotation_idx: int) -> Dict[str, str]:
        """Get specific field mapping by rotation index"""
        return self.rotations[rotation_idx]
    
    def get_field_name(self, field: str, rotation_idx: int = 0) -> str:
        """Get field name for specific rotation"""
        return self.rotations[rotation_idx][field]

class ReconstructionProcessor:
    def __init__(self, args: SimpleNamespace, comm: MPI.Comm):
        self.args = args
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.mapper = FieldMapping()
        self.size_mappa = hp.Alm.getsize(args.lmax_qlm)
        
        # Create output directory if needed
        if self.rank == 0:
            os.makedirs(args.saving_directory, exist_ok=True)

    def read_alm(self, path: str) -> np.ndarray:
        """Read alm from file with proper error handling"""
        if not os.path.exists(path):
            return np.zeros(self.size_mappa, dtype=complex)
        return utils.alm_copy(hp.read_alm(path), lmax=self.args.lmax_qlm)

    def get_lensed_alpha(self, alm0_input: np.ndarray, plm0_input: np.ndarray) -> np.ndarray:
        """Compute lensed alpha field"""
        dlm, dclm, lmax_dlm, mmax_dlm = self._get_dlm(plm0_input)
        lmax_map = hp.Alm.getlmax(alm0_input.size)
        nside_lens = 2048
        
        a0_len = lenspyx.alm2lenmap(
            alm0_input, [dlm, None],
            geometry=('healpix', {'nside': nside_lens}),
            epsilon=1e-8, verbose=0
        )
        alm0_len = shts.map2alm(a0_len, lmax=lmax_map)
        return utils.alm_copy(alm0_len, lmax=self.args.lmax_qlm)

    def _get_dlm(self, dlm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
        """Process displacement field"""
        dclm = np.zeros_like(dlm)
        lmax_dlm = hp.Alm.getlmax(dlm.size, -1)
        mmax_dlm = lmax_dlm
        p2d = np.sqrt(np.arange(lmax_dlm + 1) * np.arange(1, lmax_dlm + 2))
        hp.almxfl(dlm, p2d, mmax_dlm, inplace=True)
        hp.almxfl(dclm, p2d, mmax_dlm, inplace=True)
        return dlm, dclm, lmax_dlm, mmax_dlm

    def get_input_fields(self, idx: int) -> Dict[str, np.ndarray]:
        """Get all input fields for a simulation"""
        if self.args.unlensed:
            return {s: np.zeros(self.size_mappa, dtype=complex) 
                   for s in self.args.selected}
        
        base_path = opj(self.args.directory, self.args.cmb_version, 
                       "simswalpha", f"sim_{idx:04d}")
        
        fields = {}
        for field in self.args.selected:
            for rotation in self.mapper.rotations:
                name = rotation[field]
                if name not in fields:
                    path = f"{base_path}_{name}.fits"
                    fields[name] = self.read_alm(path)
        
        return fields

    def process_simulation(self, idx: int) -> SimulationResult:
        """Process a single simulation"""
        # Get input fields
        fields = self.get_input_fields(idx)
        
        # Get reconstruction path
        rec_path = opj(
            self.args.directory,
            f"{self.args.cmb_version}_version_{self.args.v}_recs",
            f"{self.args.qe_key}_sim{idx:04d}{self.args.v}"
        )
        
        # Load reconstructions
        recs = [np.load(opj(rec_path, f"{s}lm0_norm.npy")) 
                for s in self.args.selected]
        recs_combined = np.concatenate(recs)
        recs_split = np.split(recs_combined, len(self.args.selected))
        
        # Compute auto spectra
        auto_spectra = np.concatenate([hp.alm2cl(rec) for rec in recs_split])
        
        # Compute cross spectra for each rotation
        cross_spectra = []
        for rotation in self.mapper.rotations:
            cross = np.concatenate([
                hp.alm2cl(rec, fields[rotation[field]]) 
                for rec, field in zip(recs_split, self.args.selected)
            ])
            cross_spectra.append(cross)
        
        # Compute lensed spectra if needed
        lensed_spectra = None
        if not self.args.unlensed:
            lensed_fields = []
            for field in self.args.selected:
                if field == "a":
                    lensed = self.get_lensed_alpha(
                        fields[self.mapper.nome["a"]], 
                        fields[self.mapper.nome["p"]]
                    )
                else:
                    lensed = np.zeros(self.size_mappa, dtype=np.complex128)
                lensed_fields.append(lensed)
            
            lensed_spectra = np.concatenate([
                hp.alm2cl(rec, lensed)
                for rec, lensed in zip(recs_split, lensed_fields)
            ])
        
        return SimulationResult(auto_spectra, cross_spectra, lensed_spectra)

    def process_all_simulations(self) -> Dict[str, np.ndarray]:
        """Process all simulations and combine results"""
        # Split simulations across processes
        all_sims = np.arange(self.args.imin, self.args.imax)
        local_sims = np.array_split(all_sims, self.size)[self.rank]
        
        # Process local simulations
        local_results = []
        for idx in local_sims:
            if self.rank == 0:
                print(f"Processing simulation {idx}")
            result = self.process_simulation(idx)
            local_results.append(result)
        
        # Gather results from all processes
        all_results = self.comm.gather(local_results, root=0)
        
        if self.rank == 0:
            # Combine results
            combined_results = []
            for results in all_results:
                combined_results.extend(results)
            
            # Convert to output format
            output = {}
            for key in combined_results[0].to_dict().keys():
                output[key] = np.stack([r.to_dict()[key] for r in combined_results])
            
            return output
        return None

    def save_results(self, results: Dict[str, np.ndarray]):
        """Save results to disk"""
        if self.rank == 0:
            base_path = opj(
                self.args.saving_directory,
                f"{{}}_{self.args.qe_key}_{self.args.v}_"
                f"{self.args.cmb_version}_{self.args.imin}_"
                f"{self.args.imax}_{self.args.itmax}"
            )
            
            for key, data in results.items():
                np.save(base_path.format(key), data)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Analyse reconstruction output.')
    # Add your argument definitions here
    args = parser.parse_args()
    
    # Initialize processor
    comm = MPI.COMM_WORLD
    processor = ReconstructionProcessor(args, comm)
    
    # Process simulations
    results = processor.process_all_simulations()
    
    # Save results
    processor.save_results(results)

if __name__ == "__main__":
    main()
