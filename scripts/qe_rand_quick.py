import healpy as hp
import numpy as np


maindir = "/home/users/d/darwish/scratch/JOINTRECONSTRUCTION/apo_new_v2_nonzero_version_apo_new_v2_nonzero_official_check_factor_2_vofficial_recs/"

ncomps = 3
Ns = 128

estindex = 0

all = []
for index in range(Ns):
    print(index)
    dir = f"{maindir}p_p_sim{index:04}apo_new_v2_nonzero_official_check_factor_2_vofficial"
    plm = np.split(np.load(f"{dir}/phi_plm_it000_1000_1000.npy"), ncomps)[estindex]
    cl = hp.alm2cl(plm)
    all.append(cl)

all = np.array(all)
np.save("qe_rand.npy", all)
