#!/bin/bash

# Ensure the script stops on errors
set -e



python plot_xcorr.py --configs config_alpha_alpha config_full_alpha --subset_selected a --itrs 0 1 5 --outname xcorr_alpha_alpha.pdf
python plot_xcorr.py --configs config_full_alpha_lensing config_full_alpha_disabled_lensing --subset_selected a p o --itrs 0 1 6 15 --outname xcorr_alpha_lensing.pdf