## Running code

`python param_joint.py -c configs/spt3g.yaml`

There is also QE bias-hardening in a different parameter file

`python param_joint_bh.py -c configs/spt3g.yaml`

## Making plots

Before making plots, please edit the `utils_data.py` file. 

`python plot_noise.py --config spt3g`

`python plot_wiener.py --config spt3g`

`python plot_xcorr.py --config spt3g`

## To add

* Choosing your own stepper from the configuration file
* T and MV support