## Running code

`python param_joint.py -c spt3g.yaml`

There is also QE bias-hardening in a different parameter file

`python param_joint_bh.py -c spt3g.yaml`

## Making plots

`python plot_noise.py --config spt3g`
`python plot_wiener.py --config spt3g`
`python plot_xcorr.py --config spt3g`
