



def process_simulation(config_path, subset_selected, itrs, simidx):
    """Process a single simulation index."""
    config = load_config(config_path)
    scratch = os.getenv("SCRATCH") + "/JOINTRECONSTRUCTION/"
    cmbversion = config["cmb_version"]
    version = config["v"]
    its_folder = f"{scratch}/{cmbversion}_version_{version}_recs/p_p_sim{simidx:04}/"
    recs = rec()
    plms = recs.load_plms(its_folder, itrs=itrs)

    Nselected = len(config["selected"])
    lmax_rec = hp.Alm.getlmax(np.split(plms[0], Nselected)[0].shape[0])
    selected = list(map(lambda s: s[0] if len(s) == 2 else s, config["selected"]))
    subset_selected = [k for k in subset_selected if k in selected]
    inputs = load_inputs(config, selected, scratch, cmbversion, lmax_rec)
    title = config.get("title", f"Config: {os.path.basename(config_path)}")
    return selected, subset_selected, plms, inputs, lmax_rec, title


def compute_average_cross(config_paths, subset_selected, itrs, imin, imax):
    """Compute the average cross-correlation coefficients across simulation indices."""
    N_configs = len(config_paths)
    rows = len(subset_selected)
    cols = N_configs
    names_fields = {"p": r"$\phi$", "o": r"$\omega$", "f": r"$\tau$", "a": r"$\alpha$"}

    # Initialize results dictionary for averaging
    average_results = {config: {k: None for k in subset_selected} for config in config_paths}

    for config_path in config_paths:
        for k in subset_selected:
            accumulated_cross = None
            accumulated_ell = None
            sim_count = 0

            for simidx in range(imin, imax + 1):  # Iterate through simulation indices
                selected, subset_selected, plms, inputs, lmax_rec, title = process_simulation(
                    config_path, subset_selected, itrs, simidx
                )
                original_idx = selected.index(k)

                for idx, itr in enumerate(itrs):  # Iterate over iterations
                    splits = np.split(plms[idx], len(selected))
                    el, cross = cross_corr_coeff(splits[original_idx], inputs[k], plot=False)

                    if accumulated_cross is None:
                        accumulated_cross = np.zeros_like(cross)
                        accumulated_ell = el

                    accumulated_cross += cross
                    sim_count += 1

            # Compute the average
            average_results[config_path][k] = (accumulated_ell, accumulated_cross / sim_count)

    return average_results


def plot_average_cross(average_results, subset_selected, outname):
    """Plot the average cross-correlation coefficients."""
    N_configs = len(average_results)
    rows = len(subset_selected)
    cols = N_configs
    plot = CMBLensingPlot(rows=rows, cols=cols, figsize=(12, 6), sharex=True, sharey="row", outdir="../plots/")

    for i, k in enumerate(subset_selected):  # Iterate over estimators
        for j, (config_path, results) in enumerate(average_results.items()):  # Iterate over configurations
            el, cross = results[k]
            plot.add_curve(el, cross, label="Average", row=i, col=j, linewidth=2)
            plot.set_labels(xlabel=r"$L$", ylabel=r"$\rho_L$" + f" ({k})", row=i, col=j)

            if i == 0:
                plot.set_title(config_path, row=i, col=j)

    plot.save_plot(outname, dpi=300)
    plot.show_plot()


# Example Usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", type=str, help="Paths to configuration files")
    parser.add_argument("--subset_selected", nargs="+", type=str, help="Fields to plot")
    parser.add_argument("--itrs", nargs="+", type=int, help="Iterations to plot")
    parser.add_argument("--imin", type=int, help="Minimum simulation index")
    parser.add_argument("--imax", type=int, help="Maximum simulation index")
    parser.add_argument("--outname", type=str, help="Output file name", default="xcorr_average_cmb_lensing.pdf")

    args = parser.parse_args()
    config_paths = args.configs
    config_paths = [f"configs/{config}.yaml" for config in config_paths]
    subset_selected = args.subset_selected
    itrs = args.itrs
    imin = args.imin
    imax = args.imax
    outname = args.outname

    # Compute average cross-correlation coefficients
    average_results = compute_average_cross(config_paths, subset_selected, itrs, imin, imax)

    # Plot the results
    plot_average_cross(average_results, subset_selected, outname)
