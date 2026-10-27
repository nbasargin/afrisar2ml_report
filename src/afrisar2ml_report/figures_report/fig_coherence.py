import matplotlib.pyplot as plt
import numpy as np

from afrisar2ml_report import metadata, open_dataset, paths, sites
from afrisar2ml_report.figures_report import config_report
from afrisar2ml_report.plot_utils import figwidth, fmt, layout_single, mpl_styles


def fig_coherence(site, band, pol):
    print("fig_coherence", site, band, pol)
    mpl_styles.setup_matplotlib_report_figure_styles()

    pass_a, pass_b = config_report.get_pass_pair(site)
    pass_ae, pass_be = pass_a.encode("ascii"), pass_b.encode("ascii")
    pol_a = pol.encode("ascii")
    pol_b = pol.encode("ascii")
    cov_name = f"cov_{band.lower()}"

    # get covariance matrices
    ds = open_dataset.open_dataset(site)
    cov_ab = ds[cov_name].sel(pass_a=pass_ae, pass_b=pass_be, pol_a=pol_a, pol_b=pol_b).values
    cov_aa = ds[cov_name].sel(pass_a=pass_ae, pass_b=pass_ae, pol_a=pol_a, pol_b=pol_a).values
    cov_bb = ds[cov_name].sel(pass_a=pass_be, pass_b=pass_be, pol_a=pol_b, pol_b=pol_b).values

    # calculate coherence
    coherence = np.abs(cov_ab) / np.sqrt(np.abs(cov_aa) * np.abs(cov_bb))

    # plot
    aspect = coherence.shape[1] / coherence.shape[0]
    fig, ax, cax = layout_single.map_layout(aspect, fig_width=figwidth.WIDTH_REPORT)
    im = ax.imshow(coherence, interpolation="nearest", vmin=0, vmax=1, cmap="Grays_r")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, cax=cax)

    # title
    h_bl, v_bl = metadata.get_spatial_baselines_between(ds, pass_a, pass_b, band)
    bl_str = fmt.format_baseline(h_bl, v_bl)
    ax.set_title(f"{fmt.format_name(site)}, InSAR coherence, {band}-band, {pol.upper()} polarization, {bl_str}")

    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_coherence_{site}_{band}_{pol}.jpg", dpi=600)
    plt.close("all")


if __name__ == "__main__":
    fig_coherence(site=sites.LOPE, band="L", pol="hh")
