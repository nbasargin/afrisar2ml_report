import matplotlib.pyplot as plt
import numpy as np

from afrisar2ml_report import open_dataset, paths, sites
from afrisar2ml_report.figures_report import config_report
from afrisar2ml_report.plot_utils import figwidth, fmt, layout_single, mpl_styles


def fig_power(site, band, pol):
    print("fig_power", site, band, pol)
    mpl_styles.setup_matplotlib_report_figure_styles()

    # get the primary pass for the site
    pass_a, _pass_b = config_report.get_pass_pair(site)
    name_ascii = pass_a.encode("ascii")
    pol_ascii = pol.encode("ascii")
    cov_name = f"cov_{band.lower()}"

    # get covariance matrix
    ds = open_dataset.open_dataset(site)
    cov = ds[cov_name].sel(pass_a=name_ascii, pass_b=name_ascii, pol_a=pol_ascii, pol_b=pol_ascii).values

    # calculate power in dB
    power = np.abs(cov)
    power_db = 10 * np.log10(power)

    # plot
    aspect = power_db.shape[1] / power_db.shape[0]
    fig, ax, cax = layout_single.map_layout(aspect, fig_width=figwidth.WIDTH_REPORT)
    im = ax.imshow(power_db, interpolation="nearest", vmin=-16, vmax=0, cmap="Grays_r")
    ax.set_xticks([])
    ax.set_yticks([])

    # colorbar
    fig.colorbar(im, cax=cax)
    cax.set_title("dB", pad=5)

    # title
    ax.set_title(f"{fmt.format_name(site)}, power, {band}-band, {pol.upper()} polarization")

    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_power_{site}_{band}_{pol}.jpg", dpi=600)
    plt.close("all")


if __name__ == "__main__":
    fig_power(site=sites.LOPE, band="L", pol="hh")
