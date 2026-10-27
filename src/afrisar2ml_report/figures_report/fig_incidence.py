import matplotlib.pyplot as plt
import numpy as np

from afrisar2ml_report import open_dataset, paths, sites
from afrisar2ml_report.plot_utils import figwidth, fmt, layout_single, mpl_styles


def fig_incidence(site, band):
    print("fig_incidence", site, band)
    mpl_styles.setup_matplotlib_report_figure_styles()

    # get incidence angle
    ds = open_dataset.open_dataset(site)
    inc = ds[f"incidence_{band.lower()}"].values

    # plot
    aspect = inc.shape[1] / inc.shape[0]
    fig, ax, cax = layout_single.map_layout(aspect, fig_width=figwidth.WIDTH_REPORT)
    im = ax.imshow(np.rad2deg(inc), interpolation="nearest", vmin=10, vmax=70, cmap="plasma")
    ax.set_xticks([])
    ax.set_yticks([])

    # colorbar
    fig.colorbar(im, cax=cax)
    cax.set_title("deg", pad=5)

    # title
    ax.set_title(f"{fmt.format_name(site)}, local incidence angle, {band}-band")

    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_incidence_{site}_{band}.jpg", dpi=600)
    plt.close("all")


if __name__ == "__main__":
    fig_incidence(site=sites.LOPE, band="L")
