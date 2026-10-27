import matplotlib.pyplot as plt
import numpy as np

from afrisar2ml_report import open_dataset, paths, sites
from afrisar2ml_report.plot_utils import figwidth, fmt, layout_single, mpl_styles


def fig_offnadir(site, band):
    print("fig_offnadir", site, band)
    mpl_styles.setup_matplotlib_report_figure_styles()

    # get off-nadir angle
    ds = open_dataset.open_dataset(site)
    offnadir = ds[f"offnadir_{band.lower()}"].values

    # plot
    aspect = offnadir.shape[1] / offnadir.shape[0]
    fig, ax, cax = layout_single.map_layout(aspect, fig_width=figwidth.WIDTH_REPORT)
    im = ax.imshow(np.rad2deg(offnadir), interpolation="nearest", vmin=10, vmax=70, cmap="plasma")
    ax.set_xticks([])
    ax.set_yticks([])

    # colorbar
    fig.colorbar(im, cax=cax)
    cax.set_title("deg", pad=5)

    # title
    ax.set_title(f"{fmt.format_name(site)}, off-nadir angle, {band}-band")

    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_offnadir_{site}_{band}.jpg", dpi=600)
    plt.close("all")


if __name__ == "__main__":
    fig_offnadir(site=sites.LOPE, band="L")
