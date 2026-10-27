import matplotlib.pyplot as plt

from afrisar2ml_report import open_dataset, paths, sites
from afrisar2ml_report.plot_utils import figwidth, fmt, layout_single, mpl_styles


def fig_dem(site, band):
    print("fig_dem", site, band)
    mpl_styles.setup_matplotlib_report_figure_styles()

    # get slant range DEM
    ds = open_dataset.open_dataset(site)
    dem_name = f"slantdem_{band.lower()}"
    dem = ds[dem_name].values

    # plot
    aspect = dem.shape[1] / dem.shape[0]
    fig, ax, cax = layout_single.map_layout(aspect, fig_width=figwidth.WIDTH_REPORT)
    im = ax.imshow(dem, interpolation="nearest", cmap="terrain", vmin=0, vmax=760)
    ax.set_xticks([])
    ax.set_yticks([])

    # colorbar
    fig.colorbar(im, cax=cax)
    cax.set_title("m", pad=5)

    # title
    ax.set_title(f"{fmt.format_name(site)}, digital elevation model (DEM), {band}-band mask")

    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_dem_{site}_{band}.jpg", dpi=600)
    plt.close("all")


if __name__ == "__main__":
    fig_dem(site=sites.LOPE, band="L")
