import matplotlib.pyplot as plt

from afrisar2ml_report import open_dataset, paths, sites
from afrisar2ml_report.plot_utils import figwidth, fmt, layout_single, mpl_styles


def fig_dem_diff(site):
    print("fig_dem_diff", site)
    mpl_styles.setup_matplotlib_report_figure_styles()

    # get slant range DEM
    ds = open_dataset.open_dataset(site)
    dem_l = ds["slantdem_l"].values
    dem_p = ds["slantdem_p"].values

    # plot
    aspect = dem_l.shape[1] / dem_l.shape[0]
    fig, ax, cax = layout_single.map_layout(aspect, fig_width=figwidth.WIDTH_REPORT)
    im = ax.imshow(dem_l - dem_p, interpolation="nearest", cmap="PiYG", vmin=-10, vmax=10)
    ax.set_xticks([])
    ax.set_yticks([])

    # colorbar
    fig.colorbar(im, cax=cax)
    cax.set_title("m", pad=5)

    # title
    ax.set_title(f"{fmt.format_name(site)}, DEM difference")

    # save
    folder = paths.get_report_figure_folder() / "extras"
    folder.mkdir(exist_ok=True, parents=True)
    fig.savefig(folder / f"fig_dem_diff_{site}.jpg", dpi=600)
    plt.close("all")


if __name__ == "__main__":
    fig_dem_diff(site=sites.LOPE)
    fig_dem_diff(site=sites.MABOUNIE)
    fig_dem_diff(site=sites.MITZIC)
    fig_dem_diff(site=sites.MONDAH)
    fig_dem_diff(site=sites.PONGARA)
