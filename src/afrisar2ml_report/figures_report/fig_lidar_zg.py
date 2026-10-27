import matplotlib.pyplot as plt
import numpy as np

from afrisar2ml_report import open_dataset, paths, sites
from afrisar2ml_report.plot_utils import figwidth, fmt, layout_single, mpl_styles


def fig_lidar_zg(site, band_mask):
    print("fig_lidar_zg", site, band_mask)
    mpl_styles.setup_matplotlib_report_figure_styles()

    # get lidar ZG
    ds = open_dataset.open_dataset(site)
    zg = ds["lvis_zg"].values.astype(float)

    # load band-specific mask and set invalid pixels to nan
    mask_name = f"mask_{band_mask.lower()}"
    mask = ds[mask_name].values.astype(bool)
    zg[~mask] = np.nan

    # plot
    aspect = zg.shape[1] / zg.shape[0]
    fig, ax, cax = layout_single.map_layout(aspect, fig_width=figwidth.WIDTH_REPORT)
    im = ax.imshow(zg, interpolation="nearest", cmap="terrain", vmin=0, vmax=760)
    ax.set_xticks([])
    ax.set_yticks([])

    # colorbar
    fig.colorbar(im, cax=cax)
    cax.set_title("m", pad=5)

    # title
    ax.set_title(f"{fmt.format_name(site)}, LVIS ZG, {band_mask}-band mask")

    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_lidar_zg_{site}_{band_mask}.jpg", dpi=600)
    plt.close("all")


if __name__ == "__main__":
    fig_lidar_zg(site=sites.LOPE, band_mask="L")
