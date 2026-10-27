import matplotlib.pyplot as plt
import numpy as np

from afrisar2ml_report import open_dataset, paths, sites
from afrisar2ml_report.plot_utils import figwidth, fmt, layout_single, mpl_styles


def fig_lidar_rh98(site, band_mask):
    print("fig_lidar_rh98", site, band_mask)
    mpl_styles.setup_matplotlib_report_figure_styles()

    # get lidar RH98
    ds = open_dataset.open_dataset(site)
    rh98 = ds["lvis_rh98"].values.astype(float)

    # load band-specific mask and set invalid pixels to nan
    mask_name = f"mask_{band_mask.lower()}"
    mask = ds[mask_name].values.astype(bool)
    rh98[~mask] = np.nan

    # plot
    aspect = rh98.shape[1] / rh98.shape[0]
    fig, ax, cax = layout_single.map_layout(aspect, fig_width=figwidth.WIDTH_REPORT)
    im = ax.imshow(rh98, interpolation="nearest", vmin=0, vmax=50, cmap="Greens")
    ax.set_xticks([])
    ax.set_yticks([])

    # colorbar
    fig.colorbar(im, cax=cax)
    cax.set_title("m", pad=5)

    # title
    ax.set_title(f"{fmt.format_name(site)}, LVIS RH98, {band_mask}-band mask")

    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_lidar_rh98_{site}_{band_mask}.jpg", dpi=600)
    plt.close("all")


if __name__ == "__main__":
    fig_lidar_rh98(site=sites.LOPE, band_mask="L")