import matplotlib.pyplot as plt
import numpy as np

from afrisar2ml_report import open_dataset, paths, sites
from afrisar2ml_report.plot_utils import figwidth, fmt, layout_single, mpl_styles


def fig_lidar_counts(site, band_mask):
    print("fig_lidar_counts", site, band_mask)
    mpl_styles.setup_matplotlib_report_figure_styles()

    # get lidar counts (shots per pixel)
    ds = open_dataset.open_dataset(site)
    counts = ds["lvis_counts"].values.astype(float)

    # load band-specific mask and set invalid pixels to nan
    mask_name = f"mask_{band_mask.lower()}"
    mask = ds[mask_name].values.astype(bool)
    counts[~mask] = np.nan

    # plot
    aspect = counts.shape[1] / counts.shape[0]
    fig, ax, cax = layout_single.map_layout(aspect, fig_width=figwidth.WIDTH_REPORT)
    im = ax.imshow(counts, interpolation="nearest", vmin=0, vmax=10, cmap="Blues")
    ax.set_xticks([])
    ax.set_yticks([])

    # colorbar
    fig.colorbar(im, cax=cax)

    # title
    ax.set_title(f"{fmt.format_name(site)}, LVIS shot density, {band_mask}-band mask")

    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_lidar_counts_{site}_{band_mask}.jpg", dpi=600)
    plt.close("all")


if __name__ == "__main__":
    fig_lidar_counts(site=sites.LOPE, band_mask="L")
