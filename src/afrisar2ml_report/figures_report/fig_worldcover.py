import matplotlib.pyplot as plt
import numpy as np

from afrisar2ml_report import open_dataset, paths, sites
from afrisar2ml_report.plot_utils import esa_worldcover_cmap, figwidth, fmt, layout_single, mpl_styles


def fig_worldcover(site, band_mask):
    print("fig_worldcover", site, band_mask)
    mpl_styles.setup_matplotlib_report_figure_styles()
    cmap, norm, class_values, class_names = esa_worldcover_cmap.get_worldcover_cmap_and_norm()

    # get worldcover map
    ds = open_dataset.open_dataset(site)
    worldcover = ds["worldcover"].values.astype(np.float32)

    # load band-specific mask and set invalid pixels to nan
    mask_fsar = ds[f"mask_{band_mask.lower()}"].values.astype(bool)
    worldcover[~mask_fsar] = np.nan

    # plot
    aspect = worldcover.shape[1] / worldcover.shape[0]
    fig, ax, cax = layout_single.map_layout(aspect, fig_width=figwidth.WIDTH_REPORT)
    im = ax.imshow(worldcover, cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    # colorbar
    cbar = fig.colorbar(im, cax=cax, ticks=class_values, drawedges=False)
    cbar.ax.set_yticklabels([f"{n} ({v})" for n, v in zip(class_names, class_values)])
    cbar.ax.yaxis.set_tick_params(which="minor", left=False, right=False)
    cax.set_title("Map\ncode", linespacing=1.0)

    # title
    ax.set_title(f"{fmt.format_name(site)}, ESA WorldCover 2021, {band_mask}-band mask")

    # move to the left
    pos = ax.get_position()
    ax.set_position((pos.x0 - 0.1, pos.y0, pos.width, pos.height))
    pos = cax.get_position()
    cax.set_position((pos.x0 - 0.1, pos.y0, pos.width, pos.height))

    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_worldcover_{site}_{band_mask}.jpg", dpi=600)
    plt.close("all")


if __name__ == "__main__":
    fig_worldcover(site=sites.LOPE, band_mask="L")
