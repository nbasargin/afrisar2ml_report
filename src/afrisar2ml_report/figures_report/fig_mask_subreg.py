import matplotlib.pyplot as plt
import numpy as np

from afrisar2ml_report import open_dataset, paths, sites
from afrisar2ml_report.plot_utils import colors, figwidth, fmt, layout_single, mpl_styles


def fig_mask_subreg(site, band):
    print("fig_mask_subreg", site, band)
    mpl_styles.setup_matplotlib_report_figure_styles()
    inside_color, outside_color = colors.get_subregion_colors()

    # get masks
    ds = open_dataset.open_dataset(site)
    mask_fsar = ds[f"mask_{band.lower()}"].values.astype(bool)
    mask_subregion = ds["mask_subregion"].values.astype(bool)
    inside = mask_fsar & mask_subregion
    outside = mask_fsar & ~mask_subregion

    # colors
    rgb = np.zeros((*mask_fsar.shape, 3), dtype=np.float32)
    rgb[outside] = outside_color
    rgb[inside] = inside_color
    rgb[~mask_fsar] = np.nan

    # plot
    aspect = mask_fsar.shape[1] / mask_fsar.shape[0]
    fig, ax, cax = layout_single.map_layout(aspect, fig_width=figwidth.WIDTH_REPORT)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    # no colorbar
    cax.axis("off")

    # title
    ax.set_title(f"{fmt.format_name(site)}, F-SAR coverage (gray) and subregion mask (green), {band}-band")

    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_mask_subreg_{site}_{band}.jpg", dpi=600)
    plt.close("all")


if __name__ == "__main__":
    fig_mask_subreg(site=sites.LOPE, band="L")
