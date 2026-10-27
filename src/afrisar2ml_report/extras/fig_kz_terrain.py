import matplotlib.pyplot as plt
import numpy as np

from afrisar2ml_report import metadata, open_dataset, paths, sites
from afrisar2ml_report.figures_report import config_report
from afrisar2ml_report.plot_utils import figwidth, fmt, layout_single, mpl_styles


def get_terrain_corrected_kz(kz, inc, offnadir):
    return kz * np.sin(offnadir) / np.sin(inc)


def fig_kz_terrain(site, band):
    print("fig_kz_terrain", site, band)
    mpl_styles.setup_matplotlib_report_figure_styles()

    pass_a, pass_b = config_report.get_pass_pair(site)
    pass_ae, pass_be = pass_a.encode("ascii"), pass_b.encode("ascii")
    kz_name = f"kz_{band.lower()}"

    # get kz
    ds = open_dataset.open_dataset(site)
    kz = ds[kz_name].sel(pass_a=pass_ae, pass_b=pass_be).values

    # correct kz by the terrain
    inc = ds[f"incidence_{band.lower()}"].values
    offnadir = ds[f"offnadir_{band.lower()}"].values
    kz_corr = get_terrain_corrected_kz(kz, inc, offnadir)

    # plot
    aspect = kz_corr.shape[1] / kz_corr.shape[0]
    fig, ax, cax = layout_single.map_layout(aspect, fig_width=figwidth.WIDTH_REPORT)
    vmin, vmax = (0.08, 0.19) if band == "L" else (0.03, 0.06)
    im = ax.imshow(kz_corr, interpolation="nearest", vmin=vmin, vmax=vmax, cmap="viridis")
    ax.set_xticks([])
    ax.set_yticks([])

    # colorbar
    fig.colorbar(im, cax=cax)
    cax.set_title("rad / m", pad=5)

    # title
    h_bl, v_bl = metadata.get_spatial_baselines_between(ds, pass_a, pass_b, band)
    bl_str = fmt.format_baseline(h_bl, v_bl)
    ax.set_title(f"{fmt.format_name(site)}, $k_z$ (terrain-corrected), {band}-band, {bl_str}")

    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_kz_terrain_{site}_{band}.jpg", dpi=600)
    plt.close("all")


if __name__ == "__main__":
    fig_kz_terrain(site=sites.LOPE, band="L")
