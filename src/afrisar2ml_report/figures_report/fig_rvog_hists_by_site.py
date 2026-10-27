import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LogNorm

from afrisar2ml_report import metadata, open_dataset, paths, sites
from afrisar2ml_report.benchmarks import benchmark_rvog
from afrisar2ml_report.plot_utils import axis, figwidth, fmt, hist_metrics, layout_hists, mpl_styles


def fig_rvog_hists_by_site(band, site_pass_dict):
    print("fig_rvog_hists_by_site", band)
    mpl_styles.setup_matplotlib_report_figure_styles()

    num_rows = 1
    num_cols = len(site_pass_dict)
    padding_left = 0.20
    padding_right = 0.40
    horizontal_gap = 0.07
    fig_width = figwidth.WIDTH_REPORT * 1.2
    fig, axs = layout_hists.histogram_grid_layout(
        num_rows=num_rows,
        num_cols=num_cols,
        fig_width=fig_width,
        padding_top=0.30,
        padding_right=padding_right,
        padding_bottom=0.80,
        padding_left=padding_left,
        vertical_gap=0.07,
        horizontal_gap=horizontal_gap,
    )
    norm = LogNorm(vmin=1e-5, vmax=1e-3)

    # plot histograms
    for i, (site, (pass_a, pass_b)) in enumerate(site_pass_dict.items()):
        rvog_height = benchmark_rvog.load_rvog_height(site, band, pass_a, pass_b)
        ds = open_dataset.open_dataset(site)
        rh98 = open_dataset.get_non_water_rh98(ds)
        ax = axs[0, i]
        hb = hist_metrics.plot_hist_with_metrics_into_axis(ax, torch.tensor(rh98), torch.tensor(rvog_height), norm)
        h_bl, v_bl = metadata.get_spatial_baselines_between(ds, pass_a, pass_b, band)
        ax.set_title(f"{fmt.format_name(site)}, {band}-band\n{fmt.format_baseline(h_bl, v_bl)}", linespacing=1.0)
        ax.set_xlabel("LVIS RH98")

    axs[0, 0].set_ylabel("RVOG prediction")
    axs[0, num_cols - 1].yaxis.set_label_position("right")
    axs[0, num_cols - 1].set_ylabel("Predicted RVoG height")

    # colorbar
    ax_width = (fig_width - padding_right - padding_left - horizontal_gap * (num_cols - 1)) / num_cols
    cax_width = ax_width * (num_cols - 1) + horizontal_gap * (num_cols - 2)
    cax = axis.add_axes(fig, padding_left, 0.25, cax_width, 0.1)
    fig.colorbar(hb, cax=cax, orientation="horizontal")  # type:ignore
    cax.text(1.05, 0.5, "Fraction of samples", transform=cax.transAxes, va="center", ha="left")

    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_rvog_hists_by_site_{band}.jpg")
    plt.close("all")


def fig_rvog_hists_by_site_main():
    site_list = [sites.LOPE, sites.MABOUNIE, sites.MITZIC, sites.MONDAH, sites.PONGARA]
    for band in ["L", "P"]:
        site_pass_dict = {site: benchmark_rvog.get_pass_names_by_site(site, band) for site in site_list}
        fig_rvog_hists_by_site(band, site_pass_dict)


if __name__ == "__main__":
    fig_rvog_hists_by_site_main()
