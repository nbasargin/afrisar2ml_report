import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from afrisar2ml_report import config, paths, sites
from afrisar2ml_report.benchmarks import benchmark_tsnn, data_split
from afrisar2ml_report.plot_utils import axis, figwidth, fmt, hist_metrics, layout_hists, mpl_styles


def get_tsnn_heights(band, seed):
    tsnn_height_dict = {}
    for site in [sites.LOPE, sites.MABOUNIE, sites.MITZIC, sites.MONDAH, sites.PONGARA]:
        tsnn_height_dict[site] = benchmark_tsnn.load_tsnn_height(site, band, seed)
    return tsnn_height_dict


def fig_tsnn_hists(band, seed):
    print("fig_tsnn_hists", band, seed)
    mpl_styles.setup_matplotlib_report_figure_styles()

    num_rows = 1
    num_cols = 3
    padding_left = 0.20
    padding_right = 0.40
    horizontal_gap = 0.07
    fig, axs = layout_hists.histogram_grid_layout(
        num_rows=num_rows,
        num_cols=num_cols,
        fig_width=figwidth.WIDTH_REPORT,
        padding_top=0.50,
        padding_right=padding_right,
        padding_bottom=0.80,
        padding_left=padding_left,
        vertical_gap=0.07,
        horizontal_gap=horizontal_gap,
    )
    ax_train, ax_val, ax_test = axs[0]
    norm = LogNorm(vmin=1e-5, vmax=1e-3)

    # plot histograms
    height_dict = get_tsnn_heights(band, seed)
    train_pred, train_trg, val_pred, val_trg, test_pred, test_trg = data_split.split_data_into_train_val_test(
        height_dict, band
    )
    hb = hist_metrics.plot_hist_with_metrics_into_axis(ax_train, train_trg, train_pred, norm)
    hb = hist_metrics.plot_hist_with_metrics_into_axis(ax_val, val_trg, val_pred, norm)
    hb = hist_metrics.plot_hist_with_metrics_into_axis(ax_test, test_trg, test_pred, norm)

    # titles and labels
    site_str = [fmt.format_name(s) for s in config.TRAIN_VAL_SITES]
    site_title = ", ".join(site_str[0:2]) + ",\n" + ", ".join(site_str[2:])
    test_str = fmt.format_name(config.TEST_SITE)
    ax_train.set_ylabel("TSNN prediction")
    ax_train.set_title(site_title + f"\nTrain area, {band}-band", linespacing=1.0)
    ax_val.set_title(site_title + f"\nValidation area, {band}-band", linespacing=1.0)
    ax_test.set_title(f"\n{test_str}\nTest area, {band}-band", linespacing=1.0)
    for ax in [ax_train, ax_val, ax_test]:
        ax.set_xlabel("LVIS RH98")
    ax_test.yaxis.set_label_position("right")
    ax_test.set_ylabel("Predicted TSNN height")

    # colorbar
    ax_width = (figwidth.WIDTH_REPORT - padding_right - padding_left - horizontal_gap * (num_cols - 1)) / num_cols
    cax_width = ax_width * 2 + horizontal_gap
    cax = axis.add_axes(fig, padding_left, 0.25, cax_width, 0.1)
    fig.colorbar(hb, cax=cax, orientation="horizontal")  # type:ignore
    cax.text(1.05, 0.5, "Fraction of samples", transform=cax.transAxes, va="center", ha="left")

    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_tsnn_hists_{band}_s{seed}.jpg")
    plt.close("all")


if __name__ == "__main__":
    fig_tsnn_hists("L", config.SEED)
    fig_tsnn_hists("P", config.SEED)
