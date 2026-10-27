import matplotlib.pyplot as plt
import numpy as np
import torch

from afrisar2ml_report import config, metrics, open_dataset, paths
from afrisar2ml_report.benchmarks import benchmark_tsnn
from afrisar2ml_report.plot_utils import fmt, layout_maps, mpl_styles


def add_metrics_to_ax(ax, predicted, target):
    valid_pixels = torch.isfinite(predicted) & torch.isfinite(target)
    metrics_dict = metrics.get_combined_metrics_dict(predicted[valid_pixels], target[valid_pixels])
    str1 = f"RMSE: {metrics_dict['rmse']:.2f} m\nBias: {metrics_dict['bias']:.2f} m"
    str2 = f"$R^2$: {metrics_dict['r2']:.2f}\nPearson: {metrics_dict['pearson']:.2f}"
    metrics_styles = {
        "transform": ax.transAxes,
        "fontsize": 7,
        "bbox": {"facecolor": "#ffffff", "alpha": 0.9, "pad": 2, "linewidth": 0.5, "edgecolor": "#aaaaaa"},
        "linespacing": 1.0,
    }
    ax.text(0.04, 0.05, str1, verticalalignment="bottom", **metrics_styles)
    ax.text(0.96, 0.05, str2, horizontalalignment="right", verticalalignment="bottom", **metrics_styles)


def fig_tsnn_maps(site, seed):
    print("fig_tsnn_maps", site, seed)
    mpl_styles.setup_matplotlib_report_figure_styles()
    # lvis
    ds = open_dataset.open_dataset(site)
    target_np = open_dataset.get_non_water_rh98(ds)
    water_pixels = ds["worldcover"].values == 80
    # figure
    site_shape = target_np.shape
    aspect = site_shape[1] / site_shape[0]
    num_rows = 2
    num_cols = 2
    fig, axs, caxs = layout_maps.maps_grid_layout(num_rows=num_rows, num_cols=num_cols, aspect=aspect)
    pred_styles = {"interpolation": "nearest", "vmin": 0, "vmax": 50, "cmap": "Greens"}
    err_styles = {"interpolation": "nearest", "vmin": -20, "vmax": 20, "cmap": "turbo"}  # PuOr_r
    pred_cb_ticks = [0, 10, 20, 30, 40]
    for row in range(num_rows):
        # plot maps
        ax_pred, ax_err = axs[row]
        ax_pred.set_xticks([])
        ax_pred.set_yticks([])
        ax_err.set_xticks([])
        ax_err.set_yticks([])
        band = "P" if row == 0 else "L"
        ax_pred.set_ylabel(f"TSNN, {band}-band")
        predictions = benchmark_tsnn.load_tsnn_height(site, band, seed)
        predictions[water_pixels] = np.nan
        err = predictions - target_np
        im_pred = ax_pred.imshow(predictions, **pred_styles)
        im_err = ax_err.imshow(err, **err_styles)
        add_metrics_to_ax(ax_err, torch.tensor(predictions), torch.tensor(target_np))

        # titles
        if row == 0:
            ax_pred.set_title(f"Predicted TSNN height, {fmt.format_name(site)}")
            ax_err.set_title("Error ($h_{pred} - h_{LVIS}$)")

        # colorbars
        cax_pred, cax_err = caxs
        fig.colorbar(im_pred, cax=cax_pred, orientation="horizontal", ticks=pred_cb_ticks)  # type:ignore
        fig.colorbar(im_err, cax=cax_err, orientation="horizontal")  # type:ignore
        cax_pred.set_ylabel("m")

    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_tsnn_maps_{site}_s{seed}.jpg")
    plt.close("all")


if __name__ == "__main__":
    fig_tsnn_maps(config.TEST_SITE, config.SEED)
