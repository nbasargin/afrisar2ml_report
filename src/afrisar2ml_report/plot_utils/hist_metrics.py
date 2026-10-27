import numpy as np
import torch

from afrisar2ml_report import metrics


def plot_hist_with_metrics_into_axis(ax, target, predicted, norm, cmap="viridis"):
    if len(target) == 0:
        ax.axis("off")
        ax.text(0.5, 0.5, "No points", horizontalalignment="center", verticalalignment="center", transform=ax.transAxes)
        return
    vmin, vmax, ticklabels = 0, 70, [5, 25, 45, 65]
    
    # select valid samples
    valid_samples = torch.isfinite(target) & torch.isfinite(predicted)
    target_valid = target[valid_samples]
    predicted_valid = predicted[valid_samples]
    # Convert to numpy
    x = target_valid.cpu().detach().numpy().ravel()
    y = predicted_valid.cpu().detach().numpy().ravel()
    # Normalized hexbin density plot: normalize counts by the total number of samples
    c = np.ones_like(x) / len(x)
    reduce_C_function = np.sum
    hb = ax.hexbin(
        x,
        y,
        C=c,
        reduce_C_function=reduce_C_function,
        gridsize=140,
        extent=(vmin, vmax, vmin, vmax),
        cmap=cmap,
        mincnt=1,
        norm=norm,
        linewidths=0.5,
    )
    # Reference lines
    ax.plot((vmin, vmax), (vmin, vmax), c="black", alpha=0.2, linewidth=0.5)
    ax.plot((vmin, vmax), (vmin - 10, vmax - 10), c="black", alpha=0.2, linewidth=0.5, linestyle="dotted")
    ax.plot((vmin, vmax), (vmin + 10, vmax + 10), c="black", alpha=0.2, linewidth=0.5, linestyle="dotted")
    # Limits
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    # Ticks
    ax.yaxis.tick_right()
    ax.set_xticks(ticklabels)
    ax.set_yticks(ticklabels)
    # Metrics
    metrics_dict = metrics.get_combined_metrics_dict(predicted_valid, target_valid)
    str1 = f"RMSE: {metrics_dict['rmse']:.2f} m\nBias: {metrics_dict['bias']:.2f} m"
    str2 = f"$R^2$: {metrics_dict['r2']:.2f}\nPearson: {metrics_dict['pearson']:.2f}"
    metrics_styles = {
        "transform": ax.transAxes,
        "fontsize": 7,
        "bbox": {"facecolor": "#ffffff", "alpha": 0.9, "pad": 2, "linewidth": 0.5, "edgecolor": "#aaaaaa"},
        "linespacing": 1.0,
    }
    ax.text(0.05, 0.95, str1, verticalalignment="top", **metrics_styles)
    ax.text(0.95, 0.05, str2, horizontalalignment="right", verticalalignment="bottom", **metrics_styles)
    return hb
