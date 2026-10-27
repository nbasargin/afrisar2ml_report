import matplotlib.pyplot as plt
import pandas as pd


def plot_train_loss(loss_hist, save_to, loss_max=200):
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 4))
    plt.subplots_adjust(top=0.9, bottom=0.12, left=0.1, right=0.97, hspace=0.3)
    fig.suptitle(save_to.name)
    # loss
    smoothed_loss = pd.Series(loss_hist).rolling(window=32).mean()
    ax.plot(loss_hist, label="loss", alpha=0.5)
    ax.plot(smoothed_loss, label="loss (smoothed)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_ylim(0, loss_max)
    ax.legend(loc="upper right")
    ax.grid()
    # save
    fig.savefig(save_to, dpi=300)
    plt.close("all")


def plot_eval_metrics(eval_dict, title, save_to):
    """
    eval_dict: dict[int, dict[str, float]]
        Example:
        {
            1: {"rmse": ..., "bias": ..., "pearson": ..., "r2": ...},
            2: {...},
            ...
        }
    save_to: pathlib.Path or str
    """
    if len(eval_dict) == 0:
        return
    # sort epochs and map to evenly spaced x positions
    epochs = sorted(eval_dict.keys())
    x = list(range(len(epochs)))  # equally spaced positions
    # extract metrics
    rmse = [eval_dict[e].get("rmse", None) for e in epochs]
    bias = [eval_dict[e].get("bias", None) for e in epochs]
    pearson = [eval_dict[e].get("pearson", None) for e in epochs]
    r2 = [eval_dict[e].get("r2", None) for e in epochs]
    # plotting
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(13, 8))
    plt.subplots_adjust(top=0.92, bottom=0.09, left=0.07, right=0.98, hspace=0.3, wspace=0.2)
    fig.suptitle(title)
    plots = [
        ("RMSE", rmse, axes[0, 0]),
        ("Bias", bias, axes[0, 1]),
        ("Pearson", pearson, axes[1, 0]),
        ("R²", r2, axes[1, 1]),
    ]
    for axtitle, values, ax in plots:
        ax.scatter(x, values)
        ax.set_title(axtitle)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(axtitle)
        # evenly spaced ticks but labeled with actual epochs
        ax.set_xticks(x)
        ax.set_xticklabels(epochs, rotation=45)
        ax.grid()
    # save
    fig.savefig(save_to, dpi=300)
    plt.close("all")
