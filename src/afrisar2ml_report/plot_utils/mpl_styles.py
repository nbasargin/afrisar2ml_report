import matplotlib.pyplot as plt


def setup_matplotlib_report_figure_styles():
    """Setup global matplotlib styles for paper figures."""
    # plt.rcParams["font.family"] = "serif"
    # plt.rcParams["font.serif"] = ["STIXGeneral"]
    plt.rcParams["mathtext.fontset"] = "stix"  # "cm"
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams["axes.titlesize"] = 9
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["axes.titlepad"] = 3
    plt.rcParams["axes.labelpad"] = 2
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["xtick.major.width"] = 0.5
    plt.rcParams["ytick.major.width"] = 0.5
    plt.rcParams["xtick.major.pad"] = 2
    plt.rcParams["ytick.major.pad"] = 2
    plt.rcParams["figure.dpi"] = 600


def reset_matplotlib_styles():
    plt.rcdefaults()
