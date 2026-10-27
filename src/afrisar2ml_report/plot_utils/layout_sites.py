import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from afrisar2ml_report import sites
from afrisar2ml_report.plot_utils import axis, figwidth


def combined_site_layout_report():
    fig_width = figwidth.WIDTH_REPORT
    fig_height = 5.18
    fig = plt.figure(figsize=(fig_width, fig_height))
    scale_factor = 0.00431  # pixels per inch
    # Format: (left_inch, bottom_inch, width_px, height_px)
    layout = {
        sites.LOPE: (0.11, 2.23, 561, 581),
        sites.MABOUNIE: (4.56, 1.78, 347, 639),
        sites.MITZIC: (2.74, 2.32, 330, 539),
        sites.MONDAH: (4.34, 0.18, 440, 343),
        sites.PONGARA: (0.00, 0.00, 982, 486),
    }
    axes: dict[str, Axes] = {}
    for name, (x, y, w_px, h_px) in layout.items():
        w = w_px * scale_factor
        h = h_px * scale_factor
        ax = axis.add_axes(fig, x, y, w, h)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        ax.set_frame_on(False)
        axes[name] = ax
    return fig, axes


def add_site_labels_report(fig):
    fig_width, fig_height = fig.get_size_inches()
    # in inch
    top_row_y = 4.67
    bottom_row_y = 0.16
    label_positions = {
        sites.LOPE: (1.31, top_row_y),
        sites.MITZIC: (3.45, top_row_y),
        sites.MABOUNIE: (5.32, top_row_y),
        sites.PONGARA: (2.23, bottom_row_y),
        sites.MONDAH: (5.30, bottom_row_y),
    }
    for site, (x_in, y_in) in label_positions.items():
        site_str = site[0] + site[1:].lower()
        fig.text(x_in / fig_width, y_in / fig_height, site_str, ha="center", va="center", linespacing=1.0)
