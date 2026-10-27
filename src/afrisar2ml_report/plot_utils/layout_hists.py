import matplotlib.pyplot as plt
import numpy as np

from afrisar2ml_report.plot_utils import axis


def histogram_grid_layout(
    num_rows,
    num_cols,
    fig_width,
    padding_top,
    padding_right,
    padding_bottom,
    padding_left,
    vertical_gap,
    horizontal_gap,
):
    cell_w = (fig_width - horizontal_gap * (num_cols - 1) - padding_left - padding_right) / num_cols
    cell_h = cell_w  # * 1.1  # cell_w / (sm_max + sm_min) * (sm_max + sm_min + sm_top_buffer)
    fig_height = padding_top + cell_h * num_rows + vertical_gap * (num_rows - 1) + padding_bottom
    fig = plt.figure(figsize=(fig_width, fig_height))
    axs = []
    # axis grid
    for row in range(num_rows):
        bottom_inch = padding_bottom + (cell_h + vertical_gap) * (num_rows - row - 1)
        row_axs = []
        for col in range(num_cols):
            left_inch = padding_left + (cell_w + horizontal_gap) * col
            row_axs.append(axis.add_axes(fig, left_inch, bottom_inch, cell_w, cell_h))
        axs.append(row_axs)
    return fig, np.array(axs)
