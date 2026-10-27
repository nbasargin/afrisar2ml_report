import matplotlib.pyplot as plt
import numpy as np

from afrisar2ml_report.plot_utils import axis, figwidth


def maps_grid_layout(num_rows, num_cols, aspect, fig_width=figwidth.WIDTH_REPORT):
    padding_top, padding_right, padding_bottom, padding_left = 0.20, 0.10, 0.25, 0.20
    gap = 0.07
    cell_w = (fig_width - gap * (num_cols - 1) - padding_left - padding_right) / num_cols
    cell_h = cell_w / aspect
    cax_h = 0.08
    fig_height = padding_top + cell_h * num_rows + gap * num_rows + cax_h + padding_bottom
    fig = plt.figure(figsize=(fig_width, fig_height))
    # axis grid
    axs = []
    for row in range(num_rows):
        bottom_inch = padding_bottom + (cell_h + gap) * (num_rows - row - 1) + cax_h + gap
        row_axs = []
        for col in range(num_cols):
            left_inch = padding_left + (cell_w + gap) * col
            row_axs.append(axis.add_axes(fig, left_inch, bottom_inch, cell_w, cell_h))
        axs.append(row_axs)
    # cax row
    caxs = []
    for col in range(num_cols):        
        left_inch = padding_left + (cell_w + gap) * col
        bottom_inch = padding_bottom
        caxs.append(axis.add_axes(fig, left_inch, bottom_inch, cell_w, cax_h))
    return fig, np.array(axs), np.array(caxs)
