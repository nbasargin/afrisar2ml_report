from matplotlib.axes import Axes
from matplotlib.figure import Figure


def add_axes(fig: Figure, left_inch, bottom_inch, width_inch, height_inch) -> Axes:
    fig_w, fig_h = fig.get_size_inches()
    axis_box = [left_inch / fig_w, bottom_inch / fig_h, width_inch / fig_w, height_inch / fig_h]
    ax = fig.add_axes(axis_box)  # type: ignore
    return ax
