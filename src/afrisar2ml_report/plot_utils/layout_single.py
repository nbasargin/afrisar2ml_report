import matplotlib.pyplot as plt

from afrisar2ml_report.plot_utils import axis, figwidth


def map_layout(
    aspect,
    fig_width=figwidth.WIDTH_REPORT,
    max_height=4.0,
    padding_top=0.2,
    padding_right=0.25,
    padding_bottom=0.01,
    padding_left=0.15,
    gap=0.15,
    cax_w=0.1,
):
    # image dimensions based on the requested aspect ratio
    available_w = fig_width - padding_left - padding_right - gap - cax_w  # available width for image + colorbar
    image_w = available_w
    image_h = image_w / aspect

    fig_height = padding_top + image_h + padding_bottom

    if max_height is not None and fig_height > max_height:
        image_h = max_height - padding_top - padding_bottom  # maximum height available for the image
        image_w = image_h * aspect  # preserve aspect ratio
        # Recalculate figure width usage
        # keep the image + colorbar centered vertically/horizontally within the available figure width
        total_w = image_w + gap + cax_w
        left_inch = (fig_width - total_w) / 2
        fig_height = max_height
    else:
        left_inch = padding_left

    fig = plt.figure(figsize=(fig_width, fig_height))
    bottom_inch = padding_bottom
    ax = axis.add_axes(fig, left_inch, bottom_inch, image_w, image_h)
    cax_left = left_inch + image_w + gap
    cax_h = image_h * 0.8
    cax_bottom = bottom_inch + (image_h - cax_h) / 2
    cax = axis.add_axes(fig, cax_left, cax_bottom, cax_w, cax_h)

    return fig, ax, cax
