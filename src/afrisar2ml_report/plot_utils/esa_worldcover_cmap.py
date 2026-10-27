from matplotlib.colors import BoundaryNorm, ListedColormap

# ESA WorldCover 2021 full specification: Value -> (Hex Color, Label)
WORLDCOVER_SPECS = {
    10: ("#006400", "Tree cover"),
    20: ("#ffbb22", "Shrubland"),
    30: ("#ffff4c", "Grassland"),
    40: ("#f096ff", "Cropland"),
    50: ("#fa0000", "Built-up"),
    60: ("#b4b4b4", "Bare / sparse vegetation"),
    70: ("#f0f0f0", "Snow / ice"),
    80: ("#0064c8", "Permanent water"),
    90: ("#0096a0", "Herbaceous wetland"),
    95: ("#00cf75", "Mangroves"),
    100: ("#fae6a0", "Moss / lichen"),
}


def get_worldcover_cmap_and_norm():
    class_values = sorted(WORLDCOVER_SPECS.keys())
    class_colors = [WORLDCOVER_SPECS[v][0] for v in class_values]
    class_names = [WORLDCOVER_SPECS[v][1] for v in class_values]
    # create colormap and set NaN (masked out) pixels to transparent
    cmap = ListedColormap(class_colors)
    cmap.set_bad(color="none")
    # fixed boundaries around discrete class values
    boundaries = [5, 15, 25, 35, 45, 55, 65, 75, 85, 94.9, 95.1, 105]
    norm = BoundaryNorm(boundaries, cmap.N)
    return cmap, norm, class_values, class_names
