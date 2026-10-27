from afrisar2ml_report import open_dataset, paths, sites
from afrisar2ml_report.plot_utils import fmt


def tab_extent():
    extent_str = "% tab_extent \n"
    extent_str += "\\begin{tabular}{lcccc} \n"
    extent_str += "\\hline \n"
    extent_str += "\\textbf{Site} & \\textbf{Shape (Y, X)} & \\textbf{CRS}"
    extent_str += " & \\textbf{Northing bounds (m)} & \\textbf{Easting bounds (m)}  \\\\ \n"
    extent_str += "\\hline \n"
    for site in [sites.LOPE, sites.MABOUNIE, sites.MITZIC, sites.MONDAH, sites.PONGARA]:
        ds = open_dataset.open_dataset(site)
        crs = ds.rio.crs
        min_east, min_north, max_east, max_north = ds.rio.bounds()
        height = ds.rio.height  # number of rows / pixels in Y
        width = ds.rio.width  # number of columns / pixels in X
        extent_str += f"{fmt.format_name(site)}"
        extent_str += f" & ({height}, {width})"
        extent_str += f" & {crs}"
        extent_str += f" & [{min_north:g}, {max_north:g}]"
        extent_str += f" & [{min_east:g}, {max_east:g}]"
        extent_str += " \\\\ \n"
    extent_str += "\\hline \n"
    extent_str += "\\end{tabular} \n"
    # save
    folder = paths.get_report_figure_folder()
    with open(folder / "tab_extent.tex", "w") as file:
        file.write(extent_str)


if __name__ == "__main__":
    tab_extent()
