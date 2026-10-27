from afrisar2ml_report import metadata, open_dataset, paths, sites


def tab_passes(site, band):
    print("tab_passes", site, band)
    ds = open_dataset.open_dataset(site)
    pass_names = ds.attrs["pass_names"]
    date = metadata.get_pass_start_time(ds, pass_names[0], band).date()
    passes_str = f"% tab_passes, {site}, {band} band, acquired on {date} \n"
    passes_str += "\\begin{tabular}{lccc} \n"
    passes_str += "\\hline \n"
    passes_str += "\\textbf{Pass name} & \\textbf{Time (UTC)}"
    passes_str += " & \\textbf{$B_{hor}$ (m)} & \\textbf{$B_{ver}$ (m)} \\\\ \n"
    passes_str += "\\hline \n"
    for ps_name in pass_names:
        dt = metadata.get_pass_start_time(ds, ps_name, band)
        h_bl, v_bl = metadata.get_pass_spatial_baselines(ds, ps_name, band)
        passes_str += f"{ps_name} & {dt.time():%H:%M} & {h_bl:.0f} & {v_bl:.0f} \\\\ \n"
    passes_str += "\\hline \n"
    passes_str += "\\end{tabular} \n"
    # save
    folder = paths.get_report_figure_folder()
    with open(folder / f"tab_passes_{site}_{band}.tex", "w") as file:
        file.write(passes_str)


def tab_passes_main():
    band = "P"
    tab_passes(sites.LOPE, band)
    tab_passes(sites.MABOUNIE, band)
    tab_passes(sites.MITZIC, band)
    tab_passes(sites.MONDAH, band)
    tab_passes(sites.PONGARA, band)


if __name__ == "__main__":
    tab_passes_main()
