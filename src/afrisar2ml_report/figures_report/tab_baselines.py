from afrisar2ml_report import metadata, open_dataset, paths, sites


def tab_baselines(site, band):
    print("tab_baselines", site, band)
    ds = open_dataset.open_dataset(site)
    pass_names = ds.attrs["pass_names"]

    h_bl_str = r"B_{hor}"
    v_bl_str = r"B_{ver}"

    for ps_name in pass_names:
        h_bl, v_bl = metadata.get_pass_spatial_baselines(ds, ps_name, band)
        h_bl_str += f" & {h_bl:.0f}"
        v_bl_str += f" & {v_bl:.0f}"

    passes_str = f"% tab_baselines, {site}, {band} band \n"
    passes_str += r"\begin{equation}" + "\n"
    passes_str += r"  \begin{matrix} " + "\n"
    passes_str += "    " + h_bl_str + r" \\" + "\n"
    passes_str += "    " + v_bl_str + r" \\" + "\n"
    passes_str += r"  \end{matrix} " + "\n"
    passes_str += r"\end{equation}" + "\n"

    # save
    folder = paths.get_report_figure_folder()
    with open(folder / f"tab_baselines_{site}_{band}.tex", "w") as file:
        file.write(passes_str)


def tab_baselines_main():
    band = "P"
    tab_baselines(sites.LOPE, band)
    tab_baselines(sites.MABOUNIE, band)
    tab_baselines(sites.MITZIC, band)
    tab_baselines(sites.MONDAH, band)
    tab_baselines(sites.PONGARA, band)


if __name__ == "__main__":
    tab_baselines_main()
