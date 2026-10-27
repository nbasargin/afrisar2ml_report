from afrisar2ml_report import open_dataset, paths, sites


def tab_proc_param(site):
    ds = open_dataset.open_dataset(site)
    primary_name = ds.attrs["primary_pass_name"]
    details_l = ds.attrs["processing_details_l"]
    details_p = ds.attrs["processing_details_p"]
    # LaTeX header
    param_str = f"% tab_proc_param, {site}, primary {primary_name} \n"
    param_str += "\\begin{tabular}{lcc} \n"
    param_str += "\\hline \n"
    param_str += "\\textbf{Parameter} & \\textbf{L-Band} & \\textbf{P-band} \\\\ \n"
    param_str += "\\hline \n"
    # range params
    cbw_l, cbw_p = details_l["chirp_bandwidth_hz"], details_p["chirp_bandwidth_hz"]
    res_rg_l, res_rg_p = details_l["range_slc_resolution_m"], details_p["range_slc_resolution_m"]
    ps_rg_l, ps_rg_p = details_l["range_slc_pixel_spacing_m"], details_p["range_slc_pixel_spacing_m"]
    pixels_rg_l, pixels_rg_p = details_l["range_slc_window_size_pixels"], details_p["range_slc_window_size_pixels"]
    meters_rg_l, meters_rg_p = details_l["range_slc_window_size_m"], details_p["range_slc_window_size_m"]
    looks_rg_l, looks_rg_p = details_l["range_independent_looks"], details_p["range_independent_looks"]
    param_str += f"Chirp bandwidth & {cbw_l / 1e6:.0f} MHz & {cbw_p / 1e6:.0f} MHz \\\\ \n"
    param_str += f"Range resolution & {res_rg_l:.2f} m & {res_rg_p:.2f} m \\\\ \n"
    param_str += f"Range SLC pixel spacing & {ps_rg_l:.2f} m & {ps_rg_p:.2f} m \\\\ \n"
    param_str += f"Range SLC window size & {pixels_rg_l} pixel & {pixels_rg_p} pixel \\\\ \n"
    param_str += f"Range SLC window size & {meters_rg_l:.1f} m & {meters_rg_p:.1f} m \\\\ \n"
    param_str += f"Range independent looks & {looks_rg_l:.1f} & {looks_rg_p:.1f} \\\\ \n"
    param_str += "\\hline \n"
    # azimuth params
    prf_l = details_l["processed_pulse_repetition_frequency_hz"]
    prf_p = details_p["processed_pulse_repetition_frequency_hz"]
    res_az_l, res_az_p = details_l["azimuth_slc_resolution_m"], details_p["azimuth_slc_resolution_m"]
    ps_az_l, ps_az_p = details_l["azimuth_slc_pixel_spacing_m"], details_p["azimuth_slc_pixel_spacing_m"]
    pixels_az_l, pixels_az_p = details_l["azimuth_slc_window_size_pixels"], details_p["azimuth_slc_window_size_pixels"]
    meters_az_l, meters_az_p = details_l["azimuth_slc_window_size_m"], details_p["azimuth_slc_window_size_m"]
    looks_az_l, looks_az_p = details_l["azimuth_independent_looks"], details_p["azimuth_independent_looks"]
    param_str += f"Processed PRF & {prf_l:.0f} Hz & {prf_p:.0f} Hz \\\\ \n"
    param_str += f"Azimuth resolution & {res_az_l:.2f} m & {res_az_p:.2f} m \\\\ \n"
    param_str += f"Azimuth SLC pixel spacing & {ps_az_l:.2f} m & {ps_az_p:.2f} m \\\\ \n"
    param_str += f"Azimuth SLC window size & {pixels_az_l} pixel & {pixels_az_p} pixel \\\\ \n"
    param_str += f"Azimuth SLC window size & {meters_az_l:.1f} m & {meters_az_p:.1f} m \\\\ \n"
    param_str += f"Azimuth independent looks & {looks_az_l:.1f} & {looks_az_p:.1f} \\\\ \n"
    param_str += "\\hline \n"
    # total looks
    total_looks_l, total_looks_p = details_l["total_independent_looks"], details_p["total_independent_looks"]
    param_str += f"Total independent looks & {total_looks_l:.1f} & {total_looks_p:.1f} \\\\ \n"
    param_str += "\\hline \n"
    param_str += "\\end{tabular} \n"
    # save
    folder = paths.get_report_figure_folder()
    with open(folder / f"tab_proc_param_{site}.tex", "w") as file:
        file.write(param_str)


def tab_proc_param_main():
    tab_proc_param(sites.LOPE)
    tab_proc_param(sites.MABOUNIE)
    tab_proc_param(sites.MITZIC)
    tab_proc_param(sites.MONDAH)
    tab_proc_param(sites.PONGARA)


if __name__ == "__main__":
    tab_proc_param_main()
