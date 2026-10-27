import matplotlib

from afrisar2ml_report import config, hardware, paths, sites
from afrisar2ml_report.benchmarks import benchmark_rvog, benchmark_tsnn
from afrisar2ml_report.figures_report import (
    fig_coherence,
    fig_data_split,
    fig_dem,
    fig_incidence,
    fig_kz,
    fig_lidar_counts,
    fig_lidar_rh98,
    fig_lidar_zg,
    fig_mask_subreg,
    fig_offnadir,
    fig_pauli_rgb,
    fig_phase,
    fig_power,
    fig_rvog_hists_by_site,
    fig_rvog_maps,
    fig_tsnn_hists,
    fig_tsnn_maps,
    fig_worldcover,
    tab_baselines,
    tab_extent,
    tab_metrics_tsnn_rvog,
    tab_passes,
    tab_proc_param,
)


def afrisar2ml_report_main():
    matplotlib.use("Agg")

    print("afrisar2ml_report package")
    print("  dataset version   ", config.DS_VERSION)
    print("  benchmark version ", config.REPORT_VERSION)
    print("  project folder    ", paths.get_project_folder())
    print("  cache folder      ", paths.get_cache_folder())
    print("  figures folder    ", paths.get_report_figure_folder())
    print("  device            ", hardware.get_device())

    print("running benchmarks")
    benchmark_rvog.main_benchmark_rvog()
    benchmark_tsnn.main_benchmark_tsnn()

    print("generating figures")
    fig_coherence.fig_coherence(sites.LOPE, "L", "hh")
    fig_data_split.fig_data_split("L")
    fig_dem.fig_dem(sites.LOPE, "L")
    fig_incidence.fig_incidence(sites.LOPE, "L")
    fig_kz.fig_kz(sites.LOPE, "L")
    fig_lidar_counts.fig_lidar_counts(sites.LOPE, "L")
    fig_lidar_rh98.fig_lidar_rh98(sites.LOPE, "L")
    fig_lidar_zg.fig_lidar_zg(sites.LOPE, "L")
    fig_mask_subreg.fig_mask_subreg(sites.LOPE, "L")
    fig_offnadir.fig_offnadir(sites.LOPE, "L")
    fig_pauli_rgb.fig_pauli_rgb("L")
    fig_phase.fig_phase(sites.LOPE, "L", "hh")
    fig_power.fig_power(sites.LOPE, "L", "hh")
    fig_rvog_hists_by_site.fig_rvog_hists_by_site_main()
    fig_rvog_maps.fig_rvog_maps(config.TEST_SITE)
    fig_tsnn_hists.fig_tsnn_hists("L", config.SEED)
    fig_tsnn_hists.fig_tsnn_hists("P", config.SEED)
    fig_tsnn_maps.fig_tsnn_maps(config.TEST_SITE, config.SEED)
    fig_worldcover.fig_worldcover(sites.LOPE, "L")

    print("generating tables")
    tab_baselines.tab_baselines(sites.LOPE, "P")
    tab_baselines.tab_baselines(sites.MABOUNIE, "P")
    tab_baselines.tab_baselines(sites.MITZIC, "P")
    tab_baselines.tab_baselines(sites.MONDAH, "P")
    tab_baselines.tab_baselines(sites.PONGARA, "P")
    tab_extent.tab_extent()
    tab_metrics_tsnn_rvog.tab_metrics_tsnn(config.TEST_SITE, config.SEED, config.N_RUNS)
    tab_metrics_tsnn_rvog.tab_metrics_rvog(config.TEST_SITE)
    tab_passes.tab_passes_main()
    tab_proc_param.tab_proc_param_main()

    print("done")


if __name__ == "__main__":
    afrisar2ml_report_main()
