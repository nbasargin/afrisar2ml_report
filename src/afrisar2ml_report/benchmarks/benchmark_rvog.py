import numpy as np

from afrisar2ml_report import config, matrix_transform, open_dataset, paths, sites
from afrisar2ml_report.rvog import rvog


def _rvog_height_path(site, band, pass_a, pass_b):
    fname = f"rvog_{site}_{band}_height_{pass_a}_{pass_b}_v{config.REPORT_VERSION}.npy"
    return paths.get_cache_folder() / fname


def get_pass_names_by_site(site, band):
    """
    Single-baseline results strongly depend on the choice of the tracks and the spatial baseline.
    Here, we choose the tracks that result in the mean (absolute) kz across the scene of about 0.09 - 0.13.
    For the most sites, this corresponds to a vertical baseline of 40m for P-band, and 20m for L-band.
    For Pongara, horizontal baselines have been flown, and we choose 30m for P-band and 15m for L-band.
    """
    pass_a, pass_b = {
        (sites.LOPE, "P"): ("23gabonx0402", "23gabonx0405"),  # mean_kz  0.090   h_bl   0.3   v_bl  40.7
        (sites.LOPE, "L"): ("23gabonx0402", "23gabonx0407"),  # mean_kz  0.130   h_bl   0.2   v_bl  19.9
        (sites.MABOUNIE, "P"): ("23gabonx0802", "23gabonx0807"),  # mean_kz  0.090   h_bl   0.4   v_bl  40.8
        (sites.MABOUNIE, "L"): ("23gabonx0802", "23gabonx0808"),  # mean_kz  0.127   h_bl   0.5   v_bl  19.8
        (sites.MITZIC, "P"): ("23gabonx0302", "23gabonx0305"),  # mean_kz  0.092   h_bl  -0.0   v_bl  40.7
        (sites.MITZIC, "L"): ("23gabonx0302", "23gabonx0306"),  # mean_kz  0.138   h_bl  -0.0   v_bl  20.3
        (sites.MONDAH, "P"): ("23gabonx0504", "23gabonx0507"),  # mean_kz  0.088   h_bl   0.2   v_bl  40.2
        (sites.MONDAH, "L"): ("23gabonx0504", "23gabonx0508"),  # mean_kz  0.130   h_bl   0.1   v_bl  19.9
        (sites.PONGARA, "P"): ("23gabonx0606", "23gabonx0609"),  # mean_kz -0.087   h_bl  30.1   v_bl  -0.7
        (sites.PONGARA, "L"): ("23gabonx0606", "23gabonx0605"),  # mean_kz  0.122   h_bl -14.9   v_bl  -0.6
    }[site, band]
    return pass_a, pass_b


def extract_dual_pol_cov(cov4):
    """
    Convert full-pol 4x4 lexicographic covariance to dual-pol (dihedral and volume) 2x2 Pauli covariance.
    """
    c3 = matrix_transform.cov4_to_cov3(cov4)  # convert 4x4 to 3x3 covariance (both full-pol lexicographic)
    t3 = matrix_transform.cov3_to_coh3(c3)  # convert to 3x3 coherency matrix (Pauli basis)
    t2 = t3[..., 1:3, 1:3]  # get covariance at two polarizations: HH-VV and HV
    return t2


def cache_rvog_height(site, band, pass_a, pass_b):
    height_path = _rvog_height_path(site, band, pass_a, pass_b)
    if height_path.exists():
        print(f"RVoG canopy height for {site}, {band}, {pass_a} - {pass_b} is already cached")
        return
    print(f"Caching RVoG canopy height for {site}, {band}, {pass_a} - {pass_b}")

    # load data
    ds = open_dataset.open_dataset(site)
    pass_ae = pass_a.encode("ascii")
    pass_be = pass_b.encode("ascii")
    ds_cov = ds[f"cov_{band.lower()}"]
    cov4_a = ds_cov.sel(pass_a=pass_ae, pass_b=pass_ae).values
    cov4_b = ds_cov.sel(pass_a=pass_be, pass_b=pass_be).values
    cov4_omega = ds_cov.sel(pass_a=pass_ae, pass_b=pass_be).values
    inc = ds[f"incidence_{band.lower()}"].values
    kz = ds[f"kz_{band.lower()}"].sel(pass_a=pass_ae, pass_b=pass_be).values

    # extract dual-pol covariance (Pauli basis, dihedral and volume)
    t2_a = extract_dual_pol_cov(cov4_a)
    t2_b = extract_dual_pol_cov(cov4_b)
    t2_omega = extract_dual_pol_cov(cov4_omega)

    # estimate height and cache results
    Hest = rvog.estimate_rvog_height(t2_a, t2_b, t2_omega, inc, kz)
    np.save(height_path, Hest)


def load_rvog_height(site, band, pass_a, pass_b):
    return np.load(_rvog_height_path(site, band, pass_a, pass_b))


def main_benchmark_rvog():
    print("Running RVoG benchmarks")
    # run inversion on all sites and bands, cache results
    for site in [sites.LOPE, sites.MABOUNIE, sites.MITZIC, sites.MONDAH, sites.PONGARA]:
        for band in ["L", "P"]:
            pass_a, pass_b = get_pass_names_by_site(site, band)
            cache_rvog_height(site, band, pass_a, pass_b)


if __name__ == "__main__":
    main_benchmark_rvog()
