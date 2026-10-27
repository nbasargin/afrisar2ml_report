import matplotlib.pyplot as plt
import numpy as np

from afrisar2ml_report import matrix_transform, open_dataset, paths, sites
from afrisar2ml_report.figures_report import config_report
from afrisar2ml_report.plot_utils import layout_sites, mpl_styles, pauli_rgb


def fig_pauli_rgb(band):
    print("fig_pauli_rgb", band)
    mpl_styles.setup_matplotlib_report_figure_styles()

    site_names = [sites.LOPE, sites.MABOUNIE, sites.MITZIC, sites.MONDAH, sites.PONGARA]
    fig, axes = layout_sites.combined_site_layout_report()
    layout_sites.add_site_labels_report(fig)
    fig.suptitle(f"Pauli RGB, {band}-band")
    for site in site_names:
        print("  working on", site)
        ds = open_dataset.open_dataset(site)
        # use first pass for preview
        pass_a, _pass_b = config_report.get_pass_pair(site)
        name_ascii = pass_a.encode("ascii")
        cov = ds[f"cov_{band.lower()}"].sel(pass_a=name_ascii, pass_b=name_ascii).values
        inc = ds[f"incidence_{band.lower()}"].values
        cov_s0 = cov * np.sin(inc[:, :, np.newaxis, np.newaxis])  # convert from beta0 to sigma0
        c3 = matrix_transform.cov4_to_cov3(cov_s0)  # convert to 3x3 covariance
        t3 = matrix_transform.cov3_to_coh3(c3)  # convert to coherency matrix (Pauli basis)
        rgb = pauli_rgb.coherency_matrix_to_pauli_rgb(t3)  # convert to RGB
        # plot
        ax = axes[site]
        ax.imshow(rgb, interpolation="nearest")
    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_pauli_rgb_{band}.jpg")
    plt.close("all")


if __name__ == "__main__":
    fig_pauli_rgb(band="L")
