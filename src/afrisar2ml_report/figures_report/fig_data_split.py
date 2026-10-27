import matplotlib.pyplot as plt
import numpy as np

from afrisar2ml_report import config, open_dataset, paths, sites
from afrisar2ml_report.plot_utils import layout_sites, mpl_styles


def fig_data_split(band):
    print("fig_data_split", band)
    mpl_styles.setup_matplotlib_report_figure_styles()

    site_names = [sites.LOPE, sites.MABOUNIE, sites.MITZIC, sites.MONDAH, sites.PONGARA]
    fig, axes = layout_sites.combined_site_layout_report()
    layout_sites.add_site_labels_report(fig)
    fig.suptitle(f"Training (blue), validation (orange), test (red) areas, {band}-band")
    for site in site_names:
        print("  working on", site)
        ds = open_dataset.open_dataset(site)
        # get masks
        mask_fsar = ds[f"mask_{band.lower()}"].values.astype(bool)
        mask_subregion = ds["mask_subregion"].values.astype(bool)
        # colors
        rgb = np.zeros((*mask_fsar.shape, 3), dtype=np.float32)
        train_color = [0.2, 0.4, 0.9]
        val_color = [1.0, 0.5, 0.0]
        test_color = [1.0, 0.1, 0.1]
        if site == config.TEST_SITE:
            # full site used for testing
            rgb[mask_fsar] = test_color
        else:
            # validation area: valid pixels AND inside subregion
            # training area: valid pixels AND outside subregion
            validation = mask_fsar & mask_subregion
            training = mask_fsar & ~mask_subregion
            rgb[training] = train_color
            rgb[validation] = val_color
        rgb[~mask_fsar] = np.nan
        # plot
        ax = axes[site]
        ax.imshow(rgb, interpolation="nearest")
    # save
    folder = paths.get_report_figure_folder()
    fig.savefig(folder / f"fig_data_split_{band}.jpg")
    plt.close("all")


if __name__ == "__main__":
    fig_data_split(band="L")
