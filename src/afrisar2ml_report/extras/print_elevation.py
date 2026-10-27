import numpy as np

from afrisar2ml_report import open_dataset, sites


def print_min_max_dem(site):
    ds = open_dataset.open_dataset(site)
    dem = ds["slantdem_l"].values
    print(f"{site:8}  min = {np.nanmin(dem):.0f},  mean = {np.nanmean(dem):.0f},  max = {np.nanmax(dem):.0f}")


if __name__ == "__main__":
    print_min_max_dem(sites.LOPE)
    print_min_max_dem(sites.MABOUNIE)
    print_min_max_dem(sites.MITZIC)
    print_min_max_dem(sites.MONDAH)
    print_min_max_dem(sites.PONGARA)
