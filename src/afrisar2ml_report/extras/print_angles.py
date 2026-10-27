import numpy as np

from afrisar2ml_report import open_dataset, sites


def print_min_max_offnadir(site, variable):
    ds = open_dataset.open_dataset(site)
    offnadir = ds[variable].values
    offnadir_min = np.rad2deg(np.nanmin(offnadir))
    offnadir_mean = np.rad2deg(np.nanmean(offnadir))
    offnadir_max = np.rad2deg(np.nanmax(offnadir))
    print(f"{site:8}  {variable}  min = {offnadir_min:.0f}°,  mean = {offnadir_mean:.0f}°,  max = {offnadir_max:.0f}°")


def main():
    print("Offnadir")
    print_min_max_offnadir(sites.LOPE, "offnadir_l")
    print_min_max_offnadir(sites.MABOUNIE, "offnadir_l")
    print_min_max_offnadir(sites.MITZIC, "offnadir_l")
    print_min_max_offnadir(sites.MONDAH, "offnadir_l")
    print_min_max_offnadir(sites.PONGARA, "offnadir_l")
    print("Incidence")
    print_min_max_offnadir(sites.LOPE, "incidence_l")
    print_min_max_offnadir(sites.MABOUNIE, "incidence_l")
    print_min_max_offnadir(sites.MITZIC, "incidence_l")
    print_min_max_offnadir(sites.MONDAH, "incidence_l")
    print_min_max_offnadir(sites.PONGARA, "incidence_l")


if __name__ == "__main__":
    main()
