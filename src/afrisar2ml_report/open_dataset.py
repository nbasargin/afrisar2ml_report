import numpy as np
import rioxarray as rioxarray  # noqa: PLC0414
import xarray as xr

from afrisar2ml_report import paths


def open_dataset(site):
    zarr_path = paths.get_dataset_path(site)
    ds = xr.open_dataset(zarr_path, engine="zarr")
    return ds


def get_non_water_rh98(ds):
    """
    Read LVIS RH98 metrics, remove all pixels that belong to water, according to ESA WorldCover.
    """
    rh98 = ds["lvis_rh98"].values.copy()
    worldcover = ds["worldcover"].values
    rh98[worldcover == 80] = np.nan
    return rh98
