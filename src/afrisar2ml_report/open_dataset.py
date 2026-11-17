from pathlib import Path

import numpy as np
import rioxarray as rioxarray  # noqa: PLC0414  (registers the .rio accessor)
import xarray as xr
from zarr.storage import ZipStore


def open_dataset(path: str | Path) -> xr.Dataset:
    """Open a site dataset lazily from a zipped Zarr v3 store (or an unzipped .zarr folder)."""
    path = Path(path)
    store = path if path.is_dir() else ZipStore(path, mode="r")
    return xr.open_dataset(store, engine="zarr", consolidated=True)  # type: ignore


def get_non_water_rh98(ds):
    """
    Read LVIS RH98 metrics, remove all pixels that belong to water, according to ESA WorldCover.
    """
    rh98 = ds["lvis_rh98"].values.copy()
    worldcover = ds["worldcover"].values
    rh98[worldcover == 80] = np.nan
    return rh98
