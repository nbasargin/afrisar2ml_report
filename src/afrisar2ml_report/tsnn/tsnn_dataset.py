import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from afrisar2ml_report import open_dataset


class TSNNDataset(Dataset):
    def __init__(
        self,
        mpmb: Tensor,  # shape (D, N*P, N*P) complex64
        height: Tensor,  # shape (D,) float32
        ds_id: str,
    ):
        self.mpmb = mpmb
        self.height = height
        self.ds_id = ds_id

    def __len__(self):
        return self.mpmb.shape[0]

    def __getitem__(self, idx):
        return self.mpmb[idx], self.height[idx]

    def check_shapes(self):
        assert len(self.mpmb.shape) == 3
        D, NP1, NP2 = self.mpmb.shape
        assert NP1 == NP2
        assert self.height.shape == (D,)

    def check_dtypes(self):
        assert self.mpmb.dtype == torch.complex64
        assert self.height.dtype == torch.float32


def repeat_tensor_tracks(small_tensor: np.ndarray, n_tracks: int):
    """
    Repeat first track to pad the tensor to the specific number of tracks.
    """
    x1, x2, n1, n2, *p = small_tensor.shape
    assert n1 == n2
    assert n_tracks >= n1
    if n_tracks == n1:
        return small_tensor  # tensor already has exactly n tracks
    expanded_shape = (x1, x2, n_tracks, n_tracks, *p)
    expanded_tensor = np.zeros(expanded_shape, dtype=small_tensor.dtype)
    for track_i in range(n_tracks):
        for track_j in range(n_tracks):
            # repeat first track to reach min_n_tracks in total
            src_i = track_i if track_i < n1 else 0
            src_j = track_j if track_j < n2 else 0
            expanded_tensor[:, :, track_i, track_j] = small_tensor[:, :, src_i, src_j]
    return expanded_tensor


def load_tsnn_site_data(site, band, n_tracks: int | None):
    print(f"load_tsnn_data, {site}, {band}-band")
    ds = open_dataset.open_dataset(site)
    cov: np.ndarray = ds[f"cov_{band.lower()}"].values
    rh98 = ds["lvis_rh98"].values
    mask_subregion = ds["mask_subregion"].values.astype(np.bool)
    # repeat tracks if necessary
    if n_tracks is not None:
        cov = repeat_tensor_tracks(cov, n_tracks)
    # reshape to mpmb matrix
    d1, d2, n1, n2, p1, p2 = cov.shape
    cov = cov.transpose(0, 1, 2, 4, 3, 5)  # shape (d1, d2, n1, p1, n2, p2)
    cov = cov.reshape((d1, d2, n1 * p1, n2 * p2))
    mpmb = torch.tensor(cov)
    height = torch.tensor(rh98)
    mask_subregion = torch.tensor(mask_subregion)
    return mpmb, height, mask_subregion


def convert_data_to_dataset(mpmb, height, mask_subregion, site, band, area_subset):
    # masks
    finite_pixels = torch.isfinite(height) & torch.isfinite(mpmb[:, :, 0, 0])
    if area_subset == "all":
        ds_pixels = finite_pixels
    else:
        if area_subset == "train":
            ds_pixels = finite_pixels & (~mask_subregion)
        elif area_subset == "validation":
            ds_pixels = finite_pixels & mask_subregion
        else:
            raise ValueError(f"Unknown area_subset = {area_subset}, must be one of 'train', 'validation', or 'all'")
    mpmb = mpmb[ds_pixels]
    height = height[ds_pixels]
    ds = TSNNDataset(
        mpmb=mpmb,
        height=height,
        ds_id=f"ds_{site}_{band}_{area_subset}",
    )
    ds.check_shapes()
    ds.check_dtypes()
    return ds


def merge_tsnn_datasets(ds_list: list[TSNNDataset], merged_id: str):
    mpmb = torch.cat([ds.mpmb for ds in ds_list])
    height = torch.cat([ds.height for ds in ds_list])
    ds = TSNNDataset(
        mpmb=mpmb,
        height=height,
        ds_id=merged_id,
    )
    ds.check_shapes()
    ds.check_dtypes()
    return ds
