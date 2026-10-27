import numpy as np

from afrisar2ml_report.tsnn import tsnn_dataset


def _check_tsnn_track_repeat():
    cov = np.random.randn(10, 20, 6, 6, 4, 4) + 1j * np.random.randn(10, 20, 6, 6, 4, 4)
    cov = cov.astype(np.complex64)
    cov_out = tsnn_dataset.repeat_tensor_tracks(cov, n_tracks=8)
    assert cov_out.shape == (10, 20, 8, 8, 4, 4)
    assert cov_out.dtype == np.complex64
    assert np.allclose(cov_out[:, :, 0:6, 0:6, :, :], cov)
    kz = np.random.randn(10, 20, 6, 6)
    kz = kz.astype(np.float32)
    kz_out = tsnn_dataset.repeat_tensor_tracks(kz, n_tracks=8)
    assert kz_out.shape == (10, 20, 8, 8)
    assert kz_out.dtype == np.float32
    assert np.allclose(kz_out[:, :, 0:6, 0:6], kz)
    print("TSNN track repeat ok")


if __name__ == "__main__":
    _check_tsnn_track_repeat()
