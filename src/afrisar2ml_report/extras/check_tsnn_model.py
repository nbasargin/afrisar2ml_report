import torch

from afrisar2ml_report import paths
from afrisar2ml_report.tsnn import tsnn_model


def _check_shapes():
    n_samples = 64
    n_pols = 3
    n_tracks = 8
    d = n_pols * n_tracks
    mpmb = torch.randn((n_samples, d, d)) + 1j * torch.randn((n_samples, d, d))
    mpmb = mpmb @ mpmb.mH  # make matrix hermitian
    # check feature size
    features = tsnn_model.tsnn_mpmb_matrix_to_feature_vector(mpmb)
    n_features = d + 2 * (d - 1)
    assert features.shape == (n_samples, n_features)
    # check model out shape
    model = tsnn_model.TSNNnet9LRegression(n_tracks, n_pols, 400)
    out = model.forward(features)
    assert out.shape == (n_samples,)
    print("TSNN forward pass shapes ok")


def _check_load_save():
    n_samples = 64
    n_pols = 3
    n_tracks = 8
    d = n_pols * n_tracks
    mpmb = torch.randn((n_samples, d, d)) + 1j * torch.randn((n_samples, d, d))
    mpmb = mpmb @ mpmb.mH  # make matrix hermitian
    features = tsnn_model.tsnn_mpmb_matrix_to_feature_vector(mpmb)
    model = tsnn_model.TSNNnet9LRegression(n_tracks, n_pols, 400)
    # save and load
    pth = paths.get_cache_folder() / "_test_model.pth"
    tsnn_model.save_tsnn(model, pth)
    loaded_model = tsnn_model.load_tsnn(pth)
    # check that forward pass results are equal
    result_1 = model.forward(features)
    result_2 = loaded_model.forward(features)
    assert torch.allclose(result_1, result_2)
    print("TSNN load save ok")


if __name__ == "__main__":
    _check_shapes()
    _check_load_save()
