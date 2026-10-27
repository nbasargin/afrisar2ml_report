"""
Model adapted from https://github.com/wenyuyangit/TSNN

W. Yang, S. Vitale, H. Aghababaei, G. Ferraioli, V. Pascazio and G. Schirinzi
"A Deep Learning Solution for Height Estimation on a Forested Area Based on Pol-TomoSAR Data"
IEEE Transactions on Geoscience and Remote Sensing, vol. 61, 2023
doi: 10.1109/TGRS.2023.3274395

Notable changes:
- replace classification head with regression
"""

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TSNNnet9LRegression(nn.Module):
    def __init__(self, n_tracks, n_pols, n_nodes, model_id=None):
        super().__init__()
        self.n_tracks = n_tracks  # number of flight tracks
        self.n_pols = n_pols  # number of polarizations
        self.n_nodes = n_nodes  # model size (nodes/neurons per layer)
        self.model_id = model_id
        mpmb_size = n_tracks * n_pols
        input_channels = mpmb_size + 2 * (mpmb_size - 1)
        self.fc1 = nn.Linear(input_channels, n_nodes)
        self.fc2 = nn.Linear(n_nodes, n_nodes)
        self.fc3 = nn.Linear(n_nodes, n_nodes)
        self.fc4 = nn.Linear(n_nodes, n_nodes)
        self.fc5 = nn.Linear(n_nodes, n_nodes)
        self.fc6 = nn.Linear(n_nodes, n_nodes)
        self.fc7 = nn.Linear(n_nodes, n_nodes)
        self.fc8 = nn.Linear(n_nodes, n_nodes)
        self.fc9 = nn.Linear(n_nodes, 1)
        self._init_weights()

    def _init_weights(self):
        gain = nn.init.calculate_gain("relu")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = self.fc9(x).squeeze(-1)
        return x


def save_tsnn(model: TSNNnet9LRegression, save_to):
    model_state = model.state_dict()
    model_metadata = {
        "n_tracks": model.n_tracks,
        "n_pols": model.n_pols,
        "n_nodes": model.n_nodes,
        "model_id": model.model_id,
    }
    combined_state = {
        "model_state": model_state,
        "model_metadata": model_metadata,
    }
    torch.save(combined_state, save_to)


def load_tsnn(load_from, device: torch.device | None = None):
    combined_state = torch.load(load_from, weights_only=True, map_location=device)
    model_state = combined_state["model_state"]
    model_metadata = combined_state["model_metadata"]
    model = TSNNnet9LRegression(
        n_tracks=model_metadata["n_tracks"],
        n_pols=model_metadata["n_pols"],
        n_nodes=model_metadata["n_nodes"],
        model_id=model_metadata["model_id"],
    )
    model.load_state_dict(model_state)
    if device is not None:
        model.to(device=device)
    model.eval()
    return model


def tsnn_mpmb_matrix_to_feature_vector(mpmb: Tensor) -> Tensor:
    """
    Convert a batch of complex-valued MPMB matrices to a batch of feature vectors.

    mpmb : torch.Tensor
        Complex tensor of shape (*B, D, D), where D = N * P (N track, P polarizations)

    Returns
    -------
    torch.Tensor
        Real-valued tensor of shape (*B, M),
        where M = D + 2 * (D - 1)
    """
    # Diagonal: real part only
    diag_real = mpmb.diagonal(dim1=-2, dim2=-1).real  # (*B, D)
    # First row excluding (0,0)
    first_row = mpmb[..., 0, 1:]  # (*B, D-1)
    # Concatenate features
    features = torch.cat([diag_real, first_row.real, first_row.imag], dim=-1)
    return features
