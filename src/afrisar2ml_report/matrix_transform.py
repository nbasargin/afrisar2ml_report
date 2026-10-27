import numpy as np


def cov4_to_cov3(C4: np.ndarray) -> np.ndarray:
    """
    Batched conversion from 4x4 lexicographic covariance to 3x3 reduced covariance.
    Input shape:  (*B, 4, 4)
    Output shape: (*B, 3, 3)
    """
    HH, HV, VH, VV = 0, 1, 2, 3
    C3 = np.zeros((*C4.shape[:-2], 3, 3), dtype=C4.dtype)
    # diagonal, reciprocity enforced for HV-HV block
    C3[..., 0, 0] = C4[..., HH, HH]
    C3[..., 1, 1] = 0.5 * (C4[..., HV, HV] + C4[..., VH, VH] + C4[..., HV, VH] + C4[..., VH, HV])
    C3[..., 2, 2] = C4[..., VV, VV]
    # cross terms
    C3[..., 0, 1] = (C4[..., HH, HV] + C4[..., HH, VH]) / np.sqrt(2)
    C3[..., 0, 2] = C4[..., HH, VV]
    C3[..., 1, 2] = (C4[..., HV, VV] + C4[..., VH, VV]) / np.sqrt(2)
    # cross terms, hermitian copy
    C3[..., 1, 0] = np.conj(C3[..., 0, 1])
    C3[..., 2, 0] = np.conj(C3[..., 0, 2])
    C3[..., 2, 1] = np.conj(C3[..., 1, 2])
    return C3


def cov3_to_coh3(C3: np.ndarray) -> np.ndarray:
    """
    Batched conversion from 3x3 covariance to 3x3 coherency (Pauli basis).
    Input shape:  (*B, 3, 3)
    Output shape: (*B, 3, 3)
    """
    U = (1 / np.sqrt(2)) * np.array([[1, 0, 1], [1, 0, -1], [0, np.sqrt(2), 0]], dtype=C3.dtype)
    return U @ C3 @ U.conj().T  # np.matmul handles broadcasting automatically
