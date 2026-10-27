import numpy as np


def coherency_matrix_to_pauli_rgb(t3):
    hh_plus_vv = np.sqrt(2 * np.abs(t3[..., 0, 0]))
    hh_minus_vv = np.sqrt(2 * np.abs(t3[..., 1, 1]))
    two_hv = np.sqrt(2 * np.abs(t3[..., 2, 2]))
    r_max = np.nanmean(hh_minus_vv) * 2
    g_max = np.nanmean(two_hv) * 1.7
    b_max = np.nanmean(hh_plus_vv) * 2
    pauli_r = hh_minus_vv / r_max
    pauli_g = two_hv / g_max
    pauli_b = hh_plus_vv / b_max
    rgb = np.stack([pauli_r, pauli_g, pauli_b], axis=-1)
    gamma = 1.1
    rgb = rgb**gamma
    return np.clip(rgb, 0, 1).astype(np.float32)
