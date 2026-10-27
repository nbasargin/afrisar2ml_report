"""
The code is adapted from the DLR/ESA PolInSAR course (2026 edition).
https://eo4society.esa.int/event/10th-dlr-esa-open-polinsar-training-course-2026/

Single-baseline dual-pol PolInSAR RVoG canopy height inversion.
"""

import numpy as np

MAX_HEIGHT = 50


def matmul_2(a11, a12, a21, a22, b11, b12, b21, b22):
    """
    Implements the multiplication A.B of two 2 x 2 matrices A and B.
    Inputs are the elements of matrices (row-wise).
    Outputs are the elements of the matrix product (row-wise).
    """
    c11 = a11 * b11 + a12 * b21
    c12 = a11 * b12 + a12 * b22
    c21 = a21 * b11 + a22 * b21
    c22 = a21 * b12 + a22 * b22
    return c11, c12, c21, c22


def sqrt_inverse(a11, a12, a22):
    """
    Given a 2 x 2 Hermitian matrix A, calculates A**(-1/2) using eigendecomposition
    Inputs are the upper-diagonal matrix elements (row-wise) - NumPy arrays.
    Outputs are the elements of the sqrt inverse matrix.
    Uses function matmul_2.
    """
    # Calculate eigenvalues (l1, l2)
    sigma = 0.5 * (a11 + a22)
    D = a11 * a22 - a12 * np.conj(a12)
    delta = np.sqrt(sigma**2 - D)
    l1 = sigma + delta
    l2 = sigma - delta

    # Calculate eigenvector matrix elements (eigenvectors are the columns)
    norm_ = np.sqrt(a12 * np.conj(a12) + ((a11 - a22) / 2 - delta) ** 2)
    v11 = -a12 / norm_
    v21 = ((a11 - a22) / 2 - delta) / norm_
    v12 = ((a11 - a22) / 2 - delta) / norm_
    v22 = np.conj(a12) / norm_

    # calculate sqrt inverse matrix elements
    zz = np.zeros(l1.shape, np.float32)
    aux11, aux12, aux21, aux22 = matmul_2(v11, v12, v21, v22, 1 / np.sqrt(l1), zz, zz, 1 / np.sqrt(l2))
    i11, i12, i21, i22 = matmul_2(aux11, aux12, aux21, aux22, np.conj(v11), np.conj(v21), np.conj(v12), np.conj(v22))
    return i11, i12, i21, i22


def eigenvalvect_2(a11, a12, a21, a22):
    """
    Calculates eigenvalues / eigenvectors of a generic complex-valued 2 x 2 matrix.
    Inputs are the matrix elements (row-wise) - NumPy arrays.
    Outputs are the eigenvalues (l1, l2), and the eigenvector (E1, E2) elements - NumPy arrays.
    E1 = [e11, e21]
    E2 = [e12, e22]
    """
    # eigenvalues
    l1 = 0.5 * ((a11 + a22) + np.sqrt((a11 - a22) ** 2 + 4 * a12 * a21))
    l2 = 0.5 * ((a11 + a22) - np.sqrt((a11 - a22) ** 2 + 4 * a12 * a21))

    # eigenvector 1
    e11 = -a12
    e21 = a11 - l1

    # normalize eigenvector 1
    norm_ = np.sqrt(np.abs(e11) ** 2 + np.abs(e21) ** 2)
    e11 = e11 / norm_
    e21 = e21 / norm_

    # eigenvector 2
    e12 = a22 - l2
    e22 = -a21

    # normalize eigenvector 2
    norm_ = np.sqrt(np.abs(e12) ** 2 + np.abs(e22) ** 2)
    e12 = e12 / norm_
    e22 = e22 / norm_
    return l1, l2, e11, e21, e12, e22


def pre_whitening(t2_a, t2_b, t2_omega):
    """
    Apply pre-whitening to the dual-pol covariance and cross-covariances.
    t2_a, t2_b are the 2x2 covariances of individual tracks.
    t2_omega is the interferometric 2x2 cross-covariance between tracks.
    """
    # average of t2_a and t2_b, element-wise
    t2_ab_avg = 0.5 * (t2_a + t2_b)
    T11 = t2_ab_avg[..., 0, 0]
    T12 = t2_ab_avg[..., 0, 1]
    T22 = t2_ab_avg[..., 1, 1]

    # omega, element-wise
    Om_11 = t2_omega[..., 0, 0]
    Om_12 = t2_omega[..., 0, 1]
    Om_21 = t2_omega[..., 1, 0]
    Om_22 = t2_omega[..., 1, 1]

    # calculate the elements of T**(-1/2)
    iT11, iT12, iT21, iT22 = sqrt_inverse(T11, T12, T22)

    # whiten, two 2 x 2 multiplications ==> Matrix P
    aux11, aux12, aux21, aux22 = matmul_2(iT11, iT12, iT21, iT22, Om_11, Om_12, Om_21, Om_22)
    P11, P12, P21, P22 = matmul_2(aux11, aux12, aux21, aux22, iT11, iT12, iT21, iT22)
    return P11, P12, P21, P22


def optimize_coherence_and_phase(P11, P12, P21, P22):
    """
    Performs the Polar decomposition of the 2 by 2 matrix P with the given elements
    and estimates coherences with min max phase and absolute value.

    The polar decomposition of P is P = U * J.
    The max & min coherence is extracted from the eigenvectors of J and the
    max & min phase from the eigenvectors of U.

    See: https://en.wikipedia.org/wiki/Polar_decomposition
    """

    # Calculate J^2 = P^(*T) * P
    J11, J12, J21, J22 = matmul_2(np.conj(P11), np.conj(P21), np.conj(P12), np.conj(P22), P11, P12, P21, P22)

    # Calculate eigenvalues and eigenvectors of J
    l1, l2, v11, v21, v12, v22 = eigenvalvect_2(J11, J12, J21, J22)

    # Calculate V^(*T) * P * V --> gamma max & min from there
    aux11, aux12, aux21, aux22 = matmul_2(np.conj(v11), np.conj(v21), np.conj(v12), np.conj(v22), P11, P12, P21, P22)
    gamma_max, gamma12, gamma21, gamma_min = matmul_2(aux11, aux12, aux21, aux22, v11, v12, v21, v22)

    # Compute U = P * J^(-1)
    Ji11, Ji1, Ji21, Ji22 = sqrt_inverse(J11, J12, J22)
    U11, U12, U21, U22 = matmul_2(P11, P12, P21, P22, Ji11, Ji1, Ji21, Ji22)

    # Calculate eigenvalues and eigenvectors of U
    l1, l2, w11, w21, w12, w22 = eigenvalvect_2(U11, U12, U21, U22)

    # Calculate W^(*T) * P * W --> gamma phamax & phamin from there
    aux11, aux12, aux21, aux22 = matmul_2(np.conj(w11), np.conj(w21), np.conj(w12), np.conj(w22), P11, P12, P21, P22)
    gamma_phamax, gamma12, gamma21, gamma_phamin = matmul_2(aux11, aux12, aux21, aux22, w11, w12, w21, w22)

    # Return optimum coherences
    return gamma_phamax, gamma_phamin, gamma_max, gamma_min


def order_phase(gamma_phamax, gamma_phamin, kz):
    """
    Order phase values
      assumption: phase = kz * height
      swap if height(gamma_phamax) > height(gamma_phamin)
    """
    height_diff = np.angle(gamma_phamax * np.conj(gamma_phamin)) / kz
    idx = np.where(height_diff < 0)
    gamma_phamax = gamma_phamax.copy()
    gamma_phamin = gamma_phamin.copy()
    gamma_phamax[idx], gamma_phamin[idx] = gamma_phamin[idx], gamma_phamax[idx]
    return gamma_phamax, gamma_phamin


def calc_ground_phase(gamma_v, gamma_phamin):
    v = gamma_phamin - gamma_v
    x = (
        -np.real(np.conj(v) * gamma_v) / np.abs(v) ** 2
        + np.sqrt(np.real(np.conj(v) * gamma_v) ** 2 - (np.abs(gamma_v) ** 2 - 1) * np.abs(v) ** 2) / np.abs(v) ** 2
    )
    gamma_g = gamma_v + v * x
    phi0 = np.angle(gamma_g)
    return phi0, gamma_g


def invert_pixel_with_lut(Hmat, smat, incang, kz, gamma_v0):
    """
    Invert height for a single given pixel.
    Construct LUT, find the best match.
    """
    p1 = 2 * smat / np.cos(incang)
    p2 = 2 * smat / np.cos(incang) + 1j * kz
    lut = (p1 * (np.exp(p2 * Hmat) - 1)) / (p2 * (np.exp(p1 * Hmat) - 1))
    lut[Hmat == 0] = 1
    HH = Hmat[np.logical_and(smat < 1e-6, Hmat != 0)]

    lut[np.logical_and(smat < 1e-6, Hmat != 0)] = np.sinc(kz * HH / 2 / np.pi) * np.exp(1j * kz * HH / 2)

    # get estimate
    func = np.abs(gamma_v0 - lut)
    imin = np.unravel_index(np.argmin(func, axis=None), func.shape)
    return Hmat[imin]


def prepare_lut_parameter_meshgrids():
    # make meshgrids for LUT parameters
    H = np.linspace(0, MAX_HEIGHT, MAX_HEIGHT * 2)
    s = np.tan(np.linspace(0.00000001, 0.99, 31) * np.pi / 2) / 30
    Hmat, smat = np.meshgrid(H, s)
    return Hmat, smat


def estimate_rvog_height(t2_a, t2_b, t2_omega, inc, kz):
    """
    Estimate height from single-baseline PolInSAR data using the RVoG model.
    Dual-pol inversion is implemented here, the polarizations are (HH - VV) and HV.
    """
    with np.errstate(over="ignore", invalid="ignore"):
        # get optimized coherences
        P11, P12, P21, P22 = pre_whitening(t2_a, t2_b, t2_omega)
        gamma_phamax, gamma_phamin, gamma_max, gamma_min = optimize_coherence_and_phase(P11, P12, P21, P22)

        # order phase
        gamma_phamax, gamma_phamin = order_phase(gamma_phamax, gamma_phamin, kz)

        # decide gamma_v
        gamma_v = gamma_phamax

        # ground phase and coherence
        phi0, gamma_g = calc_ground_phase(gamma_v, gamma_phamin)

        # volume-only coherence
        gamma_v0 = gamma_v * np.conj(gamma_g)

        # make meshgrids for LUT parameters
        Hmat, smat = prepare_lut_parameter_meshgrids()

        # define output array
        Hest = np.full_like(kz, fill_value=np.nan)

        # invert each pixel
        for i in range(Hest.shape[0]):
            for j in range(Hest.shape[1]):
                if np.isnan(kz[i, j]):
                    continue
                Hest[i, j] = invert_pixel_with_lut(Hmat, smat, inc[i, j], kz[i, j], gamma_v0[i, j])

        # remove estimations with max height
        Hest[Hest >= MAX_HEIGHT] = np.nan

    return Hest
