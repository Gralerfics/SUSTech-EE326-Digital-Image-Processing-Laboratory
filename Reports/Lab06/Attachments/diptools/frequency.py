import cv2
import numpy as np

from .histogram import gamma_correction
from .regulator import NonRegulator, GrayScalingRegulator


def shifting_mat(dim):
    return (-1) ** sum(np.meshgrid(range(dim[1]), range(dim[0])))


def pad_fh_in_pair(f, h, mode='min'):
    f_r, f_c = f.shape
    h_r, h_c = h.shape
    f_p, h_p = f, h
    if mode == 'min':
        fac_r, fac_c = f_r % 2, f_c % 2
        f_p = np.pad(f_p, ((0, h_r - 1 + fac_r), (0, h_c - 1 + fac_c)))
        h_p = np.pad(h_p, ((f_r // 2 + fac_r, f_r // 2 - 1 + fac_r), (f_c // 2 + fac_c, f_c // 2 - 1 + fac_c)))
    elif mode == 'double':
        f_p = np.pad(f_p, ((0, f_r), (0, f_c)))
        h_p = np.pad(h_p, ((f_r - h_r // 2, f_r - 1 - h_r // 2), (f_c - h_c // 2, f_c - 1 - h_c // 2)))
    return (f_p, h_p)


def magnitude_spectrum_disp(S, gamma=0.15):
    return gamma_correction(np.abs(S), gamma, regulator=GrayScalingRegulator)


def magnitude_phase_disp(S): # TODO
    return GrayScalingRegulator(np.angle(S))


def fft2d(f):
    # shifter = shifting_mat(f.shape)
    # return np.fft.fft2(f * shifter)
    return np.fft.fftshift(np.fft.fft2(f))


def ifft2d(G):
    # shifter = shifting_mat(G.shape)
    # return regulator(np.real(np.fft.ifft2(G)) * shifter)
    return np.real(np.fft.ifft2(np.fft.ifftshift(G)))


def compute_FH_in_pair(f, h, pad_mode='min'): # Shifted
    f_p, h_p = pad_fh_in_pair(f, h, pad_mode)
    shifter = shifting_mat(f_p.shape)
    return (fft2d(f_p), 1j * np.imag(fft2d(h_p)) * shifter)


def convolution_freq(f, h, regulator=NonRegulator):
    F, H = compute_FH_in_pair(f, h)
    g_p = ifft2d(F * H)
    return regulator(g_p[0 : f.shape[0], 0 : f.shape[1]])


def filter_freq(f, H, regulator=NonRegulator): # No padding.
    f_r, f_c = f.shape
    F = fft2d(f)
    G = F * H
    g = ifft2d(G)
    return regulator(g)


def filter_freq_pad(f, H, is_H_padded=False, regulator=NonRegulator): # Double padded.
    f_r, f_c = f.shape
    f_p = np.pad(f, ((0, f_r), (0, f_c)))
    H_p = H
    if not is_H_padded:
        H_p = np.pad(H_p, (((f_r + 1) // 2, f_r // 2), ((f_c + 1) // 2, f_c // 2)))
    return regulator(filter_freq(f_p, H_p, regulator=regulator)[:f_r, :f_c])


def full_inverse_filter(F, H, regulator=GrayScalingRegulator):
    return regulator(ifft2d(F / H))


def radially_limited_inverse_filter(F, H, radius, regulator=GrayScalingRegulator):
    H_tmp = np.copy(H)
    r, c = F.shape
    H_tmp[np.sum((np.array(np.meshgrid(range(c), range(r))) - np.array([np.ones((r, c)) * c // 2, np.ones((r, c)) * r // 2])) ** 2, axis=0) > radius * radius] = 1
    return full_inverse_filter(F, H_tmp, regulator=regulator)


def wiener_filter(F, H, K, regulator=GrayScalingRegulator):
    H2 = H * np.conj(H)
    return regulator(ifft2d(F * H2 / (H * (H2 + K))))


def H_ideal_LPF(dim, radius, double_pad=False):
    if double_pad:
        dim = (dim[0] * 2, dim[1] * 2)
    o_x, o_y = dim[1] // 2 + 1, dim[0] // 2 + 1
    res = np.zeros(dim)
    cv2.circle(res, (o_x, o_y), radius, 1, -1)
    return res


def H_butterworth_LPF(dim, order, d_cutoff, double_pad=False):
    if double_pad:
        dim = (dim[0] * 2, dim[1] * 2)
    o_x, o_y = dim[1] // 2 + 1, dim[0] // 2 + 1
    x_mesh, y_mesh = np.meshgrid(range(dim[1]), range(dim[0]))
    D2 = (x_mesh - o_x) ** 2 + (y_mesh - o_y) ** 2
    return 1 / (1 + (D2 / d_cutoff ** 2) ** order)


def H_gauss_LPF(dim, sigma, double_pad=False):
    if double_pad:
        dim = (dim[0] * 2, dim[1] * 2)
    o_x, o_y = dim[1] // 2 + 1, dim[0] // 2 + 1
    x_mesh, y_mesh = np.meshgrid(range(dim[1]), range(dim[0]))
    D2 = (x_mesh - o_x) ** 2 + (y_mesh - o_y) ** 2
    return np.exp(-D2 / (2 * sigma ** 2))


def H_butterworth_NRF(dim, position_wrt_center, order, d_cutoff, double_pad=False): # Notch reject filter.
    if double_pad:
        dim = (dim[0] * 2, dim[1] * 2)
        position_wrt_center = [position_wrt_center[0] * 2, position_wrt_center[1] * 2]
        d_cutoff = d_cutoff * 2 # TODO: ?
    o_x, o_y = dim[1] // 2 + 1, dim[0] // 2 + 1
    x_mesh, y_mesh = np.meshgrid(range(dim[1]), range(dim[0]))
    D2 = (x_mesh - o_x + 0.5 - position_wrt_center[0]) ** 2 + (y_mesh - o_y + 0.5 - position_wrt_center[1]) ** 2
    res = 1 / (1 + (d_cutoff ** 2 / D2) ** order)
    res = res * np.fliplr(np.flipud(res))
    return res

