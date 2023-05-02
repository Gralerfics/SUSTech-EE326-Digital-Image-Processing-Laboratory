import numpy as np

from .regulator import NonRegulator


def window_process(img, kernel, center=None, process_func=None, pad_mode='constant', inter_type=np.int32, regulator=NonRegulator):
    def pf_sum(x): return np.sum(x, axis=0)
    size = kernel.shape
    rad = np.array(size) // 2
    if center is None: center = rad
    if process_func is None: process_func = pf_sum
    img_pad = np.pad(img.astype(inter_type), ((rad[0], rad[0]), (rad[1], rad[1])), mode=pad_mode)
    kernel_spanned = np.expand_dims(np.reshape(kernel, -1), axis=(1, 2))
    pts = [(center[0] - idx // size[1], center[1] - idx % size[1]) for idx in range(size[0] * size[1])]
    moved = np.array([np.roll(np.roll(img_pad, x, axis=0), y, axis=1) for (x, y) in pts])
    res = process_func(moved * kernel_spanned)
    return regulator(res[rad[0]:-rad[0], rad[1]:-rad[1]])


def convolution(img, kernel, pad_mode='constant', regulator=NonRegulator):
    size = kernel.shape[0]
    rad = size // 2
    return window_process(img, kernel, center=(rad, rad), pad_mode=pad_mode, regulator=regulator)

    # size = kernel.shape[0]
    # rad = size // 2
    # img = np.pad(img.astype(np.int32), ((rad, rad), (rad, rad)), mode=pad_mode)
    # kernel_spanned = np.expand_dims(np.reshape(kernel, -1), axis=(1, 2))
    # pts = [(idx // size - rad, idx % size - rad) for idx in range(size * size)]
    # moved = np.array([np.roll(np.roll(img, x, axis=0), y, axis=1) for (x, y) in pts])
    # res = np.sum(moved * kernel_spanned, axis=0)
    # return regulator(res[rad:-rad, rad:-rad])


def arithmetic_mean_filter(img, kernel_dim, pad_mode='constant', regulator=NonRegulator):
    return window_process(img, np.ones(kernel_dim) / kernel_dim[0] / kernel_dim[1], pad_mode=pad_mode, regulator=regulator)


def geometric_mean_filter(img, kernel_dim, pad_mode='constant', regulator=NonRegulator):
    def pf_prod(x): return np.prod(x, axis=0)
    return regulator(window_process(img ** (1.0 / kernel_dim[0] / kernel_dim[1]), np.ones(kernel_dim), process_func=pf_prod, pad_mode=pad_mode, inter_type=np.float32))


def harmonic_mean_filter(img, kernel_dim, pad_mode='constant', regulator=NonRegulator):
    return regulator(kernel_dim[0] * kernel_dim[1] / window_process(1.0 / img, np.ones(kernel_dim), pad_mode=pad_mode, inter_type=np.float32))


def contraharmonic_mean_filter(img, kernel_dim, q, pad_mode='constant', regulator=NonRegulator):
    u = window_process(np.float_power(img, q + 1), np.ones(kernel_dim), pad_mode=pad_mode, inter_type=np.float32)
    d = window_process(np.float_power(img, q), np.ones(kernel_dim), pad_mode=pad_mode, inter_type=np.float32)
    return regulator(u / d)

def median_filter(img, kernel_dim, pad_mode='constant', regulator=NonRegulator):
    def pf_median(x): return np.median(x, axis=0)
    return window_process(img, np.ones(kernel_dim), process_func=pf_median, pad_mode=pad_mode, regulator=regulator)


def max_filter(img, kernel_dim, pad_mode='constant', regulator=NonRegulator):
    def pf_max(x): return np.max(x, axis=0)
    return window_process(img, np.ones(kernel_dim), process_func=pf_max, pad_mode=pad_mode, regulator=regulator)


def min_filter(img, kernel_dim, pad_mode='constant', regulator=NonRegulator):
    def pf_min(x): return np.min(x, axis=0)
    return window_process(img, np.ones(kernel_dim), process_func=pf_min, pad_mode=pad_mode, regulator=regulator)


def midpoint_filter(img, kernel_dim, pad_mode='constant', regulator=NonRegulator): # ?
    def pf_midpoint(x): return (np.min(x, axis=0) + np.min(x, axis=0)) / 2
    return window_process(img, np.ones(kernel_dim), process_func=pf_midpoint, pad_mode=pad_mode, regulator=regulator)


def alpha_trimmed_mean_filter(img, kernel_dim, d, pad_mode='constant', regulator=NonRegulator):
    pass


def adaptive_local_noise_reduction_filter(img, kernel_dim, V_noise, pad_mode='constant', regulator=NonRegulator):
    def pf_var(x): return np.var(x, axis=0)
    m_L = window_process(img, np.ones(kernel_dim) / kernel_dim[0] / kernel_dim[1], pad_mode=pad_mode, inter_type=np.float32)
    V_L = window_process(img, np.ones(kernel_dim), process_func=pf_var, pad_mode=pad_mode, inter_type=np.float32)
    return regulator(img - (img - m_L) * V_noise / V_L)


def adaptive_median_filter(img, s_max, pad_mode='constant', regulator=NonRegulator):
    r_max = s_max // 2
    img_pad = np.pad(img.astype(np.int32), ((r_max, r_max), (r_max, r_max)), mode=pad_mode)
    res = np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r, c = i + r_max, j + r_max
            r_now = 1
            while True:
                z_xy = img_pad[r, c]
                z_win = img_pad[r - r_now : r + r_now + 1, c - r_now : c + r_now + 1]
                z_min = np.min(z_win)
                z_max = np.max(z_win)
                z_med = np.median(z_win).astype(np.int32)
                if z_med == z_min or z_med == z_max:
                    r_now += 1
                    if r_now > r_max:
                        res[i, j] = z_med
                        break
                else:
                    if z_xy == z_min or z_xy == z_max:
                        res[i, j] = z_med
                    else:
                        res[i, j] = z_xy
                    break
    return regulator(res)

