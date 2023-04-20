import numpy as np

from .regulator import NonRegulator





def window_process(img, kernel, center=None, process_func=None, pad_mode='constant', inter_type=np.int32, regulator=NonRegulator):
    def pf_sum(x): return np.sum(x, axis=0)
    size = kernel.shape
    rad = np.array(size) // 2
    if center is None: center = rad
    if process_func is None: process_func = pf_sum
    img = np.pad(img.astype(inter_type), ((rad[0], rad[0]), (rad[1], rad[1])), mode=pad_mode)
    kernel_spanned = np.expand_dims(np.reshape(kernel, -1), axis=(1, 2))
    pts = [(center[0] - idx // size[1], center[1] - idx % size[1]) for idx in range(size[0] * size[1])]
    moved = np.array([np.roll(np.roll(img, x, axis=0), y, axis=1) for (x, y) in pts])
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

