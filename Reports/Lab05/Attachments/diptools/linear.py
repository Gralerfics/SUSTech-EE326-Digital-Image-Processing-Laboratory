import numpy as np

from .regulator import NonRegulator


def convolution(img, kernel, pad_mode='constant', regulator=NonRegulator):
    size = kernel.shape[0]
    rad = size // 2
    img = np.pad(img.astype(np.int32), ((rad, rad), (rad, rad)), mode=pad_mode)
    kernel_spanned = np.expand_dims(np.reshape(kernel, -1), axis=(1, 2))
    pts = [(idx // size - rad, idx % size - rad) for idx in range(size * size)]
    moved = np.array([np.roll(np.roll(img, x, axis=0), y, axis=1) for (x, y) in pts])
    res = np.sum(moved * kernel_spanned, axis=0)
    return regulator(res[rad:-rad, rad:-rad])

