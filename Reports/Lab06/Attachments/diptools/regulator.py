import numpy as np


def NonRegulator(x):
    return x


def GrayCuttingRegulator(x):
    return np.clip(x, 0, 255).astype(np.uint8)


def GrayScalingRegulator(x):
    l, r = np.min(x), np.max(x)
    return GrayCuttingRegulator(256 * (x - l) / (r - l))


def GrayScalingToRegulator(low, high):
    def func_raw(x):
        l, r = np.min(x), np.max(x)
        return GrayCuttingRegulator(low + (high - low) * (x - l) / (r - l))

    return func_raw

