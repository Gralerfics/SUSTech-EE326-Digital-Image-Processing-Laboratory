import numpy as np

from .spatial import convolution
from .regulator import NonRegulator, GrayCuttingRegulator


def laplacian(img, regulator=NonRegulator):
    res = convolution(img, np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ]))
    return regulator(res)


def sobel_grad(img, regulator=NonRegulator):
    kernel_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    kernel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    res = np.abs(convolution(img, kernel_x)) + np.abs(convolution(img, kernel_y))
    return regulator(res)


def average_filter(img, radius, regulator=NonRegulator):
    size = radius * 2 - 1
    kernel = np.ones((size, size))
    res = convolution(img, kernel) / (size * size)
    return regulator(res)


def combined_spatial_sharpening(img, k, regulator=GrayCuttingRegulator):
    img_laplacian = laplacian(img)
    img_addition = img + (-1) * img_laplacian
    img_sobel = sobel_grad(img)
    img_average = average_filter(img_sobel, 3)
    img_product = img_addition * img_average / 255
    res = img + k * img_product
    return regulator(res)

