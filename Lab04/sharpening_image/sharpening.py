import numpy as np


def convolution(img, kernel):
    assert kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2 == 1
    size = kernel.shape[0]
    radius = size // 2 + 1
    img = np.pad(img.astype(np.int32), ((radius - 1, radius - 1), (radius - 1, radius - 1)), 'reflect')

    pts = [(idx // size - radius + 1, idx % size - radius + 1) for idx in range(size * size)]
    mvd = np.array([np.roll(np.roll(img, x, axis=0), y, axis=1) for (x, y) in pts])
    ker = np.expand_dims(np.reshape(kernel, -1), axis=(1, 2))
    res = np.sum(mvd * ker, axis=0)

    res = res[(radius - 1):(1 - radius), (radius - 1):(1 - radius)]
    return res


def laplacian(img):
    kernel = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ])
    res = convolution(img.astype(np.int32), kernel)
    return res


def sobel_gradient(img):
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
    return res


def average_filtering(img, radius):
    assert radius % 2 == 1
    size = radius * 2 - 1
    kernel = np.ones((size, size))
    res = convolution(img, kernel) / (size * size)
    return res


def gamma_correction(img, gamma):
    img = img.astype(np.int32)
    res = (img / 255) ** gamma * 255
    return res.astype(np.uint8)


def combined_spatial_sharpening(img, k):
    img = img.astype(np.int32)

    img_laplacian = laplacian(img)
    img_addition = img + (-1) * img_laplacian
    img_sobel = sobel_gradient(img)
    img_average = average_filtering(img_sobel, 3)
    img_product = img_addition * img_average / 255
    res = img + k * img_product

    # return (res * 255 / np.max(res)).astype(np.uint8)
    return np.clip(res, 0, 255).astype(np.uint8)

