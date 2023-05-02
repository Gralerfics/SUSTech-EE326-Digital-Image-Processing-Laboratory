import numpy as np
from matplotlib import pyplot

from .regulator import GrayCuttingRegulator


def gamma_correction(img, gamma, regulator=GrayCuttingRegulator):
    res = (img.astype(np.int32) / 255) ** gamma * 255
    return regulator(res)


def compute_pdf(img):
    res, bins = np.histogram(img, bins=256, range=(-0.5, 255.5))
    return res / (img.shape[0] * img.shape[1])


def hist_equ(img, regulator=GrayCuttingRegulator):
    img = img.astype(np.int32)

    img_pdf = compute_pdf(img)  # PDF (regardless of size)
    img_cdf = np.cumsum(img_pdf)  # Cumulative Sum
    res = 255 * img_cdf[img]

    return regulator(res)


def hist_match(img, spec_hist, regulator=GrayCuttingRegulator):  # spec_hist should be normalized to a PDF
    img = img.astype(np.int32)

    img_pdf = compute_pdf(img)
    img_cdf = np.cumsum(img_pdf)
    spec_cdf = np.cumsum(spec_hist)

    # tran = np.searchsorted(255 * spec_cdf, 255 * img_cdf + 0.5, side='left') - 1
    tran = np.searchsorted(spec_cdf, img_cdf + 1e-6, side='left') - 1
    res = tran[img]

    return regulator(res)


def local_hist_equal(img, radius, regulator=GrayCuttingRegulator):  # radius should be an integer and larger than 1
    size = radius * 2 - 1
    img = np.pad(img.astype(np.int32), ((radius - 1, radius - 1), (radius - 1, radius - 1)), 'reflect')

    pts = [(idx // size - radius + 1, idx % size - radius + 1) for idx in range(size * size)]
    mvd = np.array([np.roll(np.roll(img, x, axis=0), y, axis=1) for (x, y) in pts])
    res = np.sum(mvd <= img, axis=0) / (size * size) * 255

    res = res[(radius - 1):(1 - radius), (radius - 1):(1 - radius)]
    return regulator(res)


# def median_filtering(img, radius, regulator=GrayCuttingRegulator):  # see spatial.median_filter
#     size = radius * 2 - 1
#     img = np.pad(img.astype(np.int32), ((radius - 1, radius - 1), (radius - 1, radius - 1)), 'reflect')
#
#     pts = [(idx // size - radius + 1, idx % size - radius + 1) for idx in range(size * size)]
#     mvd = np.array([np.roll(np.roll(img, x, axis=0), y, axis=1) for (x, y) in pts])
#     res = np.median(mvd, axis=0)
#
#     res = res[(radius - 1):(1 - radius), (radius - 1):(1 - radius)]
#     return regulator(res)


def plot(img, color='lightgray', plt=pyplot, ymode='linear'):
    hist = compute_pdf(img)
    plt.bar(np.arange(0, 256), hist, color=color)
    plt.xlim(-0.5, 255.5)
    plt.xlabel("Grayscale")
    plt.ylabel("Frequency")
    plt.yscale(ymode)
    plt.show()

