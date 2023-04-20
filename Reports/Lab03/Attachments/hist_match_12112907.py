import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time


def loadGrayscaleAsNumpy(filepath):
    img = Image.open(filepath)
    return np.array(img.convert('L'))


def plotHistogram(plt, hist, color):
    plt.bar(np.arange(0, 256), hist, color=color)
    plt.xlim(-0.5, 255.5)
    plt.xlabel("Grayscale")
    plt.ylabel("Frequency")


def getPDF(img):
    res, bins = np.histogram(img, bins=256, range=(-0.5, 255.5))
    return res / (img.shape[0] * img.shape[1])


def hist_match_12112907(img, spec_hist): # spec_hist should be normalized to a PDF
    img = img.astype(np.int32)

    img_pdf = getPDF(img)
    img_cdf = np.cumsum(img_pdf)
    spec_cdf = np.cumsum(spec_hist)

    tran = np.searchsorted(255 * spec_cdf, 255 * img_cdf + 0.5, side='left') - 1
    res = tran[img]

    res = res.astype(np.uint8)
    return (res, getPDF(res), img_pdf)


img_raw = loadGrayscaleAsNumpy("Q3_2.tif")

spec_hist = np.concatenate((
    np.linspace(0, 7, 5 - 0 + 1),
    np.linspace(7, 0.7, 20 - 5 + 1)[1:],
    np.linspace(0.7, 0, 181 - 20 + 1)[1:],
    np.linspace(0, 0.6, 203 - 181 + 1)[1:],
    np.linspace(0.6, 0, 255 - 203 + 1)[1:]
), axis=0)
spec_hist /= np.sum(spec_hist)

start = time.time()
img, img_hist, raw_hist = hist_match_12112907(img_raw, spec_hist)
print("Time Cost: {} s.".format(time.time() - start))

plt.figure(1, figsize=(25.6, 5))
plotHistogram(plt, raw_hist, "lightgray")
plotHistogram(plt, img_hist, "orange")

plt.figure(2)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)

plt.show()

