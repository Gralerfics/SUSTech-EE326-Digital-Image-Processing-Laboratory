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


def hist_equ_12112907(img):
    img = img.astype(np.int32)

    img_pdf = getPDF(img) # PDF (regardless of size)
    img_cdf = np.cumsum(img_pdf) # Cumulative Sum

    res = 255 * img_cdf[img]

    res = res.astype(np.uint8)
    return (res, getPDF(res), img_pdf)


img_raw = loadGrayscaleAsNumpy("Q3_1_1.tif")
# img_raw = loadGrayscaleAsNumpy("Q3_1_2.tif")
start = time.time()
img, img_hist, raw_hist = hist_equ_12112907(img_raw)
print("Time Cost: {} s.".format(time.time() - start))

plt.figure(1, figsize=(25.6, 5))
plotHistogram(plt, raw_hist, "lightgray")
plotHistogram(plt, img_hist, "orange")

plt.figure(2)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)

plt.show()

