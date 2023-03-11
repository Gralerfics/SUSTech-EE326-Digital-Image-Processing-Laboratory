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


def local_hist_equal_12112907(img, radius): # radius should be an integer and larger than 1
    size = radius * 2 - 1
    img = np.pad(img.astype(np.int32), ((radius - 1, radius - 1), (radius - 1, radius - 1)), 'reflect')

    pts = [(idx // size - radius + 1, idx % size - radius + 1) for idx in range(size * size)]
    mvd = np.array([np.roll(np.roll(img, x, axis=0), y, axis=1) for (x, y) in pts])
    res = np.sum(mvd <= img, axis=0) / (size * size) * 255

    res = res[(radius - 1):(1 - radius), (radius - 1):(1 - radius)].astype(np.uint8)
    return (res, getPDF(res), getPDF(img))


img_raw = loadGrayscaleAsNumpy("Q3_3.tif")
start = time.time()
img, img_hist, raw_hist = local_hist_equal_12112907(img_raw, 2)
print("Time Cost: {} s.".format(time.time() - start))

plt.figure(1, figsize=(25.6, 5))
plotHistogram(plt, raw_hist, "lightgray")
plotHistogram(plt, img_hist, "orange")

plt.figure(2)
plt.imshow(img, cmap="gray", vmin=0, vmax=255)

plt.show()

