from PIL import Image
import numpy as np


# Read image file
# - and convert it to a numpy.ndarray of numpy.uint8
def loadGrayscaleAsNumpy(filepath):
    img = Image.open(filepath)
    return np.array(img.convert('L'))

def saveGrayscale(img, filepath):
    Image.fromarray(img).save(filepath)


def plotHistogram(plt, hist, color, ymode="linear"):
    plt.bar(np.arange(0, 256), hist, color=color)
    plt.xlim(-0.5, 255.5)
    plt.xlabel("Grayscale")
    plt.ylabel("Frequency")
    plt.yscale(ymode)
