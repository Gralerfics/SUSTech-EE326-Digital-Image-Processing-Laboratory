from PIL import Image
import numpy as np


# Read image file
# - and convert it to a numpy.ndarray of numpy.uint8
def loadGrayscaleAsNumpy(filepath):
    img = Image.open(filepath)
    return np.array(img.convert('L'))

def saveGrayscale(img, filepath):
    Image.fromarray(img).save(filepath)

