import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time


def nearest_12112907(img, dim):
    img = img.astype(np.int32)
    ratio = (np.array(img.shape) - 1) / (np.array(dim) - 1)

    c_mesh, r_mesh = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]))
    round_r_mat = np.int32(np.round(r_mesh * ratio[0]))
    round_c_mat = np.int32(np.round(c_mesh * ratio[1]))

    res = img[round_r_mat, round_c_mat]
    return res.astype(np.uint8)


# Read image file and convert it to a numpy.ndarray of numpy.uint8
def loadGrayscaleAsNumpy(filepath):
    img = Image.open(filepath)
    return np.array(img.convert('L'))


# Testbench
img_raw = loadGrayscaleAsNumpy("res/rice.tif")
scalar = 4

start = time.time()
img = nearest_12112907(img_raw, (round(256 * scalar), round(256 * scalar)))
print("Time Cost: {} s.".format(time.time() - start))

plt.imshow(img, cmap="gray")
plt.show()

