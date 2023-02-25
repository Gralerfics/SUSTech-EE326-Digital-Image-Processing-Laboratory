import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time


def bilinear_12112907(img, dim):
    def inter(a, b, c):
        return a + (b - a) * c

    img = img.astype(np.int32)
    ratio = (np.array(img.shape) - 1) / (np.array(dim) - 1)

    c_mesh, r_mesh = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]))
    r_mat = r_mesh * ratio[0]
    c_mat = c_mesh * ratio[1]
    tl_r_mat = np.minimum(np.int32(np.floor(r_mat)), img.shape[0] - 2)
    tl_c_mat = np.minimum(np.int32(np.floor(c_mat)), img.shape[1] - 2)
    tl_r_mat_n = tl_r_mat + 1
    tl_c_mat_n = tl_c_mat + 1
    rdots_mat = r_mat - tl_r_mat
    cdots_mat = c_mat - tl_c_mat

    res = inter(
        inter(img[tl_r_mat, tl_c_mat], img[tl_r_mat, tl_c_mat_n], cdots_mat),
        inter(img[tl_r_mat_n, tl_c_mat], img[tl_r_mat_n, tl_c_mat_n], cdots_mat),
        rdots_mat
    )
    return res.astype(np.uint8)


# Read image file and convert it to a numpy.ndarray of numpy.uint8
def loadGrayscaleAsNumpy(filepath):
    img = Image.open(filepath)
    return np.array(img.convert('L'))


# Testbench
img_raw = loadGrayscaleAsNumpy("res/rice.tif")
scalar = 4

start = time.time()
img = bilinear_12112907(img_raw, (round(256 * scalar), round(256 * scalar)))
print("Time Cost: {} s.".format(time.time() - start))

plt.imshow(img, cmap="gray")
plt.show()

