import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time


def bicubic_12112907(img, dim, a=-0.5):
    def W(x):
        x_fabs = np.fabs(x)
        res = x_fabs
        flag = x_fabs <= 1
        res[flag] = (a + 2) * x_fabs[flag] ** 3 - (a + 3) * x_fabs[flag] ** 2 + 1
        flag = x_fabs > 1
        res[flag] = a * x_fabs[flag] ** 3 - 5 * a * x_fabs[flag] ** 2 + 8 * a * x_fabs[flag] - 4 * a
        return res

    ratio = (np.array(img.shape) - 1) / (np.array(dim) - 1)
    img = np.pad(img.astype(np.int32), ((2, 2), (2, 2)), 'reflect')

    c_mesh, r_mesh = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]))
    r_mat = r_mesh * ratio[0] + 2
    c_mat = c_mesh * ratio[1] + 2
    tl_r_mat = np.clip(np.int32(np.floor(r_mat)) - 1, 0, img.shape[0] - 4)
    tl_c_mat = np.clip(np.int32(np.floor(c_mat)) - 1, 0, img.shape[1] - 4)

    res = np.zeros(dim)
    for i in range(4):
        it_r_mat = tl_r_mat + i
        for j in range(4):
            it_c_mat = tl_c_mat + j
            res += img[it_r_mat, it_c_mat] * W(r_mat - it_r_mat) * W(c_mat - it_c_mat)
    return np.clip(res, 0, 255).astype(np.uint8)


# Read image file and convert it to a numpy.ndarray of numpy.uint8
def loadGrayscaleAsNumpy(filepath):
    img = Image.open(filepath)
    return np.array(img.convert('L'))


# Testbench
img_raw = loadGrayscaleAsNumpy("res/rice.tif")
scalar = 4

start = time.time()
img = bicubic_12112907(img_raw, (round(256 * scalar), round(256 * scalar)))
print("Time Cost: {} s.".format(time.time() - start))

plt.imshow(img, cmap="gray")
plt.show()

