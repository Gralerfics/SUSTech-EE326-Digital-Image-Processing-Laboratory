# Q6_1_1, Pepper noise

import numpy as np
import cv2

from diptools import *


img = cv2.imread('res/Q6_1_2.tiff', cv2.IMREAD_GRAYSCALE)
cv2.imshow('img', img)
cv2.waitKey(0)

# res = spatial.arithmetic_mean_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator)
# res = spatial.geometric_mean_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator)
res = spatial.harmonic_mean_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator)
cv2.imshow('result', res)
cv2.waitKey(0)

res = spatial.contraharmonic_mean_filter(img, (3, 3), -1, regulator=regulator.GrayCuttingRegulator)
cv2.imshow('result', res)
cv2.waitKey(0)
