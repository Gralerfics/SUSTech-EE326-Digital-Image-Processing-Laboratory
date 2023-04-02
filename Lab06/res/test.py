import numpy as np
import cv2


def GrayCuttingRegulator(x):
    return np.clip(x, 0, 255).astype(np.uint8)

def GrayScalingRegulator(x):
    l, r = np.min(x), np.max(x)
    return GrayCuttingRegulator(256 * (x - l) / (r - l))

def gamma_correction(img, gamma, regulator=GrayCuttingRegulator):
    res = (img.astype(np.int32) / 255) ** gamma * 255
    return regulator(res)

def magnitude_spectrum_disp(S, gamma=0.15):
    return gamma_correction(np.abs(S), gamma, regulator=GrayScalingRegulator)


r = 34
img = cv2.imread('C:\\Workplace\\SUSTech-EE326-Digital-Image-Processing-Laboratory\\Lab06\\res\\Q6_3_1.tiff', cv2.IMREAD_GRAYSCALE)
img = img.astype(np.int32)

img_r, img_c = img.shape
x_m, y_m = np.meshgrid(range(img_c), range(img_r))
shifter = (-1) ** (x_m + y_m)
exp_fac = np.exp(-2 * np.pi * 1j * (x_m / img_c + y_m / img_r))
factor = exp_fac ** (-r) - exp_fac ** (r + 1)
factor[np.abs(factor) < 1e-6] = 1e-6

img[1:, 1:] = img[1:, 1:] - img[:-1, :-1]

F_img = np.fft.fft2(img)

F_res = F_img / factor

cv2.imshow('image', magnitude_spectrum_disp(F_res))
cv2.waitKey(0)

cv2.imshow('image', GrayScalingRegulator(np.fft.ifft2(F_res)))
cv2.waitKey(0)

