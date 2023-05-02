import cv2
import numpy as np

from diptools import *


# for i in [1, 2, 3, 4]:
#     img = persistence.load_gray('res/Q6_1_{}.tiff'.format(i))
#     persistence.save_gray('output/Q6_1_{}_arithmetic_mean_3x3.jpg'.format(i), spatial.arithmetic_mean_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator))
#     persistence.save_gray('output/Q6_1_{}_geometric_mean_3x3.jpg'.format(i), spatial.geometric_mean_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator))
#     persistence.save_gray('output/Q6_1_{}_harmonic_mean_3x3.jpg'.format(i), spatial.harmonic_mean_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator))
#     persistence.save_gray('output/Q6_1_{}_contraharmonic_mean_3x3_1.5.jpg'.format(i), spatial.contraharmonic_mean_filter(img, (3, 3), 1.5, regulator=regulator.GrayCuttingRegulator))
#     persistence.save_gray('output/Q6_1_{}_contraharmonic_mean_3x3_-1.5.jpg'.format(i), spatial.contraharmonic_mean_filter(img, (3, 3), -1.5, regulator=regulator.GrayCuttingRegulator))
#     persistence.save_gray('output/Q6_1_{}_median_3x3.jpg'.format(i), spatial.median_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator))
#     persistence.save_gray('output/Q6_1_{}_max_3x3.jpg'.format(i), spatial.max_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator))
#     persistence.save_gray('output/Q6_1_{}_min_3x3.jpg'.format(i), spatial.min_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator))
#     persistence.save_gray('output/Q6_1_{}_.jpg'.format(i), )


# Q6_1_1
img = persistence.load_gray('res/Q6_1_1.tiff')
persistence.save_gray('output/ans/Q6_1_1_contraharmonic_mean_3x3_1.5.jpg', spatial.contraharmonic_mean_filter(img, (3, 3), 1.5, regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_1_contraharmonic_mean_5x5_1.5.jpg', spatial.contraharmonic_mean_filter(img, (5, 5), 1.5, regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_1_contraharmonic_mean_7x7_1.5.jpg', spatial.contraharmonic_mean_filter(img, (7, 7), 1.5, regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_1_contraharmonic_mean_3x3_0.5.jpg', spatial.contraharmonic_mean_filter(img, (3, 3), 0.5, regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_1_contraharmonic_mean_3x3_1.jpg', spatial.contraharmonic_mean_filter(img, (3, 3), 1, regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_1_median_3x3.jpg', spatial.median_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_1_median_3x3_twice.jpg', spatial.median_filter(spatial.median_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator), (3, 3), regulator=regulator.GrayCuttingRegulator))


# Q6_1_2
img = persistence.load_gray('res/Q6_1_2.tiff')
persistence.save_gray('output/ans/Q6_1_2_contraharmonic_mean_3x3_-1.5.jpg', spatial.contraharmonic_mean_filter(img, (3, 3), -1.5, regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_2_contraharmonic_mean_5x5_-1.5.jpg', spatial.contraharmonic_mean_filter(img, (5, 5), -1.5, regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_2_contraharmonic_mean_7x7_-1.5.jpg', spatial.contraharmonic_mean_filter(img, (7, 7), -1.5, regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_2_contraharmonic_mean_3x3_-1.jpg', spatial.contraharmonic_mean_filter(img, (3, 3), -1, regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_2_contraharmonic_mean_3x3_-2.jpg', spatial.contraharmonic_mean_filter(img, (3, 3), -2, regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_2_median_3x3.jpg', spatial.median_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_2_median_3x3_twice.jpg', spatial.median_filter(spatial.median_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator), (3, 3), regulator=regulator.GrayCuttingRegulator))


# Q6_1_3
img = persistence.load_gray('res/Q6_1_3.tiff')
persistence.save_gray('output/ans/Q6_1_3_median_3x3.jpg', spatial.median_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_3_median_3x3_twice.jpg', spatial.median_filter(spatial.median_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator), (3, 3), regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_3_median_3x3_thrice.jpg', spatial.median_filter(spatial.median_filter(spatial.median_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator), (3, 3), regulator=regulator.GrayCuttingRegulator), (3, 3), regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_3_median_5x5.jpg', spatial.median_filter(img, (5, 5), regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_3_median_5x5_twice.jpg', spatial.median_filter(spatial.median_filter(img, (5, 5), regulator=regulator.GrayCuttingRegulator), (5, 5), regulator=regulator.GrayCuttingRegulator))
persistence.save_gray('output/ans/Q6_1_3_adaptive_median_7.jpg', spatial.adaptive_median_filter(img, 7, regulator=regulator.GrayCuttingRegulator))


# Q6_1_4
img = persistence.load_gray('res/Q6_1_4.tiff')
res_adaptive_median_7 = spatial.adaptive_median_filter(img, 7, regulator=regulator.GrayCuttingRegulator)
persistence.save_gray('output/ans/Q6_1_4_adaptive_median_7.jpg', res_adaptive_median_7)
# res_adaptive_median_7 = persistence.load_gray('output/ans/Q6_1_4_adaptive_median_7.jpg')
persistence.save_gray('output/ans/Q6_1_4_adaptive_local_noise_reduction_7x7_after_adaptive_median_7.jpg', spatial.adaptive_local_noise_reduction_filter(res_adaptive_median_7, (7, 7), 250, regulator=regulator.GrayCuttingRegulator))


# Q6_2
img = persistence.load_gray('res/Q6_2.tiff')
r, c = img.shape
F = frequency.fft2d(img)
k = 0.0025
H = np.exp(-k * np.float_power(np.sum((np.array(np.meshgrid(range(c), range(r))) - np.array([np.ones((r, c)) * c // 2, np.ones((r, c)) * r // 2])) ** 2, axis=0), 5 / 6))

persistence.save_gray('output/ans/Q6_2_inverse.jpg', frequency.full_inverse_filter(F, H))

for R in [5, 10, 20, 50, 60, 80, 100, 120, 200]:
    persistence.save_gray('output/ans/Q6_2_radially_limited_inverse_{}.jpg'.format(R), frequency.radially_limited_inverse_filter(F, H, R))

for K in [1e-1, 1e-3, 1e-4, 5e-5, 1e-5, 1e-6]:
    persistence.save_gray('output/ans/Q6_2_wiener_{}.jpg'.format(K), frequency.wiener_filter(F, H, K))


# Q6_3
def linear_motion_deblurring_FH(img, a, b, T):
    r, c = img.shape
    F = frequency.fft2d(img)
    u, v = np.array(np.meshgrid(np.linspace(1, c, c), np.linspace(1, r, r)))
    A = u * a + v * b
    H = T / (np.pi * A) * np.sin(np.pi * A) * np.exp(-1j * np.pi * A)
    return F, H


# Noise Estimation
img = persistence.load_gray('res/Q6_3_2.tiff')
histogram.plot(img[250:357, 476:545])
Var = np.var(img[250:357, 476:545])
print(Var)


# Q6_3_1
img = persistence.load_gray('res/Q6_3_1.tiff')
F, H = linear_motion_deblurring_FH(img, 0.1, 0.1, 1.0)

persistence.save_gray('output/ans/Q6_3_1_inverse.jpg', frequency.full_inverse_filter(F, H))

for K in [1e-3, 1e-5, 1e-9, 1e-12, 1e-15, 1e-16, 1e-17, 1e-19]:
    persistence.save_gray('output/ans/Q6_3_1_wiener_{}.jpg'.format(K), frequency.wiener_filter(F, H, K))


# Q6_3_2
img = persistence.load_gray('res/Q6_3_2.tiff')

# for i in range(5):
#     print(i)
#     img = spatial.adaptive_median_filter(img, 5, regulator=regulator.GrayScalingRegulator)
# persistence.save_gray('output/Q6_3_2_adaptive_median_5_5times.tif', img)
img = persistence.load_gray('output/Q6_3_2_adaptive_median_5_5times.tif')

F, H = linear_motion_deblurring_FH(img, 0.1, 0.1, 1.0)
img = frequency.wiener_filter(F, H, 1e-16, regulator=regulator.GrayScalingRegulator)

img = histogram.hist_equ(img)

img = regulator.GrayScalingToRegulator(-100, 300)(img + 0.5 * spatial.geometric_mean_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator))
img = regulator.GrayCuttingRegulator(img)

img = spatial.adaptive_local_noise_reduction_filter(img, (5, 5), 1500, regulator=regulator.GrayScalingToRegulator(-100, 300))
img = regulator.GrayCuttingRegulator(img)

persistence.save_gray('output/ans/Q6_3_2_output.jpg', img)


# Q6_3_3
img = persistence.load_gray('res/Q6_3_3.tiff')

# for i in range(10):
#     print(i)
#     img = spatial.adaptive_median_filter(img, 5, regulator=regulator.GrayScalingRegulator)
# persistence.save_gray('output/Q6_3_3_adaptive_median_5_10times.tif', img)
img = persistence.load_gray('output/Q6_3_3_adaptive_median_5_10times.tif')

F, H = linear_motion_deblurring_FH(img, 0.1, 0.1, 1.0)
img = frequency.wiener_filter(F, H, 1e-16, regulator=regulator.GrayScalingRegulator)

img = histogram.hist_equ(img)

img = regulator.GrayScalingToRegulator(-100, 300)(img + 0.7 * spatial.geometric_mean_filter(img, (3, 3), regulator=regulator.GrayCuttingRegulator))
img = regulator.GrayCuttingRegulator(img)

img = spatial.adaptive_local_noise_reduction_filter(img, (5, 5), 2500, regulator=regulator.GrayScalingToRegulator(-200, 350))
img = regulator.GrayCuttingRegulator(img)

persistence.save_gray('output/ans/Q6_3_3_output.jpg', img)

