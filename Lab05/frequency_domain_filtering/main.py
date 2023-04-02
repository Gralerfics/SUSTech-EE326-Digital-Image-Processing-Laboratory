import numpy as np
from diptools import *


img_1 = persistance.load_gray('res/Q5_1.tif')
img_2 = persistance.load_gray('res/Q5_2.tif')
img_3 = persistance.load_gray('res/Q5_3.tif')

show = True
save = False


## Q1
kernel_sobel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

# Spatial domain
img_1_spatial = linear.convolution(img_1, kernel_sobel, regulator=regulator.GrayScalingRegulator)
if show: persistance.show(img_1_spatial)
if save: persistance.save_gray('output/Q5_1_spatial.tif', img_1_spatial)

# Frequency domain
img_1_freq = frequency.convolution_freq(img_1, kernel_sobel, regulator=regulator.GrayScalingRegulator)
if show: persistance.show(img_1_freq)
if save: persistance.save_gray('output/Q5_1_freq.tif', img_1_freq)

# Test No-shift under Doubling Padding (need to modify the diptools.frequency)
# F, H = frequency.compute_FH_in_pair(img_1, kernel_sobel, pad_mode='double')
# G = F * H
# g_p = frequency.recover_spatial(G, regulator=regulator.GrayScalingRegulator)
# persistance.show(g_p)
# persistance.save_gray('output/Q5_1_freq_H_no_shift_double.tif', g_p)

## Q2
for radius in [10, 30, 60, 160, 460]:
    H_p = frequency.H_ideal_LPF(img_2.shape, radius, double_pad=True)
    img_2_ideal = frequency.filter_freq_pad(img_2, H_p, is_H_padded=True, regulator=regulator.GrayCuttingRegulator)
    h_p = frequency.recover_spatial(H_p * frequency.shifting_mat(H_p.shape), regulator=regulator.GrayScalingRegulator)
    if show:
        persistance.show(h_p)
        persistance.show(img_2_ideal)
    if save:
        persistance.save_gray('output/Q5_2_ideal_kernel_r={}.tif'.format(radius), h_p)
        persistance.save_gray('output/Q5_2_ideal_r={}.tif'.format(radius), img_2_ideal)


## Q3
for radius in [30, 60, 160]:
    H_p = frequency.H_gauss_LPF(img_2.shape, radius, double_pad=True)
    img_2_gauss = frequency.filter_freq_pad(img_2, H_p, is_H_padded=True, regulator=regulator.GrayCuttingRegulator)
    if show: persistance.show(img_2_gauss)
    if save: persistance.save_gray('output/Q5_2_gauss_r={}.tif'.format(radius), img_2_gauss)


## Q4
order = 4
D0 = 6

img_3_F_disp = frequency.magnitude_spectrum_disp(frequency.compute_spectrum(img_3))
if show: persistance.show(img_3_F_disp)
if save: persistance.save_gray('output/Q5_3_raw_spectrum.tif', img_3_F_disp)

x0 = -28
H_p = np.ones((img_3.shape[0] * 2, img_3.shape[1] * 2))
for y0 in [-82, -41, 41, 82]:
    H_p_0 = frequency.H_butterworth_NRF(img_3.shape, (x0, y0), order, D0, double_pad=True)
    H_p = H_p * H_p_0
if show: persistance.show(regulator.GrayScalingRegulator(H_p))
if save: persistance.save_gray('output/Q5_3_nrf_H_spectrum.tif', regulator.GrayScalingRegulator(H_p))

img_3_nrf = frequency.filter_freq_pad(img_3, H_p, is_H_padded=True, regulator=regulator.GrayCuttingRegulator)
if show: persistance.show(img_3_nrf)
if save: persistance.save_gray('output/Q5_3_nrf.tif', img_3_nrf)

img_3_nrf_F_disp = frequency.magnitude_spectrum_disp(frequency.compute_spectrum(img_3_nrf))
if show: persistance.show(img_3_nrf_F_disp)
if save: persistance.save_gray('output/Q5_3_nrf_spectrum.tif', img_3_nrf_F_disp)

