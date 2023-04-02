import numpy as np

from diptools import *


img = persistance.load_gray('res/Q5_3.tif')

# H_p = frequency.H_ideal_LPF(img.shape, 30, double_pad=True)
# H_p = frequency.H_butterworth_LPF(img.shape, 2, 30, double_pad=True)
# H_p = 1 - frequency.H_gauss_LPF(img.shape, 30, double_pad=True)

f_r, f_c = img.shape
img_p = np.pad(img, ((0, f_r), (0, f_c)))

F_disp = frequency.magnitude_spectrum_disp(frequency.compute_spectrum(img_p))
persistance.show(F_disp)

D0 = 6
H_p = frequency.H_butterworth_NRF(img.shape, (-28, 82), 4, D0, double_pad=True)
H_p = H_p * frequency.H_butterworth_NRF(img.shape, (-28, 41), 4, D0, double_pad=True)
H_p = H_p * frequency.H_butterworth_NRF(img.shape, (-28, -41), 4, D0, double_pad=True)
H_p = H_p * frequency.H_butterworth_NRF(img.shape, (-28, -82), 4, D0, double_pad=True)
persistance.show(H_p)
persistance.show(frequency.magnitude_spectrum_disp(frequency.compute_spectrum(img_p) * H_p))

img = frequency.filter_freq_pad(img, H_p, is_H_padded=True, regulator=regulator.GrayCuttingRegulator)
persistance.show(img)

