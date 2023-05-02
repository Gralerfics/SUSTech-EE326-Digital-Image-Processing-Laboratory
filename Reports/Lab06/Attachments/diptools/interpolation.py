import numpy as np
# from scipy import interpolate

from .regulator import GrayCuttingRegulator


def nearest(img, dim, regulator=GrayCuttingRegulator):
    img = img.astype(np.int32)
    ratio = (np.array(img.shape) - 1) / (np.array(dim) - 1)

    c_mesh, r_mesh = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]))
    round_r_mat = np.int32(np.round(r_mesh * ratio[0]))
    round_c_mat = np.int32(np.round(c_mesh * ratio[1]))

    res = img[round_r_mat, round_c_mat]
    return regulator(res)


def bilinear(img, dim, regulator=GrayCuttingRegulator):
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
    return regulator(res)


# def bilinear_scipy(img, dim, regulator=GrayCuttingRegulator):
#     f = interpolate.interp2d(np.arange(img.shape[0]), np.arange(img.shape[1]), img, kind='linear')
#     res = f(np.linspace(0, img.shape[1] - 1, dim[1]), np.linspace(0, img.shape[0] - 1, dim[0]))
#     return regulator(res)


def bicubic(img, dim, a=-0.5, regulator=GrayCuttingRegulator):
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
    return regulator(res)


# def bicubic_scipy(img, dim, regulator=GrayCuttingRegulator):
#     f = interpolate.interp2d(np.arange(img.shape[0]), np.arange(img.shape[1]), img, kind='cubic')
#     res = f(np.linspace(0, img.shape[1] - 1, dim[1]), np.linspace(0, img.shape[0] - 1, dim[0]))
#     return regulator(res)

