import numpy as np
from scipy import interpolate


def nearest(img, dim):
    img = img.astype(np.int32)
    ratio = (np.array(img.shape) - 1) / (np.array(dim) - 1)

    c_mesh, r_mesh = np.meshgrid(np.arange(dim[1]), np.arange(dim[0]))
    round_r_mat = np.int32(np.round(r_mesh * ratio[0]))
    round_c_mat = np.int32(np.round(c_mesh * ratio[1]))

    res = img[round_r_mat, round_c_mat]
    return res.astype(np.uint8)

def nearest_v1(img, dim):
    img = img.astype(np.int32)
    ratio = (np.array(img.shape) - 1) / (np.array(dim) - 1)

    round_r_list = np.int32(np.round(np.arange(dim[0]) * ratio[0]))
    round_c_list = np.int32(np.round(np.arange(dim[1]) * ratio[1]))

    res = np.zeros(dim)
    for i in range(dim[0]):
        round_r = round_r_list[i]
        for j in range(dim[1]):
            round_c = round_c_list[j]
            res[i, j] = img[round_r, round_c]
    return res.astype(np.uint8)

def nearest_v0(img, dim):
    ratio = (np.array(img.shape) - 1) / (np.array(dim) - 1)

    res = np.zeros(dim)
    for i in range(dim[0]):
        for j in range(dim[1]):
            r, c = np.array([i, j]) * ratio
            res[i, j] = img[round(r), round(c)]
    return res

def bilinear(img, dim):
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

def bilinear_v1(img, dim):
    def inter(a, b, c):
        return a + (b - a) * c

    img = img.astype(np.int32)
    ratio = (np.array(img.shape) - 1) / (np.array(dim) - 1)

    r_list, c_list = np.arange(dim[0]) * ratio[0], np.arange(dim[1]) * ratio[1]
    topleft_r_list, topleft_c_list = np.int32(np.floor(r_list)), np.int32(np.floor(c_list)) # int32/64 快于 int16 ?
    topleft_r_list[-1] -= 1
    topleft_c_list[-1] -= 1
    rdots_list, cdots_list = r_list - topleft_r_list, c_list - topleft_c_list

    res = np.zeros(dim, dtype=np.uint8)
    for i in range(dim[0]):
        tlr = topleft_r_list[i]
        rdots = rdots_list[i]
        for j in range(dim[1]):
            tlc, cdots = topleft_c_list[j], cdots_list[j]

            res[i, j] = inter(
                inter(img[tlr, tlc], img[tlr, tlc + 1], cdots),
                inter(img[tlr + 1, tlc], img[tlr + 1, tlc + 1], cdots),
                rdots
            )
    return res.astype(np.uint8)

def bilinear_v0(img, dim):
    def inter(a, b, c):
        return np.uint8(np.int32(a) + (np.int32(b) - np.int32(a)) * c)

    ratio = (np.array(img.shape) - 1) / (np.array(dim) - 1)
    res = np.zeros(dim, dtype=np.uint8)
    for i in range(dim[0]):
        for j in range(dim[1]):
            r, c = np.array([i, j]) * ratio
            tlr, tlc = np.int32(np.minimum(np.floor([r, c]), np.array(img.shape) - 2))
            rdots, cdots = r - tlr, c - tlc

            res[i, j] = inter(
                inter(img[tlr, tlc], img[tlr, tlc + 1], cdots),
                inter(img[tlr + 1, tlc], img[tlr + 1, tlc + 1], cdots),
                rdots
            )
    return res

def bilinear_scipy(img, dim):
    f = interpolate.interp2d(np.arange(img.shape[0]), np.arange(img.shape[1]), img, kind='linear')
    res = f(np.linspace(0, img.shape[1] - 1, dim[1]), np.linspace(0, img.shape[0] - 1, dim[0]))
    return res.astype(np.uint8)

def bicubic(img, dim, a=-0.5):
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

def bicubic_scipy(img, dim):
    f = interpolate.interp2d(np.arange(img.shape[0]), np.arange(img.shape[1]), img, kind='cubic')
    res = f(np.linspace(0, img.shape[1] - 1, dim[1]), np.linspace(0, img.shape[0] - 1, dim[0]))
    return res.astype(np.uint8)

