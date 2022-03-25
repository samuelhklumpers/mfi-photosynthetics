import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sn
import scipy.optimize as so
from skimage import restoration

from psf_generator import *
from util import *


def forward(psf, image):
    return sn.convolve(image, psf, mode="constant")

def solve_deconvolve(psf, image):
    res = restoration.wiener(image, psf, 0.01)
    return res# np.abs(res)

def solve_lsq(psf, image):
    def F(x):
        x = np.reshape(x, image.shape)
        return (image - forward(psf, x)).flatten()

    x0 = image.copy().flatten()

    res = so.least_squares(F, x0, max_nfev=10, verbose=2)

    return res['x']

def solve_lsq_jac(psf, image):
    def g(x):
        x = np.reshape(x, image.shape)
        return (image - forward(psf, x)).flatten()

    def F(x):
        return np.linalg.norm(g(x))

    Jg = np.zeros((image.size, image.size))
    N, M = image.shape

    psf_pad = np.pad(psf, ((M, M), (M, M)))
    
    print("precompute J")
    K = image.size
    # this least squares gives bad results for all but small gaussian psfs
    # the jacobian might be wrong
    for (i, j), _ in np.ndenumerate(image):
        Jg[M * i + j, :] = -psf_pad[i + M:i:-1, j + M:j:-1].flatten()
            # Jg[M * i + j, M * k + l] = -take_(psf, (i - k, j - l))

            # -psf[i:i - M, j:j - M].flatten()[M * k + l] = -psf[i:i - M, j:j - M][k, l] = -psf[i - k][j - l]
            # Jg[M * i + j, :] = -psf[i:i - M, j:j - M].flatten()

        if (M * i + j) % (K // 10) == 0:
            print("progress", M * i + j, "out of", K)

    def jac(x):
        return 2 * np.dot(g(x), Jg)

    x0 = image.flatten()

    res = so.least_squares(F, x0, jac=jac, max_nfev=100, verbose=2)

    return res['x'].reshape(image.shape)

def solve_demo(solver):
    #psf = load_im(".\psfs\gaussian3.png")
    psf = np.load(".\psfs\psf2.npy")

    psf = normalize(psf)
    psf = downscale(psf, 8)

    image = load_im(".\images\demo2.png")
    image = upscale(image, 2)
    #image = zero_pad(image, 4)

    
    preimage = solver(psf, image)
    result = forward(psf, preimage)

    show_field(psf, "psf")
    show_field(image, "desired output")
    show_field(preimage, "optimal input")
    show_field(result, "actual output")

    plt.show(block=True)

def psf_demo():
    z = 0 #2000e-9

    r = 20
    lims = 2 * r + 1

    psf = GLA_psf(z, lims, lims, **defaults)
    np.save(".\psfs\GLA1e-6real.npy", psf, allow_pickle=False)

    show_field(psf, "psf")
    plt.show(block=True)

if __name__ == "__main__":
    #psf_demo()
    solve_demo(solve_lsq_jac)
