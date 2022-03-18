import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sn
import scipy.optimize as so
from skimage import restoration

from psf_generator import *


def load_im(fn):
    im = cv2.imread(fn)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype(float) / 255

def save_im(fn, im):
    im = (im * 255).astype(int)
    im = cv2.imwrite(fn, im)

def show_field(im, title):
    plt.figure()
    r = plt.pcolormesh(im, shading='auto')
    plt.colorbar(r)
    plt.title(title)
    plt.show(block=False)

def forward(psf, image):
    return sn.convolve(image, psf, mode="constant")

def solve_deconvolve(psf, image):
    res = restoration.wiener(image, psf, 0.01)
    return res# np.abs(res)

# deprecated
# (almost always increases errors)
def solve_split_deconvolution(psf_mid, psf_neg, psf_pos, image, steps=10, beta=0.25, **kwargs):
    # beta is the aggression factor
    # if beta = 1.0, expect oscillation
    # if beta > 1.0, expect divergence
    # if beta is small, expect slow convergence/oscillation

    verbose = kwargs.get("v", False)

    mid_out = restoration.wiener(image, psf_mid, 0.01)

    def split(im):
        im_neg = im.copy()
        im_pos = im.copy()

        im_neg[im_neg > 0] = 0
        im_pos[im_pos < 0] = 0

        return (im_neg, im_pos)

    neg_in, pos_in = split(mid_out)

    for i in range(steps):
        neg_out = forward(psf_neg, neg_in)
        pos_out = forward(psf_pos, pos_in)

        mid_out = pos_out + neg_out

        err = image - mid_out
        if verbose:
            print(f"i={i}, err={np.linalg.norm(err)}")

        #neg_err, pos_err = split(err)

        #neg_in_ = beta * restoration.wiener(neg_err, psf_neg, 0.01)
        #pos_in_ = beta * restoration.wiener(pos_err, psf_neg, 0.01)
        
        neg_in_ = beta * restoration.wiener(err, psf_neg, 0.01)
        pos_in_ = beta * restoration.wiener(err, psf_pos, 0.01)

        neg_in_[neg_in_ > 0] = 0
        pos_in_[pos_in_ < 0] = 0

        neg_in = neg_in_
        pos_in = pos_in_

    neg_out = forward(psf_neg, neg_in)
    pos_out = forward(psf_pos, pos_in)

    mid_out = pos_out + neg_out

    return (neg_in, pos_in, mid_out)

def take_(a, ix):
    try:
        return a[ix]
    except:
        return 0

def bounds2(x, y):
    def f(i, j):
        return (0 <= i < x) and (0 <= j < y)

    return f

# deprecated
def solve_lsq(psf, image):
    b = image.flat
    A = np.zeros((len(b), len(b)))

    print("solving")

    bounds = bounds2(*image.shape)

    total = len(list(np.ndenumerate(image)))
    num = 0
    for ((i, j), _) in np.ndenumerate(image):
        for ((k, l), _) in np.ndenumerate(psf):

            n, m = i - k, j - l
            if bounds(n, m):
                x = np.ravel_multi_index((i, j), image.shape)
                y = np.ravel_multi_index((n, m), image.shape)

                v = take_(psf, (k, l))
                A[x, y] = v

        num += 1

        if num % 1000 == 0:
            print(num, "out of", total)

    print("matrix is", A)

    res = so.lsq_linear(A, b, bounds=(0, np.inf))

    print("solved", res)
    print(res.shape, image.shape)

    res = res.reshape(image.shape)


    return res


def upscale(im, scale=2):
    x = np.repeat(im, scale, axis=0)
    x = np.repeat(x, scale, axis=1)

    return x


def demo(psf, image):
    preimage = solve_deconvolve(psf, image)
    result = forward(psf, preimage)

    show_field(psf, "psf")
    show_field(image, "desired output")
    show_field(preimage, "optimal input")
    show_field(result, "actual output")


def normalize(im):
    return im / im.sum()


def deconv_demo():
    psf = load_im(".\strange.png")
    #psf = np.load("psf2.npy")
    psf = normalize(psf)

    image = load_im(".\demo2.png")
    image = upscale(image, 1)

    demo(psf, image)
    plt.show(block=True)

# deprecated
# (underlying functionality is broken)
def split_demo(lam_neg, lam_pos, im):
    show_psf = True
    
    im = upscale(im, 4)

    z = 0
    r = 20
    lims = 2 * r + 1

    defaults_mid = defaults.copy()
    defaults_neg = defaults.copy()
    defaults_pos = defaults.copy()

    defaults_mid["k"] = 2 * np.pi / ((lam_neg + lam_pos) / 2)
    defaults_neg["k"] = 2 * np.pi / lam_neg
    defaults_pos["k"] = 2 * np.pi / lam_pos

    psf_mid = GLA_psf(z, lims, lims, **defaults_mid)
    psf_neg = GLA_psf(z, lims, lims, **defaults_neg)
    psf_pos = GLA_psf(z, lims, lims, **defaults_pos)
    
    if show_psf:
        show_field(psf_mid, "mid")
        show_field(psf_neg, "neg")
        show_field(psf_pos, "pos")
        show_field(psf_pos - psf_neg, "dif")

        print(np.linalg.norm(psf_pos - psf_neg))

        plt.show(block=True)

    neg_in, pos_in, mid_out = solve_split_deconvolution(psf_mid, psf_neg, psf_pos, im, v=True)

    show_field(im, "in")
    show_field(neg_in, "neg in")
    show_field(pos_in, "pos in")
    show_field(mid_out, "out")
    plt.show(block=True)


def psf_demo():
    z = 0 #2000e-9

    r = 20
    lims = 2 * r + 1

    psf = GLA_psf(z, lims, lims, **defaults)
    np.save("GLA1e-6real.npy", psf, allow_pickle=False)

    show_field(psf, "psf")
    plt.show(block=True)

if __name__ == "__main__":
    im = load_im(".\demo2.png")
    split_demo(400e-6, 460e-6, im)