import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sn
import scipy.optimize as so
from skimage import restoration


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

def take_(a, ix):
    try:
        return a[ix]
    except:
        return 0

def bounds2(x, y):
    def f(i, j):
        return (0 <= i < x) and (0 <= j < y)

    return f

def upscale(im, scale=2):
    x = np.repeat(im, scale, axis=0)
    x = np.repeat(x, scale, axis=1)

    return x

def downscale(im, scale=2):
    return im[::scale, ::scale]


def zero_pad(im, scale=2):
    n = max(im.shape[0], im.shape[1])

    return np.pad(im, int((scale - 1) * n))

def normalize(im):
    return im / im.sum()
