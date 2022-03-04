import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as sn
from skimage import restoration


def load_im(fn):
    im = cv2.imread(fn)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype(float) / 255

def show_field(im, title):
    plt.figure()
    r = plt.pcolormesh(im, shading='auto')
    plt.colorbar(r)
    plt.title(title)
    plt.show(block=False)

def forward(psf, image):
    return sn.convolve(image, psf, mode="constant")

def solve(psf, image):
    res = restoration.wiener(image, psf, 0.01)
    return np.abs(res)

def solve_(psf, image):
    def f(x):
        x_ = x.reshape(image.shape)

    res = restoration.wiener(image, psf, 0.01)
    return np.abs(res)

def demo(psf, image):
    preimage = solve(psf, image)
    result = forward(psf, preimage)

    show_field(image, "desired output")
    show_field(preimage, "optimal input")
    show_field(result, "actual output")


if __name__ == "__main__":
    psf = load_im(".\gaussian3.png")
    image = load_im(".\demo1.png")

    demo(psf, image)
    plt.show(block=True)