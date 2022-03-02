import cv2

import numpy as np
from skimage import color, data, restoration


def load_im(fn):
    return cv2.imread(fn).astype(float) / 255

def show_im(im, nm):
    cv2.imshow(nm, (im * 255).astype(np.uint8))

original = load_im("original.png")[:, :, 0]
blurred = load_im("blurred.png")[:, :, 0]

def gauss(r, s):
    x = np.arange(2 * r + 1) - r
    
    c = np.exp(-x ** 2 / 2 / s ** 2) / np.sqrt(2 * np.pi) / s

    r = np.outer(c, c)
    r /= r.sum()

    return r



#for r in range(5, 20):
#    for s in range(1, 6):
#        for b in np.linspace(0.01, 1, num=5):
#            psf = gauss(r, s)
#            deblurred = restoration.wiener(blurred, psf, b)
#    
#            print(r, s, b, np.linalg.norm(original - deblurred))

psf = gauss(10, 3)
deblurred = restoration.wiener(blurred, psf, 0.01)

show_im(original, "original")
show_im(blurred, "blurred")
show_im(deblurred, "deblurred")
