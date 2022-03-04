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
    return np.abs(res)

def take_(a, ix):
    try:
        return a[ix]
    except:
        return 0

def bounds2(x, y):
    def f(i, j):
        return (0 <= i < x) and (0 <= j < y)

    return f

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

def demo(psf, image):
    preimage = solve_deconvolve(psf, image)
    result = forward(psf, preimage)

    show_field(image, "desired output")
    show_field(preimage, "optimal input")
    show_field(result, "actual output")


def normalize(im):
    return im / im.sum()


if __name__ == "__main__":
    psf = load_im(".\gaussian3.png")
    psf = normalize(psf)

    image = load_im(".\demo2.png")

    demo(psf, image)
    plt.show(block=True)