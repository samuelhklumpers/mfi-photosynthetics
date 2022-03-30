import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

from skimage import restoration
from jax import jit, value_and_grad

import os

from psf_generator import *
from util import *


# see: https://stackoverflow.com/a/64776128/5895919
def gen_idx_conv1d(in_size, ker_size):
    f = lambda dim1, dim2, axis: jnp.reshape(jnp.tile(jnp.expand_dims(jnp.arange(dim1),axis),dim2),-1)
    out_size = in_size-ker_size+1
    return f(ker_size, out_size, 0)+f(out_size, ker_size, 1)

def repeat_idx_2d(idx_list, nbof_rep, axis):
    assert axis in [0,1], "Axis should be equal to 0 or 1."
    tile_axis = (nbof_rep,1) if axis else (1,nbof_rep)
    return jnp.reshape(jnp.tile(jnp.expand_dims(idx_list, 1),tile_axis),-1)

def conv2d(im, ker): # make sure ker is odd dimensional
    ker_x = ker.shape[0]
    ker_y = ker.shape[1]

    im = jnp.pad(im, [(ker_x // 2, ker_x // 2), (ker_y // 2, ker_y // 2)])


    if len(im.shape)==2: # it the image is a 2D array, it is reshaped by expanding the last dimension
        im = jnp.expand_dims(im,-1)

    im_x, im_y, im_w = im.shape

    if len(ker.shape)==2: # if the kernel is a 2D array, it is reshaped so it will be applied to all of the image channels
        ker = jnp.tile(jnp.expand_dims(ker,-1),[1,1,im_w]) # the same kernel will be applied to all of the channels 

    assert ker.shape[-1]==im.shape[-1], "Kernel and image last dimension must match."


    # shape of the output image
    out_x = im_x - ker_x + 1 
    out_y = im_y - ker_y + 1

    # reshapes the image to (out_x, ker_x, out_y, ker_y, im_w)
    idx_list_x = gen_idx_conv1d(im_x, ker_x) # computes the indices of a 1D conv (cf. idx_conv1d doc)
    idx_list_y = gen_idx_conv1d(im_y, ker_y)

    idx_reshaped_x = repeat_idx_2d(idx_list_x, len(idx_list_y), 0) # repeats the previous indices to be used in 2D (cf. repeat_idx_2d doc)
    idx_reshaped_y = repeat_idx_2d(idx_list_y, len(idx_list_x), 1)

    im_reshaped = jnp.reshape(im[idx_reshaped_x, idx_reshaped_y, :], [out_x, ker_x, out_y, ker_y, im_w]) # reshapes

    # reshapes the 2D kernel
    ker = jnp.reshape(ker,[1, ker_x, 1, ker_y, im_w])

    # applies the kernel to the image and reduces the dimension back to the one of original input image
    return jnp.squeeze(jnp.sum(im_reshaped*ker, axis=(1,3)))

def forward(psf, image):
    return conv2d(image, psf)

def solve_deconvolve(psf, image):
    res = restoration.wiener(image, psf, 0.01)
    return res# np.abs(res)

def solve_lsq(psf, image):
    def F(x):
        x = jnp.reshape(x, image.shape)
        return (image - forward(psf, x)).flatten()

    x0 = image.copy().flatten()

    res = so.least_squares(F, x0, max_nfev=10, verbose=2)

    return res['x']

def solve_lsq_jac(psf, image):
    def g(x):
        x = jnp.reshape(x, image.shape)
        return (image - forward(psf, x)).flatten()
   
    def F(x):
        return jnp.linalg.norm(g(x))

    x0 = image.flatten()

    obj_and_grad = jit(value_and_grad(F))

    res = so.minimize(obj_and_grad, x0, jac=True, options={
        'maxiter' : 3,
        'disp'    : True
    })

    return res['x'].reshape(image.shape)


def make_odd(psf):
    if psf.shape[0] % 2 == 0:
        psf = psf[:-1]
    
    if psf.shape[1] % 2 == 0:
        psf = psf[:, :-1]
    
    return psf

def solve_demo(solver):
    #psf = load_im("./psfs/gaussian3.png")
    psf = np.load("./psfs/psf2.npy")

    psf = normalize(psf)
    psf = downscale(psf, 8)

    image = load_im("./images/demo2.png")
    image = upscale(image, 2)
    #image = zero_pad(image, 4)

    psf = make_odd(psf)

    
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
    np.save("./psfs/GLA1e-6real.npy", psf, allow_pickle=False)

    show_field(psf, "psf")
    plt.show(block=True)

if __name__ == "__main__":
    #psf_demo()
    solve_demo(solve_lsq_jac)
