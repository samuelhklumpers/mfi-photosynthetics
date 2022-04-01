import matplotlib.pyplot as plt
import scipy.special as ss
import numpy as np
from time import perf_counter


def path_difference(**kwargs):
    # Returns a function which computes the optical path difference multiplied
    # by the wave number k. The returned function f is a function of the depth
    # z and horizontal distance rho to the origin.

    k = kwargs.get("k")
    na = kwargs.get("na")

    n_s = kwargs.get("n_s")
    n_g = kwargs.get("n_g")
    n_i = kwargs.get("n_i")
    n_g_ = kwargs.get("n_g_")
    n_i_ = kwargs.get("n_i_")

    t_g = kwargs.get("t_g")
    t_i = kwargs.get("t_i")
    t_g_ = kwargs.get("t_g_")
    t_i_ = kwargs.get("t_i_")

    def f(z, rho):
        # Returns the optical path difference multiplied by the wave number k
        tot  = n_s * z * (1 - (na * rho / n_s) ** 2) ** -0.5
        tot += n_g * t_g * (1 - (na * rho / n_g) ** 2) ** -0.5
        tot += n_i * t_i * (1 - (na * rho / n_i) ** 2) ** -0.5
        tot -= n_g_ * t_g_ * (1 - (na * rho / n_g_) ** 2) ** -0.5
        tot -= n_i_ * t_i_ * (1 - (na * rho / n_g_) ** 2) ** -0.5

        return k * tot

    return f


def path_length(**kwargs):
    # Returns a function which computes the optical path length multiplied by
    # the wave number k. This function f is a function of the depth z and the
    # horizontal distance rho.

    k = kwargs.get("k")
    na = kwargs.get("na")
    n_s = kwargs.get("n_s")

    def f(z, rho):
        # Returns the optical path length multiplied by the wave number k
        return k * n_s * z * (1 - (na * rho / n_s) ** 2) ** -0.5

    return f


def GLA_int(**kwargs):
    # Gibson Lanni with Absorption integrand
    # Returns a function of the depth z and horizontal norm rho,
    # which expresses the exponential function exp(i OPD - alpha * OPL).
    # This function is part of the integrand of the Gibson-Lanni absorption model.

    alpha = kwargs.get("alpha", 1.0) # Absorption coefficient

    OPD = path_difference(**kwargs) # Optical path difference (multiplied by constant)
    OPL = path_length(**kwargs) # Optical path length (multiplied by constant)

    def f(z, rho):
        return np.exp((1.j * OPD(z, rho) - alpha * OPL(z, rho)))

    return f


def fit_bessel_functions(f, a, upper, K):
    # Approximate the function f on the interval [0, upper] as a sum of K Bessel
    # functions J_0 by minimizing the sum of squares. Returns the coefficients
    # in the sum of Bessel functions.
    # f : function to be approximated
    # a : array of scalars inside the Bessel functions
    # upper : upper bound of the interval [0, upper]
    # K : step size in the interval [0, upper]

    x = np.linspace(0, upper, K)
    J = ss.j0(np.outer(x, a)) # Values of the Bessel functions on [0, upper]
    F = f(x)

    # Compute the minimizer of ||F - Jc||^2:
    c = np.dot(np.dot(np.linalg.inv(np.dot(J.T, J)), J.T), F)

    return c


def sum_of_bessel_functions(c, a, x):
    # Return the weighted sum over the Bessel functions c_i J_0(a_i x).
    # Inputs c, a are expected to be 1-dimensional arrays of equal size,
    # while x can be a 1-dimensional array of any size.

    return np.dot(ss.j0(np.outer(x, a)), c)


def R(r, a, upper, **kwargs):
    # Computes an array of the values of R_m(r), of which the sum are used to
    # approximate the integral.
    # r: horizontal norm sqrt(x^2 + y^2)
    # a: inner coefficients of the Bessel functions
    # upper: upper bound of the interval [0, upper] on which the integral is defined

    k = kwargs.get("k")
    NA = kwargs.get("na")

    beta = k * r * NA

    return (a * ss.j1(a * upper) * ss.j0(beta * upper) * upper - \
            beta * ss.j0(a * upper) * ss.j1(beta * upper) * upper) / \
             (a ** 2 - beta ** 2)


def PSF(rs, zs, N=50, K=100, **kwargs):
    # Computes a 2-dimensional array of the values of PSF on rs and zs
    # rs: 1-dimensional array with values r = sqrt(x^2 + y^2)
    # zs: 1-dimensional array containing the depths z
    # N: number of Bessel functions used for the approximation
    # K: number of points on which the Bessel approximation is fitted

    upper = 0.5 # Upper bound of the interval [0, upper] to be integrated over
    # Inner coefficients of the Bessel functions:
    a = (3 * np.linspace(1, N, N) - 2) / upper

    # Fit a series of Bessel functions for each depth z
    # c contains the coefficients of this series at each depth z
    cs = []
    for z in zs:
        f = lambda rho : GLA_int(**kwargs)(z, rho)
        cs.append(fit_bessel_functions(f, a, upper, K))

    # Compute the functions R used to approximate the integral
    # Rs contains these functions at each distance r
    Rs = []
    for r in rs:
        Rs.append(R(r, a, upper, **kwargs))

    # Convert to numpy arrays
    cs = np.array(cs)
    Rs = np.array(Rs)

    # Return the weighted sum over c_i R_i squared, which gives a 2-dimensional
    # array of the values of the PSF at each point (r, z)
    return np.abs(np.matmul(cs, Rs.T))**2


# Default constants
wavelength = 600e-9 # unit is meter
defaults = {
    "na"    : 1.5,                      # numerical aperture
    "n_s"   : 1.33,                     # refractive index of sample layer
    "n_g"   : 1.5,                      # actual refractive index of cover slip
    "n_i"   : 1.7,                      # actual refractive index of immersion layer
    "n_g_"  : 1.5,                      # nominal refractive index of coverslip
    "n_i_"  : 1.5,                      # nominal refractive index of immerson layer
    "t_g"   : 170e-6,                   # actual coverslip thickness
    "t_i"   : 130e-6,                   # actual distance between objective lense and coverslip
    "t_g_"  : 150e-6,                   # nominal coverslip thickness
    "t_i_"  : 150e-6,                   # nominal distance between objective lens and coverslip
    "k"     : 2 * np.pi / wavelength,   # wave number
    "alpha" : np.log(2) / (2 * np.pi / wavelength) * 10**3
            # absorption constant, assuming the illuminating halves after 1 mm
}


if __name__ == "__main__":
    # Determine horizontal and lateral boundaries
    xmin = -18e-6
    xmax = 18e-6
    ymin = 0
    ymax = 3e-5

    xs = np.linspace(xmin, xmax, 1000)
    ys = np.linspace(ymin, ymax, 1000)

    X, Y = np.meshgrid(xs, ys)

    # Compute PSF
    start_time = perf_counter()
    psf = PSF(xs, ys, **defaults)
    end_time = perf_counter()
    print("Computation time: %.3fs" % (end_time - start_time))

    # Alternatively load a PSF in:
    # psf = np.load("psf.npy")

    # Display PSF
    fig = plt.figure(figsize=(8,6))
    plt.pcolormesh(X, Y, psf, vmin=0, vmax=35e-05)
    plt.xlabel("meter")
    plt.ylabel("meter")
    plt.colorbar(format='%.e')
    # plt.savefig("psf.png" % (wavelength * 10**9)) # save figure as png
    plt.show()

    # Optionally, save PSF
    # np.save("psf.npy", psf)

