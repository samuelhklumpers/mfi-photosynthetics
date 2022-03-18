import matplotlib.pyplot as plt
import scipy.special as ss
import numpy as np
from time import perf_counter


def W(**kwargs):
    k = kwargs.get("k")
    na = kwargs.get("na")

    n_s = kwargs.get("n_s")
    n_g = kwargs.get("n_g")
    n_i = kwargs.get("n_i")
    n_g_ = kwargs.get("n_g_")
    n_i_ = kwargs.get("n_i_")

    # t_s = kwargs.get("t_s")
    t_g = kwargs.get("t_g")
    t_i = kwargs.get("t_i")
    t_g_ = kwargs.get("t_g_")
    t_i_ = kwargs.get("t_i_")

    def f(z, rho):
        tot  = n_s * z * (1 - (na * rho / n_s) ** 2) ** -0.5
        tot += n_g * t_g * (1 - (na * rho / n_g) ** 2) ** -0.5
        tot += n_i * t_i * (1 - (na * rho / n_i) ** 2) ** -0.5
        tot -= n_g_ * t_g_ * (1 - (na * rho / n_g_) ** 2) ** -0.5
        tot -= n_i_ * t_i_ * (1 - (na * rho / n_g_) ** 2) ** -0.5

        return k * tot

    return f


def path_length(**kwargs):
    k = kwargs.get("k")
    na = kwargs.get("na")
    n_s = kwargs.get("n_s")

    def f(z, rho):
        return k * n_s * z * (1 - (na * rho / n_s) ** 2) ** -0.5

    return f


def GLA_int(**kwargs):
    alpha = kwargs.get("alpha", 1.0)

    w = W(**kwargs)
    w_ = path_length(**kwargs)

    def f(z, rho):
        return np.exp((1.j * w(z, rho) - w_(z, rho) * alpha))

    return f


def fit_bessel_functions(f, a, b, K):
    # Approximate the function f on the interval [0, b] as a sum of K Bessel
    # functions J_0 by minimizing the sum of squares. Returns the coefficients
    # in the sum of Bessel functions.
    # f : function to be approximated
    # a : array of scalars inside the Bessel functions
    # b : upper bound of the interval [0, b]
    # K : step size in the interval [0, b]

    x = np.linspace(0, b, K)
    J = ss.j0(np.outer(x, a)) # Values of the Bessel functions on [0, b]
    F = f(x)

    # Compute the minimizer of ||F - Jc||^2:
    c = np.dot(np.dot(np.linalg.inv(np.dot(J.T, J)), J.T), F)

    return c


def sum_of_bessel_functions(c, a, x):
    return np.dot(ss.j0(np.outer(x, a)), c)


def R(r, a, upper, **kwargs):
    k = kwargs.get("k")
    NA = kwargs.get("na")

    beta = k * r * NA

    return (a * ss.j1(a * upper) * ss.j0(beta * upper) * upper - \
            beta * ss.j0(a * upper) * ss.j1(beta * upper) * upper) / \
             (a ** 2 - beta ** 2)


def PSF(rs, zs, **kwargs):
    N = 50
    K = 500

    #b = np.min([na, n_s, n_g, n_i, n_g_, n_i_]) / na
    upper = 0.5
    a = (3 * np.linspace(1, N, N) - 2) / upper

    c = []
    for z in zs:
        f = lambda rho : GLA_int(**kwargs)(z, rho)

        c.append(fit_bessel_functions(f, a, upper, K))

    Rs = []
    for r in rs:
        Rs.append(R(r, a, upper, **kwargs))

    c = np.array(c)
    Rs = np.array(Rs)

    return np.abs(np.matmul(c, Rs.T))**2


wavelength = 610e-9

xmax = ymax = None # 7e-4

defaults = {
    "na"    : 1.4,
    "n_s"   : 1.33,
    "n_g"   : 1.5,
    "n_i"   : 1.7,
    "n_g_"  : 1.5,
    "n_i_"  : 1.5,
    "t_s"   : 0,
    "t_g"   : 170e-6,
    "t_i"   : 130e-6,
    "t_g_"  : 170e-6,
    "t_i_"  : 150e-6,
    "C"     : 1e6,
    "k"     : 2 * np.pi / wavelength,
    "alpha" : 9e-4,
    "zd"    : 2000e-9
}


N = 50
K = 500

b = 1
a = (3 * np.linspace(1, N, N) - 2) / b

xmin = -7e-6
xmax = 7e-6

ymin = 0
ymax = 4e-5

xs = np.linspace(xmin, xmax, 255)
ys = np.linspace(ymin, ymax, 255)

X, Y = np.meshgrid(xs, ys)

start_time = perf_counter()

psf = PSF(xs, ys, **defaults)
#psf = [[ PSF(r, z, **defaults) for r in xs] for z in ys]

#psf = np.load("psf_alpha=9e-3.npy")

end_time = perf_counter()
print(end_time - start_time)

fig = plt.figure(figsize=(8,6))
plt.pcolormesh(X, Y, psf, vmin=0, vmax=np.percentile(psf, 97))
#plt.pcolormesh(X, Y, psf)
plt.show()

np.save("psf_focused.npy", psf)

print(np.percentile(psf, 95))
