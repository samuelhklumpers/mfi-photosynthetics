import numpy as np
import sympy as sy

import matplotlib.pyplot as plt

from abc import *
from sympy.integrals.integrals import *

# F = float

k = 1 # wavenumber
lims = (-1, 1)

def gauss(s):
    def u(x, y):
        return sy.exp(-(x ** 2 + y ** 2) / 2 / s ** 2) / np.sqrt(2 * np.pi) / s
    return u

def R(d): # d : F
    def f(u): # u : (F, F) -> F
        def v(x, y): # x, y : F
            xi, eta = sy.symbols("c,h")

            m = k / (2 * sy.pi * 1.j * d)
            p = (x - xi) ** 2 + (y - eta) ** 2
            integrand = u(xi, eta) * sy.exp(1.j * k / (2 * d) * p)

            z = m * Integral(Integral(integrand, (xi, *lims)), (eta, *lims))

            return z
        return v
    return f

def image(u, X, Y):
    x, y = sy.symbols("x,y")
    u = u(x, y)

    U = np.zeros(X.shape, dtype=complex)

    for i, _ in np.ndenumerate(U):
        xi, yi = X[i], Y[i]

        v = u.subs({x : xi, y : yi})
        w = v.doit()
        z = w.n(2)
        r = complex(z)

        U[i] = r

    return np.absolute(U) ** 2


def main():
    u1 = gauss(0.2)

    xs = np.linspace(-1, 1, num=20)
    X, Y = np.meshgrid(xs, xs)

    def show_im(im, title):
        plt.figure()
        plt.title(title)
        r = plt.pcolormesh(X, Y, im, shading='auto')
        plt.colorbar(r)
        plt.show(block=False)

    R1 = R(0.5)
    #Q1 = Q(-1 / 2)

    u2 = R1(u1)
    u3 = R1(u2)
    
    show_im(image(u1, X, Y), title="u1")
    show_im(image(u2, X, Y), title="u2")
    show_im(image(u3, X, Y), title="u3")
    plt.show()

    return u1, u2



if __name__ == "__main__":
    u1, u2 = main()
