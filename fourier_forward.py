import numpy as np
import matplotlib.pyplot as plt
from abc import *


# classes
class Field:
    def __init__(self, u, x, y):
        self.u = u      # U as array [[float]]
        self.x = x
        self.y = y

        self.l = 1.0
        self.k = 2 * np.pi / self.l

class OP(ABC):
    def __init__(self):
        ...

    def __call__(self, f):
        return Field(self.act(f), f.x, f.y)

    @abstractmethod
    def act(self, f):
        ...

class FunOP(OP):
    def __init__(self, fun):
        self.fun = fun

    def act(self, f):
        return self.fun(f.u)

class R(OP):
    def __init__(self, d):
        self.d = d      # distance

    def act(self, f):
        X, Y = np.meshgrid(f.x, f.y)

        def u_(x, y):
            m = (x - X) ** 2 + (y - Y) ** 2

            integrand = f.u * np.exp(1.j * f.k / (2 * self.d) * m)

            return int2(Field(integrand, f.x, f.y))

        v = np.vectorize(u_)(X, Y)

        return v / (1.j * f.l * self.d)

class Q(OP):
    def __init__(self, a):
        self.a = a      # phase

    def act(self, f):
        X, Y = np.meshgrid(f.x, f.y)

        return np.exp(1.j * f.k * self.a * (X ** 2 + Y ** 2) / 2) * f.u


# utility functions
def gauss2d(r, s):
    x = np.arange(2 * r + 1) - r
    
    c = np.exp(-x ** 2 / 2 / s ** 2) / np.sqrt(2 * np.pi) / s
    c /= c.sum()

    return np.outer(c, c)

def centered(u, dx=1, dy=1, cx=0, cy=0):
    (lx, ly) = u.shape
    x = np.linspace(-dx + cx, dx + cx, num=lx)
    y = np.linspace(-dy + cy, dy + cy, num=ly)

    return Field(u, x, y)

def int2(f):
    return np.trapz(np.trapz(f.u, x=f.x, axis=0), x=f.y, axis=0)


# demo
U = centered(gauss2d(30, 3), cx=0, cy=0)

R1 = R(0.5)
Q1 = Q(-1 / 2)

U2 = R1(R1(U))

toI = FunOP(lambda x: np.absolute(x) ** 2)

I = toI(U)
I2 = toI(U2)

def show_field(f):
    plt.figure()
    r = plt.pcolormesh(f.x, f.y, f.u, shading='auto')
    plt.colorbar(r)
    plt.show(block=False)

show_field(I)
show_field(I2)

plt.show()