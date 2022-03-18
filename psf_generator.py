import scipy.special as ss
import numpy as np
import scipy.integrate as si


def W(**kwargs):
    k = kwargs.get("k")
    na = kwargs.get("na")

    n_s = kwargs.get("n_s")
    n_g = kwargs.get("n_g")
    n_i = kwargs.get("n_i")
    n_g_ = kwargs.get("n_g_")
    n_i_ = kwargs.get("n_i_")

    t_s = kwargs.get("t_s")
    t_g = kwargs.get("t_g")
    t_i = kwargs.get("t_i")
    t_g_ = kwargs.get("t_g_")
    t_i_ = kwargs.get("t_i_")

    def f(rho):
        tot  = n_s * t_s * (1 - (na * rho / n_s) ** 2) ** -0.5
        tot += n_g * t_g * (1 - (na * rho / n_g) ** 2) ** -0.5
        tot += n_i * t_i * (1 - (na * rho / n_i) ** 2) ** -0.5
        tot -= n_g_ * t_g_ * (1 - (na * rho / n_g_) ** 2) ** -0.5
        tot -= n_i_ * t_i_ * (1 - (na * rho / n_g_) ** 2) ** -0.5

        return k * tot

    return f

def GLA_int(z, r, **kwargs):
    k = kwargs.get("k", 1.0)
    na = kwargs.get("na", 1.0)
    alpha = kwargs.get("alpha", 1.0)
    C = kwargs.get("C", 1.0)
    zd = kwargs.get("zd", 1.0)

    w = W(**kwargs)

    def f(rho):
        # moved C and zd here because of numerical issues
        # return C / zd * ss.j0(k * a * rho * r / z) * np.exp((1.j - alpha) * w(rho)) * rho
        return C / zd * ss.j0(k * na * rho * r) * np.exp((1.j - alpha) * w(rho)) * rho

    return f

def iquad(f, a, b, **kwargs):
    def fr(x):
        return f(x).real
    def fi(x):
        return f(x).imag

    (Fr, _) = si.quad(fr, a, b, **kwargs)
    (Fi, _) = si.quad(fi, a, b, **kwargs)

    return Fr + 1.j * Fi

def GLA_psf(z, xlim, ylim, **kwargs):
    #dx = kwargs.get("dx", 1.0)
    #dy = kwargs.get("dy", 1.0)

    na = kwargs.get("na")
    n_s = kwargs.get("n_s")
    n_g = kwargs.get("n_g")
    n_i = kwargs.get("n_i")
    n_g_ = kwargs.get("n_g_")
    n_i_ = kwargs.get("n_i_")

    xmax = kwargs.get("xmax", 1.0)
    ymax = kwargs.get("ymax", 1.0)

    if kwargs.get("dx", None) is not None:
        dx = dy = kwargs.get("dx")
    else:
        dx = 2 * xmax / xlim
        dy = 2 * ymax / ylim 

    zd = kwargs.get("zd", 1.0)
    C = kwargs.get("C", 1.0)

    shape = (xlim, ylim)

    psf = np.zeros(shape)

    xc, yc = -xlim / 2, -ylim / 2

    num = psf.size
    ind = 0

    upper = np.min([na, n_s, n_g, n_i, n_g_, n_i_]) / na

    for ((i, j), _) in np.ndenumerate(psf):
        x, y = (xc + i) * dx, (yc + j) * dy
        r = (x ** 2 + y ** 2) ** 0.5
        
        f = GLA_int(z, r, **kwargs)

        v = iquad(f, 0, upper)
        # print(v)

        psf[i, j] = np.abs(v) ** 2

        ind += 1

        if ind % (num // 10) == 0:
            print(ind, num)

    C = psf.sum()
    print(C)
    psf /= C

    return psf

wavelength = 610e-9

xmax = ymax = None # 7e-4
dx = dy = 800e-9

defaults = {
    "na"    : 1.4,
    "n_s"   : 1.33,
    "n_g"   : 1.5,
    "n_i"   : 1.5,
    "n_g_"  : 1.5,
    "n_i_"  : 1.5,
    "t_s"   : 0,
    "t_g"   : 170e-6,
    "t_i"   : 130e-6,
    "t_g_"  : 170e-6,
    "t_i_"  : 150e-6,
    "xmax"  : xmax,
    "ymax"  : ymax,
    "dx"    : dx,
    "C"     : 1e6,
    "k"     : 2 * np.pi / wavelength,
    "alpha" : 1e-6,
    "zd"    : 2000e-9
}

