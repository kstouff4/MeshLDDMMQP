import logging

from numba import jit, prange, int64
import numpy as np
cupy_available = True
try:
    import cupy as cp
except:
    print('cannot import cupy')
    cupy_available = False

from math import pi
from pykeops.numpy import Genred, LazyTensor
import pykeops

pkfloat = 'float64'

c_ = np.array([[1,0,0,0,0],
               [1,1,0,0,0],
               [1,1,1/3,0,0],
               [1,1,0.4,1/15,0],
               [1,1,3/7,2/21,1/105]])

c1_ = np.array([[0,0,0,0],
                [1,0,0,0],
                [1/3,1/3,0,0],
                [1/5, 1/5, 1/15, 0],
                [1/7,1/7,2/35,1/105]])

c2_ = np.array([[0,0,0],
               [0,0,0],
               [1/3,0,0],
               [1/15,1/15,0],
               [1/35,1/35,1/105]])


# Polynomial factor for Laplacian kernel
@jit(nopython=True)
def lapPol(u, ord):
    u2 = u*u
    return c_[ord,0] + c_[ord,1] * u + c_[ord,2] * u2 + c_[ord,3] * u*u2 + c_[ord,4] * u2*u2

@jit(nopython=True)
def lapPolDiff(u, ord):
    u2 = u*u
    return c1_[ord,0] + c1_[ord,1] * u + c1_[ord,2] * u2 + c1_[ord,3] * u*u2

@jit(nopython=True)
def lapPolDiff2(u, ord):
    return c2_[ord,0] + c2_[ord,1] * u + c2_[ord,2] * u*u

@jit(nopython=True)
def atanK(u):
    return np.arctan(u) + pi/2

@jit(nopython=True)
def atanKDiff(u):
    return 1/(1 + u**2)

@jit(nopython=True)
def logcoshK(u):
    v = np.fabs(u)
    return u + v + np.log1p(np.exp(-2*v))

@jit(nopython=True)
def logcoshKDiff(u):
    return 1 + np.tanh(u)

@jit(nopython=True)
def ReLUK(u):
    return (1 + np.sign(u))*u/2

@jit(nopython=True)
def ReLUKDiff(u):
    return heaviside(u)

@jit(nopython=True)
def heaviside(u):
    return (np.sign(u - 1e-8) + np.sign(u + 1e-8) + 2) / 4


## Kernel functions for numba
## Computes K(u,v)a
@jit(nopython=True)
def gauss_fun(u_, v_, a_, order):
    uv_ = ((u_ - v_) ** 2).sum() / 2
    return np.exp(-uv_) * a_

@jit(nopython=True)
def lap_fun(u_, v_, a_, order):
    uv_ = np.sqrt(((u_ - v_) ** 2).sum())
    u1 = lapPol(uv_, order)
    return u1 * np.exp(-uv_) * a_

@jit(nopython=True)
def euclidean_fun(u_, v_, a_, order):
    return (u_*v_).sum() * a_

@jit(nopython=True)
def poly_fun(u_, v_, a_, order):
    g = (u_*v_).sum()
    gk = 1.
    res = 1.
    for i in range(order):
        gk *= g
        res += gk
    return res * a_

@jit(nopython=True)
def const_fun(u_, v_, a_, order):
    return 1
@jit(nopython=True)
def min_fun(u_,v_, a_, order):
    uv_ = np.minimum(u_, v_)
    return ReLUK(uv_) * a_

def makeKij(ys_, xs_, name, order):
    if 'min' in name:
        Kij = (ys_ - xs_).ifelse(ys_.relu(), xs_.relu())
    elif 'gauss' in name:
        Dij = ((ys_ - xs_) ** 2).sum(-1)
        Kij = (-0.5 * Dij).exp()
    elif 'lap' in name:
        Dij = ((ys_ - xs_) ** 2).sum(-1).sqrt()
        polij = c_[order, 0] + c_[order, 1] * Dij + c_[order, 2] * Dij * Dij + c_[order, 3] * Dij * Dij * Dij \
                + c_[order, 4] * Dij * Dij * Dij * Dij
        Kij = polij * (-Dij).exp()
    elif 'polyCauchy' in name:
        Dij = ((ys_ - xs_) ** 2).sum(-1)
        g = (ys_ * xs_).sum(-1)
        if order == 1:
            Kij = 1 + g
        elif order == 2:
            Kij = 1 + g + g*g
        elif order == 3:
            Kij = 1 + g + g*g + g*g*g
        elif order == 4:
            Kij = 1 + g + g*g + g*g*g + g*g*g*g
        else:
            Kij = 1 + g + g*g + g*g*g + g*g*g*g + g*g*g*g*g
        Kij /= (1 + Dij)
    elif 'poly' in name:
        g = (ys_ * xs_).sum(-1)
        if order == 1:
            Kij = 1 + g
        elif order == 2:
            Kij = 1 + g + g*g
        elif order == 3:
            Kij = 1 + g + g*g + g*g*g
        elif order == 4:
            Kij = 1 + g + g*g + g*g*g + g*g*g*g
        else:
            Kij = 1 + g + g*g + g*g*g + g*g*g*g + g*g*g*g*g
    elif 'constant' in name:
        # logging.info('constant')
        Kij = 1
    else:  # Applying Euclidean kernel
        Kij = (ys_ * xs_).sum(-1)
    return Kij


## Kernel functions: differential
## compute (d_1K(u,v) * a).sum()
@jit(nopython=True)
def const_fun_diff(u_, v_, a_, order):
    return 0

@jit(nopython=True)
def euclidean_fun_diff(u_, v_, a_, order):
    return (v_ * a_).sum()

@jit(nopython=True)
def poly_fun_diff(u_, v_, a_, order):
    g = (u_*v_).sum()
    gk = 1.
    res = 1.
    for i in range(1, order):
        gk *= g
        res += (i+1) * g
    return res * (v_ * a_).sum()

# @jit(nopython=True)
# def min_fun_diffT(u_,v_, a_, order):
#     uv_ = np.minimum(u_, v_)
#     # res[k, :] += (heaviside(x[k,:]-y[l,:])*a1[k,:]*a2[l,:])*logcoshKDiff(u)/s
#     return (heaviside(v_ - u_) * a_) * ReLUKDiff(uv_)

@jit(nopython=True)
def gauss_fun_diff(u_, v_, a_, order):
    d_ = u_ - v_
    uv_ = (d_ ** 2).sum() / 2
    return - np.exp(-uv_) * (d_*a_).sum()

@jit(nopython=True)
def lap_fun_diff(u_, v_, a_, order):
    d_ = u_ - v_
    uv_ = np.sqrt((d_ ** 2).sum())
    u1 = lapPolDiff(uv_, order)
    return  - (u1 * np.exp(-uv_) * (d_*a_).sum())




## Kernel functions: differential transposed
## compute d_1K(u,v) * a.sum()
@jit(nopython=True)
def const_fun_diffT(u_, v_, a_, order):
    return 0

@jit(nopython=True)
def euclidean_fun_diffT(u_, v_, a_, order):
    return v_ * a_.sum()

@jit(nopython=True)
def poly_fun_diffT(u_, v_, a_, order):
    g = (u_*v_).sum()
    gk = 1.
    res = 1.
    for i in range(1, order):
        gk *= g
        res += (i+1) * g
    return res * v_ * a_.sum()

@jit(nopython=True)
def min_fun_diffT(u_,v_, a_, order):
    uv_ = np.minimum(u_, v_)
    # res[k, :] += (heaviside(x[k,:]-y[l,:])*a1[k,:]*a2[l,:])*logcoshKDiff(u)/s
    return (heaviside(v_ - u_) * a_) * ReLUKDiff(uv_)

@jit(nopython=True)
def gauss_fun_diffT(u_, v_, a_, order):
    d_ = u_ - v_
    uv_ = (d_ ** 2).sum() / 2
    return - d_ * np.exp(-uv_) * a_.sum()

@jit(nopython=True)
def lap_fun_diffT(u_, v_, a_, order):
    d_ = u_ - v_
    uv_ = np.sqrt((d_ ** 2).sum())
    u1 = lapPolDiff(uv_, order)
    return  - d_ * (u1 * np.exp(-uv_) * a_.sum())

def makeDiffKij(ys_, xs_, name, order):
    if name == 'min':
        Kij = (ys_ - xs_).ifelse(ys_.ifelse(1, 0), 0)
    elif 'gauss' in name:
        diffij = ys_ - xs_
        Dij = (diffij ** 2).sum(-1)
        Kij = - diffij * (-0.5 * Dij).exp()
    elif 'lap' in name:
        # if lddmm:
        diffij = ys_ - xs_
        Dij = (diffij ** 2).sum(-1).sqrt()
        Kij = - diffij * (c1_[order, 0] + c1_[order, 1] * Dij + c1_[order, 2] * Dij * Dij
                          + c1_[order, 3] * Dij * Dij * Dij) * (-Dij).exp()
    elif 'polyCauchy' in name:
        diffij = ys_ - xs_
        Dij = (diffij ** 2).sum(-1).sqrt()
        g = (ys_ * xs_).sum(-1)
        if order == 1:
            Kij = 1 + g
            dKij = 1
        elif order == 2:
            Kij = 1 + g + g*g
            dKij = 1 + 2*g
        elif order == 3:
            Kij = 1 + g + g*g + g*g*g
            dKij = 1 + 2*g + 3*g*g
        elif order == 4:
            Kij = 1 + g + g*g + g*g*g + g*g*g*g
            dKij = 1 + 2*g + 3*g*g + 4*g*g*g
        else:
            Kij = 1 + g + g*g + g*g*g + g*g*g*g + g*g*g*g*g
            dKij = 1 + 2*g + 3*g*g + 4*g*g*g + 5*g*g*g*g
        Kij = LazyTensor(np.ones(ys_.shape)) * xs_*  dKij/(1 + Dij) - 2 * diffij * Kij / (1+Dij)**2
    elif 'poly' in name:
        g = (ys_ * xs_).sum(-1)
        if order == 1:
            dKij = 1
        elif order == 2:
            dKij = 1 + 2*g
        elif order == 3:
            dKij = 1 + 2*g + 3*g*g
        elif order == 4:
            dKij = 1 + 2*g + 3*g*g + 4*g*g*g
        else:
            dKij = 1 + 2*g + 3*g*g + 4*g*g*g + 5*g*g*g*g
        Kij = LazyTensor(np.ones(ys_.shape)) * xs_ * dKij
    elif 'constant' in name:
        Kij = 0
    else:  # Euclidean kernel
        Kij = LazyTensor(np.ones(ys_.shape)) * xs_
    return Kij

## Kernel functions: second derivatives
@jit(nopython=True)
def gauss_fun_ddiff(u_, v_, a_, p_, order):
    d_ = u_ - v_
    uv_ = (d_ ** 2).sum() / 2
    h = -np.eye(u_.shape[0])
    for i in range(u_.shape[0]):
        for j in range(u_.shape[0]):
            h[i,j] += d_[i]*d_[j]
    return ( (d_*p_).sum() *d_ - p_) * np.exp(-uv_) * a_.sum()

@jit(nopython=True)
def lap_fun_ddiff(u_, v_, a_, p_, order):
    d_ = u_ - v_
    uv_ = np.sqrt((d_ ** 2).sum())
    u1 = -lapPolDiff(uv_, order)
    u2 = lapPolDiff2(uv_, order)
    return  (u2 * (d_*p_).sum() * d_ + u1 *p_) * np.exp(-uv_) * a_.sum()

def makeDDiffKij(ys_, xs_, name, order):
    if 'gauss' in name:
        diffij = ys_ - xs_
        Dij = (diffij ** 2).sum(-1)
        K1ij = (-0.5 * Dij).exp()
        K2ij = K1ij
    elif 'lap' in name:
        diffij = ys_ - xs_
        Dij = (diffij ** 2).sum(-1).sqrt()
        K1ij = (c1_[order, 0] + c1_[order, 1] * Dij + c1_[order, 2] * Dij * Dij
                          + c1_[order, 3] * Dij * Dij * Dij) * (-Dij).exp()
        K2ij = (c2_[order,0] + c2_[order,1] * Dij + c2_[order,2] * Dij * Dij) * (-Dij).exp()
    else:
        print('unknown kernel type DDiff')
        K1ij = None
        K2ij = None
        diffij = None

    return K1ij, K2ij, diffij




# @jit(nopython=True)
# def euclidean_fun_diff(u_, v_, a_, order):
#     return v_ * a_.sum()
#
# @jit(nopython=True)
# def poly_fun_diff(u_, v_, a_, order):
#     g = (u_*v_).sum()
#     gk = 1.
#     res = 1.
#     for i in range(1, order):
#         gk *= g
#         res += (i+1) * g
#     return res * v_ * a_.sum()
#

@jit(nopython=True, parallel=True)
def kernelmatrix(y, x, name, scale, ord, KP=-1):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    f = np.zeros((num_nodes_y, num_nodes))


    wsig = 0
    for s in scale:
        wsig +=  s**KP

    if 'gauss' in name:
        for k in prange(num_nodes_y):
            for l in prange(num_nodes):
                ut0 = np.sum((y[k,:] - x[l,:])**2)
                Kh = 0
                for s in scale:
                    ut = ut0 / s*s
                    if ut < 1e-8:
                        Kh += s**KP
                    else:
                        Kh += np.exp(-0.5*ut) * s**KP
                Kh /= wsig
                f[k,l] = Kh
    elif 'lap' in name:
        for k in prange(num_nodes_y):
            for l in prange(num_nodes):
                ut0 = np.sqrt(np.sum((y[k, :] - x[l, :]) ** 2))
                Kh = 0
                for s in scale:
                    ut = ut0 / s
                    if ut < 1e-8:
                        Kh += s ** KP
                    else:
                        lpt = lapPol(ut, ord)
                        Kh += lpt * np.exp(-ut) * s ** KP
                Kh /= wsig
                f[k,l] = Kh
    return f


def pick_fun(name, mode = 'base'):
    if mode == 'base':
        if 'gauss' in name:
            fun = gauss_fun
        elif 'lap' in name:
            fun = lap_fun
        elif 'min' in name:
            fun = min_fun
        elif 'poly' in name:
            fun = poly_fun
        elif 'constant' in name:
            fun = const_fun
        else:
            fun = euclidean_fun
    elif mode == 'DiffT':
        if 'gauss' in name:
            fun = gauss_fun_diffT
        elif 'lap' in name:
            fun = lap_fun_diffT
        elif 'min' in name:
            fun = min_fun_diffT
        elif 'poly' in name:
            fun = poly_fun_diffT
        elif 'constant' in name:
            fun = const_fun_diffT
        else:
            fun = euclidean_fun_diffT
    elif mode == 'Diff':
        if 'gauss' in name:
            fun = gauss_fun_diff
        elif 'lap' in name:
            fun = lap_fun_diff
        elif 'poly' in name:
            fun = poly_fun_diff
        elif 'constant' in name:
            fun = const_fun_diff
        else:
            fun = euclidean_fun_diff
    elif mode == 'DDiff':
        if 'gauss' in name:
            fun = gauss_fun_ddiff
        else:
            fun = lap_fun_ddiff
    else:
        fun = None

    return fun

def applyK(y, x, a, name, scale, order, cpu=False, dtype='float64', KP=-1):
    if not cpu and pykeops.config.gpu_available:
        return applyK_pykeops(y, x, a, name, scale, order, dtype=dtype, KP=KP)
    else:
        fun = pick_fun(name)
        return applyK_numba(y, x, a, fun , scale, order, KP=KP)


@jit(nopython=True, parallel=True)
def applyK_numba(y, x, a, fun, scale, order, KP=-1):
    res = np.zeros((y.shape[0], a.shape[1]))
    ns = len(scale)
    sKP = scale**KP
    wsig = sKP.sum()
    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        for k in prange(y.shape[0]):
            resk = np.zeros(a.shape[1])
            for l in range(x.shape[0]):
                resk += fun(ys[k, :], xs[l, :], a[l, :], order) *  sKP[s]
            res[k] += resk
    res /= wsig
    return res


def make_Kij_pykeops(y, x, name, scale, order, dtype='float64'):
    Kij_ = []
    for s in range(len(scale)):
        ys = y/scale[s]
        xs = x/scale[s]
        ys_ = LazyTensor(ys.astype(dtype)[:, None, :])
        xs_ = LazyTensor(xs.astype(dtype)[None, :, :])
        if name == 'min':
            Kij = (ys_ - xs_).ifelse(ys_.relu(), xs_.relu())
        elif 'gauss' in name:
            Dij = ((ys_ - xs_)**2).sum(-1)
            Kij = (-0.5*Dij).exp()
        elif 'lap' in name:
            Dij = ((ys_ - xs_)**2).sum(-1).sqrt()
            polij = c_[order, 0] + c_[order, 1] * Dij + c_[order, 2] * Dij * Dij + c_[order, 3] * Dij*Dij*Dij\
                    + c_[order, 4] *Dij*Dij*Dij*Dij
            Kij = polij * (-Dij).exp()
        elif 'poly' in name:
            g = (ys_*xs_).sum(-1)
            gk = LazyTensor(np.ones(ys_.shape))
            Kij = LazyTensor(np.ones(ys_.shape))
            for i in range(order):
                gk *= g
                Kij += gk
        else: #Applying Euclidean kernel
            Kij = (ys_*xs_).sum(-1)
        Kij_.append(Kij)
    return Kij_

def applyK_pykeops(y, x, a, name, scale, order, dtype='float64', KP=-1):
    res = np.zeros((y.shape[0], a.shape[1]))
    ns = len(scale)
    sKP = scale**KP
    wsig = sKP.sum()
    a_ = a.astype(dtype)
    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        ys_ = LazyTensor(ys.astype(dtype)[:, None, :])
        xs_ = LazyTensor(xs.astype(dtype)[None, :, :])
        Kij = makeKij(ys_, xs_, name, order)
        # if name == 'min':
        #     Kij = (ys_ - xs_).ifelse(ys_.relu(), xs_.relu())
        # elif 'gauss' in name:
        #     Dij = ((ys_ - xs_)**2).sum(-1)
        #     Kij = (-0.5*Dij).exp()
        # elif 'lap' in name:
        #     Dij = ((ys_ - xs_)**2).sum(-1).sqrt()
        #     polij = c_[order, 0] + c_[order, 1] * Dij + c_[order, 2] * Dij * Dij + c_[order, 3] * Dij*Dij*Dij\
        #     + c_[order, 4] *Dij*Dij*Dij*Dij
        #     Kij = polij * (-Dij).exp()
        # elif 'poly' in name:
        #     g = (ys_*xs_).sum(-1)
        #     gk = LazyTensor(np.ones(ys_.shape))
        #     Kij = LazyTensor(np.ones(ys_.shape))
        #     for i in range(order):
        #         gk *= g
        #         Kij += gk
        # else: #Applying Euclidean kernel
        #     Kij = (ys_*xs_).sum(-1)
        if name == min:
            res += Kij * a_ * sKP[s]
        else:
            res += Kij @ a_ * sKP[s]
    res /= wsig
    return res

def applyK1K2(y1, x1, name1, scale1, order1, y2, x2, name2, scale2, order2, a,
              cpu=False, dtype='float64', KP=-1):
    if not cpu and pykeops.config.gpu_available:
        return applyK1K2_pykeops(y1, x1, name1, scale1, order1, y2, x2, name2, scale2, order2, a,
                                 dtype=dtype, KP=KP)
    else:
        fun1 = pick_fun(name1)
        fun2 = pick_fun(name2)
        return applyK1K2_numba(y1, x1, fun1, scale1, order1, y2, x2, fun2, scale2, order2, a, KP=KP)

@jit(nopython=True, parallel=True)
def applyK1K2_numba(y1, x1, fun1, scale1, order1, y2, x2, fun2, scale2, order2, a, KP=-1):
    res = np.zeros((y1.shape[0], a.shape[1]))
    ns1 = len(scale1)
    s1KP = scale1**KP
    ns2 = len(scale2)
    s2KP = scale2**KP
    wsig = s1KP.sum() * s2KP.sum()
    for s1 in range(ns1):
        ys1 = y1/scale1[s1]
        xs1 = x1/scale1[s1]
        for s2 in range(ns2):
            ys2 = y2/scale2[s2]
            xs2 = x2/scale2[s2]
            for k in prange(y1.shape[0]):
                for l in range(x1.shape[0]):
                    u = fun2(ys2[k, :], xs2[l, :], a[l, :], order2)
                    res[k, :] += fun1(ys1[k, :], xs1[l, :], u, order1) * s1KP[s1] * s2KP[s2]
    res /= wsig
    return res


def applyK1K2_pykeops(y1, x1, name1, scale1, order1, y2, x2, name2, scale2, order2, a,
                                 dtype='float64', KP=-1):
    res = np.zeros((y1.shape[0], a.shape[1]))
    ns1 = len(scale1)
    s1KP = scale1**KP
    ns2 = len(scale2)
    s2KP = scale2**KP
    wsig = s1KP.sum() * s2KP.sum()
    a_ = a.astype(dtype)

    for s1 in range(ns1):
        ys1 = y1/scale1[s1]
        xs1 = x1/scale1[s1]
        ys1_ = LazyTensor(ys1.astype(dtype)[:, None, :])
        xs1_ = LazyTensor(xs1.astype(dtype)[None, :, :])
        K1ij = makeKij(ys1_, xs1_, name1, order1)
        for s2 in range(ns2):
            ys2 = y2/scale2[s2]
            xs2 = x2/scale2[s2]
            ys2_ = LazyTensor(ys2.astype(dtype)[:, None, :])
            xs2_ = LazyTensor(xs2.astype(dtype)[None, :, :])
            K2ij = makeKij(ys2_, xs2_, name2, order2)
            res += (K1ij * K2ij) @ a_ * s1KP[s1] * s2KP[s2]
    res /= wsig
    return res




## Kernel sums: first derivative
def applyDiffK(x, a1, a2, name, scale, order, cpu=False, dtype='float64', KP=-1, option='1'):
    if not cpu and pykeops.config.gpu_available:
        return applyDiffK_pykeops(x, a1, a2, name, scale, order, dtype=dtype, KP=KP, option=option)
    else:
        fun = pick_fun(name, mode = 'Diff')
        return applyDiffK_numba(x, a1, a2, fun, scale, order, KP=KP, option=option)


@jit(nopython=True, parallel=True)
def applyDiffK_numba(x, a1, a2, fun, scale, order, KP=-1, option='1'):
    res = np.zeros(x.shape)
    ns = len(scale)
    sKP1 = scale ** (KP - 1)
    sKP = scale ** (KP)
    wsig = sKP.sum()

    for s in range(ns):
        xs = x / scale[s]
        if option=='1':
            for k in prange(x.shape[0]):
                for l in range(x.shape[0]):
                    res[k, :] += fun(xs[k, :], xs[l, :], a1[k, :], order) * a2[l, :] * sKP1[s]
        elif option=='2':
            for k in prange(x.shape[0]):
                for l in range(x.shape[0]):
                    res[k, :] += fun(xs[k, :], xs[l, :], a1[l, :], order) * a2[l, :] * sKP1[s]
        elif option=='1and2':
            for k in prange(x.shape[0]):
                for l in range(x.shape[0]):
                    res[k, :] += fun(xs[k, :], xs[l, :], a1[k,:] - a1[l, :], order) * a2[l, :] * sKP1[s]
        else:
            print('DiffK: unknown option')

    res /= wsig
    return res

# def makeDiffKij(ys_, xs_, name, order):
#     if name == 'min':
#         Kij = (ys_ - xs_).ifelse(ys_.ifelse(1, 0), 0)
#     elif 'gauss' in name:
#         diffij = ys_ - xs_
#         Dij = (diffij ** 2).sum(-1)
#         Kij = - diffij * (-0.5 * Dij).exp()
#     elif 'lap' in name:
#         # if lddmm:
#         diffij = ys_ - xs_
#         Dij = (diffij ** 2).sum(-1).sqrt()
#         Kij = - diffij * (c1_[order, 0] + c1_[order, 1] * Dij + c1_[order, 2] * Dij * Dij
#                           + c1_[order, 3] * Dij * Dij * Dij) * (-Dij).exp()
#     elif 'polyCauchy' in name:
#         diffij = ys_ - xs_
#         Dij = (diffij ** 2).sum(-1).sqrt()
#         g = (ys_ * xs_).sum(-1)
#         if order == 1:
#             Kij = 1 + g
#             dKij = 1
#         elif order == 2:
#             Kij = 1 + g + g*g
#             dKij = 1 + 2*g
#         elif order == 3:
#             Kij = 1 + g + g*g + g*g*g
#             dKij = 1 + 2*g + 3*g*g
#         elif order == 4:
#             Kij = 1 + g + g*g + g*g*g + g*g*g*g
#             dKij = 1 + 2*g + 3*g*g + 4*g*g*g
#         else:
#             Kij = 1 + g + g*g + g*g*g + g*g*g*g + g*g*g*g*g
#             dKij = 1 + 2*g + 3*g*g + 4*g*g*g + 5*g*g*g*g
#         Kij = LazyTensor(np.ones(ys_.shape)) * xs_*  dKij/(1 + Dij) - 2 * diffij * Kij / (1+Dij)**2
#     elif 'poly' in name:
#         g = (ys_ * xs_).sum(-1)
#         if order == 1:
#             dKij = 1
#         elif order == 2:
#             dKij = 1 + 2*g
#         elif order == 3:
#             dKij = 1 + 2*g + 3*g*g
#         elif order == 4:
#             dKij = 1 + 2*g + 3*g*g + 4*g*g*g
#         else:
#             dKij = 1 + 2*g + 3*g*g + 4*g*g*g + 5*g*g*g*g
#         Kij = LazyTensor(np.ones(ys_.shape)) * xs_ * dKij
#     elif 'constant' in name:
#         Kij = 0
#     else:  # Euclidean kernel
#         Kij = LazyTensor(np.ones(ys_.shape)) * xs_
#     return Kij

def applyDiffK_pykeops(x, a1, a2, name, scale, order, dtype='float64', KP=-1, option = '1'):
    res = np.zeros(x.shape)
    ns = len(scale)
    sKP1 = scale**(KP-1)
    sKP = scale**(KP)
    wsig = sKP.sum()
    a1_ = a1.astype(dtype)
    a2_ = a2.astype(dtype)
    a1i_ = LazyTensor(a1_[:, None, :])
    a1j_ = LazyTensor(a1_[None, :, ])
    a2j_ = LazyTensor(a2_[None, :, :])

    for s in range(ns):
        xs = x/scale[s]
        xsi = LazyTensor(xs.astype(dtype)[:, None, :])
        xsj = LazyTensor(xs.astype(dtype)[None, :, :])
        Kij = makeDiffKij(xsi, xsj, name, order)
        if option == '1':
            res += (sKP1[s]) * ((Kij * a1i_).sum(-1)*a2j_).sum(1)
        elif option == '2':
            res += (sKP1[s]) * ((Kij * a1j_).sum(-1) * a2j_).sum(1)
        elif option == '1and2':
            res += (sKP1[s]) * ((Kij * (a1i_- a1j_)).sum(-1) * a2j_).sum(1)
        else:
            print('DiffK: unknown option')
    res /= wsig
    return res


##Transposed
def applyDiffKT(y, x, p, a, name, scale, order, regweight=1., lddmm=False, cpu=False, dtype='float64', KP=-1):
    if not cpu and pykeops.config.gpu_available:
        return applyDiffKT_pykeops(y, x, p, a, name, scale, order, regweight=regweight, lddmm=lddmm, dtype=dtype, KP=KP)
    else:
        fun = pick_fun(name, mode = 'DiffT')
        return applyDiffKT_numba(y, x, p, a, fun, scale, order, regweight=regweight, lddmm=lddmm, KP=KP)

@jit(nopython=True, parallel=True)
def applyDiffKT_numba(y, x, p, a, fun, scale, order, regweight=1., lddmm=False, KP=-1):
    res = np.zeros(y.shape)
    ns = len(scale)
    sKP1 = scale**(KP-1)
    sKP = scale**(KP)
    wsig = sKP.sum()
    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        for k in prange(y.shape[0]):
            for l in range(x.shape[0]):
                if lddmm:
                    akl = p[k, :] * a[l, :] + a[k, :] * p[l, :] - 2 * regweight * a[k, :] * a[l, :]
                else:
                    akl = p[k, :] * a[l, :]
                res[k, :] += fun(ys[k, :], xs[l, :], akl, order) * sKP1[s]
    res /= wsig
    return res


def applyDiffKT_pykeops(y, x, p, a, name, scale, order, regweight=1., lddmm=False, dtype='float64', KP=-1):
    res = np.zeros(y.shape)
    ns = len(scale)
    sKP1 = scale**(KP-1)
    sKP = scale**(KP)
    wsig = sKP.sum()
    p_ = p.astype(dtype)
    a_ = a.astype(dtype)
    pi_ = LazyTensor(p_[:, None, :])
    aj_ = LazyTensor(a_[None, :, :])
    if lddmm:
        pj_ = LazyTensor(p_[None, :, :])
        ai_ = LazyTensor(a_[:, None, :])
        ap_ = (pi_ * aj_ + ai_ * pj_ - 2 * regweight * ai_ * aj_).sum(-1)
    else:
        ap_ = (pi_*aj_).sum(-1)

    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        ys_ = LazyTensor(ys.astype(dtype)[:, None, :])
        xs_ = LazyTensor(xs.astype(dtype)[None, :, :])
        Kij = makeDiffKij(ys_, xs_, name, order)
        # if name == 'min':
        #     Kij = (ys_-xs_).ifelse(ys_.ifelse(1,0), 0)
        # elif 'gauss' in name:
        #     diffij = ys_ - xs_
        #     Dij = (diffij ** 2).sum(-1)
        #     Kij = diffij * (-0.5 * Dij).exp()
        # elif 'lap' in name:
        #     diffij = ys_ - xs_
        #     Dij = (diffij ** 2).sum(-1).sqrt()
        #     Kij = diffij * (c1_[order, 0] + c1_[order, 1] * Dij + c1_[order, 2] * Dij * Dij
        #     + c1_[order, 3] * Dij * Dij * Dij) * (-Dij).exp()
        # elif 'poly' in name:
        #     g = (ys_ * xs_).sum(-1)
        #     gk = LazyTensor(np.ones(ys_.shape))
        #     Kij = LazyTensor(np.ones(ys_.shape))
        #     for i in range(1, order):
        #         gk *= g
        #         Kij += (i+1) * gk
        # else: # Euclidean kernel
        #     Kij = LazyTensor(np.ones(ys_.shape)) * xs_
        res += (sKP1[s]) * (Kij * ap_).sum(1)
    res /= wsig
    return res

def applyDiffK1K2T(y1, x1, name1, scale1, order1, y2, x2, name2, scale2, order2, p, a,
              regweight=1., lddmm=False, cpu=False, dtype='float64', KP=-1):
    if not cpu and pykeops.config.gpu_available:
        return applyDiffK1K2T_pykeops(y1, x1, name1, scale1, order1, y2, x2, name2, scale2, order2, p, a,
                                 regweight=regweight, lddmm=lddmm, dtype=dtype, KP=KP)
    else:
        fun1 = pick_fun(name1, mode='DiffT')
        fun2 = pick_fun(name2)
        return applyDiffK1K2T_numba(y1, x1, fun1, scale1, order1, y2, x2, fun2, scale2, order2, p, a,
                                    regweight=regweight, lddmm=lddmm, KP=KP)

@jit(nopython=True, parallel=True)
def applyDiffK1K2T_numba(y1, x1, fun1, scale1, order1, y2, x2, fun2, scale2, order2,
                         p, a, regweight=1., lddmm=False, KP=-1):
    res = np.zeros(y1.shape)
    ns1 = len(scale1)
    s1KP = scale1 ** (KP-1)
    ns2 = len(scale2)
    s2KP = scale2 ** KP
    wsig = (scale1**KP).sum() * (scale2**KP).sum()
    for s1 in range(ns1):
        ys1 = y1 / scale1[s1]
        xs1 = x1 / scale1[s1]
        for s2 in range(ns2):
            ys2 = y2 / scale2[s2]
            xs2 = x2 / scale2[s2]
            for k in prange(y1.shape[0]):
                for l in range(x1.shape[0]):
                    if lddmm:
                        akl = p[k, :] * a[l, :] + a[k, :] * p[l, :] - 2 * regweight * a[k, :] * a[l, :]
                    else:
                        akl = p[k, :] * a[l, :]
                    u = fun2(ys2[k, :], xs2[l, :], akl, order2)
                    res[k, :] += fun1(ys1[k, :], xs1[l, :], u, order1) * s1KP[s1] * s2KP[s2]
    res /= wsig
    return res

def applyDiffK1K2T_pykeops(y1, x1, name1, scale1, order1, y2, x2, name2, scale2, order2, p, a,
                                 regweight=1., lddmm=False, dtype='float64', KP=-1):
    res = np.zeros(y1.shape)
    ns1 = len(scale1)
    s1KP = scale1 ** (KP-1)
    ns2 = len(scale2)
    s2KP = scale2 ** KP
    wsig = (scale1**KP).sum() * (scale2**KP).sum()
    a_ = a.astype(dtype)
    p_ = p.astype(dtype)
    pi_ = LazyTensor(p_[:, None, :])
    aj_ = LazyTensor(a_[None, :, :])
    if lddmm:
        pj_ = LazyTensor(p_[None, :, :])
        ai_ = LazyTensor(a_[:, None, :])
        ap_ = (pi_ * aj_ + ai_ * pj_ - 2 * regweight * ai_ * aj_).sum(-1)
    else:
        ap_ = (pi_*aj_).sum(-1)


    for s1 in range(ns1):
        ys1 = y1/scale1[s1]
        xs1 = x1/scale1[s1]
        ys1_ = LazyTensor(ys1.astype(dtype)[:, None, :])
        xs1_ = LazyTensor(xs1.astype(dtype)[None, :, :])
        K1ij = makeDiffKij(ys1_, xs1_, name1, order1)
        for s2 in range(ns2):
            ys2 = y2/scale2[s2]
            xs2 = x2/scale2[s2]
            ys2_ = LazyTensor(ys2.astype(dtype)[:, None, :])
            xs2_ = LazyTensor(xs2.astype(dtype)[None, :, :])
            K2ij = makeKij(ys2_, xs2_, name2, order2)
            res += ((K1ij * K2ij) * ap_).sum(1) * s1KP[s1] * s2KP[s2]
    res /= wsig
    return res


@jit(nopython=True, parallel=True)
def applyDiv(y, x, a, name, scale, order, KP=-1):
    res = np.zeros((y.shape[0], 1))
    sc = scale**(KP-1)
    if name == 'min':
        for k in prange(y.shape[0]):
            for l in range(x.shape[0]):
                for kk,s in enumerate(scale):
                    u = np.minimum(y[k,:],x[l,:]) / s
                    #res[k, :] += (heaviside(x[k,:]-y[l,:])*a[l,:]*logcoshKDiff(u)/s).sum()
                    res[k, :] += (heaviside(x[k,:]-y[l,:])*a[l,:]*ReLUKDiff(u)/s).sum()
    elif 'gauss' in name:
        for k in prange(y.shape[0]):
            for l in range(x.shape[0]):
                for kk,s in enumerate(scale):
                    res[k, :] += ((y[k,:]-x[l,:])*a[l,:]).sum() * \
                                 (-np.exp(- ((y[k,:]-x[l,:])**2).sum()/(2*s**2)))*sc[kk]
    elif 'lap' in name:
        for k in prange(y.shape[0]):
            for l in range(x.shape[0]):
                for kk,s in enumerate(scale):
                    u = np.sqrt(((y[k,:] - x[l,:]) ** 2).sum()) / s
                    res[k, :] += ((y[k,:]-x[l,:])*a[l,:]).sum() * (-lapPolDiff(u, order) * np.exp(- u) *sc[kk])
    res /= (scale**KP).sum()
    return res


def applykdiff1(x, a1, a2, name, scale, order, KP=-1):
    return applyDiffK(x, a1, a2, name, scale, order, KP=-1, option='1')

def applykdiff2(x, a1, a2, name, scale, order, KP=-1):
    return applyDiffK(x, a1, a2, name, scale, order, KP=-1, option='2')

def applykdiff1and2(x, a1, a2, name, scale, order, KP=-1):
    return applyDiffK(x, a1, a2, name, scale, order, KP=-1, option='1and2')

@jit(nopython=True, parallel=True)
def applykdiff1_(x, a1, a2, name, scale, order, KP=-1):
    num_nodes = x.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes, dim))
    wsig = 0
    for s in scale:
        wsig += s ** KP

    if 'gauss' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
             ut0 = ((x[k,:] - x[l,:])**2).sum()
             Kh_diff = 0
             for s in scale:
                ut = ut0 / s**2
                if ut < 1e-8:
                    Kh_diff -= 0.5*s**(KP-2)
                else:
                    Kh_diff -= 0.5*np.exp(-0.5*ut)*s**(KP-2)
                Kh_diff /= wsig
                df += Kh_diff * 2 * ((x[k, :] - x[l, :]) * a1[k, :]).sum() * a2[l, :]
            f[k, :] += df
    elif 'lap' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                ut0 = np.sqrt(((x[k, :] - x[l, :]) ** 2).sum())
                Kh_diff = 0
                for s in scale:
                    ut = ut0 / s
                    if ut < 1e-8:
                        Kh_diff -= 0.5*lapPolDiff(0,order)*s**(KP-2)
                    else:
                        Kh_diff -= 0.5 * lapPolDiff(ut,order) * np.exp(-1.0*ut)*s**(KP-2)
                Kh_diff /= wsig
                df += Kh_diff * 2*((x[k,:]-x[l,:])*a1[k,:]).sum() * a2[l,:]
            f[k, :] += df
    return f

@jit(nopython=True, parallel=True)
def applykdiff2_(x, a1, a2, name, scale, order, KP=-1):
    num_nodes = x.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes, dim))
    wsig = 0
    for s in scale:
        wsig += s ** KP

    if 'gauss' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                ut0 = ((x[k, :] - x[l, :]) ** 2).sum()
                Kh_diff = 0
                for s in scale:
                    ut = ut0 / s ** 2
                    if ut < 1e-8:
                        Kh_diff -= 0.5 * s ** (KP - 2)
                    else:
                        Kh_diff -= 0.5 * np.exp(-0.5 * ut) * s ** (KP - 2)
                    Kh_diff /= wsig
                    df += Kh_diff * 2 * ((x[k, :] - x[l, :]) * a1[l, :]).sum() * a2[l, :]
            f[k, :] += df
    elif 'lap' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                ut0 = np.sqrt(((x[k, :] - x[l, :]) ** 2).sum())
                Kh_diff = 0
                for s in scale:
                    ut = ut0 / s
                    if ut < 1e-8:
                        Kh_diff -= 0.5 * lapPolDiff(0, order) * s ** (KP - 2)
                    else:
                        Kh_diff -= 0.5 * lapPolDiff(ut, order) * np.exp(-1.0 * ut) * s ** (KP - 2)
                Kh_diff /= wsig
                df += Kh_diff * 2 * ((x[k, :] - x[l, :]) * a1[l, :]).sum() * a2[l, :]
            f[k, :] += df
    return f

@jit(nopython=True, parallel=True)
def applykdiff1and2_(x, a1, a2, name, scale, order, KP=-1):
    num_nodes = x.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes, dim))
    wsig = 0
    for s in scale:
        wsig += s ** KP

    if 'gauss' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                dx = x[k,:] - x[l,:]
                ut0 = ((x[k, :] - x[l, :]) ** 2).sum()
                Kh_diff = 0
                for s in scale:
                    ut = ut0 / s ** 2
                    if ut < 1e-8:
                        Kh_diff -= 0.5 * s ** (KP - 2)
                    else:
                        Kh_diff -= 0.5 * np.exp(-0.5 * ut) * s ** (KP - 2)
                Kh_diff /= wsig
                df += Kh_diff * 2 * (dx*(a1[k, :] - a1[l, :])).sum() * a2[l, :]
            f[k, :] += df
    elif 'lap' in name:
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in range(num_nodes):
                dx = x[k,:] - x[l,:]
                ut0 = np.sqrt(((x[k, :] - x[l, :]) ** 2).sum())
                Kh_diff = 0
                for s in scale:
                    ut = ut0 / s
                    if ut < 1e-8:
                        Kh_diff -= 0.5 * lapPolDiff(0, order) * s ** (KP - 2)
                    else:
                        Kh_diff -= 0.5 * lapPolDiff(ut, order) * np.exp(-1.0 * ut) * s ** (KP - 2)
                Kh_diff /= wsig
                df += Kh_diff * 2 * (dx*(a1[k, :] - a1[l, :])).sum() * a2[l, :]
            f[k, :] += df
    return f


## Second derivatives
def applyDDiffK(x, a1, a2, p, name, scale, order, cpu=False, dtype='float64', KP=-1, option='11'):
    if not cpu and pykeops.config.gpu_available:
        return applyDDiffK_pykeops(x, a1, a2, p, name, scale, order, dtype=dtype, KP=KP, option=option)
    else:
        fun = pick_fun(name, mode = 'DDiff')
        return applyDDiffK_numba(x, a1, a2, p, fun, scale, order, KP=KP, option=option)

@jit(nopython=True, parallel=True)
def applyDDiffK_numba(x, a1, a2, p, fun, scale, order, option = '11and12', KP=-1):
    num_nodes = x.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes, dim))
    wsig = 0
    for s in scale:
        wsig += s ** KP

    if option == '11':
        for s in scale:
            xs = x/s
            KS = s**(KP-2)
            for k in prange(num_nodes):
                df = np.zeros(dim)
                for l in prange(num_nodes):
                    df += fun(xs[k,:], xs[l,:], a1[k,:] * a2[l,:], p[k,:], order)
                f[k, :] += df * KS
    elif option == '12':
        for s in scale:
            xs = x/s
            KS = s**(KP-2)
            for k in prange(num_nodes):
                df = np.zeros(dim)
                for l in prange(num_nodes):
                    df -= fun(xs[k,:], xs[l,:], a1[k,:] * a2[l,:], p[l,:], order)
                f[k, :] += df * KS
    elif option=='11and12':
        for s in scale:
            xs = x/s
            KS = s**(KP-2)
            for k in prange(num_nodes):
                df = np.zeros(dim)
                for l in prange(num_nodes):
                    df += fun(xs[k,:], xs[l,:], a1[k,:] * a2[l,:], p[k,:]-p[l, :], order)
                f[k, :] += df * KS
    else:
        print('Unknown option in second kernel derivative')
    return f/wsig

def applyDDiffK_pykeops(x, a1, a2, p, name, scale, order, option = '11and12', dtype='float64', KP=-1):
    num_nodes = x.shape[0]
    dim = x.shape[1]
    res = np.zeros((num_nodes, dim))
    wsig = 0
    for s in scale:
        wsig += s ** KP

    a1_ = a1.astype(dtype)
    a2_ = a2.astype(dtype)
    p_ = p.astype(dtype)
    pi_ = LazyTensor(p_[:, None, :])
    pj_ = LazyTensor(p_[None, :, ])
    a1i_ = LazyTensor(a1_[:, None, :])
    a2j_ = LazyTensor(a2_[None, :, :])
    aij = (a1i_*a2j_).sum(-1)

    for s in scale:
        xs = x/s
        KS = s**(KP-2)
        xsi = LazyTensor(xs.astype(dtype)[:, None, :])
        xsj = LazyTensor(xs.astype(dtype)[None, :, :])
        K1ij, K2ij, Dij = makeDDiffKij(xsi, xsj, name, order)
        # print(K1ij.shape, K2ij.shape, Dij.shape)
        # print('Tensors 1', (K2ij * (Dij * (pi_ -pj_)).sum(-1) * Dij).shape)
        # print('Tensors 2', K1ij * (pi_-pj_))
        # print('Tensors 3', ((K2ij * (Dij * (pi_ -pj_)).sum(-1) * Dij - K1ij * (pi_-pj_)) * (a1i_ * a2j_).sum(-1) * KS).shape)
        if option == '11':
            res += ((K2ij * (Dij*pi_).sum(-1) * Dij - K1ij*pi_) *(a1i_ * a2j_).sum(-1)).sum(axis=1)* KS
        elif option == '12':
            res -= ((K2ij * (Dij * pj_).sum(-1) * Dij - K1ij * pj_) * (a1i_ * a2j_).sum(-1)).sum(axis=1) * KS
        elif option=='11and12':
            dres = (K2ij * (Dij * (pi_ -pj_)).sum(-1) * Dij - K1ij * (pi_-pj_)) * (a1i_ * a2j_).sum(-1) * KS
            res = res + dres.sum(axis=1)
        else:
            print('Unknown option in second kernel derivative')
            return res
    return res/wsig

@jit(nopython=True, parallel=True)
def applykdiff12_numba(x, a1, a2, p, fun, scale, order, KP=-1):
    num_nodes = x.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes, dim))
    wsig = 0
    for s in scale:
        wsig += s ** KP

    for s in scale:
        xs = x/s
        KS = s**(KP-2)
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in prange(num_nodes):
                df -= fun(xs[k,:], xs[l,:], a1[k,:] * a2[l,:], p[l,:], order)
            f[k, :] += df * KS
    return  f

@jit(nopython=True, parallel=True)
def applykdiff11and12_numba(x, a1, a2, p, fun, scale, order, KP=-1):
    num_nodes = x.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes, dim))
    wsig = 0
    for s in scale:
        wsig += s ** KP

    for s in scale:
        xs = x/s
        KS = s**(KP-2)
        for k in prange(num_nodes):
            df = np.zeros(dim)
            for l in prange(num_nodes):
                df = fun(xs[k,:], xs[l,:], a1[k,:] * a2[l,:], p[k,:] - p[l,:], order)
            f[k, :] += df * KS
    return  f

def applykdiff11(x, a1, a2, p, name, scale, order, KP=-1):
    return applyDDiffK(x, a1, a2, p, name, scale, order, KP=KP, option='11')

def applykdiff12(x, a1, a2, p, name, scale, order, KP=-1):
    return applyDDiffK(x, a1, a2, p, name, scale, order, KP=KP, option='12')

def applykdiff11and12(x, a1, a2, p, name, scale, order, KP=-1):
    return applyDDiffK(x, a1, a2, p, name, scale, order, KP=KP, option='11and12')

#
# @jit(nopython=True, parallel=True)
# def applykdiff11_(x, a1, a2, p, name, scale, order, KP=-1):
#     num_nodes = x.shape[0]
#     dim = x.shape[1]
#     f = np.zeros((num_nodes, dim))
#     wsig = 0
#     for s in scale:
#         wsig += s ** KP
#
#     if 'gauss' in name:
#         for k in prange(num_nodes):
#             df = np.zeros(dim)
#             for l in range(num_nodes):
#                 ut0 = ((x[k,:] - x[l,:])**2).sum()
#                 Kh_diff = 0
#                 Kh_diff2 = 0
#                 for s in scale:
#                     ut = ut0/s**2
#                     if ut < 1e-8:
#                         Kh_diff -= 0.5*s**(KP-2)
#                         Kh_diff2 += 0.25 * s**(KP-4)
#                     else:
#                         Kh_diff -= 0.5*np.exp(-0.5*ut)*s**(KP-2)
#                         Kh_diff2 += 0.25*np.exp(-0.5*ut)*s**(KP-4)
#                 Kh_diff /= wsig
#                 Kh_diff2 /= wsig
#                 df += Kh_diff2 * 4 * (a1[k, :] * a2[l, :]).sum() \
#                       * ((x[k, :] - x[l, :]) * p[k, :]).sum() * (x[k, :] - x[l, :]) \
#                       + Kh_diff * 2 * (a1[k, :] * a2[l, :]).sum() * p[k, :]
#             f[k, :] += df
#     if 'lap' in name:
#         for k in prange(num_nodes):
#             df = np.zeros(dim)
#             for l in range(num_nodes):
#                 ut0 = np.sqrt(((x[k, :] - x[l, :]) ** 2).sum())
#                 Kh_diff = 0
#                 Kh_diff2 = 0
#                 for s in scale:
#                     ut = ut0 / s
#                     if ut < 1e-8:
#                         Kh_diff -= 0.5*lapPolDiff(0, order)*s**(KP-2)
#                         Kh_diff2 += 0.25*lapPolDiff2(0, order)*s**(KP-4)
#                     else:
#                         Kh_diff -= 0.5*lapPolDiff(ut, order) * np.exp(-ut)*s**(KP-2)
#                         Kh_diff2 += 0.25*lapPolDiff2(ut, order) * np.exp(-ut)*s**(KP-4)
#                 Kh_diff /= wsig
#                 Kh_diff2 /= wsig
#                 df += Kh_diff2 * 4*(a1[k,:] * a2[l,:]).sum() \
#                       * ((x[k,:]-x[l,:])*p[k,:]).sum() *(x[k,:]-x[l,:]) \
#                       + Kh_diff * 2 * (a1[k,:] * a2[l,:]).sum() * p[k,:]
#             f[k, :] += df
#
#     return  f
#
#
#
# @jit(nopython=True, parallel=True)
# def applykdiff12_(x, a1, a2, p, name, scale, order, KP=-1):
#     num_nodes = x.shape[0]
#     dim = x.shape[1]
#     f = np.zeros((num_nodes, dim))
#     wsig = 0
#     for s in scale:
#         wsig += s ** KP
#
#     if 'gauss' in name:
#         for k in prange(num_nodes):
#             df = np.zeros(dim)
#             for l in range(num_nodes):
#                 ut0 = ((x[k, :] - x[l, :]) ** 2).sum()
#                 Kh_diff = 0
#                 Kh_diff2 = 0
#                 for s in scale:
#                     ut = ut0 / s ** 2
#                     if ut < 1e-8:
#                         Kh_diff -= 0.5 * s ** (KP - 2)
#                         Kh_diff2 += 0.25 * s ** (KP - 4)
#                     else:
#                         Kh_diff -= 0.5 * np.exp(-0.5 * ut) * s ** (KP - 2)
#                         Kh_diff2 += 0.25 * np.exp(-0.5 * ut) * s ** (KP - 4)
#                 Kh_diff /= wsig
#                 Kh_diff2 /= wsig
#                 df += -Kh_diff2 * 4 * (a1[k, :] * a2[l, :]).sum() \
#                       * ((x[k, :] - x[l, :]) * p[l, :]).sum() * (x[k, :] - x[l, :]) \
#                       - Kh_diff * 2 * (a1[k, :] * a2[l, :]).sum() * p[l, :]
#             f[k, :] += df
#     if 'lap' in name:
#         for k in prange(num_nodes):
#             df = np.zeros(dim)
#             for l in range(num_nodes):
#                 ut0 = np.sqrt(((x[k, :] - x[l, :]) ** 2).sum())
#                 Kh_diff = 0
#                 Kh_diff2 = 0
#                 for s in scale:
#                     ut = ut0 / s
#                     if ut < 1e-8:
#                         Kh_diff -= 0.5 * lapPolDiff(0, order) * s ** (KP - 2)
#                         Kh_diff2 += 0.25 * lapPolDiff2(0, order) * s ** (KP - 4)
#                     else:
#                         Kh_diff -= 0.5 * lapPolDiff(ut, order) * np.exp(-ut) * s ** (KP - 2)
#                         Kh_diff2 += 0.25 * lapPolDiff2(ut, order) * np.exp(-ut) * s ** (KP - 4)
#                 Kh_diff /= wsig
#                 Kh_diff2 /= wsig
#                 df += -Kh_diff2 * 4 * (a1[k, :] * a2[l, :]).sum() \
#                       * ((x[k, :] - x[l, :]) * p[l, :]).sum() * (x[k, :] - x[l, :]) \
#                       - Kh_diff * 2 * (a1[k, :] * a2[l, :]).sum() * p[l, :]
#             f[k, :] += df
#
#     return f
#
#
# @jit(nopython=True, parallel=True)
# def applykdiff11and12_(x, a1, a2, p, name, scale, order, KP=-1):
#     num_nodes = x.shape[0]
#     dim = x.shape[1]
#     f = np.zeros((num_nodes, dim))
#     wsig = 0
#     for s in scale:
#         wsig += s ** KP
#
#     if 'gauss' in name:
#         for k in prange(num_nodes):
#             df = np.zeros(dim)
#             for l in range(num_nodes):
#                 dx = x[k, :] - x[l, :]
#                 dp = p[k, :] - p[l, :]
#                 ut0 = ((x[k, :] - x[l, :]) ** 2).sum()
#                 Kh_diff = 0
#                 Kh_diff2 = 0
#                 for s in scale:
#                     ut = ut0 / s ** 2
#                     if ut < 1e-8:
#                         Kh_diff -= 0.5 * s ** (KP - 2)
#                         Kh_diff2 += 0.25 * s ** (KP - 4)
#                     else:
#                         Kh_diff -= 0.5 * np.exp(-0.5 * ut) * s ** (KP - 2)
#                         Kh_diff2 += 0.25 * np.exp(-0.5 * ut) * s ** (KP - 4)
#                 Kh_diff /= wsig
#                 Kh_diff2 /= wsig
#                 df += 2 * (a1[k,:]*a2[l,:]).sum() *  (2 * Kh_diff2 *(dx*dp).sum() *dx + Kh_diff * dp)
#             f[k, :] += df
#     if 'lap' in name:
#         for k in prange(num_nodes):
#             df = np.zeros(dim)
#             for l in range(num_nodes):
#                 dx = x[k, :] - x[l, :]
#                 dp = p[k, :] - p[l, :]
#                 ut0 = np.sqrt(((x[k, :] - x[l, :]) ** 2).sum())
#                 Kh_diff = 0
#                 Kh_diff2 = 0
#                 for s in scale:
#                     ut = ut0 / s
#                     if ut < 1e-8:
#                         Kh_diff -= 0.5 * lapPolDiff(0, order) * s ** (KP - 2)
#                         Kh_diff2 += 0.25 * lapPolDiff2(0, order) * s ** (KP - 4)
#                     else:
#                         Kh_diff -= 0.5 * lapPolDiff(ut, order) * np.exp(-ut) * s ** (KP - 2)
#                         Kh_diff2 += 0.25 * lapPolDiff2(ut, order) * np.exp(-ut) * s ** (KP - 4)
#                 Kh_diff /= wsig
#                 Kh_diff2 /= wsig
#                 df += 2 * (a1[k,:]*a2[l,:]).sum() *  (2 * Kh_diff2 *(dx*dp).sum() *dx + Kh_diff * dp)
#             f[k, :] += df
#
#     return f

def applyktensor(y, x, ay, ax, betay, betax, name, scale, order, cpu=False, dtype='float64', KP=-1):
    if not cpu and pykeops.config.gpu_available:
       # res1 = applyktensor_numba(y, x, ay, ax, betay, betax, name, scale, order)
        res2 = applyktensor_pykeops(y, x, ay, ax, betay, betax, name, scale, order, dtype=dtype, KP=KP)
        return res2 
    else:
        return applyktensor_numba(y, x, ay, ax, betay, betax, name, scale, order, KP=KP)



def applyktensor_pykeops(y, x, ay, ax, betay, betax, name, scale, order, dtype='float64', KP=-1):
    dim = x.shape[1]
    num_nodes_y = y.shape[0]
    res = np.zeros(num_nodes_y)

    sKP = scale**KP
    wsig = sKP.sum()
    ns = len(scale)
    ay_ = LazyTensor(ay.astype(dtype)[:, None], axis=0)
    ax_ = LazyTensor(ax.astype(dtype)[:, None], axis=1)
    betay_ = LazyTensor(betay.astype(dtype)[:, None, :])
    betax_ = LazyTensor(betax.astype(dtype)[None, :, :])
    ayx = (ay_ * ax_).sum(-1)
#    ayx_n = (ay[:, None, None] * ax[None, :, None]).sum(axis=-1)
    betayx =  ((betay_ * betax_).sum(axis=2))**2
#    betayx_n = (betay[:, None, :] * betax[None, :, :]).sum(axis=2)**2

    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        ys_ = LazyTensor(ys.astype(dtype)[:, None, :])
        xs_ = LazyTensor(xs.astype(dtype)[None, :, :])
        if 'gauss' in name:
            Dij = ((ys_ - xs_)**2).sum(-1)
            Kij = (-0.5*Dij).exp()
        elif 'lap' in name:
            Dij = ((ys_ - xs_) ** 2).sum(-1).sqrt()
            polij = c_[order, 0] + c_[order, 1] * Dij + c_[order, 2] * Dij * Dij + c_[order, 3] * Dij * Dij * Dij \
                    + c_[order, 4] * Dij * Dij * Dij * Dij
            Kij = polij * (-Dij).exp()
        else: #Applying Euclidean kernel
            Kij = (ys_*xs_).sum(-1)

#        Kij_n = np.exp(-((ys[:, None, :] - xs[None, :, :])**2).sum(axis=-1)/2)
#        dres_n = (Kij_n *(ayx_n + betayx_n)).sum(axis=1)*sKP[s]
        dres = (Kij * (ayx + betayx)).sum(1) * sKP[s]
        res += dres[:, 0]

    res /= wsig
    return np.array(res)

@jit(nopython=True, parallel=True)
def applyktensor_numba(y, x, ay, ax, betay, betax, name, scale, order, KP=-1):
    dim = x.shape[1]
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dimb = betax.shape[1]
    f = np.zeros(num_nodes_y)

    sKP = scale**KP
    wsig = sKP.sum()
    ns = len(scale)

    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        if 'gauss' in name:
            for k in prange(num_nodes_y):
                fk = 0
                for l in range(num_nodes):
                    u = ((ys[k,:] - xs[l,:])**2).sum()/2
                    fk += np.exp(-u)*(ay[k]*ax[l] + (betay[k,:]*betax[l, :]).sum()**2)*sKP[s]
                f[k] += fk
        elif 'lap' in name:
            for k in prange(num_nodes_y):
                fk = 0
                for l in range(num_nodes):
                    u = np.sqrt(((ys[k, :] - xs[l, :]) ** 2).sum())
                    u1 = lapPol(u, order) * np.exp(-u) * sKP[s]
                    fk += u1*(ay[k]*ax[l] + (betay[k,:]*betax[l, :]).sum()**2)
                f[k] += fk
    f /= wsig
    return f


@jit(nopython=True, parallel=True)
def applykmat(y, x, beta, name, scale, order, KP=-1):
    dim = x.shape[1]
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dimb = beta.shape[2]
    f = np.zeros((num_nodes_y, dimb))

    sKP = scale**KP
    wsig = sKP.sum()
    ns = len(scale)

    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        if 'gauss' in name:
            for k in prange(num_nodes_y):
                fk = np.zeros(dimb)
                for l in range(num_nodes):
                    u = ((ys[k,:] - xs[l,:])**2).sum()/2
                    fk += np.exp(-u)*beta[k,l,:]*sKP[s]
                f[k,:] += fk
        elif 'lap' in name:
            for k in prange(num_nodes_y):
                fk = np.zeros(dimb)
                for l in range(num_nodes):
                    u = np.sqrt(((ys[k, :] - xs[l, :]) ** 2).sum())
                    u1 = lapPol(u, order) * np.exp(-u) * sKP[s]
                    fk += u1*beta[k,l,:]
                f[k, :] += fk
    f /= wsig
    return f

def applydiffktensor(y, x, ay, ax, betay, betax, name, scale, order, cpu=False, dtype='float64', KP=-1):
    if not cpu and pykeops.config.gpu_available:
        return applydiffktensor_pykeops(y, x, ay, ax, betay, betax, name, scale, order, dtype=dtype, KP=KP)
    else:
        return applydiffktensor_numba(y, x, ay, ax, betay, betax, name, scale, order, KP=KP)


def applydiffktensor_pykeops(y, x, ay, ax, betay, betax, name, scale, order, dtype='float64', KP=-1):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    res = np.zeros((num_nodes_y, dim))
    dim = x.shape[1]
    strdim = str(dim)
    dimb = betax.shape[1]
    strdimb = str(dimb)

    sKP = scale**KP
    sKP1 = scale**(KP-1)
    wsig = sKP.sum()
    ns = len(scale)
    ay_ = LazyTensor(ay.astype(dtype)[:, None, None])
    ax_ = LazyTensor(ax.astype(dtype)[None, :, None])
    betay_ = LazyTensor(betay.astype(dtype)[:, None, :])
    betax_ = LazyTensor(betax.astype(dtype)[None, :, :])

    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        ys_ = LazyTensor(ys.astype(dtype)[:, None, :])
        xs_ = LazyTensor(xs.astype(dtype)[None, :, :])
        g = np.array([0.5])
        sKP1s = np.array([sKP1[s]])
        if 'gauss' in name:
            diffij = ys_ - xs_
            Dij = (diffij ** 2).sum(-1)
            Kij = diffij * (-0.5 * Dij).exp()
        elif 'lap' in name:
            diffij = ys_ - xs_
            Dij = (diffij ** 2).sum(-1).sqrt()
            Kij = diffij * (c1_[order, 0] + c1_[order, 1] * Dij + c1_[order, 2] * Dij * Dij
            + c1_[order, 3] * Dij * Dij * Dij) * (-Dij).exp()
        else: # Euclidean kernel
            Kij = LazyTensor(np.ones(ys_.shape)) * xs_

        res += (-sKP1[s]) * (Kij * ((ax_*ay_) + (betax_ * betay_).sum(axis=-1)**2 )).sum(1)


    res/=wsig
    return res




@jit(nopython=True, parallel=True)
def applydiffktensor_numba(y, x, ay, ax, betay, betax, name, scale, order, KP=-1):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes_y, dim))

    sKP = scale**KP
    sKP1 = scale**(KP-1)
    wsig = sKP.sum()
    ns = len(scale)

    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        if 'gauss' in name:
            for k in prange(num_nodes_y):
                fk = np.zeros(dim)
                for l in range(num_nodes):
                    u = ((ys[k,:] - xs[l,:])**2).sum()/2
                    fk -= sKP1[s] * np.exp(-u) * (ys[k, :] - xs[l, :]) * (ay[k]*ax[l]+ (betay[k,:]*betax[l, :]).sum()**2)
                f[k,:] += fk
        elif 'lap' in name:
            for k in prange(num_nodes_y):
                fk = np.zeros(dim)
                for l in range(num_nodes):
                    u = np.sqrt(((ys[k, :] - xs[l, :]) ** 2).sum())
                    u1 = lapPolDiff(u, order) * np.exp(-u) * sKP1[s]
                    fk -= u1 * (ys[k, :] - xs[l, :]) * (ay[k]*ax[l] + (betay[k,:]*betax[l, :]).sum()**2)
                f[k, :] += fk
    f/=wsig
    return f



def applykdiffmat(y, x, beta, name, scale, order, cpu=False, dtype='float64', KP=-1):
    if not cpu and cupy_available and pykeops.config.gpu_available:
        return applykdiffmat_cupy(y, x, beta, name, scale, order, dtype=dtype, KP=KP)
    else:
        return applykdiffmat_numba(y, x, beta, name, scale, order, KP=KP)


@jit(nopython=True, parallel=True)
def applykdiffmat_numba(y, x, beta, name, scale, order, KP=-1):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes_y, dim))

    sKP = scale**KP
    sKP1 = scale**(KP-1)
    wsig = sKP.sum()
    ns = len(scale)

    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        if 'gauss' in name:
            for k in prange(num_nodes_y):
                fk = np.zeros(dim)
                for l in range(num_nodes):
                    u = ((ys[k,:] - xs[l,:])**2).sum()/2
                    fk -= sKP1[s] * np.exp(-u) * (ys[k, :] - xs[l, :]) * beta[k,l]
                f[k,:] += fk
        elif 'lap' in name:
            for k in prange(num_nodes_y):
                fk = np.zeros(dim)
                for l in range(num_nodes):
                    u = np.sqrt(((ys[k, :] - xs[l, :]) ** 2).sum())
                    u1 = lapPolDiff(u, order) * np.exp(-u) * sKP1[s]
                    fk -= u1 * (ys[k, :] - xs[l, :]) * beta[k,l]
                f[k, :] += fk
    f/=wsig
    return f

def applykdiffmat_pykeops(y, x, beta, name, scale, order, dtype='float64', KP=-1):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    res = np.zeros((num_nodes_y, dim))

    sKP = scale**KP
    sKP1 = scale**(KP-1)
    wsig = sKP.sum()
    ns = len(scale)

    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        ys_ = LazyTensor(ys.astype(dtype)[:, None, :])
        xs_ = LazyTensor(xs.astype(dtype)[None, :, :])
        g = np.array([0.5])
        sKP1s = np.array([sKP1[s]])
        if 'gauss' in name:
            diffij = ys_ - xs_
            Dij = (diffij ** 2).sum(-1)
            Kij = diffij * (-0.5 * Dij).exp()
        elif 'lap' in name:
            diffij = ys_ - xs_
            Dij = (diffij ** 2).sum(-1).sqrt()
            Kij = diffij * (c1_[order, 0] + c1_[order, 1] * Dij + c1_[order, 2] * Dij * Dij
            + c1_[order, 3] * Dij * Dij * Dij) * (-Dij).exp()
        else: # Euclidean kernel
            Kij = LazyTensor(np.ones(ys_.shape)) * xs_

        print(np.array(Kij))
        res += (-sKP1[s]) * (Kij * beta).sum(1)

    res/=wsig
    return res


def applykdiffmat_cupy(y, x, beta, name, scale, order, dtype='float64', KP=-1):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    res = cp.zeros((num_nodes_y, dim))

    sKP = scale**KP
    sKP1 = scale**(KP-1)
    wsig = sKP.sum()
    ns = len(scale)

    for s in range(ns):
        ys = y/scale[s]
        xs = x/scale[s]
        ys_ = cp.asarray(ys.astype(dtype)[:, None, :])
        xs_ = cp.asarray(xs.astype(dtype)[None, :, :])
        g = np.array([0.5])
        sKP1s = np.array([sKP1[s]])
        if 'gauss' in name:
            diffij = ys_ - xs_
            Dij = cp.square(diffij).sum(-1)
            Kij = diffij * cp.exp(-0.5 * Dij)[:,:,None]
        elif 'lap' in name:
            diffij = ys_ - xs_
            Dij = (diffij ** 2).sum(-1).sqrt()
            Kij = diffij * (c1_[order, 0] + c1_[order, 1] * Dij + c1_[order, 2] * Dij * Dij
            + c1_[order, 3] * Dij * Dij * Dij) * cp.exp(-Dij)
        else: # Euclidean kernel
            Kij = cp.ones(ys_.shape) * xs_

        res += (-sKP1[s]) * (Kij * cp.asarray(beta[:,:,None])).sum(axis=1)

    res/=wsig
    return cp.asnumpy(res)



## Local kernels for machine learning
def applylocalk(y, x, a, name, scale, order, neighbors, num_neighbors, KP=-1):
    if 'lap' in name:
        return applylocalk_lap(y,x,a,scale,order,neighbors,num_neighbors, KP=KP)
    elif 'gauss' in name:
        return applylocalk_gauss(y,x,a,scale,neighbors,num_neighbors, KP=KP)

@jit(nopython=True, parallel=True)
def applylocalk_gauss(y, x, a, scale, neighbors, num_neighbors, KP=-1):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes_y, dim))
    tot_neighbors = neighbors.shape[0]

    startSet = np.zeros(dim, dtype=int64)
    endSet = np.zeros(dim, dtype=int64)


    wsig = 0
    for s in scale:
        wsig +=  s**KP


    nSets = 0
    startSet[0] = 0
    for k in range(tot_neighbors):
        if neighbors[k] == -1:
            endSet[nSets] = k
            if k < tot_neighbors-1:
                nSets = nSets+1
                startSet[nSets] = k+1


    for k in prange(num_nodes_y):
        sqdist = np.zeros(dim)
        for l in range(num_nodes):
            for kk in range(nSets):
                ut0 = 0
                for jj in range(startSet[kk],endSet[kk]):
                    ut0 += (y[k,neighbors[jj]]-x[l,neighbors[jj]])**2
                Kv = 0
                for s in scale:
                    ut = ut0/s**2
                    if ut < 1e-8:
                        Kv += s**KP
                    else:
                        Kv += np.exp(-0.5*ut) * s**KP
                sqdist[kk] = Kv/wsig
            for j in range(dim):
                f[k,j] += sqdist[num_neighbors[j]] * a[l,j]
    return f

@jit(nopython=True, parallel=True)
def applylocalk_lap(y, x, a, scale, order, neighbors, num_neighbors, KP=-1):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes_y, dim))
    tot_neighbors = neighbors.shape[0]

    startSet = np.zeros(dim, dtype=int64)
    endSet = np.zeros(dim, dtype=int64)


    wsig = 0
    for s in scale:
        wsig +=  s**KP


    nSets = 0
    startSet[0] = 0
    for k in range(tot_neighbors):
        if neighbors[k] == -1:
            endSet[nSets] = k
            if k < tot_neighbors-1:
                nSets = nSets+1
                startSet[nSets] = k+1


    for k in prange(num_nodes_y):
        sqdist = np.zeros(dim)
        for l in range(num_nodes):
            for kk in range(nSets):
                ut0 = 0
                for jj in range(startSet[kk], endSet[kk]):
                    ut0 += (y[k, neighbors[jj]] - x[l, neighbors[jj]]) ** 2
                ut0 = np.sqrt(ut0)
                Kv = 0
                for s in scale:
                    ut = ut0 / s
                    if ut < 1e-8:
                        Kv += s ** KP
                    else:
                        lpt = lapPol(ut, order)
                        Kv += lpt * np.exp(-ut) * s ** KP
                sqdist[kk] = Kv / wsig
            for j in range(dim):
                f[k, j] += sqdist[num_neighbors[j]] * a[l, j]
    return f

@jit(nopython=True, parallel=True)
def applylocalk_naive(y, x, a, name, scale, order, KP=-1):
    f = np.zeros(y.shape)
    wsig = 0
    for s in scale:
        wsig += s**KP

    s = scale[0]
    ys = y / s
    xs = x / s
    if 'lap' in name:
        for k in prange(y.shape[0]):
            for l in range(x.shape[0]):
                ut = np.fabs(ys[k, :] - xs[l, :])
                f[k, :] += lapPol(ut, order) * np.exp(-ut) * a[l, :]
    elif 'gauss' in name:
        for k in prange(y.shape[0]):
            for l in range(x.shape[0]):
                ut = np.maximum(np.minimum(np.fabs(ys[k, :] - xs[l, :]), 10),-10)
                f[k,:] += np.exp(-ut*ut/2) * a[l,:]
    return f


def applylocalkdifft(y, x, p, a, name, scale, order, neighbors, num_neighbors, regweight=1., lddmm=False, KP=-1):
    if 'lap' in name:
        return applylocalkdifft_lap(y, x, p, a, scale, order, neighbors, num_neighbors, regweight, lddmm, KP=KP)
    elif 'gauss' in name:
        return applylocalkdifft_gauss(y, x, p, a, scale, neighbors, num_neighbors, regweight, lddmm, KP=KP)

@jit(nopython=True, parallel=True)
def applylocalkdifft_gauss(y, x, p, a, scale, neighbors, num_neighbors, regweight=1., lddmm=False, KP=-1):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes_y, dim))
    tot_neighbors = neighbors.shape[0]

    startSet = np.zeros(dim, dtype=int64)
    endSet = np.zeros(dim, dtype=int64)


    wsig = 0
    for s in scale:
        wsig +=  s**KP

    nSets = 0
    startSet[0] = 0
    for k in range(tot_neighbors):
        if neighbors[k] == -1:
            endSet[nSets] = k
            if k < tot_neighbors-1:
                nSets = nSets+1
                startSet[nSets] = k+1

    for k in prange(num_nodes_y):
        for l in range(num_nodes):
            sqdist = np.zeros(dim)
            for kk in range(nSets):
                ut0 = 0
                for jj in range(startSet[kk],endSet[kk]):
                    ut0 += (y[k,neighbors[jj]]-x[l,neighbors[jj]])**2
                Kv_diff = 0
                for s in scale:
                    ut = ut0/s**2
                    if ut < 1e-8:
                        Kv_diff -= s**(KP-2)
                    else:
                        Kv_diff -= np.exp(-0.5*ut)*s**(KP-2)
                sqdist[kk] = Kv_diff/wsig
            for j in range(dim):
                kk = num_neighbors[j]
                if lddmm:
                    akl = p[k, j] * a[l, j] + p[l, j] * a[k, j] - 2 * regweight * a[k, j] * a[l, j]
                else:
                    akl = p[k, j] * a[l,j]
                for jj in range(startSet[kk], endSet[kk]):
                    ii = neighbors[jj]
                    f[k,ii] += sqdist[kk] * (y[k,ii]-x[l,ii])* akl
    return f

@jit(nopython=True, parallel=True)
def applylocalkdifft_lap(y, x, p, a, scale, order, neighbors, num_neighbors, regweight=1., lddmm=False, KP=-1):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes_y, dim))
    tot_neighbors = neighbors.shape[0]

    startSet = np.zeros(dim, dtype=int64)
    endSet = np.zeros(dim, dtype=int64)


    wsig = 0
    for s in scale:
        wsig +=  s**KP

    nSets = 0
    startSet[0] = 0
    for k in range(tot_neighbors):
        if neighbors[k] == -1:
            endSet[nSets] = k
            if k < tot_neighbors-1:
                nSets = nSets+1
                startSet[nSets] = k+1

    for k in prange(num_nodes_y):
        sqdist = np.zeros(dim)
        for l in range(num_nodes):
            for kk in range(nSets):
                ut0 = 0
                for jj in range(startSet[kk],endSet[kk]):
                    ut0 += (y[k,neighbors[jj]]-x[l,neighbors[jj]])**2
                ut0 = np.sqrt(ut0)
                Kv_diff = 0
                for s in scale:
                    ut = ut0/s**2
                    if ut < 1e-8:
                        Kv_diff -= s**(KP-2)
                    else:
                        Kv_diff -= lapPolDiff(ut, order)*np.exp(-ut)*s**(KP-2)
                sqdist[kk] = Kv_diff/wsig
            for j in range(dim):
                kk = num_neighbors[j]
                if lddmm:
                    akl = p[k, j] * a[l, j] + p[l, j] * a[k, j] - 2 * regweight * a[k, j] * a[l, j]
                else:
                    akl = p[k, j] * a[l,j]
                for jj in range(startSet[kk], endSet[kk]):
                    ii = neighbors[jj]
                    f[k,ii] += sqdist[kk] * (y[k,ii]-x[l,ii])* akl
    return f

@jit(nopython=True, parallel=True)
def applylocalkdiv(x, y, a, name, scale, order, neighbors, num_neighbors, KP=-1):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes,1))
    tot_neighbors = neighbors.shape[0]

    startSet = np.zeros(dim, dtype=int64)
    endSet = np.zeros(dim, dtype=int64)


    wsig = 0
    for s in scale:
        wsig +=  s**KP

    nSets = 0
    startSet[0] = 0
    for k in range(tot_neighbors):
        if neighbors[k] == -1:
            endSet[nSets] = k
            if k < tot_neighbors-1:
                nSets = nSets+1
                startSet[nSets] = k+1

    if 'gauss' in name:
        for k in prange(num_nodes):
            df = 0
            sqdist = np.zeros(dim)
            for l in range(num_nodes_y):
                for kk in range(nSets):
                    ut0 = 0
                    for jj in range(startSet[kk],endSet[kk]):
                        ut0 += (x[k,neighbors[jj]]-y[l,neighbors[jj]])**2
                    Kv_diff = 0
                    for s in scale:
                        ut = ut0/s**2
                        if ut < 1e-8:
                            Kv_diff -= 0.5*s**(KP-2)
                        else:
                            Kv_diff -= 0.5*np.exp(-0.5*ut)*s**(KP-2)
                    sqdist[kk] = Kv_diff/wsig
                for j in range(dim):
                    kk = num_neighbors[j]
                    for jj in range(startSet[kk], endSet[kk]):
                        ii = neighbors[jj]
                        df += sqdist[kk] * 2*(x[k,ii]-y[l,ii])* a[l,ii]
            f[k] = df
    elif 'lap' in name:
        for k in prange(num_nodes):
            df = 0
            sqdist = np.zeros(dim)
            for l in range(num_nodes_y):
                for kk in range(nSets):
                    ut0 = 0
                    for jj in range(startSet[kk],endSet[kk]):
                        ut0 += (x[k,neighbors[jj]]-y[l,neighbors[jj]])**2
                    ut0 = np.sqrt(ut0)
                    Kv_diff = 0
                    for s in scale:
                        ut = ut0/s**2
                        if ut < 1e-8:
                            Kv_diff -= 0.5*s**(KP-2)
                        else:
                            Kv_diff -= 0.5*lapPolDiff(ut, order)+np.exp(-ut)*s**(KP-2)
                    sqdist[kk] = Kv_diff/wsig
                for j in range(dim):
                    kk = num_neighbors[j]
                    for jj in range(startSet[kk], endSet[kk]):
                        ii = neighbors[jj]
                        df += sqdist[kk] * 2*(x[k,ii]-y[l,ii])* a[l,ii]
            f[k] = df
    return f

@jit(nopython=True, parallel=True)
def applylocalk_naivedifft(y, x, p, a, name, scale, order, regweight=1., lddmm=False, KP=-1):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros(y.shape)

    wsig = 0
    for s in scale:
        wsig += s ** KP
    if lddmm:
        lddmm_ = 1.0
    else:
        lddmm_ = 0

    if 'lap' in name:
        for k in prange(num_nodes_y):
            for l in range(num_nodes):
                xy = y[k,:] - x[l,:]
                ut0 = np.fabs(xy)
                Kv_diff = np.zeros(dim)
                for s in scale:
                    ut = ut0 / s
                    Kv_diff -= lapPolDiff(ut, order) * np.exp(-ut) * s ** (KP - 2)
                sqdist = Kv_diff/wsig
                akl = p[k, :] * a[l, :] + lddmm_ * (a[k, :] * p[l, :] - 2 * regweight * a[k, :] * a[l, :])
                f[k, :] += sqdist * xy * akl
    elif 'gauss' in name:
        for k in prange(num_nodes_y):
            for l in range(num_nodes):
                xy = y[k,:] - x[l,:]
                ut0 = np.fabs(xy)
                Kv_diff = np.zeros(dim)
                for s in scale:
                    ut = ut0/s
                    Kv_diff -= np.exp(-0.5*ut*ut)*s**(KP-2)
                sqdist = Kv_diff/wsig
                akl = p[k, :] * a[l, :] + lddmm_ * (a[k, :] * p[l, :] - 2 * regweight * a[k, :] * a[l, :])
                f[k,:] +=  sqdist * xy* akl
    return f

@jit(nopython=True, parallel=True)
def applylocalk_naivediv(x, y, a, name, scale, order, KP=-1):
    num_nodes = x.shape[0]
    num_nodes_y = y.shape[0]
    dim = x.shape[1]
    f = np.zeros((num_nodes,1))

    wsig = 0
    for s in scale:
        wsig += s ** KP

    if 'gauss' in name:
        for k in prange(num_nodes):
            df = 0
            for l in range(num_nodes_y):
                ut0 = np.fabs(x[k,:] - y[l,:])
                Kv_diff = np.zeros(dim)
                for s in scale:
                    ut = ut0/s
                    Kv_diff -= 0.5*np.exp(-0.5*ut*ut)*s**(KP-2)
                sqdist = Kv_diff/wsig
                for j in range(dim):
                    df -=  sqdist[j] * 2*(x[k,j]-y[l,j])* a[l,j]
            f[k] = df

    elif 'lap' in name:
        for k in prange(num_nodes):
            df = 0
            for l in range(num_nodes_y):
                ut0 = np.fabs(x[k, :] - y[l, :])
                Kv_diff = np.zeros(dim)
                for s in scale:
                    ut = ut0 / s
                    Kv_diff -= 0.5 * lapPolDiff(ut, order) * np.exp(-ut) * s ** (KP - 2)
                sqdist = Kv_diff/wsig
                for j in range(dim):
                    df -= sqdist[j] * 2*(x[k,j]-y[l,j])* a[l,j]
            f[k] = df
    return f
