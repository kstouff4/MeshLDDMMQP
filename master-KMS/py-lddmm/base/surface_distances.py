import numpy as np
from numba import jit
from . import diffeo
import logging
from pykeops.numpy import LazyTensor
import pykeops
from . import kernelFunctions_util as ku

def haussdorffDist(fv1, fv2):
    dst = ((fv1.vertices[:, np.newaxis, :] - fv2.vertices[np.newaxis, :, :] )**2).sum(axis=2)
    res = np.amin(dst, axis=1).max() + np.amin(dst, axis=0).max()
    return res


# Current norm of fv1
def currentMagnitude(fv1, KparDist):
    c2 = fv1.centers
    cr2 = np.copy(fv1.surfel) * fv1.face_weights[:, None]
    obj = (cr2 *KparDist.applyK(c2, cr2)).sum()
    return obj

# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def currentMagnitudeGradient(fvDef, KparDist, with_weights=False):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    w1 = fvDef.face_weights
    cr1 = np.copy(fvDef.surfel)
    dim = c1.shape[1]
    wcr1 = w1 * cr1

    z1 = KparDist.applyK(c1, wcr1)
    wz1 = w1[:, None] * z1
    dz1 = ( 2 /3) * KparDist.applyDiffKT(c1, wcr1[np.newaxis ,...], wcr1[np.newaxis ,...])

    xDef1 = xDef[fvDef.faces[:, 0], :]
    xDef2 = xDef[fvDef.faces[:, 1], :]
    xDef3 = xDef[fvDef.faces[:, 2], :]

    px = np.zeros([xDef.shape[0], dim])

    I = fvDef.faces[: ,0]
    crs = np.cross(xDef3 - xDef2, wz1)
    for k in range(I.size):
        px[I[k], :] += dz1[k, :] -  crs[k, :]

    I = fvDef.faces[: ,1]
    crs = np.cross(xDef1 - xDef3, wz1)
    for k in range(I.size):
        px[I[k], :] += dz1[k, :] -  crs[k, :]

    I = fvDef.faces[: ,2]
    crs = np.cross(xDef2 - xDef1, wz1)
    for k in range(I.size):
        px[I[k], :] += dz1[k, :] -  crs[k, :]

    if with_weights:
        z1w = ( 2 /3) * (cr1 *z1).sum(axis=1)
        pxw = np.zeros(xDef.shape[0])
        for j in range(dim):
            I = fvDef.faces[: ,j]
            for k in range(I.size):
                pxw[I[k]] += z1w[k]
        return  px, pxw
    else:
        return px


def currentNorm0(fv1, KparDist, weight=1.):
    return currentMagnitude(fv1, KparDist)


# Computes |fvDef|^2 - 2 fvDef * fv1 with current dot produuct
def currentNormDef(fvDef, fv1, KparDist, weight=1.):
    c1 = fvDef.centers
    cr1 = fvDef.face_weights[:, None] * fvDef.surfel
    c2 = fv1.centers
    cr2 = fv1.face_weights[:, None] * fv1.surfel
    obj = (cr1 *KparDist.applyK(c1, cr1)).sum() - 2* (cr1 * KparDist.applyK(c2, cr2, firstVar=c1)).sum()
    return obj


# Returns |fvDef - fv1|^2 for current norm
def currentNorm(fvDef, fv1, KparDist, weight=1.):
    return currentNormDef(fvDef, fv1, KparDist) + currentNorm0(fv1, KparDist)


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def currentNormGradient(fvDef, fv1, KparDist, weight=1., with_weights=False):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    cr1 = np.copy(fvDef.surfel)
    w1 = fvDef.face_weights
    c2 = fv1.centers
    cr2 = np.copy(fv1.surfel)
    w2 = fv1.face_weights
    dim = c1.shape[1]
    wcr1 = w1[:, None] * cr1
    wcr2 = w2[:, None] * cr2

    z1 = KparDist.applyK(c1, wcr1) - KparDist.applyK(c2, wcr2, firstVar=c1)
    wz1 = w1[:, None] * z1
    dz1 = (2 / 3) * (KparDist.applyDiffKT(c1, wcr1, wcr1) -
                     KparDist.applyDiffKT(c2, wcr1, wcr2, firstVar=c1))

    xDef1 = xDef[fvDef.faces[:, 0], :]
    xDef2 = xDef[fvDef.faces[:, 1], :]
    xDef3 = xDef[fvDef.faces[:, 2], :]

    px = np.zeros([xDef.shape[0], dim])
    I = fvDef.faces[:, 0]
    crs = np.cross(xDef3 - xDef2, wz1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - crs[k, :]

    I = fvDef.faces[:, 1]
    crs = np.cross(xDef1 - xDef3, wz1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - crs[k, :]

    I = fvDef.faces[:, 2]
    crs = np.cross(xDef2 - xDef1, wz1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - crs[k, :]

    if with_weights:
        z1w = (2 / 3) * (cr1 * z1).sum(axis=1)
        pxw = np.zeros(xDef.shape[0])
        for j in range(dim):
            I = fvDef.faces[:, j]
            for k in range(I.size):
                pxw[I[k]] += z1w[k]
        return px, pxw
    else:
        return px



# Measure norm of fv1
def measureNormPS0(fv1, KparDist):
    c2 = fv1.points
    cr2 = fv1.weights[:, None]
    return (cr2*KparDist.applyK(c2, cr2)).sum()


# Computes |fvDef|^2 - 2 fvDef * fv1 with measure dot produuct
def measureNormPSDef(fvDef, fv1, KparDist):
    c1 = fvDef.centers
    cr1 = np.sqrt((fvDef.surfel ** 2).sum(axis=1) + 1e-10)[:, None] * fvDef.face_weights[:, None]
    c2 = fv1.points
    cr2 = fv1.weights[:, None]
    obj = ((cr1 * KparDist.applyK(c1, cr1)).sum()
           - 2 * (cr1 * KparDist.applyK(c2, cr2, firstVar=c1)).sum())
    return obj


def measureNormPS(fvDef, fv1, KparDist):
    return measureNormPSDef(fvDef, fv1, KparDist) + measureNormPS0(fv1, KparDist)


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (measure norm)
def measureNormPSGradient(fvDef, fv1, KparDist):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    c2 = fv1.points
    dim = c1.shape[1]
    a1 = np.sqrt((fvDef.surfel ** 2).sum(axis=1) + 1e-10)
    a2 = fv1.weights
    cr1 = fvDef.surfel / a1[:, np.newaxis]
    # cr2 = fv1.surfel / a2[:, np.newaxis]

    z1 = KparDist.applyK(c1, a1[:, np.newaxis]) - KparDist.applyK(c2, a2[:, np.newaxis], firstVar=c1)
    z1 = z1 * cr1
    # print a1.shape, c1.shape
    dz1 = (2. / 3.) * (KparDist.applyDiffKT(c1, a1[:, np.newaxis], a1[:, np.newaxis]) -
                       KparDist.applyDiffKT(c2, a1[:, np.newaxis], a2[:, np.newaxis], firstVar=c1))

    xDef1 = xDef[fvDef.faces[:, 0], :]
    xDef2 = xDef[fvDef.faces[:, 1], :]
    xDef3 = xDef[fvDef.faces[:, 2], :]

    px = np.zeros([xDef.shape[0], dim])
    I = fvDef.faces[:, 0]
    crs = np.cross(xDef3 - xDef2, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - crs[k, :]

    I = fvDef.faces[:, 1]
    crs = np.cross(xDef1 - xDef3, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - crs[k, :]

    I = fvDef.faces[:, 2]
    crs = np.cross(xDef2 - xDef1, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - crs[k, :]
    return px




# Measure norm of fv1
def measureNorm0(fv1, KparDist):
    c2 = fv1.centers
    cr2 = np.sqrt((fv1.surfel ** 2).sum(axis=1) + 1e-10)[:, np.newaxis] * fv1.face_weights[:, None]
    return np.multiply(cr2, KparDist.applyK(c2, cr2)).sum()


# Computes |fvDef|^2 - 2 fvDef * fv1 with measure dot produuct
def measureNormDef(fvDef, fv1, KparDist):
    c1 = fvDef.centers
    cr1 = np.sqrt((fvDef.surfel ** 2).sum(axis=1) + 1e-10)[:, None] * fvDef.face_weights[:, None]
    c2 = fv1.centers
    cr2 = np.sqrt((fv1.surfel ** 2).sum(axis=1) + 1e-10)[:, None] * fv1.face_weights[:, None]
    obj = ((cr1 * KparDist.applyK(c1, cr1)).sum()
           - 2 * (cr1 * KparDist.applyK(c2, cr2, firstVar=c1)).sum())
    return obj


def measureNorm(fvDef, fv1, KparDist):
    return measureNormDef(fvDef, fv1, KparDist) + measureNorm0(fv1, KparDist)


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (measure norm)
def measureNormGradient(fvDef, fv1, KparDist, with_weights=False):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    c2 = fv1.centers
    dim = c1.shape[1]
    w1 = fvDef.face_weights
    w2 = fv1.face_weights
    a1 = np.sqrt((fvDef.surfel ** 2).sum(axis=1) + 1e-10)
    a2 = np.sqrt((fv1.surfel ** 2).sum(axis=1) + 1e-10)
    cr1 = fvDef.surfel / a1[:, np.newaxis]
    # cr2 = fv1.surfel / a2[:, np.newaxis]
    a1w = a1 * w1
    a2w = a2 * w2

    z1 = KparDist.applyK(c1, a1w[:, np.newaxis]) - KparDist.applyK(c2, a2w[:, np.newaxis], firstVar=c1)
    wz1 = w1[:, None] * z1 * cr1
    # print a1.shape, c1.shape
    dz1 = (2. / 3.) * (KparDist.applyDiffKT(c1, a1w[:, np.newaxis], a1w[:, np.newaxis]) -
                       KparDist.applyDiffKT(c2, a1w[:, np.newaxis], a2w[:, np.newaxis], firstVar=c1))

    xDef1 = xDef[fvDef.faces[:, 0], :]
    xDef2 = xDef[fvDef.faces[:, 1], :]
    xDef3 = xDef[fvDef.faces[:, 2], :]

    px = np.zeros([xDef.shape[0], dim])
    I = fvDef.faces[:, 0]
    crs = np.cross(xDef3 - xDef2, wz1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - crs[k, :]

    I = fvDef.faces[:, 1]
    crs = np.cross(xDef1 - xDef3, wz1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - crs[k, :]

    I = fvDef.faces[:, 2]
    crs = np.cross(xDef2 - xDef1, wz1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - crs[k, :]

    if with_weights:
        z1w = (2 / 3) * a1 * z1
        pxw = np.zeros(xDef.shape[0])
        for j in range(dim):
            I = fvDef.faces[:, j]
            for k in range(I.size):
                pxw[I[k]] += z1w[k]
        return px, pxw
    else:
        return px

def f0_(u):
    return 1 + u**2

def df0_(u):
    return 2*u

def varifoldNorm0(fv1, KparDist, fun = None, cpu=False, dtype='float64'):
    if not cpu and pykeops.config.gpu_available:
        return varifoldNorm0_pykeops(fv1, KparDist, dtype = dtype)
    else:
        return varifoldNorm0_numpy(fv1, KparDist, fun = fun)

def varifoldNorm0_numpy(fv1, KparDist, fun = None):
    if fun is None:
        fun = (f0_, df0_)
    c2 = fv1.centers
    a2 = np.sqrt((fv1.surfel ** 2).sum(axis=1) + 1e-10)
    cr2 = fv1.surfel / a2[:, np.newaxis]
    a2 *= fv1.face_weights
    f0 = fun[0]

    cr2_ = cr2 * np.sqrt(a2[:, None])
    return KparDist.applyKTensor(c2, a2, a2, cr2_, cr2_).sum()

def varifoldNorm0_pykeops(fv1, KparDist, dtype = 'float64'):
    c2 = fv1.centers
    a2 = np.sqrt((fv1.surfel ** 2).sum(axis=1) + 1e-10)
    cr2 = fv1.surfel / a2[:, np.newaxis]
    a2 *= fv1.face_weights

    y = LazyTensor(c2[:, None, :].astype(dtype))
    x = LazyTensor(c2[None, :, :].astype(dtype))
    cry = LazyTensor(cr2[:, None, :].astype(dtype))
    crx = LazyTensor(cr2[None,:,  :].astype(dtype))
    ay = LazyTensor(a2[:, None].astype(dtype), axis=0)
    ax = LazyTensor(a2[:, None].astype(dtype), axis=1)
    multiplier = (1 + (cry * crx).sum(axis=-1)**2) * (ay*ax).sum(axis=-1)
    res = 0
    for s in KparDist.sigma:
        Kij = ku.makeKij(y/s, x/s, KparDist.name, KparDist.order)
        res += (Kij* multiplier).sum(1).sum(0)[0] / s
    res /= (1/KparDist.sigma).sum()
    return res




# Computes |fvDef|^2 - 2 fvDef * fv1 with current dot produuct
def varifoldNormDef(fvDef, fv1, KparDist, fun = None, cpu=False, dtype='float64'):
    if not cpu and pykeops.config.gpu_available:
        return varifoldNormDef_pykeops(fvDef, fv1, KparDist, dtype = KparDist.pk_dtype)
    else:
        return varifoldNormDef_numpy(fvDef, fv1, KparDist, fun = fun)


def varifoldNormDef_numpy(fvDef, fv1, KparDist, fun = None):
    if fun is None:
        fun = (f0_, df0_)
    c1 = fvDef.centers
    c2 = fv1.centers
    a1 = np.sqrt((fvDef.surfel ** 2).sum(axis=1) + 1e-10)
    a2 = np.sqrt((fv1.surfel ** 2).sum(axis=1) + 1e-10)
    cr1 = fvDef.surfel / a1[:, np.newaxis]
    cr2 = fv1.surfel / a2[:, np.newaxis]
    a1 *= fvDef.face_weights
    a2 *= fv1.face_weights
    f0 = fun[0]

    cr1_ = cr1 * np.sqrt(a1[:, None])
    cr2_ = cr2 * np.sqrt(a2[:, None])
    obj1 = KparDist.applyKTensor(c1, a1, a1, cr1_, cr1_).sum()
    obj2 = KparDist.applyKTensor(c2, a1, a2, cr1_, cr2_,firstVar=c1).sum()
    obj = obj1 - 2*obj2
    return obj

def varifoldNormDef_pykeops(fvDef, fv1, KparDist, dtype='float64'):
    c1 = fvDef.centers
    c2 = fv1.centers
    a1 = np.sqrt((fvDef.surfel ** 2).sum(axis=1) + 1e-10)
    a2 = np.sqrt((fv1.surfel ** 2).sum(axis=1) + 1e-10)
    cr1 = fvDef.surfel / a1[:, np.newaxis]
    cr2 = fv1.surfel / a2[:, np.newaxis]
    a1 *= fvDef.face_weights
    a2 *= fv1.face_weights

    yi = LazyTensor(c1[:, None, :].astype(dtype))
    yj = LazyTensor(c1[None, :, :].astype(dtype))
    x = LazyTensor(c2[None, :, :].astype(dtype))
    cryi = LazyTensor(cr1[:, None, :].astype(dtype))
    cryj = LazyTensor(cr1[None, :,  :].astype(dtype))
    crx = LazyTensor(cr2[None,:,  :].astype(dtype))
    ayi = LazyTensor(a1[:, None].astype(dtype), axis=0)
    ayj = LazyTensor(a1[:, None].astype(dtype), axis=1)
    ax = LazyTensor(a2[:, None].astype(dtype), axis=1)
    multiplieryy = (1 + (cryi * cryj).sum(axis=-1)**2) * (ayi*ayj).sum(axis=-1)
    multiplieryx = (1 + (cryi * crx).sum(axis=-1)**2) * (ayi*ax).sum(axis=-1)
    res = 0
    for s in KparDist.sigma:
        Kijyy = ku.makeKij(yi/s, yj/s, KparDist.name, KparDist.order)
        Kijyx = ku.makeKij(yi/s, x/s, KparDist.name, KparDist.order)
        res += (Kijyy* multiplieryy).sum(1).sum(0)[0] / s - 2* (Kijyx* multiplieryx).sum(1).sum(0)[0] / s
    res /= (1/KparDist.sigma).sum()
    return res




def varifoldNorm(fvDef, fv1, KparDist, fun=None):
    if fun is None:
        fun = (f0_, df0_)
    return varifoldNormDef(fvDef, fv1, KparDist, fun=fun) + varifoldNorm0(fv1, KparDist, fun=fun)


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def varifoldNormGradient(fvDef, fv1, KparDist, with_weights=False, fun = None, cpu=False, dtype='float64'):
    if not cpu and pykeops.config.gpu_available:
#        varifoldNormGradient_numpy(fvDef, fv1, KparDist, with_weights=with_weights, fun = fun)
        return varifoldNormGradient_pykeops(fvDef, fv1, KparDist, with_weights=with_weights, dtype = KparDist.pk_dtype)
    else:
        return varifoldNormGradient_numpy(fvDef, fv1, KparDist, with_weights=with_weights, fun = fun)



def varifoldNormGradient_numpy(fvDef, fv1, KparDist, with_weights=False, fun=None):
    if fun is None:
        fun = (f0_, df0_)
    xDef = fvDef.vertices
    c1 = fvDef.centers
    c2 = fv1.centers
    dim = c1.shape[1]

    a1 = np.sqrt((fvDef.surfel ** 2).sum(axis=1) + 1e-10)
    a2 = np.sqrt((fv1.surfel ** 2).sum(axis=1) + 1e-10)
    cr1 = fvDef.surfel / a1[:, np.newaxis]
    cr2 = fv1.surfel / a2[:, np.newaxis]
    a1w = a1 * fvDef.face_weights
    a2w = a2 * fv1.face_weights
    f0 = fun[0]
    df0 = fun[1]

    cr1cr1 = (cr1[:, np.newaxis, :] * cr1[np.newaxis, :, :]).sum(axis=2)
    cr1cr2 = (cr1[:, np.newaxis, :] * cr2[np.newaxis, :, :]).sum(axis=2)

    beta1 = a1w[:, np.newaxis] * a1w[np.newaxis, :] * f0(cr1cr1)
    beta2 = a1w[:, np.newaxis] * a2w[np.newaxis, :] * f0(cr1cr2)

    u1 = (df0(cr1cr1[:,:, np.newaxis]) * (cr1[None, :, :] - cr1cr1[:,:, None] * cr1[:, None, :])
          + f0(cr1cr1[:,:,None]) * cr1[:,None,:]) * a1w[np.newaxis, :, np.newaxis]
    u2 = (df0(cr1cr2[:,:, np.newaxis]) * (cr2[np.newaxis, :, :] - cr1cr2[:,:, None] * cr1[:, None, :])
          + f0(cr1cr2[:,:,None]) * cr1[:,None,:]) * a2w[np.newaxis, :, np.newaxis]

    z1 = fvDef.face_weights[:, None] * (KparDist.applyK(c1, u1, matrixWeights=True)
                                        - KparDist.applyK(c2, u2, firstVar=c1, matrixWeights=True))
    dz1 = (2. / 3.) * (KparDist.applyDiffKmat(c1, beta1) - KparDist.applyDiffKmat(c2, beta2, firstVar=c1))

    xDef1 = xDef[fvDef.faces[:, 0], :]
    xDef2 = xDef[fvDef.faces[:, 1], :]
    xDef3 = xDef[fvDef.faces[:, 2], :]

    px = np.zeros([xDef.shape[0], dim])
    I = fvDef.faces[:, 0]
    crs = np.cross(xDef3 - xDef2, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - crs[k, :]

    I = fvDef.faces[:, 1]
    crs = np.cross(xDef1 - xDef3, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - crs[k, :]

    I = fvDef.faces[:, 2]
    crs = np.cross(xDef2 - xDef1, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - crs[k, :]

    if with_weights:
        beta1 = f0(cr1cr1) * a1[:, np.newaxis] * a1w[np.newaxis, :]
        beta2 = f0(cr1cr2) * a1[:, np.newaxis] * a2w[np.newaxis, :]
        z1w = (2 / 3) * fvDef.face_weights * (KparDist.applyK(c1, beta1[..., np.newaxis], matrixWeights=True)
                                              - KparDist.applyK(c2, beta2[..., np.newaxis], firstVar=c1,
                                                                matrixWeights=True))
        pxw = np.zeros(xDef.shape[0])
        for j in range(dim):
            I = fvDef.faces[:, j]
            for k in range(I.size):
                pxw[I[k]] += z1w[k]
        return px, pxw
    else:
        return px

def varifoldNormGradient_pykeops(fvDef, fv1, KparDist, with_weights=False, dtype = 'float64'):
    xDef = fvDef.vertices
    c1i = LazyTensor(fvDef.centers[:, None, :].astype(dtype))
    c1j = LazyTensor(fvDef.centers[None, :, :].astype(dtype))
    c2j = LazyTensor(fv1.centers[None, :, :].astype(dtype))
    c2 = fv1.centers
    dim = fv1.centers.shape[1]

    a1 = np.sqrt((fvDef.surfel ** 2).sum(axis=1) + 1e-10)
    a2 = np.sqrt((fv1.surfel ** 2).sum(axis=1) + 1e-10)
    a1i = LazyTensor(a1[:, None].astype(dtype), axis=0)
    # a1j = LazyTensor(a1[:, None].astype(dtype), axis=1)
    # a2j = LazyTensor(a2[:, None].astype(dtype), axis=1)
    cr1i = LazyTensor( (fvDef.surfel / a1[:, np.newaxis])[:,None,:].astype(dtype))
    cr1j = LazyTensor( (fvDef.surfel / a1[:, np.newaxis])[None,:, :].astype(dtype))
    cr2j = LazyTensor( (fv1.surfel / a2[:, np.newaxis])[None,:, :].astype(dtype))
    a1wi = LazyTensor( (a1 * fvDef.face_weights)[:, None].astype(dtype), axis=0)
    a1wj = LazyTensor( (a1 * fvDef.face_weights)[:, None].astype(dtype), axis=1)
    a2wj = LazyTensor( (a2 * fv1.face_weights)[:, None].astype(dtype), axis=1)

    cr1cr1 = (cr1i*cr1j).sum(axis=2)
    cr1cr2 = (cr1i*cr2j).sum(axis=2)
#    cr1cr1 = (cr1[:, np.newaxis, :] * cr1[np.newaxis, :, :]).sum(axis=2)
#    cr1cr2 = (cr1[:, np.newaxis, :] * cr2[np.newaxis, :, :]).sum(axis=2)

    f011 = 1 + cr1cr1**2
    f012 = 1 + cr1cr2**2
    beta1 = (a1wi*a1wj) * f011 
    beta2 = (a1wi*a2wj) * f012 
#    beta1 = a1w[:, np.newaxis] * a1w[np.newaxis, :] * f0(cr1cr1)
#    beta2 = a1w[:, np.newaxis] * a2w[np.newaxis, :] * f0(cr1cr2)

    u1 = ((2*cr1cr1) * (cr1j - cr1cr1*cr1i) + f011 * cr1i) * a1wj
    u2 = ((2*cr1cr2) * (cr2j - cr1cr2*cr1i) + f012 * cr1i) * a2wj
    # u1 = (df0(cr1cr1[:,:, np.newaxis]) * (cr1[np.newaxis, :, :] - cr1cr1[:,:, None] * cr1[:, None, :])
    #       + f0(cr1cr1[:,:,None]) * cr1[:,None,:]) * a1w[np.newaxis, :, np.newaxis]
    # u2 = (df0(cr1cr2[:,:, np.newaxis]) * (cr2[np.newaxis, :, :] - cr1cr2[:,:, None] * cr1[:, None, :])
    #       + f0(cr1cr2[:,:,None]) * cr1[:,None,:]) * a2w[np.newaxis, :, np.newaxis]

    z1 = np.zeros(fvDef.centers.shape)
    for s in KparDist.sigma:
        Kijyy = ku.makeKij(c1i/s, c1j/s, KparDist.name, KparDist.order)
        Kijyx = ku.makeKij(c1i/s, c2j/s, KparDist.name, KparDist.order)
        z1 += (Kijyy * u1).sum(1) / s - (Kijyx * u2).sum(1) / s
    z1 /= (1/KparDist.sigma).sum()
    z1 *= fvDef.face_weights[:, None]

#    z1 = fvDef.face_weights[:, None] * (KparDist.applyK(c1, u1, matrixWeights=True)
#                                        - KparDist.applyK(c2, u2, firstVar=c1, matrixWeights=True))
    dz1 = np.zeros(fvDef.centers.shape)
    for s in KparDist.sigma:
        dKijyy = ku.makeDiffKij(c1i/s, c1j/s, KparDist.name, KparDist.order)
        dKijyx = ku.makeDiffKij(c1i/s, c2j/s, KparDist.name, KparDist.order)
        dz1 += (dKijyy * beta1).sum(1) / (s**2) - (dKijyx * beta2).sum(1) / (s**2)
    dz1 *= (2/3) / (1/KparDist.sigma).sum()

    #dz1 = (2. / 3.) * (KparDist.applyDiffKmat(c1, beta1) - KparDist.applyDiffKmat(c2, beta2, firstVar=c1))

    xDef1 = xDef[fvDef.faces[:, 0], :]
    xDef2 = xDef[fvDef.faces[:, 1], :]
    xDef3 = xDef[fvDef.faces[:, 2], :]

    px = np.zeros([xDef.shape[0], dim])
    I = fvDef.faces[:, 0]
    crs = np.cross(xDef3 - xDef2, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - crs[k, :]

    I = fvDef.faces[:, 1]
    crs = np.cross(xDef1 - xDef3, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - crs[k, :]

    I = fvDef.faces[:, 2]
    crs = np.cross(xDef2 - xDef1, z1)
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - crs[k, :]

    if with_weights:
        beta1 = f011 * a1i * a1wj
        beta2 = f012 * a1i * a2wj
#        beta1 = f0(cr1cr1) * a1[:, np.newaxis] * a1w[np.newaxis, :]
#        beta2 = f0(cr1cr2) * a1[:, np.newaxis] * a2w[np.newaxis, :]

        z1w = np.zeros(fvDef.centers.shape)
        for s in KparDist.sigma:
            Kijyy = ku.makeKij(c1i/s, c1j/s, KparDist.name, KparDist.order)
            Kijyx = ku.makeKij(c1i/s, c2j/s, KparDist.name, KparDist.order)
            z1w += (Kijyy * beta1).sum(1) / s - (Kijyx * beta2).sum(1) / s
        z1w /= (1/KparDist.sigma).sum()
        z1w *= (2/3)*fvDef.face_weights[:, None]


#        z1w = (2 / 3) * fvDef.face_weights * (KparDist.applyK(c1, beta1[..., np.newaxis], matrixWeights=True)
#                                              - KparDist.applyK(c2, beta2[..., np.newaxis], firstVar=c1,
#                                                                matrixWeights=True))
        pxw = np.zeros(xDef.shape[0])
        for j in range(dim):
            I = fvDef.faces[:, j]
            for k in range(I.size):
                pxw[I[k]] += z1w[k]
        return px, pxw
    else:
        return px


def L2Norm0(fv1):
    return np.fabs(fv1.surfVolume())


def L2Norm(fvDef, vfld):
    vf = np.zeros((fvDef.centers.shape[0], 3))
    for k in range(3):
        vf[:, k] = diffeo.multilinInterp(vfld[k, ...], fvDef.centers.T)
    # vf = fvDef.centers/3 - 2*vf
    # vf =  - 2*vf
    # print 'volume: ', fvDef.surfVolume(), (vf*fvDef.surfel).sum()
    return np.fabs(fvDef.surfVolume()) - 2 * (vf * fvDef.surfel).sum()


def L2NormGradient(fvDef, vfld):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    cr1 = fvDef.surfel
    dvf = np.zeros((fvDef.faces.shape[0], 3, 3))
    for k in range(3):
        dvf[:, k, :] = diffeo.multilinInterpGradient(vfld[k, ...], c1.T).T
    gradc = cr1 / 3
    for k in range(3):
        gradc[:, k] -= 2 * (dvf[:, :, k] * cr1).sum(axis=1)
    gradc = gradc / 3

    gradn = np.zeros((fvDef.faces.shape[0], 3))
    for k in range(3):
        gradn[:, k] = diffeo.multilinInterp(vfld[k, ...], c1.T)
    gradn = c1 / 3 - 2 * gradn
    gradn = gradn / 2

    grad = np.zeros((xDef.shape[0], 3))
    xDef0 = xDef[fvDef.faces[:, 0], :]
    xDef1 = xDef[fvDef.faces[:, 1], :]
    xDef2 = xDef[fvDef.faces[:, 2], :]

    crs = np.cross(xDef2 - xDef1, gradn)
    I = fvDef.faces[:, 0]
    for k in range(I.size):
        grad[I[k], :] = grad[I[k], :] + gradc[k, :] - crs[k, :]

    crs = np.cross(xDef0 - xDef2, gradn)
    I = fvDef.faces[:, 1]
    for k in range(I.size):
        grad[I[k], :] = grad[I[k], :] + gradc[k, :] - crs[k, :]

    crs = np.cross(xDef1 - xDef0, gradn)
    I = fvDef.faces[:, 2]
    for k in range(I.size):
        grad[I[k], :] = grad[I[k], :] + gradc[k, :] - crs[k, :]

    return grad


def normGrad(fv, phi):
    v1 = fv.vertices[fv.faces[:, 0], :]
    v2 = fv.vertices[fv.faces[:, 1], :]
    v3 = fv.vertices[fv.faces[:, 2], :]
    l1 = ((v2 - v3) ** 2).sum(axis=1)
    l2 = ((v1 - v3) ** 2).sum(axis=1)
    l3 = ((v1 - v2) ** 2).sum(axis=1)
    phi1 = phi[fv.faces[:, 0], :]
    phi2 = phi[fv.faces[:, 1], :]
    phi3 = phi[fv.faces[:, 2], :]
    a = 4 * np.sqrt((fv.surfel ** 2).sum(axis=1))
    u = l1 * ((phi2 - phi1) * (phi3 - phi1)).sum(axis=1) + l2 * ((phi3 - phi2) * (phi1 - phi2)).sum(axis=1) + l3 * (
                (phi1 - phi3) * (phi2 - phi3)).sum(axis=1)
    res = (u / a).sum()
    return res


def laplacian(fv, phi, weighted=False):
    res = np.zeros(phi.shape)
    v1 = fv.vertices[fv.faces[:, 0], :]
    v2 = fv.vertices[fv.faces[:, 1], :]
    v3 = fv.vertices[fv.faces[:, 2], :]
    l1 = (((v2 - v3) ** 2).sum(axis=1))[..., np.newaxis]
    l2 = (((v1 - v3) ** 2).sum(axis=1))[..., np.newaxis]
    l3 = (((v1 - v2) ** 2).sum(axis=1))[..., np.newaxis]
    phi1 = phi[fv.faces[:, 0], :]
    phi2 = phi[fv.faces[:, 1], :]
    phi3 = phi[fv.faces[:, 2], :]
    a = 8 * (np.sqrt((fv.surfel ** 2).sum(axis=1)))[..., np.newaxis]
    r1 = (l1 * (phi2 + phi3 - 2 * phi1) + (l2 - l3) * (phi2 - phi3)) / a
    r2 = (l2 * (phi1 + phi3 - 2 * phi2) + (l1 - l3) * (phi1 - phi3)) / a
    r3 = (l3 * (phi1 + phi2 - 2 * phi3) + (l2 - l1) * (phi2 - phi1)) / a
    for k, f in enumerate(fv.faces):
        res[f[0], :] += r1[k, :]
        res[f[1], :] += r2[k, :]
        res[f[2], :] += r3[k, :]
    if weighted:
        av = fv.computeVertexArea()
        return res / av[0]
    else:
        return res


def diffNormGrad(fv, phi, variables='both'):
    v1 = fv.vertices[fv.faces[:, 0], :]
    v2 = fv.vertices[fv.faces[:, 1], :]
    v3 = fv.vertices[fv.faces[:, 2], :]
    l1 = (((v2 - v3) ** 2).sum(axis=1))
    l2 = (((v1 - v3) ** 2).sum(axis=1))
    l3 = (((v1 - v2) ** 2).sum(axis=1))
    phi1 = phi[fv.faces[:, 0], :]
    phi2 = phi[fv.faces[:, 1], :]
    phi3 = phi[fv.faces[:, 2], :]
    # a = ((fv.surfel**2).sum(axis=1))
    a = 2 * np.sqrt((fv.surfel ** 2).sum(axis=1))
    a2 = 2 * a[..., np.newaxis]
    if variables == 'both' or variables == 'phi':
        r1 = (l1[:, np.newaxis] * (phi2 + phi3 - 2 * phi1) + (l2 - l3)[:, np.newaxis] * (phi2 - phi3)) / a2
        r2 = (l2[:, np.newaxis] * (phi1 + phi3 - 2 * phi2) + (l1 - l3)[:, np.newaxis] * (phi1 - phi3)) / a2
        r3 = (l3[:, np.newaxis] * (phi1 + phi2 - 2 * phi3) + (l2 - l1)[:, np.newaxis] * (phi2 - phi1)) / a2
        gradphi = np.zeros(phi.shape)
        for k, f in enumerate(fv.faces):
            gradphi[f[0], :] -= r1[k, :]
            gradphi[f[1], :] -= r2[k, :]
            gradphi[f[2], :] -= r3[k, :]

    if variables == 'both' or variables == 'x':
        gradx = np.zeros(fv.vertices.shape)
        u = (l1 * ((phi2 - phi1) * (phi3 - phi1)).sum(axis=1) + l2 * ((phi3 - phi2) * (phi1 - phi2)).sum(axis=1)
             + l3 * ((phi1 - phi3) * (phi2 - phi3)).sum(axis=1))
        # u = (2*u/a**2)[...,np.newaxis]
        u = (u / (a ** 3))[..., np.newaxis]
        r1 = (- u * np.cross(v2 - v3, fv.surfel) + 2 * (
                    (v1 - v3) * (((phi3 - phi2) * (phi1 - phi2)).sum(axis=1))[:, np.newaxis]
                    + (v1 - v2) * (((phi1 - phi3) * (phi2 - phi3)).sum(axis=1)[:, np.newaxis])) / a2)
        r2 = (- u * np.cross(v3 - v1, fv.surfel) + 2 * (
                    (v2 - v1) * (((phi1 - phi3) * (phi2 - phi3)).sum(axis=1))[:, np.newaxis]
                    + (v2 - v3) * (((phi2 - phi1) * (phi3 - phi1)).sum(axis=1))[:, np.newaxis]) / a2)
        r3 = (- u * np.cross(v1 - v2, fv.surfel) + 2 * (
                    (v3 - v2) * (((phi2 - phi1) * (phi3 - phi1)).sum(axis=1))[:, np.newaxis]
                    + (v3 - v1) * (((phi3 - phi2) * (phi1 - phi2)).sum(axis=1)[:, np.newaxis])) / a2)
        for k, f in enumerate(fv.faces):
            gradx[f[0], :] += r1[k, :]
            gradx[f[1], :] += r2[k, :]
            gradx[f[2], :] += r3[k, :]

    if variables == 'both':
        return (gradphi, gradx)
    elif variables == 'phi':
        return gradphi
    elif variables == 'x':
        return gradx
    else:
        logging.info('Incorrect option in diffNormGrad')

@jit(nopython=True)
def elasticNorm_nb(fc, ver, phi, surfel):
    v1 = ver[fc[:, 0], :]
    v2 = ver[fc[:, 1], :]
    v3 = ver[fc[:, 2], :]
    l1 = ((v2 - v3) ** 2).sum(axis=1)
    l2 = ((v1 - v3) ** 2).sum(axis=1)
    l3 = ((v1 - v2) ** 2).sum(axis=1)
    phi1 = phi[fc[:, 0], :]
    phi2 = phi[fc[:, 1], :]
    phi3 = phi[fc[:, 2], :]
    a = 4 * np.sqrt((surfel ** 2).sum(axis=1))
    u = l1 * ((phi2 - phi1) * (phi3 - phi1)).sum(axis=1) + l2 * ((phi3 - phi2) * (phi1 - phi2)).sum(axis=1) + \
        l3 * ((phi1 - phi3) * (phi2 - phi3)).sum(axis=1) - \
        0.5 * (((v1-v3)*(phi1-phi2)).sum(axis=1) - ((v1-v2)*(phi1-phi3)).sum(axis=1))**2
    res = (u / a).sum()
    return res

def elasticNorm(fv, phi):
    return elasticNorm_nb(fv.faces, fv.vertices, phi, fv.surfel)



@jit(nopython=True)
def diffElasticNorm_nb(fc, ver, phi, surfel, variables='both'):
    v1 = ver[fc[:, 0], :]
    v2 = ver[fc[:, 1], :]
    v3 = ver[fc[:, 2], :]
    l1 = np.expand_dims(((v2 - v3) ** 2).sum(axis=1),1)
    l2 = np.expand_dims(((v1 - v3) ** 2).sum(axis=1),1)
    l3 = np.expand_dims(((v1 - v2) ** 2).sum(axis=1),1)
    phi1 = phi[fc[:, 0], :]
    phi2 = phi[fc[:, 1], :]
    phi3 = phi[fc[:, 2], :]
    # a = ((fv.surfel**2).sum(axis=1))
    a = 2 * np.expand_dims(np.sqrt((surfel ** 2).sum(axis=1)),1)
    a2 = 2 * a#[:, np.newaxis]
    correct =  np.expand_dims(((v1-v3)*(phi1-phi2)).sum(axis=1) - ((v1-v2)*(phi1-phi3)).sum(axis=1),1)
    gradx = np.zeros(ver.shape)
    gradphi = np.zeros(phi.shape)
    if variables == 'both' or variables == 'phi':
        r1 = (l1* (phi2 + phi3 - 2 * phi1) + (l2 - l3) * (phi2 - phi3)) / a2 \
        + .5 * (v2-v3) * correct / a
        r2 = (l2 * (phi1 + phi3 - 2 * phi2) + (l1 - l3) * (phi1 - phi3)) / a2 \
        + .5 * (v3-v1) * correct / a
        r3 = (l3 * (phi1 + phi2 - 2 * phi3) + (l2 - l1) * (phi2 - phi1)) / a2 \
        + .5 * (v1 -v2) * correct / a
        for k, f in enumerate(fc):
            gradphi[f[0], :] -= r1[k, :]
            gradphi[f[1], :] -= r2[k, :]
            gradphi[f[2], :] -= r3[k, :]

    if variables == 'both' or variables == 'x':
        u = (l1[:,0] * ((phi2 - phi1) * (phi3 - phi1)).sum(axis=1) + l2[:,0] * ((phi3 - phi2) * (phi1 - phi2)).sum(axis=1)
             + l3[:,0] * ((phi1 - phi3) * (phi2 - phi3)).sum(axis=1))
        u = u - 0.5 * (((v1-v3)*(phi1-phi2)).sum(axis=1) - ((v1-v2)*(phi1-phi3)).sum(axis=1))**2
        # u = (2*u/a**2)[...,np.newaxis]
        u = np.expand_dims(u,1) / (a ** 3)
        r1 = (- u * np.cross(v2 - v3, surfel) + 2 * (
                    (v1 - v3) * np.expand_dims(((phi3 - phi2) * (phi1 - phi2)).sum(axis=1),1)
                    + (v1 - v2) * np.expand_dims(((phi1 - phi3) * (phi2 - phi3)).sum(axis=1), 1)) / a2)
        r2 = (- u * np.cross(v3 - v1, surfel) + 2 * (
                    (v2 - v1) * np.expand_dims(((phi1 - phi3) * (phi2 - phi3)).sum(axis=1), 1)
                    + (v2 - v3) * np.expand_dims(((phi2 - phi1) * (phi3 - phi1)).sum(axis=1), 1)) / a2)
        r3 = (- u * np.cross(v1 - v2, surfel) + 2 * (
                    (v3 - v2) * np.expand_dims(((phi2 - phi1) * (phi3 - phi1)).sum(axis=1),1)
                    + (v3 - v1) * np.expand_dims(((phi3 - phi2) * (phi1 - phi2)).sum(axis=1), 1)) / a2)
        r1 -= .5 * (phi3 - phi2) * correct / a
        r2 -= .5 * (phi1 - phi3) * correct / a
        r3 -= .5 * (phi2 - phi1) * correct / a
        for k in range(fc.shape[0]):
            gradx[fc[k, 0], :] = gradx[fc[k,0], :] + r1[k, :]
            gradx[fc[k,1], :] = gradx[fc[k,1], :] + r2[k, :]
            gradx[fc[k,2], :] = gradx[fc[k,2], :] + r3[k, :]

    return gradphi, gradx
def diffElasticNorm(fv, phi, variables='both'):
    grad = diffElasticNorm_nb(fv.faces, fv.vertices, phi, fv.surfel, variables=variables)
    if variables == 'both':
        return grad
    elif variables == 'phi':
        return grad[0]
    elif variables == 'x':
        return grad[1]
    else:
        logging.info('Incorrect option in diffElasticNorm')

def diffElasticNorm_old(fv, phi, variables='both'):
    v1 = fv.vertices[fv.faces[:, 0], :]
    v2 = fv.vertices[fv.faces[:, 1], :]
    v3 = fv.vertices[fv.faces[:, 2], :]
    l1 = (((v2 - v3) ** 2).sum(axis=1))
    l2 = (((v1 - v3) ** 2).sum(axis=1))
    l3 = (((v1 - v2) ** 2).sum(axis=1))
    phi1 = phi[fv.faces[:, 0], :]
    phi2 = phi[fv.faces[:, 1], :]
    phi3 = phi[fv.faces[:, 2], :]
    # a = ((fv.surfel**2).sum(axis=1))
    a = 2 * np.sqrt((fv.surfel ** 2).sum(axis=1))
    a2 = 2 * a[..., np.newaxis]
    correct =  ((v1-v3)*(phi1-phi2)).sum(axis=1) - ((v1-v2)*(phi1-phi3)).sum(axis=1)
    if variables == 'both' or variables == 'phi':
        r1 = (l1[:, np.newaxis] * (phi2 + phi3 - 2 * phi1) + (l2 - l3)[:, np.newaxis] * (phi2 - phi3)) / a2 \
        + .5 * (v2-v3) * correct[:, None] / a[:, None]
        r2 = (l2[:, np.newaxis] * (phi1 + phi3 - 2 * phi2) + (l1 - l3)[:, np.newaxis] * (phi1 - phi3)) / a2 \
        + .5 * (v3-v1) * correct[:, None] / a[:, None]
        r3 = (l3[:, np.newaxis] * (phi1 + phi2 - 2 * phi3) + (l2 - l1)[:, np.newaxis] * (phi2 - phi1)) / a2 \
        + .5 * (v1 -v2) * correct[:, None] / a[:, None]
        gradphi = np.zeros(phi.shape)
        for k, f in enumerate(fv.faces):
            gradphi[f[0], :] -= r1[k, :]
            gradphi[f[1], :] -= r2[k, :]
            gradphi[f[2], :] -= r3[k, :]

    if variables == 'both' or variables == 'x':
        gradx = np.zeros(fv.vertices.shape)
        u = (l1 * ((phi2 - phi1) * (phi3 - phi1)).sum(axis=1) + l2 * ((phi3 - phi2) * (phi1 - phi2)).sum(axis=1)
             + l3 * ((phi1 - phi3) * (phi2 - phi3)).sum(axis=1)) - \
            0.5 * (((v1-v3)*(phi1-phi2)).sum(axis=1) - ((v1-v2)*(phi1-phi3)).sum(axis=1))**2
        # u = (2*u/a**2)[...,np.newaxis]
        u = (u / (a ** 3))[..., np.newaxis]
        r1 = (- u * np.cross(v2 - v3, fv.surfel) + 2 * (
                    (v1 - v3) * (((phi3 - phi2) * (phi1 - phi2)).sum(axis=1))[:, np.newaxis]
                    + (v1 - v2) * (((phi1 - phi3) * (phi2 - phi3)).sum(axis=1)[:, np.newaxis])) / a2)
        r2 = (- u * np.cross(v3 - v1, fv.surfel) + 2 * (
                    (v2 - v1) * (((phi1 - phi3) * (phi2 - phi3)).sum(axis=1))[:, np.newaxis]
                    + (v2 - v3) * (((phi2 - phi1) * (phi3 - phi1)).sum(axis=1))[:, np.newaxis]) / a2)
        r3 = (- u * np.cross(v1 - v2, fv.surfel) + 2 * (
                    (v3 - v2) * (((phi2 - phi1) * (phi3 - phi1)).sum(axis=1))[:, np.newaxis]
                    + (v3 - v1) * (((phi3 - phi2) * (phi1 - phi2)).sum(axis=1)[:, np.newaxis])) / a2)
        r1 -= .5 * (phi3 - phi2) * correct[:, None] / a[:, None]
        r2 -= .5 * (phi1 - phi3) * correct[:, None] / a[:, None]
        r3 -= .5 * (phi2 - phi1) * correct[:, None] / a[:, None]
        for k, f in enumerate(fv.faces):
            gradx[f[0], :] += r1[k, :]
            gradx[f[1], :] += r2[k, :]
            gradx[f[2], :] += r3[k, :]

    if variables == 'both':
        return (gradphi, gradx)
    elif variables == 'phi':
        return gradphi
    elif variables == 'x':
        return gradx
    else:
        logging.info('Incorrect option in diffElasticNorm')

