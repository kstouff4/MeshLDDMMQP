import numpy as np
from .curves import Curve


def L2Norm0(fv1):
    # return ((fv1.vertices**2).sum(axis=1)*fv1.diffArcLength()).sum()
    return ((fv1.vertices ** 2).sum(axis=1)).sum()


def L2NormDef(fvDef, fv1):
    # a1 = fv1.diffArcLength()
    # aDef = fvDef.diffArcLength()
    # return (-2*(fvDef.vertices*fv1.vertices).sum(axis=1)*np.sqrt(a1*aDef) + (fvDef.vertices**2).sum(axis=1)*aDef ).sum()
    return (-2 * (fvDef.vertices * fv1.vertices).sum(axis=1) + (fvDef.vertices ** 2).sum(axis=1)).sum()


def L2NormGradient(fvDef, fv1):
    # a1 = fv1.diffArcLength()[:, np.newaxis]
    # aDef = fvDef.diffArcLength()[:, np.newaxis]
    # z1 = 2*(fvDef.vertices*aDef-fv1.vertices * np.sqrt(a1*aDef))
    z1 = 2 * (fvDef.vertices - fv1.vertices)
    return z1


def L2Norm(fvDev, fv1):
    return L2NormDef(fvDev, fv1) + L2Norm0(fv1)


# Current norm of fv1
def currentNorm0(fv1, KparDist=None, weight=None):
    c2 = fv1.centers
    cr2 = fv1.linel * fv1.line_weights[:, None]
    obj = (cr2 * KparDist.applyK(c2, cr2)).sum()
    if weight:
        cr2n = np.sqrt((cr2 ** 2).sum(axis=1))[:, np.newaxis]
        obj += weight * (cr2n * KparDist.applyK(c2, cr2n)).sum()
    return obj


# Computes |fvDef|^2 - 2 fvDef * fv1 with current dot produuct
def currentNormDef(fvDef, fv1, KparDist=None, weight=None):
    c1 = fvDef.centers
    cr1 = fvDef.linel * fvDef.line_weights[:, None]
    c2 = fv1.centers
    cr2 = fv1.linel * fv1.line_weights[:, None]
    obj = ((cr1 * KparDist.applyK(c1, cr1)).sum() - 2 * (cr1 * KparDist.applyK(c2, cr2, firstVar=c1)).sum())
    if weight:
        cr1n = np.sqrt((cr1 ** 2).sum(axis=1) + 1e-10)[:, np.newaxis]
        cr2n = np.sqrt((cr2 ** 2).sum(axis=1) + 1e-10)[:, np.newaxis]
        obj += weight * ((cr1n * KparDist.applyK(c1, cr1n)).sum() - 2 * (
                    cr1n * KparDist.applyK(c2, cr2n, firstVar=c1)).sum())
    return obj


# Returns |fvDef - fv1|^2 for current norm
def currentNorm(fvDef, fv1, KparDist=None, weight=None):
    return currentNormDef(fvDef, fv1, KparDist, weight=weight) + currentNorm0(fv1, KparDist, weight=weight)


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def currentNormGradient(fvDef, fv1, KparDist=None, weight=None, with_weights=False):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    cr1 = fvDef.linel
    w1 = fvDef.line_weights
    c2 = fv1.centers
    cr2 = fv1.linel
    w2 = fv1.line_weights
    dim = c1.shape[1]
    wcr1 = w1[:, None] * cr1
    wcr2 = w2[:, None] * cr2

    z1 = 2 * (KparDist.applyK(c1, wcr1) - KparDist.applyK(c2, wcr2, firstVar=c1))
    wz1 = w1[:, None] * z1
    dz1 = KparDist.applyDiffKT(c1, wcr1, wcr1) - KparDist.applyDiffKT(c2, wcr1, wcr2, firstVar=c1)

    if weight:
        a1 = np.sqrt((cr1 ** 2).sum(axis=1) + 1e-10)
        a1w = w1 * a1
        a2 = np.sqrt((cr2 ** 2).sum(axis=1) + 1e-10)
        a2w = w2 * a2
        cr1n = cr1 / a1[:, np.newaxis]
        z01 = 2 * (KparDist.applyK(c1, a1w[:, np.newaxis]) - KparDist.applyK(c2, a2w[:, np.newaxis], firstVar=c1))
        wz01 = w1[:, None] * z01
        wz1 += weight * (wz01 * cr1n)
        dz1 += weight * (KparDist.applyDiffKT(c1, a1w[:, np.newaxis], a1w[:, np.newaxis]) -
                         KparDist.applyDiffKT(c2, a1w[:, np.newaxis], a2w[:, np.newaxis], firstVar=c1))

    px = np.zeros([xDef.shape[0], dim])

    I = fvDef.faces[:, 0]
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - wz1[k, :]

    I = fvDef.faces[:, 1]
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] + wz1[k, :]

    if with_weights:
        z1w = (cr1 * z1).sum(axis=1)
        if weight:
            z1w += weight * a1 * z01[:, 0]
        pxw = np.zeros(xDef.shape[0])
        for j in range(2):
            I = fvDef.faces[:, j]
            for k in range(I.size):
                pxw[I[k]] += z1w[k] / 2
        return px, pxw
    else:
        return px


# Measure norm of fv1
def measureNorm0(fv1, KparDist=None, cpu=False):
    c2 = fv1.centers
    cr2 = fv1.linel
    cr2 = np.sqrt((cr2 ** 2).sum(axis=1))[:, np.newaxis] * fv1.line_weights[:, None]
    return np.multiply(cr2, KparDist.applyK(c2, cr2, cpu=cpu)).sum()


# Computes |fvDef|^2 - 2 fvDef * fv1 with measure dot produuct
def measureNormDef(fvDef, fv1, KparDist=None, cpu=False):
    c1 = fvDef.centers
    cr1 = fvDef.linel
    cr1 = (fvDef.line_weights * np.sqrt((cr1 ** 2).sum(axis=1) + 1e-10))[:, np.newaxis]
    c2 = fv1.centers
    cr2 = fv1.linel
    cr2 = (fv1.line_weights * np.sqrt((cr2 ** 2).sum(axis=1) + 1e-10))[:, np.newaxis]
    obj = (cr1 * KparDist.applyK(c1, cr1, cpu=cpu)).sum() \
          - 2 * (cr1 * KparDist.applyK(c2, cr2, firstVar=c1,cpu=cpu)).sum()
    return obj


# Returns |fvDef - fv1|^2 for measure norm
def measureNorm(fvDef, fv1, KparDist=None, cpu=False):
    return measureNormDef(fvDef, fv1, KparDist, cpu=cpu) + measureNorm0(fv1, KparDist, cpu=cpu)


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (measure norm)
def measureNormGradient(fvDef, fv1, KparDist=None, with_weights=False, cpu=False):
    xDef = fvDef.vertices
    c1 = fvDef.centers
    cr1 = fvDef.linel
    w1 = fvDef.line_weights
    c2 = fv1.centers
    cr2 = fv1.linel
    w2 = fv1.line_weights
    dim = c1.shape[1]
    a1 = np.sqrt((cr1 ** 2).sum(axis=1) + 1e-10)
    a2 = np.sqrt((cr2 ** 2).sum(axis=1) + 1e-10)
    cr1 = cr1 / a1[:, np.newaxis]
    # cr2 = cr2 / a2[:, np.newaxis]
    a1w = w1 * a1
    a2w = w2 * a2

    z1 = 2 * (KparDist.applyK(c1, a1w[:, np.newaxis], cpu=cpu) - KparDist.applyK(c2, a2w[:, np.newaxis], firstVar=c1, cpu=cpu))
    wz1 = z1 * cr1 * w1[:, None]

    dz1 = (KparDist.applyDiffKT(c1, a1w[:, np.newaxis], a1w[:, np.newaxis], cpu=cpu) -
           KparDist.applyDiffKT(c2, a1w[:, np.newaxis], a2w[:, np.newaxis], firstVar=c1, cpu=cpu))
    # dz1 = (np.multiply(dg11.sum(axis=1).reshape((-1,1)), c1) - np.dot(dg11,c1) - np.multiply(dg12.sum(axis=1).reshape((-1,1)), c1) + np.dot(dg12,c2))

    px = np.zeros([xDef.shape[0], dim])
    ###########

    I = fvDef.faces[:, 0]
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] - wz1[k, :]

    I = fvDef.faces[:, 1]
    for k in range(I.size):
        px[I[k], :] = px[I[k], :] + dz1[k, :] + wz1[k, :]

    if with_weights:
        z1w = a1 * z1[:, 0] / 2
        pxw = np.zeros(xDef.shape[0])
        for j in range(2):
            I = fvDef.faces[:, j]
            for k in range(I.size):
                pxw[I[k]] += z1w[k]
        return px, pxw
    else:
        return px


def _varifoldNorm0(c2, cr2, KparDist=None, weight=1.):
    d = weight
    a2 = np.sqrt((cr2 ** 2).sum(axis=1) + 1e-10)
    cr2 = cr2 / a2[:, np.newaxis]
    cr2cr2 = (cr2[:, np.newaxis, :] * cr2[np.newaxis, :, :]).sum(axis=2)
    a2a2 = a2[:, np.newaxis] * a2[np.newaxis, :]
    beta2 = (1 + d * cr2cr2 ** 2) * a2a2
    return KparDist.applyK(c2, beta2[..., np.newaxis], matrixWeights=True).sum()


def varifoldNorm0(fv1, KparDist=None, weight=1.):
    c2 = fv1.centers
    cr2 = fv1.line_weights[:, None] * fv1.linel
    return _varifoldNorm0(c2, cr2, KparDist=KparDist, weight=weight)


def varifoldNormComponent0(fv1, KparDist=None, weight=1.):
    c2 = fv1.centers
    cr2 = fv1.line_weights[:, None] * fv1.linel
    cp = fv1.component
    ncp = cp.max() + 1
    obj = 0
    for k in range(ncp):
        I = np.nonzero(cp == k)[0]
        obj += _varifoldNorm0(c2[I], cr2[I], KparDist=KparDist, weight=weight)
    return obj


# Computes |fvDef|^2 - 2 fvDef * fv1 with current dot product
def _varifoldNormDef(c1, c2, cr1, cr2, KparDist=None, weight=1.):
    d = weight
    a1 = np.sqrt((cr1 ** 2).sum(axis=1) + 1e-10)
    a2 = np.sqrt((cr2 ** 2).sum(axis=1) + 1e-10)
    cr1 = cr1 / a1[:, np.newaxis]
    cr2 = cr2 / a2[:, np.newaxis]

    cr1cr1 = (cr1[:, np.newaxis, :] * cr1[np.newaxis, :, :]).sum(axis=2)
    a1a1 = a1[:, np.newaxis] * a1[np.newaxis, :]
    cr1cr2 = (cr1[:, np.newaxis, :] * cr2[np.newaxis, :, :]).sum(axis=2)
    a1a2 = a1[:, np.newaxis] * a2[np.newaxis, :]

    beta1 = (1 + d * cr1cr1 ** 2) * a1a1
    beta2 = (1 + d * cr1cr2 ** 2) * a1a2

    obj = (KparDist.applyK(c1, beta1[..., np.newaxis], matrixWeights=True).sum()
           - 2 * KparDist.applyK(c2, beta2[..., np.newaxis], firstVar=c1, matrixWeights=True).sum())
    return obj


def varifoldNormDef(fvDef, fv1, KparDist=None, weight=1.):
    c1 = fvDef.centers
    cr1 = fvDef.line_weights[:, None] * fvDef.linel
    c2 = fv1.centers
    cr2 = fv1.line_weights[:, None] * fv1.linel
    return _varifoldNormDef(c1, c2, cr1, cr2, KparDist=KparDist, weight=weight)


def varifoldNormComponentDef(fvDef, fv1, KparDist=None, weight=1.):
    c1 = fvDef.centers
    cr1 = fvDef.line_weights[:, None] * fvDef.linel
    c2 = fv1.centers
    cr2 = fv1.line_weights[:, None] * fv1.linel
    cp1 = fvDef.component
    cp2 = fv1.component
    ncp = cp1.max() + 1
    obj = 0
    for k in range(ncp):
        I1 = np.nonzero(cp1 == k)[0]
        I2 = np.nonzero(cp2 == k)[0]
        obj += _varifoldNormDef(c1[I1], c2[I2], cr1[I1], cr2[I2], KparDist=KparDist, weight=weight)
    return obj


# Returns |fvDef - fv1|^2 for current norm
def varifoldNorm(fvDef, fv1, KparDist=None, weight=1.):
    return varifoldNormDef(fvDef, fv1, KparDist=KparDist, weight=weight) + varifoldNorm0(fv1, KparDist=KparDist,
                                                                                         weight=weight)


def varifoldNormComponent(fvDef, fv1, KparDist=None, weight=1.):
    return varifoldNormComponentDef(fvDef, fv1, KparDist=KparDist, weight=weight) + varifoldNormComponent0(fv1,
                                                                                                           KparDist=KparDist,
                                                                                                           weight=weight)


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def _varifoldNormGradient(c1, c2, cr1, cr2, w1, w2, KparDist=None, weight=1., with_weights=False):
    d = weight

    a1 = np.sqrt((cr1 ** 2).sum(axis=1) + 1e-10)
    a2 = np.sqrt((cr2 ** 2).sum(axis=1) + 1e-10)
    cr1 = cr1 / a1[:, np.newaxis]
    cr2 = cr2 / a2[:, np.newaxis]
    a1w = a1 * w1
    a2w = a2 * w2
    cr1cr1 = (cr1[:, np.newaxis, :] * cr1[np.newaxis, :, :]).sum(axis=2)
    cr1cr2 = (cr1[:, np.newaxis, :] * cr2[np.newaxis, :, :]).sum(axis=2)

    beta1 = a1w[:, np.newaxis] * a1w[np.newaxis, :] * (1 + d * cr1cr1 ** 2)
    beta2 = a1w[:, np.newaxis] * a2w[np.newaxis, :] * (1 + d * cr1cr2 ** 2)

    u1 = (2 * d * cr1cr1[..., np.newaxis] * cr1[np.newaxis, ...]
          - d * (cr1cr1 ** 2)[..., np.newaxis] * cr1[:, np.newaxis, :]
          + cr1[:, np.newaxis, :]) * a1w[np.newaxis, :, np.newaxis]
    u2 = (2 * d * cr1cr2[..., np.newaxis] * cr2[np.newaxis, ...]
          - d * (cr1cr2 ** 2)[..., np.newaxis] * cr1[:, np.newaxis, :]
          + cr1[:, np.newaxis, :]) * a2w[np.newaxis, :, np.newaxis]

    z1 = 2 * (KparDist.applyK(c1, u1, matrixWeights=True) - KparDist.applyK(c2, u2, firstVar=c1, matrixWeights=True))
    # print a1.shape, c1.shape
    dz1 = KparDist.applyDiffKmat(c1, beta1) - KparDist.applyDiffKmat(c2, beta2, firstVar=c1)

    if with_weights:
        beta1 = (1 + d * cr1cr1 ** 2) * a1[:, np.newaxis] * a1w[np.newaxis, :]
        beta2 = (1 + d * cr1cr2 ** 2) * a1[:, np.newaxis] * a2w[np.newaxis, :]
        z1w = w1 * (KparDist.applyK(c1, beta1[..., np.newaxis], matrixWeights=True)
                    - KparDist.applyK(c2, beta2[..., np.newaxis], firstVar=c1, matrixWeights=True))
        return z1, dz1, z1w
    else:
        return z1, dz1


def varifoldNormGradient(fvDef, fv1, KparDist=None, weight=1., with_weights=False):
    c1 = fvDef.centers
    cr1 = fvDef.linel
    w1 = fvDef.line_weights
    c2 = fv1.centers
    cr2 = fv1.linel
    w2 = fv1.line_weights
    foo = _varifoldNormGradient(c1, c2, cr1, cr2, w1, w2, KparDist=KparDist, weight=weight, with_weights=with_weights)
    z1 = foo[0]
    wz1 = z1 * w1[:, None]
    dz1 = foo[1]
    dim = c1.shape[1]

    px = np.zeros([fvDef.vertices.shape[0], dim])
    # I = fvDef.faces[:,0]
    # crs = np.cross(xDef3 - xDef2, z1)
    for k in range(fvDef.faces.shape[0]):
        px[fvDef.faces[k, 0], :] += dz1[k, :] - wz1[k, :]
        px[fvDef.faces[k, 1], :] += dz1[k, :] + wz1[k, :]

    if with_weights:
        z1w = foo[2][:, 0]
        pxw = np.zeros(fvDef.vertices.shape[0])
        for k in range(fvDef.faces.shape[0]):
            for j in range(2):
                pxw[fvDef.faces[k, j]] += z1w[k] / 2
        return px, pxw
    else:
        return px


def varifoldNormComponentGradient(fvDef, fv1, KparDist=None, weight=1., with_weights=False):
    c1 = fvDef.centers
    cr1 = fvDef.linel
    w1 = fvDef.line_weights
    c2 = fv1.centers
    cr2 = fv1.linel
    w2 = fv1.line_weights
    cp1 = fvDef.component
    cp2 = fv1.component
    ncp = cp1.max() + 1
    dim = c1.shape[1]

    z1 = np.zeros(c1.shape)
    wz1 = np.zeros(c1.shape)
    dz1 = np.zeros(c1.shape)
    if with_weights:
        z1w = np.zeros(c1.shape[0])
    else:
        z1w = None
    for k in range(ncp):
        I1 = np.nonzero(cp1 == k)[0]
        I2 = np.nonzero(cp2 == k)[0]
        foo = _varifoldNormGradient(c1[I1], c2[I2], cr1[I1], cr2[I2], w1[I1], w2[I2], KparDist=KparDist, weight=weight,
                                    with_weights=with_weights)
        z1[I1, :] = foo[0]
        wz1[I1, :] = z1[I1, :] * w1[I1, None]
        dz1[I1, :] = foo[1]
        if with_weights:
            z1w[I1] = foo[2]

    px = np.zeros([fvDef.vertices.shape[0], dim])
    for k in range(fvDef.faces.shape[0]):
        px[fvDef.faces[k, 0], :] += dz1[k, :] - wz1[k, :]
        px[fvDef.faces[k, 1], :] += dz1[k, :] + wz1[k, :]

    if with_weights:
        pxw = np.zeros(fvDef.vertices.shape[0])
        for k in range(fvDef.faces.shape[0]):
            for j in range(dim):
                pxw[fvDef.faces[k, j]] += z1w[k]
        return px, pxw
    else:
        return px

