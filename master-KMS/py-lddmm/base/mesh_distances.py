import numpy as np
from .kernelFunctions_util import applyK1K2, applyDiffK1K2T



def varifoldNorm0(fv1, KparDist, KparIm, imKernel = None):
    c2 = fv1.centers
    a2 = fv1.weights * fv1.volumes
    if imKernel is None:
        A1 = fv1.image
    else:
        A1 = np.dot(fv1.image, imKernel)

    return ((applyK1K2(c2, c2, KparDist.name, KparDist.sigma, KparDist.order,
                     A1, fv1.image, KparIm.name, KparIm.sigma, KparIm.order, a2[:, None], cpu=False))*a2[:, None]).sum()

# def varifoldNorm0_old(fv1, KparDist, imKernel = None):
#     c2 = fv1.centers
#     a2 = fv1.weights * fv1.volumes
#     if imKernel is None:
#         A1 = fv1.image
#     else:
#         A1 = np.dot(fv1.image, imKernel)
#
#     return  (betax*KparDist.applyK(c2, betay)).sum()


# def varifoldNorm0_old(fv1, KparDist, imKernel = None):
#     c2 = fv1.centers
#     a2 = fv1.weights * fv1.volumes
#     if imKernel is None:
#         cr2cr2 = (fv1.image[:, None, :] * fv1.image[None, :, :]).sum(axis=2)
#     else:
#         A1 = np.dot(fv1.image, imKernel)
#         cr2cr2 = (A1[:, None, :] * fv1.image[None, :, :]).sum(axis=2)
#
#     a2a2 = a2[:, None] * a2[None, :]
#     beta2 = cr2cr2 * a2a2
#     return KparDist.applyK(c2, beta2[..., np.newaxis], matrixWeights=True).sum()

# Computes |fvDef|^2 - 2 fvDef * fv1 with current dot product
def varifoldNormDef(fvDef, fv1, KparDist, KparIm, imKernel = None):
    c1 = fvDef.centers
    c2 = fv1.centers
    a1 = fvDef.weights * fvDef.volumes
    a2 = fv1.weights * fv1.volumes
    if imKernel is None:
        A1 = fvDef.image
    else:
        A1 = np.dot(fvDef.image, imKernel)

    obj = ((applyK1K2(c1, c1, KparDist.name, KparDist.sigma, KparDist.order,
                     A1, fvDef.image, KparIm.name, KparIm.sigma, KparIm.order, a1[:, None], cpu=False) -
           2*applyK1K2(c1, c2, KparDist.name, KparDist.sigma, KparDist.order,
                     A1, fv1.image, KparIm.name, KparIm.sigma, KparIm.order, a2[:, None], cpu=False)) * a1[:, None]).sum()

    return obj

def varifoldNormDef_old(fvDef, fv1, KparDist, imKernel = None):
    c1 = fvDef.centers
    c2 = fv1.centers
    a1 = fvDef.weights * fvDef.volumes
    a2 = fv1.weights * fv1.volumes
    if imKernel is None:
        betax1 = a1[:, None] * fvDef.image
        betay1 = betax1
        betay2 = a2[:, None] * fv1.image
    else:
        A1 = np.dot(fvDef.image, imKernel)
        betax1 = a1[:, None] * A1
        betay1 = a1[:, None] * fvDef.image
        betay2 = a2[:, None] * fv1.image

    obj = (betax1*KparDist.applyK(c1, betay1)).sum() \
          - 2*(betax1*KparDist.applyK(c2,betay2, firstVar=c1)).sum()
    return obj

# def varifoldNormDef_old(fvDef, fv1, KparDist, imKernel = None):
#     c1 = fvDef.centers
#     c2 = fv1.centers
#     a1 = fvDef.weights * fvDef.volumes
#     a2 = fv1.weights * fv1.volumes
#     if imKernel is None:
#         cr1cr1 = (fvDef.image[:, None, :] * fvDef.image[None, :, :]).sum(axis=2)
#         cr1cr2 = (fvDef.image[:, None, :] * fv1.image[None, :, :]).sum(axis=2)
#     else:
#         A1 = np.dot(fvDef.image, imKernel)
#         cr1cr1 = (A1[:, None, :] * fvDef.image[None, :, :]).sum(axis=2)
#         cr1cr2 = (A1[:, None, :] * fv1.image[None, :, :]).sum(axis=2)
#
#     a1a1 = a1[:, np.newaxis] * a1[np.newaxis, :]
#     a1a2 = a1[:, np.newaxis] * a2[np.newaxis, :]
#
#     beta1 = cr1cr1 * a1a1
#     beta2 = cr1cr2 * a1a2
#
#     obj = (KparDist.applyK(c1, beta1[..., np.newaxis], matrixWeights=True).sum()
#            - 2 * KparDist.applyK(c2, beta2[..., np.newaxis], firstVar=c1, matrixWeights=True).sum())
#     return obj
#


# Returns |fvDef - fv1|^2 for current norm
def varifoldNorm(fvDef, fv1, KparDist, KparIm, imKernel = None):
    return varifoldNormDef(fvDef, fv1, KparDist, KparIm, imKernel=imKernel) \
           + varifoldNorm0(fv1, KparDist, KparIm, imKernel=imKernel)


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def varifoldNormGradient(fvDef, fv1, KparDist, KparIm, with_weights=False, imKernel=None):
    c1 = fvDef.centers
    nf = c1.shape[0]
    c2 = fv1.centers
    a1 = fvDef.weights
    a2 = fv1.weights
    a1v = fvDef.weights * fvDef.volumes
    a2v = fv1.weights * fv1.volumes
    if imKernel is None:
        A1 = fvDef.image
    else:
        A1 = np.dot(fvDef.image, imKernel)

    crx1 = a1[:, None] * A1
    crx1v = a1v[:, None] * A1
    cry1v = a1v[:, None] * fvDef.image
    cry2v = a2v[:, None] * fv1.image

    dim = c1.shape[1]
    normals = np.zeros((dim+1, nf, dim))
    if dim == 2:
        k1 = 2
        k2 = 3
        #J = np.array([[0, -1], [1,0]])
        normals[0, :, :] = fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 1], :]
        normals[1, :, :] = fvDef.vertices[fvDef.faces[:, 0], :] - fvDef.vertices[fvDef.faces[:, 2], :]
        normals[2, :, :] = fvDef.vertices[fvDef.faces[:, 1], :] - fvDef.vertices[fvDef.faces[:, 0], :]
        normals = np.flip(normals, axis=2)
        normals[:,:,0] = - normals[:,:,0]
        #normals  = normals @ J
    else:
        k1 = 6
        k2 = 4
        normals[0,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 3], :] - fvDef.vertices[fvDef.faces[:, 1], :],
                                  fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 1], :])
        normals[1,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 0], :],
                                  fvDef.vertices[fvDef.faces[:, 3], :] - fvDef.vertices[fvDef.faces[:, 0], :])
        normals[2,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 3], :] - fvDef.vertices[fvDef.faces[:, 0], :],
                                  fvDef.vertices[fvDef.faces[:, 1], :] - fvDef.vertices[fvDef.faces[:, 0], :])
        normals[3,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 1], :] - fvDef.vertices[fvDef.faces[:, 0], :],
                                  fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 0], :])

    # u1 = a1a1[:,:] * cr1cr1[:,:] * fvDef.volumes[None, :]
    # u2 = a1a2[:,:] * cr1cr2[:,:] * fv1.volumes[None, :]
    #z1 = (crx1 * KparDist.applyK(c1, cry1v) - crx1 * KparDist.applyK(c2, cry2v, firstVar=c1)).sum(axis=1)
    z1 = (a1[:, None]*applyK1K2(c1, c1, KparDist.name, KparDist.sigma, KparDist.order,
                                A1, fvDef.image, KparIm.name, KparIm.sigma, KparIm.order, a1v[:, None])
          - a1[:, None]*applyK1K2(c1, c2, KparDist.name, KparDist.sigma, KparDist.order,
                                  A1, fv1.image, KparIm.name, KparIm.sigma, KparIm.order, a2v[:, None])).sum(axis=1)
    # z1_ = (KparDist.applyK(c1, u1[:,:, None], matrixWeights=True) -
    #      KparDist.applyK(c2, u2[:,:,None], firstVar=c1, matrixWeights=True))
    z1 = z1[None, :, None] * normals/k1

    # beta1 = a1a1v * cr1cr1
    # beta2 = a1a2v * cr1cr2
    dz1 = (applyDiffK1K2T(c1, c1, KparDist.name, KparDist.sigma, KparDist.order,
                         A1, fvDef.image, KparIm.name, KparIm.sigma, KparIm.order, a1v[:, None], a1v[:, None])
           - applyDiffK1K2T(c1, c2, KparDist.name, KparDist.sigma, KparDist.order,
                         A1, fv1.image, KparIm.name, KparIm.sigma, KparIm.order, a1v[:, None], a2v[:, None]))/k2

    # dz1 = (KparDist.applyDiffKT(c1, crx1v, cry1v) - KparDist.applyDiffKT(c2, crx1v, cry2v, firstVar=c1))/k2
    # dz1 = (KparDist.applyDiffKmat(c1, beta1) - KparDist.applyDiffKmat(c2, beta2, firstVar=c1))/k2


    px = np.zeros(fvDef.vertices.shape)
    for i in range(dim+1):
        I = fvDef.faces[:, i]
        for k in range(I.size):
            px[I[k], :] = px[I[k], :] + dz1[k, :] + z1[i, k, :]
    return 2*px

# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (current norm)
def varifoldNormGradient_old(fvDef, fv1, KparDist, with_weights=False, imKernel=None):
    c1 = fvDef.centers
    nf = c1.shape[0]
    c2 = fv1.centers
    a1 = fvDef.weights
    a2 = fv1.weights
    a1v = fvDef.weights * fvDef.volumes
    a2v = fv1.weights * fv1.volumes
    if imKernel is None:
        cr1cr1 = (fvDef.image[:, None, :] * fvDef.image[None, :, :]).sum(axis=2)
        cr1cr2 = (fvDef.image[:, None, :] * fv1.image[None, :, :]).sum(axis=2)
    else:
        A1 = np.dot(fvDef.image, imKernel)
        cr1cr1 = (A1[:, None, :] * fvDef.image[None, :, :]).sum(axis=2)
        cr1cr2 = (A1[:, None, :] * fv1.image[None, :, :]).sum(axis=2)

    a1a1 = a1[:, np.newaxis] * a1[np.newaxis, :]
    a1a2 = a1[:, np.newaxis] * a2[np.newaxis, :]
    a1a1v = a1v[:, np.newaxis] * a1v[np.newaxis, :]
    a1a2v = a1v[:, np.newaxis] * a2v[np.newaxis, :]
    dim = c1.shape[1]
    normals = np.zeros((dim+1, nf, dim))
    if dim == 2:
        k1 = 2
        k2 = 3
        #J = np.array([[0, -1], [1,0]])
        normals[0, :, :] = fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 1], :]
        normals[1, :, :] = fvDef.vertices[fvDef.faces[:, 0], :] - fvDef.vertices[fvDef.faces[:, 2], :]
        normals[2, :, :] = fvDef.vertices[fvDef.faces[:, 1], :] - fvDef.vertices[fvDef.faces[:, 0], :]
        normals = np.flip(normals, axis=2)
        normals[:,:,0] = - normals[:,:,0]
        #normals  = normals @ J
    else:
        k1 = 6
        k2 = 4
        normals[0,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 3], :] - fvDef.vertices[fvDef.faces[:, 1], :],
                                  fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 1], :])
        normals[1,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 0], :],
                                  fvDef.vertices[fvDef.faces[:, 3], :] - fvDef.vertices[fvDef.faces[:, 0], :])
        normals[2,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 3], :] - fvDef.vertices[fvDef.faces[:, 0], :],
                                  fvDef.vertices[fvDef.faces[:, 1], :] - fvDef.vertices[fvDef.faces[:, 0], :])
        normals[3,:,:] = np.cross(fvDef.vertices[fvDef.faces[:, 1], :] - fvDef.vertices[fvDef.faces[:, 0], :],
                                  fvDef.vertices[fvDef.faces[:, 2], :] - fvDef.vertices[fvDef.faces[:, 0], :])

    u1 = a1a1[:,:] * cr1cr1[:,:] * fvDef.volumes[None, :]
    u2 = a1a2[:,:] * cr1cr2[:,:] * fv1.volumes[None, :]
    z1 = (KparDist.applyK(c1, u1[:,:, None], matrixWeights=True) -
         KparDist.applyK(c2, u2[:,:,None], firstVar=c1, matrixWeights=True))
    z1 = z1[None, :, :] * normals/k1

    beta1 = a1a1v * cr1cr1
    beta2 = a1a2v * cr1cr2
    dz1 = (KparDist.applyDiffKmat(c1, beta1) - KparDist.applyDiffKmat(c2, beta2, firstVar=c1))/k2


    px = np.zeros(fvDef.vertices.shape)
    for i in range(dim+1):
        I = fvDef.faces[:, i]
        for k in range(I.size):
            px[I[k], :] = px[I[k], :] + dz1[k, :] + z1[i, k, :]
    return 2*px
