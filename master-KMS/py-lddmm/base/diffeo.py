import numpy as np
import logging
import pyfftw
from numba import prange, jit, int64
from multiprocessing import cpu_count

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



try:
    from vtk import vtkStructuredPointsReader
    import vtk.util.numpy_support as v2n
    gotVTK = True
except ImportError:
    print('could not import VTK functions')
    gotVTK = False

import array
from PIL.Image import core as _imaging

## Functions for images and diffeomorphisms

def convolve_(img, K, periodic=False):
    fftShape = ()
    seqPadK = ()
    seqPadI = ()
    seqPadOne = ()
    ax = np.arange(img.ndim, dtype=int)

    for i in range(K.ndim):
        if not periodic:
            delta1 = K.shape[i] - 1
        else:
            delta1 = 0
        if delta1 < img.shape[i]:
            fftShape += (img.shape[i] + delta1,)
        else:
            fftShape += (2 * delta1,)

        seqPadK += ((0, img.shape[i]-1),)
        seqPadI += ((K.shape[i] // 2, K.shape[i] // 2),)
        #seqPadI += ((0, K.shape[i] -1),)
        seqPadOne += ((K.shape[i] - 1, 0),)

    fft_in = pyfftw.empty_aligned(fftShape, dtype='complex128')
    fft_out = pyfftw.empty_aligned(fftShape, dtype='complex128')
    ifft_out = pyfftw.empty_aligned(fftShape, dtype='complex128')
    fft_fwd = pyfftw.FFTW(fft_in, fft_out, threads=cpu_count(), axes=ax)
    fft_bwd = pyfftw.FFTW(fft_out, ifft_out, threads = cpu_count(), direction='FFTW_BACKWARD', axes=ax)
    newK = np.pad(K, seqPadK)
    fK = pyfftw.empty_aligned(fftShape, dtype='complex128')
    fft_in[...] = newK
    fK[...] = fft_fwd()
    fft_in[...] = np.pad(img, seqPadI, mode='symmetric')
    newOnes = np.pad(np.ones(img.shape, dtype=bool), seqPadOne)
    fI = pyfftw.empty_aligned(fftShape, dtype='complex128')
    fI[...] = fft_fwd()
    fft_out[...] = fK * fI
    newRes = pyfftw.empty_aligned(fftShape)
    newRes[...] = np.real(fft_bwd())
    newRes = np.real(newRes[newOnes].reshape(img.shape))
    return newRes


class Kernel:
    def __init__(self, dim=3, sigma=1.0, order=1, size=50, name = 'gauss'):
        self.dim = dim
        self.sigma = sigma
        self.order = order
        self.size = int(self.getSize(name, order, sigma))#size
        size = self.size
        logging.info(f'size kernel: {size:}')
        self.type = name
        if dim == 3:
            [x, y, z] = np.mgrid[0:2 * size + 1, 0:2 * size + 1, 0:2 * size + 1]  - size
            d = np.sqrt(x**2 + y**2 + z**2)/sigma
        elif dim == 2:
            [x, y] = np.mgrid[0:2 * size + 1, 0:2 * size + 1] - size
            d = np.sqrt(x**2 + y**2)/sigma
        elif dim == 1:
            x = np.mgrid[0:2 * size + 1] - size
            d = np.fabs(x)/sigma
        else:
            logging.error('Kernels in dimensions three or less')
            # if dim == 3:
            #     [x, y, z] = np.mgrid[0:2 * size + 1, 0:2 * size + 1, 0:2 * size + 1] / size - 1
            #     d = np.sqrt(x ** 2 + y ** 2 + z ** 2) / sigma
            # elif dim == 2:
            #     [x, y] = np.mgrid[0:2 * size + 1, 0:2 * size + 1] / size - 1
            #     d = np.sqrt(x ** 2 + y ** 2) / sigma
            # elif dim == 1:
            #     x = np.mgrid[0:2 * size + 1] / size - 1
            #     d = np.fabs(x) / sigma
            # else:
            #     logging.error('Kernels in dimensions three or less')
            return

        if name=='gauss':
            self.K = np.exp(-d**2/2)
        elif name in ('laplacian', 'matern'):
            d2 = d * d
            self.K = (c_[order, 0] + c_[order, 1] * d + c_[order, 2] * d2
                      + c_[order, 3] * d * d2 + c_[order, 4] * d2 * d2)*np.exp(-d)
        else:
            logging.error('Unknown kernel')
            return


    def getSize(self, name, order, sigma, thresh = 0.001):
        if name == 'gauss':
            return np.ceil(sigma*np.sqrt(-2*np.log(thresh)))
        if name in ('laplacian', 'matern'):
            t = np.linspace(0, 100, 10000)
            K = (c_[order, 0] + c_[order, 1] * t + c_[order, 2] * t**2
                      + c_[order, 3] * t**3 + c_[order, 4] * t**4)*np.exp(-t)
            i = np.argmin(K>thresh)
            return np.ceil(sigma*t[i])



    def ApplyToImage(self, img, mask = None):
        if mask is None:
            mask = np.ones(img.shape)
        #res = mask * convolve(img*mask, self.K, mode='same')
        res = mask * convolve_(img*mask, self.K)
        return np.real(res)

    def ApplyToVectorField(self, Lv, mask = None):
        if mask is None:
            mask = np.ones(Lv.shape[1:])
        res = np.zeros(Lv.shape)
        for i in range(self.dim):
            res[i,...] = mask[i,...] * convolve_(Lv[i,...]*mask[i,...], self.K)
            #res[i, ...] = mask[i, ...] * convolve(Lv[i, ...] * mask[i, ...], self.K, mode='same')
        return res




def idMesh(shape, normalize=False, reverse = False):
    dim = len(shape)
    if dim==1:
        res = np.mgrid[0:shape[0]]
        if reverse:
            res = shape[0] - 1 -res
    elif dim == 2:
        x, y = np.mgrid[0:shape[0], 0:shape[1]]
        res = np.zeros((2, shape[0], shape[1]))
        if reverse:
            x = shape[0] - 1 - x
            y = shape[1] - 1 - y
        res[0, ...] = x
        res[1, ...] = y
    elif dim==3:
        x,y,z = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        if reverse:
            x = shape[0] - 1 - x
            y = shape[1] - 1 - y
            z = shape[1] - 1 - z
        res = np.zeros((3,shape[0], shape[1], shape[2]))
        res[0,...] = x
        res[1,...] = y
        res[2,...] = y
    else:
        logging.error('No idMesh in dimension larger than 3')
        return
    if normalize:
        for i in range(res.shape[0]):
            res[i,...]/=shape[i]
    return res

def plateau(N,a,b):
    t = np.linspace(0,1,N)
    f = np.ones(N)
    f[t<a] = 0
    J = np.logical_and(t>=a, t<b)
    u = (t[J] - a)/(b-a)
    f[J] = 3 * u**2 - 2* u**3
    J = np.logical_and(t>=1-b, t<=1-a)
    u = (1-a-t[J] )/(b-a)
    f[J] = 3 * u**2 - 2* u**3
    f[t>1-a] = 0
    return f


def makeMask(margin, S, Neumann=True, periodic = False):
    dim = len(S)
    delta = 0.05

    mask0 = np.ones((dim,) + tuple(S))
    if periodic:
        return mask0

    if Neumann:
        for k in range(dim):
            if k != 0:
                mask = np.ones(S[0])
            else:
                mask = plateau(S[0], margin / S[0], margin / S[0] + delta)
            for l in range(1, dim):
                if l != k:
                    u = np.ones(S[l])
                else:
                    u = plateau(S[k], margin / S[k], margin / S[k] + delta)
                mask = mask[..., None] * u[None, ...]
            mask0[k, ...] = mask
    else:
        mask = plateau(S[0], margin / S[0], margin / S[0] + delta)
        for k in range(1, dim):
            u = plateau(S[k], margin / S[k], margin / S[k] + delta)
            mask = mask[..., None] * u[None, ...]
        for k in range(dim):
            mask0[k,...] = mask

    return mask0


# multilinear interpolation
@jit(nopython=True, parallel=True)
def multilinInterp(img, diffeo):
    ndim = img.ndim
    if ndim > 3:
        print('interpolate only in dimensions 1 to 3')
        return

    #print diffeo.shape
    tooLarge = diffeo.min() < 0
    for k in range(img.ndim):
        if diffeo[k, ...].max() > img.shape[k]-1:
            tooLarge = True
    if tooLarge:
        #print "min", diffeo.min()
        dfo = np.copy(diffeo)
        dfo = np.maximum(dfo, 0)
        for k in range(img.ndim):
            dfo[k, ...] = np.minimum(dfo[k, ...], img.shape[k]-1)
    else:
        dfo = diffeo

    I = np.zeros(dfo.shape, dtype=int64)
    I[...] = np.floor(dfo)
    J = np.zeros(dfo.shape, dtype=int64)
    r = np.zeros(dfo.shape)
    for k in range(ndim):
        J[k, ...] = np.minimum(I[k, ...]+1, img.shape[k]-1) 
        r[k, ...] = dfo[k, ...] - I[k, ...]
    res = np.zeros(img.shape)

    if ndim ==1:
        for k in range(I.shape[1]):
            res[k] = (1-r[0, k]) * img[I[0, k]] + r[0, k] * img[J[0, k]]
    elif ndim==2:
        for k in prange(I.shape[1]):
            for l in range(I.shape[2]):
                res[k,l] = ((1-r[1, k,l]) * ((1-r[0, k,l]) * img[I[0, k,l], I[1,k,l]] + r[0, k,l] * img[J[0, k,l], I[1,k,l]])
                        + r[1, k,l] * ((1-r[0, k,l]) * img[I[0, k,l], J[1,k,l]] + r[0, k,l] * img[J[0, k,l], J[1,k,l]]))
    elif ndim==3:
        for k in prange(I.shape[1]):
            for l in range(I.shape[2]):
                for m in range(I.shape[3]):
                    res[k,l,m] = ((1-r[2,k,l, m]) * ((1-r[1, k,l, m]) * ((1-r[0, k,l, m])
                                                                  * img[I[0, k,l, m], I[1,k,l, m], I[2, k,l, m]]
                                                                  + r[0, k,l, m] * img[J[0, k,l, m], I[1,k,l, m], I[2,k,l, m]])
                                            + r[1, k,l, m] * ((1-r[0, k,l, m])
                                                              * img[I[0, k,l, m], J[1,k,l, m], I[2, k,l, m]]
                                                              + r[0, k,l, m] * img[J[0, k,l, m], J[1,k,l, m], I[2,k,l, m]]))
                            + r[2,k,l, m] * ((1-r[1, k,l, m]) * ((1-r[0, k,l, m])
                                                                 * img[I[0, k,l, m], I[1,k,l, m], J[2, k,l, m]]
                                                                 + r[0, k,l, m] * img[J[0, k,l, m], I[1,k,l, m], J[2,k,l, m]])
                                        + r[1, k,l, m] * ((1-r[0, k,l, m])
                                                          * img[I[0, k,l, m], J[1,k,l, m], J[2, k,l, m]]
                                                          + r[0, k,l, m] * img[J[0, k,l, m], J[1,k,l, m], J[2,k,l, m]])))
    else:
        print('interpolate only in dimensions 1 to 3')
        return

    return res


# multilinear interpolation
@jit(nopython=True, parallel=True)
def multilinInterpDual(img, diffeo):
    ndim = img.ndim
    if ndim > 3:
        print('interpolate only in dimensions 1 to 3')
        return

    #print diffeo.shape
    tooLarge = diffeo.min() < 0
    for k in range(img.ndim):
        if diffeo[k, ...].max() > img.shape[k]-1:
            tooLarge = True
    if tooLarge:
        #print "min", diffeo.min()
        dfo = np.copy(diffeo)
        dfo = np.maximum(dfo, 0)
        for k in range(img.ndim):
            dfo[k, ...] = np.minimum(dfo[k, ...], img.shape[k]-1)
    else:
        dfo = diffeo

    I = np.zeros(dfo.shape, dtype=int64)
    I[...] = np.floor(dfo)
    J = np.zeros(dfo.shape, dtype=int64)
    r = np.zeros(dfo.shape)
    for k in range(ndim):
        J[k, ...] = np.minimum(I[k, ...]+1, img.shape[k]-1)
        r[k, ...] = dfo[k, ...] - I[k, ...]

    res = np.zeros(img.shape)
    if ndim ==1:
        for k in range(I.shape[1]):
            res[I[0,k]] += (1-r[0, k]) * img[k]
            res[J[0,k]] += r[0, k] * img[k]
    elif ndim==2:
        for k in prange(I.shape[1]):
            for l in range(I.shape[2]):
                res[I[0,k,l], I[1,k,l]] += (1-r[1, k,l]) * (1-r[0, k,l]) * img[k,l]
                res[J[0,k,l], I[1,k,l]] += (1-r[1, k,l]) * r[0, k,l] * img[k,l]
                res[I[0,k,l], J[1,k,l]] += (1-r[0, k,l]) * r[1, k,l] * img[k,l]
                res[J[0,k,l], J[1,k,l]] += r[1, k,l] * r[0, k,l] * img[k,l]
        # res[I[0,...], I[1,...]] += (1-r[1, ...]) * (1-r[0, ...]) * img
        # res[J[0,...], I[1,...]] += (1-r[1, ...]) * r[0, ...] * img
        # res[I[0,...], J[1,...]] += (1-r[0, ...]) * r[1, ...] * img
        # res[J[0,...], J[1,...]] += r[1, ...] * r[0, ...] * img
    elif ndim==3:
        for k in prange(I.shape[1]):
            for l in range(I.shape[2]):
                for m in range(I.shape[3]):
                    res[I[0,k,l,m], I[1,k,l,m], I[2,k,l,m]] += (1-r[0, k,l,m]) * (1-r[1, k,l,m]) * (1-r[2, k,l,m]) * img[k,l,m]
                    res[J[0,k,l,m], I[1,k,l,m], I[2,k,l,m]] += r[0, k,l,m] * (1-r[1, k,l,m]) * (1-r[2, k,l,m]) * img[k,l,m]
                    res[I[0,k,l,m], J[1,k,l,m], I[2,k,l,m]] += (1-r[0, k,l,m]) * r[1, k,l,m] * (1-r[2, k,l,m]) * img[k,l,m]
                    res[I[0,k,l,m], I[1,k,l,m], J[2,k,l,m]] += (1-r[0, k,l,m]) * (1-r[1, k,l,m]) * r[2, k,l,m] * img[k,l,m]
                    res[J[0,k,l,m], J[1,k,l,m], I[2,k,l,m]] += r[0, k,l,m] * r[1, k,l,m] * (1-r[2, k,l,m]) * img[k,l,m]
                    res[J[0,k,l,m], I[1,k,l,m], J[2,k,l,m]] += r[0, k,l,m] * (1-r[1, k,l,m]) * r[2, k,l,m] * img[k,l,m]
                    res[I[0,k,l,m], J[1,k,l,m], J[2,k,l,m]] += (1-r[0, k,l,m]) * r[1, k,l,m] * r[2, k,l,m] * img[k,l,m]
                    res[J[0,k,l,m], J[1,k,l,m], J[2,k,l,m]] += r[0, k,l,m] * r[1, k,l,m] * r[2, k,l,m] * img[k,l,m]
    else:
        print('interpolate dual only in dimensions 1 to 3')
        return

    return res


def multilinInterpVectorField(v, diffeo):
    res = np.zeros(v.shape)
    for k in range(v.shape[0]):
        res[k,...] = multilinInterp(v[k,...], diffeo)
    return res

@jit(nopython=True)
def multilinInterpGradient(img, diffeo):
    ndim = img.ndim
    if ndim > 3:
        print('interpolate only in dimensions 1 to 3')
        return

    #print diffeo.shape
    tooLarge = diffeo.min() < 0
    for k in range(img.ndim):
        if diffeo[k, ...].max() > img.shape[k]-1:
            tooLarge = True
    if tooLarge:
        dfo = np.copy(diffeo)
        dfo = np.maximum(dfo, 0)
        for k in range(img.ndim):
            dfo[k, ...] = np.minimum(dfo[k, ...], img.shape[k]-1)
    else:
        dfo = diffeo

    I = np.zeros(dfo.shape, dtype=int64)
    I[...] = np.floor(dfo)
    J = np.zeros(dfo.shape, dtype=int64)
    r = np.zeros(dfo.shape)
    for k in range(ndim):
        J[k, ...] = np.minimum(I[k, ...]+1, img.shape[k]-1) 
        r[k, ...] = dfo[k, ...] - I[k, ...]

#    if tooLarge:
#        print "too large"
#    print I.min(), I.max(), J.min(), J.max()

    if ndim ==1:
        res = np.zeros(I.shape)
        for k in range(I.shape[1]):
            res[0,k] = img[J[0, k]] - img[I[0, k]]
    elif ndim==2:
        res = np.zeros(I.shape)
        for k in prange(I.shape[1]):
            for l in range(I.shape[2]):
                res[0,k,l] = ((1-r[1, k,l]) * (-img[I[0, k,l], I[1,k,l]] + img[J[0, k,l], I[1,k,l]])
                        + r[1, k,l] * (-img[I[0, k,l], J[1,k,l]] + img[J[0, k,l], J[1,k,l]]))
                res[1, k,l] = (- ((1-r[0, k,l]) * img[I[0, k,l], I[1,k,l]] + r[0, k,l] * img[J[0, k,l], I[1,k,l]])
                        + ((1-r[0, k,l]) * img[I[0, k,l], J[1,k,l]] + r[0, k,l] * img[J[0, k,l], J[1,k,l]]))
    elif ndim==3:
        #res = np.zeros(np.insert(img.shape, 0, 3))
        res = np.zeros(I.shape)
        for k in prange(I.shape[1]):
            for l in range(I.shape[2]):
                for m in range(I.shape[3]):
                    res[0,k,l,m] = ((1-r[2,k,l,m]) * ((1-r[1, k,l,m]) * (-img[I[0, k,l,m], I[1,k,l,m], I[2, k,l,m]] + img[J[0, k,l,m], I[1,k,l,m], I[2,k,l,m]])
                                            + r[1, k,l,m] * (-img[I[0, k,l,m], J[1,k,l,m], I[2, k,l,m]] + img[J[0, k,l,m], J[1,k,l,m], I[2,k,l,m]]))
                            + r[2,k,l,m] * ((1-r[1, k,l,m]) * (- img[I[0, k,l,m], I[1,k,l,m], J[2, k,l,m]] + img[J[0, k,l,m], I[1,k,l,m], J[2,k,l,m]])
                                       + r[1, k,l,m] * (-img[I[0, k,l,m], J[1,k,l,m], J[2, k,l,m]] + img[J[0, k,l,m], J[1,k,l,m], J[2,k,l,m]])))
                    res[1, k,l,m] = ((1-r[2,k,l,m]) * (-((1-r[0, k,l,m]) * img[I[0, k,l,m], I[1,k,l,m], I[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], I[1,k,l,m], I[2,k,l,m]])
                                            + ((1-r[0, k,l,m]) * img[I[0, k,l,m], J[1,k,l,m], I[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], J[1,k,l,m], I[2,k,l,m]]))
                            + r[2,k,l,m] * (-((1-r[0, k,l,m]) * img[I[0, k,l,m], I[1,k,l,m], J[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], I[1,k,l,m], J[2,k,l,m]])
                                        + ((1-r[0, k,l,m]) * img[I[0, k,l,m], J[1,k,l,m], J[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], J[1,k,l,m], J[2,k,l,m]])))
                    res[2, k,l,m] = (-((1-r[1, k,l,m]) * ((1-r[0, k,l,m]) * img[I[0, k,l,m], I[1,k,l,m], I[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], I[1,k,l,m], I[2,k,l,m]])
                                    + r[1, k,l,m] * ((1-r[0, k,l,m]) * img[I[0, k,l,m], J[1,k,l,m], I[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], J[1,k,l,m], I[2,k,l,m]]))
                            + ((1-r[1, k,l,m]) * ((1-r[0, k,l,m]) * img[I[0, k,l,m], I[1,k,l,m], J[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], I[1,k,l,m], J[2,k,l,m]])
                                + r[1, k,l,m] * ((1-r[0, k,l,m]) * img[I[0, k,l,m], J[1,k,l,m], J[2, k,l,m]] + r[0, k,l,m] * img[J[0, k,l,m], J[1,k,l,m], J[2,k,l,m]])))
    else:
        print('interpolate only in dimensions 1 to 3')
        return

    return res


def multilinInterpGradientVectorField(src, diffeo):
    res0 = np.zeros([src.shape[0]] + list(src.shape))
    for k in range(src.shape[0]):
        res0[k,...] = multilinInterpGradient(src[k,...], diffeo)
    return res0


# Computes gradient
@jit(nopython=True, parallel=True)
def imageGradient(img, resol=None):
    res = None
    if img.ndim > 3:
        print('gradient only in dimensions 1 to 3')

    if img.ndim == 3:
        if resol == None:
            resol = (1.,1.,1.)
        res = np.zeros((3,img.shape[0], img.shape[1], img.shape[2]))
        res[0,1:img.shape[0]-1, :, :] = (img[2:img.shape[0], :, :] - img[0:img.shape[0]-2, :, :])/(2*resol[0])
        res[0,0, :, :] = (img[1, :, :] - img[0, :, :])/(resol[0])
        res[0,img.shape[0]-1, :, :] = (img[img.shape[0]-1, :, :] - img[img.shape[0]-2, :, :])/(resol[0])
        res[1,:, 1:img.shape[1]-1, :] = (img[:, 2:img.shape[1], :] - img[:, 0:img.shape[1]-2, :])/(2*resol[1])
        res[1,:, 0, :] = (img[:, 1, :] - img[:, 0, :])/(resol[1])
        res[1,:, img.shape[1]-1, :] = (img[:, img.shape[1]-1, :] - img[:, img.shape[1]-2, :])/(resol[1])
        res[2,:, :, 1:img.shape[2]-1] = (img[:, :, 2:img.shape[2]] - img[:, :, 0:img.shape[2]-2])/(2*resol[2])
        res[2,:, :, 0] = (img[:, :, 1] - img[:, :, 0])/(resol[2])
        res[2,:, :, img.shape[2]-1] = (img[:, :, img.shape[2]-1] - img[:, :, img.shape[2]-2])/(resol[2])
    elif img.ndim ==2:
        if resol is None:
            resol = (1.,1.)
        res = np.zeros((2,img.shape[0], img.shape[1]))
        res[0,1:img.shape[0]-1, :] = (img[2:img.shape[0], :] - img[0:img.shape[0]-2, :])/(2*resol[0])
        res[0,0, :] = (img[1, :] - img[0, :])/(resol[0])
        res[0,img.shape[0]-1, :] = (img[img.shape[0]-1, :] - img[img.shape[0]-2, :])/(resol[0])
        res[1,:, 0] = (img[:, 1] - img[:, 0])/(resol[1])
        res[1,:, img.shape[1]-1] = (img[:, img.shape[1]-1] - img[:, img.shape[1]-2])/(resol[1])
        res[1,:, 1:img.shape[1]-1] = (img[:, 2:img.shape[1]] - img[:, 0:img.shape[1]-2])/(2*resol[1])
    elif img.ndim ==1:
        if resol == None:
            resol = 1
        res = np.zeros(img.shape[0])
        res[1:img.shape[0]-1] = (img[2:img.shape[0]] - img[0:img.shape[0]-2])/(2*resol)
        res[0] = (img[1] - img[0])/(resol)
        res[img.shape[0]-1] = (img[img.shape[0]-1] - img[img.shape[0]-2])/(resol)
    return res

# Computes Jacobian determinant
#@jit(nopython=True)
def jacobianDeterminant(diffeo, resol=(1.,1.,1.), periodic=False):
    if diffeo.ndim > 4:
        print('No jacobian in dimension larger than 3')
        return

    if diffeo.ndim == 4:
        if periodic == True:
            w = np.mgrid[0:diffeo.shape[1], 0:diffeo.shape[2], 0:diffeo.shape[3]]
            dw = diffeo-w
            for k in range(3):
                diffeo[k,:,:,:] -= np.rint(dw[k,:,:,:]/diffeo.shape[k+1])*diffeo.shape[k+1]
        grad = np.zeros([3,3,diffeo.shape[1], diffeo.shape[2], diffeo.shape[3]])
        grad[0,:,:,:,:] = imageGradient(np.squeeze(diffeo[0,:,:,:]), resol=resol)
        grad[1,:,:,:,:] = imageGradient(np.squeeze(diffeo[1,:,:,:]), resol=resol)
        grad[2,:,:,:,:] = imageGradient(np.squeeze(diffeo[2,:,:,:]), resol=resol)
        res = np.fabs(grad[0,0,:,:,:] * grad[1,1,:,:,:] * grad[2,2,:,:,:]
                        - grad[0,0,:,:,:] * grad[1,2,:,:,:] * grad[2,1,:,:,:]
                        - grad[0,1,:,:,:] * grad[1,0,:,:,:] * grad[2,2,:,:,:]
                        - grad[0,2,:,:,:] * grad[1,1,:,:,:] * grad[2,0,:,:,:]
                        + grad[0,1,:,:,:] * grad[1,2,:,:,:] * grad[2,0,:,:,:] 
                        + grad[0,2,:,:,:] * grad[1,0,:,:,:] * grad[2,1,:,:,:])
    elif diffeo.ndim == 3:
        if periodic == True:
            w = np.mgrid[0:diffeo.shape[1], 0:diffeo.shape[2]]
            dw = diffeo-w
            for k in range(2):
                diffeo[k,:,:] -= np.rint(dw[k,:,:]/diffeo.shape[k+1])*diffeo.shape[k+1]
        grad = np.zeros([2,2,diffeo.shape[1], diffeo.shape[2]])
        grad[0,:,:,:] = imageGradient(np.squeeze(diffeo[0,:,:]), resol=resol)
        grad[1,:,:,:] = imageGradient(np.squeeze(diffeo[1,:,:]), resol=resol)
        res = np.fabs(grad[0,0,:,:] * grad[1,1,:,:] - grad[0,1,:,:] * grad[1,0,:,:])
    else:
        if periodic == True:
            w = np.mgrid[0:diffeo.shape[0]]
            dw = diffeo-w
            diffeo -= np.rint(dw/diffeo.shape[0])*diffeo.shape[0]
        res =  np.fabs(imageGradient(np.squeeze(diffeo)), resol=resol)
    return res

# Computes differential
@jit(nopython=True)
def jacobianMatrix(diffeo, resol=(1.,1.,1.), periodic=False):
    if diffeo.ndim > 4:
        print('No jacobian in dimension larger than 3')
        return

    if diffeo.ndim == 4:
        if periodic == True:
            w = np.mgrid[0:diffeo.shape[1], 0:diffeo.shape[2], 0:diffeo.shape[3]]
            dw = diffeo-w
            for k in range(3):
                diffeo[k,:,:,:] -= np.rint(dw[k,:,:,:]/diffeo.shape[k+1])*diffeo.shape[k+1]
        grad = np.zeros([3,3,diffeo.shape[1], diffeo.shape[2], diffeo.shape[3]])
        grad[0,:,:,:,:] = imageGradient(np.squeeze(diffeo[0,:,:,:]), resol=resol)
        grad[1,:,:,:,:] = imageGradient(np.squeeze(diffeo[1,:,:,:]), resol=resol)
        grad[2,:,:,:,:] = imageGradient(np.squeeze(diffeo[2,:,:,:]), resol=resol)
    elif diffeo.ndim == 3:
        if periodic == True:
            w = np.mgrid[0:diffeo.shape[1], 0:diffeo.shape[2]]
            dw = diffeo-w
            for k in range(2):
                diffeo[k,:,:] -= np.rint(dw[k,:,:]/diffeo.shape[k+1])*diffeo.shape[k+1]
        grad = np.zeros([2,2,diffeo.shape[1], diffeo.shape[2]])
        grad[0,:,:,:] = imageGradient(np.squeeze(diffeo[0,:,:]), resol=resol)
        grad[1,:,:,:] = imageGradient(np.squeeze(diffeo[1,:,:]), resol=resol)
    else:
        if periodic == True:
            w = np.mgrid[0:diffeo.shape[0]]
            dw = diffeo-w
            diffeo -= np.rint(dw/diffeo.shape[0])*diffeo.shape[0]
        grad =  np.fabs(imageGradient(np.squeeze(diffeo)), resol=resol)
    return grad


def differential(v, resol):
    res = np.zeros([v.shape[0]] + list(v.shape))
    for t in range(v.shape[0]):
        res[t,...] = imageGradient(v[t,...], resol)
    return res

def differentialDual(v, resol):
    res = np.zeros((v.shape[0],) + list(v.shape))
    deter = resol.prod()
    for t in range(v.shape[0]):
        res[t, ...] = imageGradient(v[t,...], resol)/(resol[t]*deter)

def inverseDifferential(v, resol):
    res = np.zeros((v.shape[0],) + list(v.shape))
    grad = differential(v, resol)

    if v.shape[0] == 1:
        res = 1 / (grad + 0.00000000001)
    elif v.shape[0] == 2:
        jac = grad[0, 0, ...] * grad[1, 1, ...] - grad[1, 0, ...] * grad[0, 1, ...] + 0.00000000001
        res[0, 0, ...] = grad[1, 1, ...] / jac
        res[0, 1, ...] = - grad[0, 1, ...] / jac
        res[1, 0, ...] = - grad[1, 0, ...] / jac
        res[1, 1, ...] = grad[0, 0, ...] / jac
    elif v.shape[0] == 3:
        jac = grad[0, 0, :, :, :] * grad[1, 1, :, :, :] * grad[2, 2, :, :, :]\
                - grad[0, 0, :, :, :] * grad[1, 2, :, :, :] * grad[2, 1, :, :, :]\
                - grad[0, 1, :, :, :] * grad[1, 0, :, :, :] * grad[2, 2, :, :, :]\
                - grad[0, 2, :, :, :] * grad[1, 1, :, :, :] * grad[2, 0, :, :, :]\
                + grad[0, 1, :, :, :] * grad[1, 2, :, :, :] * grad[2, 0, :, :, :]\
                + grad[0, 2, :, :, :] * grad[1, 0, :, :, :] * grad[2, 1, :, :, :] + 1e-10
        res[0, 0, ...] = (grad[1, 1, ...] * grad[2, 2, ...] - grad[1, 2, ...] * grad[2, 1, ...]) / jac
        res[1, 1, ...] = (grad[0, 0, ...] * grad[2, 2, ...] - grad[0, 2, ...] * grad[2, 0, ...]) / jac
        res[2, 2, ...] = (grad[1, 1, ...] * grad[0, 0, ...] - grad[1, 0, ...] * grad[0, 1, ...]) / jac
        res[0, 1, ...] = -(grad[0, 1, ...] * grad[2, 2, ...] - grad[2, 1, ...] * grad[0, 2, ...]) / jac
        res[1, 0, ...] = -(grad[1, 0, ...] * grad[2, 2, ...] - grad[1, 2, ...] * grad[2, 0, ...]) / jac
        res[0, 2, ...] = (grad[0, 1, ...] * grad[1, 2, ...] - grad[0, 2, ...] * grad[1, 1, ...]) / jac
        res[2, 0, ...] = (grad[1, 0, ...] * grad[2, 1, ...] - grad[2, 0, ...] * grad[1, 1, ...]) / jac
        res[1, 2, ...] = -(grad[0, 0, ...] * grad[1, 2, ...] - grad[0, 2, ...] * grad[1, 0, ...]) / jac
        res[2, 1, ...] = -(grad[0, 0, ...] * grad[2, 1, ...] - grad[2, 0, ...] * grad[0, 1, ...]) / jac
    else:
        logging.error("no inverse in dimension higher than 3")
        return

    return res


def inverseMap(phi, resol, psi0=None):
    flag = 1
    d = phi.shape[1:]
    if psi0 is None:
        id = idMesh(d)
    psi = psi0.copy()
    id = idMesh(d)
    foo = id - multilinInterpVectorField(phi, psi)
    error = np.sqrt( (foo**2).sum()) / d.prod()
    for k in range(10):
        dpsi = inverseDifferential(foo, resol)
        psiTry = psi + dpsi
        foo = id - multilinInterpVectorField(phi, psi)
        errorTry = np.sqrt( (foo**2).sum()) / d.prod()
        logging.info(f"inversion error {error: 0.4f} {errorTry: 0.4f}")
        if errorTry < error:
            psi = psiTry
            error = errorTry
        else:
            break
    return psi


def laplacian(X, neumann=0):
    Xplus = np.zeros(np.array(X.shape) + 2)
    if X.ndim == 1:
        n1 = X.shape[0]
        Xplus[1:n1+1] = X
        if neumann:
            Xplus[0] = Xplus[1]
            Xplus[n1+1] = Xplus[n1]
        Y = Xplus[0:n1] + Xplus[2:n1+2] - 2*X
    elif X.ndim == 2:
        n1 = X.shape[0]
        n2 = X.shape[1]
        Xplus[1:n1+1, 1:n2+1] = X
        if neumann:
            Xplus[0,:] = Xplus[1,:]
            Xplus[:,0] = Xplus[:,1]
            Xplus[n1+1,:] = Xplus[n1,:]
            Xplus[:,n2+1] = Xplus[:,n2]
        Y = (Xplus[1:n1+1, 0:n2] + Xplus[1:n1+1, 2:n2+2] + 
            Xplus[0:n1, 1:n2+1] + Xplus[2:n1+2, 1:n2+1]) - 4*X
    elif X.ndim == 3:
        n1 = X.shape[0]
        n2 = X.shape[1]
        n3 = X.shape[2]
        Xplus[1:n1+1, 1:n2+1, 1:n3+1] = X
        if neumann:
            Xplus[0,:,:] = Xplus[1,:,:]
            Xplus[:,0,:] = Xplus[:,1,:]
            Xplus[:,:,0] = Xplus[:,:,1]
            Xplus[n1+1,:, :] = Xplus[n1,:, :]
            Xplus[:,n2+1, :] = Xplus[:,n2, :]
            Xplus[:,:,n3+1] = Xplus[:,:,n3]
        Y = (Xplus[0:n1, 1:n2+1, 1:n3+1] + Xplus[2:n1+2, 1:n2+1, 1:n3+1] + 
            Xplus[1:n1+1, 0:n2, 1:n3+1] + Xplus[1:n1+1, 2:n2+2, 1:n3+1] +
            Xplus[1:n1+1, 1:n2+1, 0:n3] + Xplus[1:n1+1, 1:n2+1, 2:n3+2]) - 6*X
    else:
        Y = None
        logging.error('Laplacian in dim less than 3 only')
        
    return Y



class DiffeoParam:
    def __init__(self, dim, timeStep = .1, KparDiff = None, sigmaKernel = 6.5, order = -1,
                 kernelSize=50, typeKernel='gauss', resol=(1,1,1)):
        self.timeStep = timeStep
        self.sigmaKernel = sigmaKernel
        self.typeKernel = typeKernel
        self.kernelNormalization = 1.
        self.maskMargin = 1
        self.dim = dim
        self.resol = resol
        if KparDiff is None:
            self.KparDiff = Kernel(name = self.typeKernel, sigma = self.sigmaKernel, order=order, size=kernelSize,
                                   dim=dim)
        else:
            self.KparDiff = KparDiff

class Diffeomorphism:
    def __init__(self, shape, param):
        self.dim = param.dim
        self.imShape = shape
        self.vfShape = [self.dim] + list(shape)
        self.timeStep = param.timeStep
        self.sigmaKernel = param.sigmaKernel
        self.typeKernel = param.typeKernel
        self.kernelNormalization = 1.
        self.maskMargin = 1
        self.nbSemi = 0
        self.dim = param.dim
        self.resol = param.resol
        if param.KparDiff is None:
            self.KparDiff = Kernel(name = self.typeKernel, sigma = self.sigmaKernel, order=param.order,
                                   size=param.kernelSize,
                                   dim=param.dim)
        else:
            self.KparDiff = param.KparDiff
        # self.param = param
        self.phi = np.zeros(self.vfShape)
        self.psi = np.zeros(self.vfShape)
        self._phi = np.zeros(self.vfShape)
        self._psi = np.zeros(self.vfShape)
        self.KShape = self.KparDiff.K.shape
        self.fftShape = ()
        for i in range(self.dim):
            self.fftShape +=  (self.KShape[i] + self.imShape[i] - 1,)
        ax = np.arange(self.dim, dtype=int)
        self.mask = makeMask(param.maskMargin, self.imShape)
        self.fft_in = pyfftw.empty_aligned(self.fftShape,dtype='complex128')
        self.fft_out = pyfftw.empty_aligned(self.fftShape,dtype='complex128')
        self.ifft_out = pyfftw.empty_aligned(self.fftShape,dtype='complex128')
        self.fft_fwd = pyfftw.FFTW(self.fft_in, self.fft_out, threads = cpu_count(), axes=ax)
        self.fft_bwd = pyfftw.FFTW(self.fft_out, self.ifft_out, direction='FFTW_BACKWARD', threads = cpu_count(), axes=ax)
        self.periodic = False
        self.seqPadK = ()
        self.seqPadI = ()
        self.seqPadOne = ()

        for i in range(self.dim):
            if not self.periodic:
                delta1 = self.KShape[i] - 1
            else:
                delta1 = 0
            # if delta1 < self.imShape[i]:
            #     newShape[i] = imShape[i] + delta1
            # else:
            #     newShape[i] = 2 * delta1
            self.seqPadK += ((0, self.imShape[i] - 1),)
            self.seqPadI += ((self.KShape[i] // 2, self.KShape[i] // 2),)
            # seqPadI += ((0, K.shape[i] -1),)
            self.seqPadOne += ((self.KShape[i] - 1, 0),)
        newK = np.pad(self.KparDiff.K, self.seqPadK)
        self.fK = pyfftw.empty_aligned(self.fftShape, dtype='complex128')
        self.fft_in[...] = newK
        self.fK[...] = self.fft_fwd()


    def kernel(self, Lv):
        res0 = np.zeros(Lv.shape)
        for i in range(Lv.shape[0]):
            self.fft_in[...] = np.pad(Lv[i,...]*self.mask[i,...], self.seqPadI)
            newOnes = np.pad(np.ones(self.imShape, dtype=bool), self.seqPadOne)
            fI = pyfftw.empty_aligned((self.fftShape), dtype='complex128')
            fI[...] = self.fft_fwd()
            self.fft_out[...] = self.fK * fI
            newRes = pyfftw.empty_aligned((self.fftShape))
            newRes[...] = np.real(self.fft_bwd())
            res0[i,...] = self.mask[i,...]*newRes[newOnes].reshape(self.imShape)

        return res0

    def initFlow(self):
        self._phi = idMesh(self.imShape)
        self._psi = idMesh(self.imShape)

    def updateFlow(self, Lv, dt):
        id = idMesh(self.imShape)
        v = self.kernel(Lv)
        res = (v*Lv).sum()
        semi = dt*v
        for jj in range(self.nbSemi):
            foo = id - semi/2
            semi = dt * multilinInterpVectorField(v, foo)
        foo = id - semi
        self._psi = multilinInterpVectorField(self._psi, foo)
        foo = id + semi
        self._phi = multilinInterpVectorField(foo, self._phi)
        # foo = id + dt*semi
        # self._phi = multilinInterpVectorField(self._phi, foo)
        # foo = id - dt * semi
        # self._psi = multilinInterpVectorField(foo, self._psi)

        return res

    def adjoint(self, v, w):
        gradv = differential(v, self.resol)
        gradw = differential(w, self.resol)
        Dvw = (gradv * w[None, ...]).sum(axis=1)
        Dwv = (gradw * v[None, ...]).sum(axis=1)
        return Dvw-Dwv

    def adjoint2(self, v, w):
        id = idMesh(v.shape[1:])
        foo1 = id + v
        foo2 = id - v
        gradv = differential(foo1, self.resol)
        foo = (gradv * w[None, ...]).sum(axis=1)
        res = multilinInterp(foo, foo2)
        return res

    def adjointStar(self, v, m):
        gradv = differential(v, self.resol)
        res = (gradv * m[:, None, ...]).sum(axis=0)
        for i in range(v.shape[0]):
            for j in range(v.shape[0]):
                foo = v[j,...] * m[i,...]
                foo1 = (np.roll(foo, -1, axis=j) - np.roll(foo, 1, axis=j))/(2*self.resol[j])
                res[i,...] += foo1
        return res

    def adjointStar2(self, v, m):
        id = idMesh(v.shape[1:])
        foo1 = id + v
        foo2 = id - v
        foo = np.zeros(m.shape)
        for i in range(v.shape[0]):
            foo[i,...] = multilinInterpDual(m[i,...], foo1)
        gradv = differential(foo2, self.resol)
        res = (gradv * foo[:, None, ...]).sum(axis=0)
        return res

    def big_adjoint(self, v, phi):
        dphi = inverseDifferential(phi, self.resol)
        Z = multilinInterp(v, phi)
        res = np.zeros(v.shape)
        for i in range(v.shape[0]):
            for j in range(v.shape[0]):
                res[i,...] += dphi[i,j,...] * Z[j,...]
        return res

    def big_adjointStar(self, m, phi):
        dphi = differential(phi, self.resol)
        jac = jacobianDeterminant(phi, self.resol)
        Z = multilinInterp(m, phi)
        res = np.zeros(m.shape)
        for i in range(m.shape[0]):
            for j in range(m.shape[0]):
                res[i,...] += dphi[j,i,...] * Z[j]
        return res * jac[None, ...]

    def GeodesicDiffeoEvolution(self, Lv, delta=1., accuracy=1., Tmax = 40, verb=True, nb_semi=3):
            v = self.kernel(Lv)
            M = delta * np.sqrt((v**2).sum(axis=0)).max()
            T = int(np.ceil(accuracy * M + 1))
            Lvt = Lv.copy()
            vt = v.copy()
            norm = np.sqrt( (vt*Lvt).sum())
                # foo = adjointStar(vt, Lvt)
                # foo2 = kernel.ApplyToVectorField(foo, mask=mask)
                # T = np.ceil(accuracy * (foo*foo2).sum() / (norm * norm + 1) + 1); * /
            if T > Tmax:
                T = Tmax

            id = idMesh(Lv.shape[1:])
            dt = delta / T
            if verb:
                logging.info(f'Evolution; T = {T:} {norm:.4f}')

            # self._phi = idMesh(self.imShape)
            # self._psi = idMesh(self.imShape)
            for t in range(T):
                semi = dt*vt
                for jj in range(nb_semi):
                    semi = dt*multilinInterpVectorField(vt, id-semi/2)

                if t > 0:
                    self._phi = multilinInterpVectorField(id+semi, self._phi)
                    self._psi = multilinInterpVectorField(self._psi, id-semi)
                else:
                    self._phi = id+semi
                    self._psi = id - semi

                Lvt = self.adjointStar2(semi, Lvt)
                vt = self.kernel(Lvt)
                norm2 = np.sqrt( (vt*Lvt).sum())

                if norm2 > 1e-6:
                    vt *= norm / norm2
                    Lvt *= norm / norm2

            return Lvt

