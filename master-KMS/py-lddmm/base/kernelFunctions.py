from numba import jit, prange
import numpy as np
from math import pi, exp, sqrt
from . import kernelFunctions_util as ku
from scipy.spatial import distance as dfun

def kernelMatrixGauss(x, firstVar=None, grid=None, par=[1], diff = False, diff2 = False, constant_plane=False, precomp=None):
    sig = par[0]
    sig2 = 2*sig*sig
    if precomp is None:
        if firstVar is None:
            if grid is None:
                u = np.exp(-dfun.pdist(x,'sqeuclidean')/sig2)
                #        K = np.eye(x.shape[0]) + np.mat(dfun.squareform(u))
                K = dfun.squareform(u, checks=False)
                np.fill_diagonal(K, 1)
                precomp = np.copy(K)
                if diff:
                    K = -K/sig2
                elif diff2:
                    K = K/(sig2*sig2)
            else:
                dst = ((grid[..., np.newaxis, :] - x)**2).sum(axis=-1)
                K = np.exp(-dst/sig2)
                if diff:
                    K = -K/sig2
                elif diff2:
                    K = K/(sig2*sig2)
        else:
            K = np.exp(-dfun.cdist(firstVar, x, 'sqeuclidean')/sig2)
            precomp = np.copy(K)
            if diff:
                K = -K/sig2
            elif diff2:
                K = K/(sig2*sig2)
    else:
        K = np.copy(precomp)
        if diff:
            K = -K/sig2
        elif diff2:
            K = K/(sig2*sig2)

    if constant_plane:
        K2 = np.exp(-dfun.pdist(x[:,x.shape[1]-1],'sqeuclidean')/sig2)
        np.fill_diagonal(K2, 1)
        if diff:
            K2 = -K2/sig2
        elif diff2:
            K2 = K2/(sig2*sig2)
        return K, K2, precomp
    else:
        return K, precomp


# Polynomial factor for Laplacian kernel
@jit(nopython=True)
def lapPol(u, ord):
    if ord == 0:
        pol = 1.
    elif ord == 1:
        pol = 1. + u
    elif ord == 2:
        pol = (3. + 3*u + u*u)/3.
    elif ord == 3:
        pol = (15. + 15 * u + 6*u*u + u*u*u)/15.
    else:
        pol = (105. + 105*u + 45*u*u + 10*u*u*u + u*u*u*u)/105.
    return pol


# Polynomial factor for Laplacian kernel (first derivative)
@jit(nopython=True)
def lapPolDiff(u, ord):
    if ord == 1:
        pol = 1.
    elif ord == 2:
        pol = (1 + u)/3.
    elif ord == 3:
        pol = (3 + 3*u + u*u)/15.
    else:
        pol = (15 + 15 * u + 6*u*u + u*u*u)/105.
    return pol


# Polynomial factor for Laplacian kernel (second derivative)
@jit(nopython=True)
def lapPolDiff2(u, ord):
    pol = 0
    if ord == 2:
        pol = 1.0/3.
    elif ord == 3:
        pol = (1 + u)/15.
    else:
        pol = (3 + 3 * u + u*u)/105.
    return pol

# @jit(nopython=True)
# def lapPol_(u, ord):
#     if ord == 0:
#         pol = 1. + np.zeros(u.shape)
#     elif ord == 1:
#         pol = 1. + u
#     elif ord == 2:
#         pol = (3. + 3*u + u*u)/3.
#     elif ord == 3:
#         pol = (15. + 15 * u + 6*u*u + u*u*u)/15.
#     else:
#         pol = (105. + 105*u + 45*u*u + 10*u*u*u + u*u*u*u)/105.
#     return pol
#
#
# # Polynomial factor for Laplacian kernel (first derivative)
# @jit(nopython=True)
# def lapPolDiff_(u, ord):
#     if ord == 1:
#         pol = 1. + 0*u
#     elif ord == 2:
#         pol = (1 + u)/3.
#     elif ord == 3:
#         pol = (3 + 3*u + u*u)/15.
#     else:
#         pol = (15 + 15 * u + 6*u*u + u*u*u)/105.
#     return pol
#


def kernelMatrixLaplacian(x, firstVar=None, grid=None, par=(1., 3), diff=False, diff2 = False, constant_plane=False, precomp = None):
    sig = par[0]
    ord=par[1]
    if precomp is None:
        precomp = kernelMatrixLaplacianPrecompute(x, firstVar, grid, par)

    u = precomp[0]
    expu = precomp[1]

    if firstVar is None and grid is None:
        if diff==False and diff2==False:
            K = dfun.squareform(lapPol(u,ord) *expu)
            np.fill_diagonal(K, 1)
        elif diff2==False:
            K = dfun.squareform(-lapPolDiff(u, ord) * expu/(2*sig*sig))
            np.fill_diagonal(K, -1./((2*ord-1)*2*sig*sig))
        else:
            K = dfun.squareform(lapPolDiff2(u, ord) *expu /(4*sig**4))
            np.fill_diagonal(K, 1./((35)*4*sig**4))
    else:
        if diff==False and diff2==False:
            K = lapPol(u,ord) * expu
        elif diff2==False:
            K = -lapPolDiff(u, ord) * expu/(2*sig*sig)
        else:
            K = lapPolDiff2(u, ord) *expu/(4*sig**4)

    if constant_plane:
        uu = dfun.pdist(x[:,x.shape[1]-1])/sig
        K2 = dfun.squareform(lapPol(uu,ord)*np.exp(-uu))
        np.fill_diagonal(K2, 1)
        return K,K2,precomp
    else:
        return K,precomp

def kernelMatrixLaplacianPrecompute(x, firstVar=None, grid=None, par=(1., 3), diff=False, diff2 = False, constant_plane=False):
    sig = par[0]
    ord=par[1]
    if firstVar is None:
        if grid is None:
            u = dfun.pdist(x)/sig
        else:
            u = np.sqrt(((grid[..., np.newaxis, :] - x)**2).sum(axis=-1))/sig
    else:
        u = dfun.cdist(firstVar, x)/sig
    precomp = [u, np.exp(-u)]
    return precomp

# Wrapper for kernel matrix computation
def  kernelMatrix(Kpar, x, firstVar=None, grid=None, diff = False, diff2=False, constant_plane = False):
    # [K, K2] = kernelMatrix(Kpar, varargin)
    # creates a kernel matrix based on kernel parameters Kpar
    # if varargin = z

    #if (Kpar.prev_x is x) and (Kpar.prev_y is y):
    if Kpar._hold:
        #print 'Kernel: not computing'
        precomp = np.copy(Kpar.precomp)
    else:
        precomp = None


    if Kpar.name == 'gauss':
        res = kernelMatrixGauss(x,firstVar=firstVar, grid=grid, par = [Kpar.sigma], diff=diff, diff2=diff2, constant_plane = constant_plane, precomp=precomp)
    elif Kpar.name == 'laplacian':
        res = kernelMatrixLaplacian(x,firstVar=firstVar, grid=grid, par = [Kpar.sigma, Kpar.order], diff=diff, diff2=diff2, constant_plane = constant_plane, precomp=precomp)
    else:
        print('unknown Kernel type')
        return []

    #Kpar.prev_x = x
    #Kpar.prev_y = y
    Kpar.precomp = res[-1]
    if constant_plane:
        return res[0:2]
    else:
        return res[0]

# def applyKmin(x, a, firstVar=None):
#     if firstVar is None:
#         return applyKmin_(x,x,a)
#     else:
#         return applyKmin_(firstVar,x,a)
#
#
# #@jit(nopython=True, parallel=True)
# def applyKmin_(y, x, a):
#     res = np.zeros(y.shape)
#     for k in prange(y.shape[0]):
#         for l in range(x.shape[0]):
#             res[k, :] += (np.arctan(np.minimum(y[k,:], x[l,:])) + pi/2) * a[l,:]
#     return res
#
#
# def applyDiffKTmin(x, a1, a2, firstVar=None):
#     if firstVar is None:
#         return applyDiffKTmin_(x,x,a1, a2)
#     else:
#         return applyDiffKTmin_(firstVar,x,a1, a1)
# # Computes array A(i) = sum_k sum_(j) nabla_1[a1(k,i). K(x(i), x(j))a2(k,j)]
# #@jit(nopython=True, parallel=True)
# def applyDiffKTmin_(y, x, a1, a2):
#     res = np.zeros(y.shape)
#     for k in prange(y.shape[0]):
#         for l in range(x.shape[0]):
#             for i in range(x.shape[1]):
#                 if y[k,i] < x[l,i] - 1e-10:
#                     res[k, i] += a1[k,i] * a2[l,i] / (1 + y[k, i]**2)
#                 elif y[k,i] < x[l,i] + 1e-10:
#                     res[k, i] += a1[k,i] * a2[l,i]/(2*(1 + y[k, i]**2))
#     return res
#

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
    return np.maximum(u, 0)

@jit(nopython=True)
def ReLUKDiff(u):
    return heaviside(u)

@jit(nopython=True)
def heaviside(u):
    return (np.sign(u - 1e-8) + np.sign(u + 1e-8) + 2) / 4

@jit(nopython=True)
def applyK_(y, x, a, name, scale, order):
    res = np.zeros(a.shape)
    if name == 'min':
        for s in scale:
            u = np.minimum(y,x)/s
            #res += atanK(u)*a
            #res += logcoshK(u)*a
            res += ReLUK(u)*a
    elif 'gauss' in name:
        for s in scale:
            res += np.exp(- ((y-x)**2).sum()/(2*s**2)) * a
    elif 'lap' in name:
        for s in scale:
            u_ = 0.
            for j in range(y.size):
                u_ += (y[j] - x[j])**2
            u = sqrt(u_)/s
            res += lapPol(u, order) * np.exp(- u) *a
    return res /len(scale)

@jit(nopython=True)
def applyDiffKT_(y, x, a1a2, name, scale, order):
    res = np.zeros(y.shape)
    if name == 'min':
        for s in scale:
            u = np.minimum(y,x)/s
            #res += (heaviside(x-y)*a1a2)*atanKDiff(u)/s
            #res += (heaviside(x-y)*a1a2)*logcoshKDiff(u)/s
            res += (heaviside(x-y)*a1a2)*ReLUKDiff(u)/s
    elif 'gauss' in name:
        for s in scale:
            res += (y-x) * (-np.exp(- ((y-x)**2).sum()/(2*s**2)) * (a1a2).sum())/(s**2)
    elif 'lap' in name:
        for s in scale:
            u = np.sqrt(((y-x)**2).sum())/s
            res += (y-x) * (-lapPolDiff(u, order) * np.exp(- u) * (a1a2).sum()/(s**2))
    return res /len(scale)


@jit(nopython=True)
def applyDiv_(y, x, a, name, scale, order):
    res = 0
    if name == 'min':
        for s in scale:
            u = np.minimum(y,x)/s
            #res += (heaviside(x-y)*a*atanKDiff(u)/s).sum()
            #res += (heaviside(x-y)*a*logcoshKDiff(u)/s).sum()
            res += (heaviside(x-y)*a*ReLUKDiff(u)/s).sum()
    elif 'gauss' in name:
        for s in scale:
            res += ((y-x)*a).sum() * (-np.exp(- ((y-x)**2).sum()/(2*s**2)))/(s**2)
    elif 'lap' in name:
        for s in scale:
            u = np.sqrt(((y-x)**2).sum())/s
            res += ((y-x)*a).sum() * (-lapPolDiff(u, order) * np.exp(- u) /(s**2))
    return res /len(scale)




# Kernel specification
# name = 'gauss' or 'laplacian'
# affine = 'affine' or 'euclidean' (affine component)
# sigma: width
# order: order for Laplacian kernel
# w1: weight for linear part; w2: weight for translation part; center: origin
# dim: dimension
class KernelSpec:
    def __init__(self, name='gauss', affine = 'none', sigma = (1.,), order = 3, w1 = 1.0,
                 w2 = 1.0, dim = 3, center = None, weight = 1.0, localMaps=None):
        self.name = name
        if np.isscalar(sigma):
            self.sigma = np.array([sigma], dtype=float)
        else:
            self.sigma = np.array(sigma, dtype=float)
        self.order = order
        self.weight=weight
        self.w1 = w1
        self.w2 = w2
        self.constant_plane=False
        if center is None:
            self.center = np.zeros(dim)
        else:
            self.center = np.array(center)
        self.affine_basis = []
        self.dim = dim
        #self.prev_x = []
        #self.prev_y = []
        self.precomp = []
        self._hold = False
        self._state = False
        self.affine = affine
        self.localMaps = localMaps
        self.kff = False
        self.pk_dtype = 'float64'
        self.ms_exponent = -1
        if name == 'laplacian':
            self.kernelMatrix = kernelMatrixLaplacian
            if self.order > 4:
                self.order = 3
            self.par = [sigma, self.order]
            self.kff = True
        elif name == 'gauss' or name == 'gaussian':
            self.kernelMatrix = kernelMatrixGauss
            self.order = 10 
            self.par = [sigma]
            self.kff = True
        elif name == 'min':
            self.kernelMatrix = None
            self.name = 'min'
            self.par = []
        elif 'poly' in name :
            self.kernelMatrix = None
            self.name = 'poly'
            self.par = [sigma, self.order]
        elif name == 'euclidean':
            self.name = 'euclidean'
            self.par = [sigma]
        elif name == 'constant':
            self.name = 'constant'
            self.par = []
        else:
            self.name = 'none'
            self.kernelMatrix = None
            self.par = []
        if self.affine=='euclidean':
            if self.dim == 3:
                s2 = np.sqrt(2.0)
                self.affine_basis.append(np.mat([ [0,1,0], [-1,0,0], [0,0,0]])/s2)
                self.affine_basis.append(np.mat([ [0,0,1], [0,0,0], [-1,0,0]])/s2)
                self.affine_basis.append(np.mat([ [0,0,0], [0,0,1], [0,-1,0]])/s2)
            elif self.dim==2:
                s2 = np.sqrt(2.0)
                self.affine_basis.append(np.mat([ [0,1], [-1,0]])/s2)
            else:
                print('Euclidian kernels only available in dimensions 2 or 3')
                return


# Main class for kernel definition
class Kernel(KernelSpec):
    def precompute(self, x,  firstVar=None, grid=None, diff=False, diff2=False):
        if not (self.kernelMatrix is None):
            if self._hold:
                precomp = self.precomp
            else:
                precomp = None

            #precomp = None
            r = self.kernelMatrix(x, firstVar=firstVar, grid = grid, par = self.par, precomp=precomp, diff=diff, diff2=diff2)
            #self.prev_x = x
            #self.prev_y = y
            self.precomp = r[1]
            #print r[0].[1,:]
            return r[0] * self.weight

    def hold(self):
        self._state = self._hold
        self._hold = True
    def release(self):
        self._state = False
        self._hold = False
    def reset(self):
        self._hold=self._state


    def getK(self, x, firstVar = None):
        z = None
        if not (self.kernelMatrix is None):
            if firstVar is None:
                z = ku.kernelmatrix(x, x, self.name, self.sigma, self.order, KP=self.ms_exponent)
            else:
                z = ku.kernelmatrix(x, firstVar, self.name, self.sigma, self.order, KP=self.ms_exponent)
        return z

        
    # Computes K(x,x)a or K(x,y)a
    def applyK(self, x, a, firstVar = None, grid=None,matrixWeights=False, cpu=False):
        if firstVar is None:
            y = np.copy(x)
            if matrixWeights:
                z = ku.applykmat(y, x, a, self.name, self.sigma, self.order, KP=self.ms_exponent)
            elif self.localMaps:
                if self.localMaps[2] == 'naive':
                    z = ku.applylocalk_naive(y ,x ,a ,self.name, self.sigma ,self.order, KP=self.ms_exponent)
                else:
                    z = ku.applylocalk(y ,x ,a ,self.name, self.sigma ,self.order, self.localMaps[0] ,
                                        self.localMaps[1], KP=self.ms_exponent)
            else:
                z = ku.applyK(y ,x ,a ,self.name, self.sigma ,self.order, cpu=cpu, dtype=self.pk_dtype, KP=self.ms_exponent)
        else:
            if matrixWeights:
                z = ku.applykmat(firstVar, x, a, self.name, self.sigma, self.order, KP=self.ms_exponent)
            elif self.localMaps:
                if self.localMaps[2] == 'naive':
                    z = ku.applylocalk_naive(firstVar ,x ,a ,self.name, self.sigma ,self.order, KP=self.ms_exponent)
                else:
                    z = ku.applylocalk(firstVar ,x ,a ,self.name, self.sigma ,self.order , self.localMaps[0] ,
                                        self.localMaps[1], KP=self.ms_exponent)
            else:
                z = ku.applyK(firstVar ,x ,a , self.name, self.sigma ,self.order, cpu=cpu, dtype=self.pk_dtype, KP=self.ms_exponent)
        if self.affine == 'affine':
            xx = x-self.center
            if firstVar is None:
                if grid is None:
                    z += self.w1 * np.dot(xx, np.dot(xx.T, a)) + self.w2 * a.sum(axis=0)
                else:
                    yy = grid -self.center
                    z += self.w1 * np.dot(grid, np.dot(xx.T, a)) + self.w2 * a.sum(axis=0)
            else:
                yy = firstVar-self.center
                z += self.w1 * np.dot(yy, np.dot(xx.T, a)) + self.w2 * a.sum(axis=0)
        elif self.affine == 'euclidean':
            xx = x-self.center
            if not (firstVar is None):
                yy = firstVar-self.center
            if not (grid is None):
                gg = grid - self.center
            z += self.w2 * a.sum(axis=0)
            for E in self.affine_basis:
                xE = np.dot(xx, E.T)
                if firstVar is None:
                    if grid is None:
                        z += self.w1 * (xE * a).sum() * xE
                    else:
                        gE = np.dot(gg, E.T)
                        z += self.w1 * (xE * a).sum() * gE
                else:
                    yE = np.dot(yy, E.T)
                    z += self.w1 * np.multiply(xE, a).sum() * yE
                #print 'In kernel: ', self.w1 * np.multiply(yy, aa).sum()
        return z

    # Computes K(x,x)a or K(x,y)a
    def applyKTensor(self, x, ay, ax, betay, betax, firstVar = None, grid=None, cpu=False):
        if firstVar is None:
            y = np.copy(x)
            z = ku.applyktensor(y ,x , ay, ax, betay, betax ,self.name, self.sigma ,self.order, cpu=cpu,
                                dtype=self.pk_dtype, KP=self.ms_exponent)
        else:
            z = ku.applyktensor(firstVar ,x , ay, ax, betay, betax , self.name, self.sigma ,self.order, cpu=cpu,
                                dtype=self.pk_dtype, KP=self.ms_exponent)
        return z

    # # Computes A(i) = sum_j D_1[K(x(i), x(j))a2(j)]a1(i)
    def applyDiffK(self, x, a1, a2):
        z = ku.applykdiff1(x, a1, a2, self.name, self.sigma, self.order, KP=self.ms_exponent)
        return z

    # Computes A(i) = sum_j D_2[K(x(i), x(j))a2(j)]a1(j)
    def applyDiffK2(self, x, a1, a2):
        z = ku.applykdiff2(x, a1, a2, self.name, self.sigma, self.order, KP=self.ms_exponent)
        return z

    def applyDiffK1and2(self, x, a1, a2):
        z = ku.applykdiff1and2(x, a1, a2, self.name, self.sigma, self.order, KP=self.ms_exponent)
        return z

    # Computes A(i) = sum_j D_2[K(x(i), x(j))a2(j)]a1(j)
    def applyDiffKmat(self, x, beta, firstVar=None):
        if firstVar is None:
            z = ku.applykdiffmat(x, x, beta, self.name, self.sigma, self.order, KP=self.ms_exponent)
        else:
            z = ku.applykdiffmat(firstVar, x, beta, self.name, self.sigma, self.order, KP=self.ms_exponent)
        return z

    # Computes A(i) = sum_j D_2[K(x(i), x(j))a2(j)]a1(j)
    def applyDiffKTensor(self, x, ay, ax, betay, betax, firstVar=None):
        if firstVar is None:
            z = ku.applydiffktensor(x, x, ay, ax, betay, betax, self.name, self.sigma, self.order, KP=self.ms_exponent)
        else:
            z = ku.applydiffktensor(firstVar, x, ay, ax, betay, betax, self.name, self.sigma, self.order, KP=self.ms_exponent)
        return z

    # Computes array A(i) = sum_k sum_(j) nabla_1[a1(k,i). K(x(i), x(j))a2(k,j)]
    def applyDiffKT(self, x, p0, a, firstVar=None, regweight=1., lddmm=False, extra_term = None, cpu=False):
        if firstVar is None:
            y = np.copy(x)
        else:
            y = firstVar
        if lddmm and (extra_term is not None):
            p = p0 + extra_term
        else:
            p = p0
        if self.localMaps:
            if self.localMaps[2] == 'naive':
                zpx = ku.applylocalk_naivedifft(y ,x , p ,a , self.name, self.sigma ,self.order,
                                                regweight=regweight, lddmm=lddmm, KP=self.ms_exponent)
                #ku.applylocalk_naivedifft.parallel_diagnostics(level=4)
            else:
                zpx = ku.applylocalkdifft(y, x, p, a, self.name,  self.sigma, self.order, self.localMaps[0],
                                           self.localMaps[1], regweight=regweight, lddmm=lddmm, KP=self.ms_exponent)
        else:
            zpx = ku.applyDiffKT(y ,x , p ,a , self.name, self.sigma ,self.order,
                                 regweight=regweight, lddmm=lddmm, cpu=cpu, dtype=self.pk_dtype, KP=self.ms_exponent)
        if self.affine == 'affine':
            xx = x-self.center

            ### TO CHECK
            zpx += self.w1 * ((a*xx).sum() * p + (p*xx).sum() * a)
        elif self.affine == 'euclidean':
            xx = x-self.center
            for E in self.affine_basis:
                yy = np.dot(xx, E.T)
                for k in range(len(a)):
                     bb = np.dot(p[k], E)
                     zpx += self.w1 * np.multiply(yy, a[k]).sum() * bb
        return zpx


    def testDiffKT(self, y, x, p, a):
        k0 = (p*self.applyK(x,a, firstVar=y)).sum()
        dy = np.random.randn(y.shape[0], y.shape[1])
        eps = 1e-8
        k1 = (p*self.applyK(x,a, firstVar=y+eps*dy)).sum()
        grd = self.applyDiffKT(x,p,a,firstVar=y)
        print('test diffKT: {0:.5f}, {1:.5f}'.format((k1-k0)/eps, (grd*dy).sum()))


    def applyDivergence(self, x,a, firstVar=None):
        if firstVar is None:
            y = x
        else:
            y = firstVar
        if self.localMaps:
            if self.localMaps[2] == 'naive':
                zpx = ku.applylocalk_naivediv(y, x,a, self.name, self.sigma, self.order, KP=self.ms_exponent)
            else:
                zpx = ku.applylocalkdiv(y, x, a, self.name, self.sigma, self.order, self.localMaps[0],
                                        self.localMaps[1], KP=self.ms_exponent)
        else:
            zpx = ku.applyDiv(y,x,a, self.name, self.sigma, self.order, KP=self.ms_exponent)
        return zpx


    def applyDDiffK11and12(self, x, n, a, p):
        z = ku.applykdiff11and12(x, n, a, p, self.name, self.sigma, self.order, KP=self.ms_exponent)
        return z

