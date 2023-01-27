import numpy as np
import numpy.linalg as LA
import scipy.linalg as spLA

try:
    from vtk import vtkQuadricClustering
    gotVTK = True
except ImportError:
    print('could not import VTK functions')
    gotVTK = False

from . import surfaces
from .pointSets import epsilonNet
from . import conjugateGradient as cg, kernelFunctions as kfun


def generateDiffeonsFromSegmentation(fv, rate):
    nc = int(np.floor(fv.vertices.shape[0] * rate))
    (idx, c) = fv.laplacianSegmentation(nc)
    for k in range(c.shape[0]):
        dst = ((c[k, :] - fv.vertices)**2).sum(axis=1)
        I = np.argmin(dst)
        c[k, :] = fv.vertices[I]
    return generateDiffeons(fv, c, idx)

def generateDiffeonsFromNet(fv, rate):
    (L, AA) =  fv.laplacianMatrix()
    #eps = rate * AA.sum()
    (D, y) = spLA.eigh(L, AA, eigvals= (L.shape[0]-10, L.shape[0]-1))
    (net, idx) = epsilonNet(y, rate)
    c = fv.vertices[net, :]
    #print c
    return generateDiffeons(fv, c, idx)

def generateDiffeonsFromDecimation(fv, target):
    if gotVTK:
        n = fv.vertices.shape[0]
        nn = fv.faces.shape[0]
        fv2 = surfaces.Surface(surf=fv)
        #dc = vtkQuadricDecimation()
        #red = 1 - min(np.float(target)/polydata.GetNumberOfPoints(), 1)
        #dc.SetTargetReduction(red)
        a = (fv.surfel**2).sum(axis=1).sum()/nn
        dx = (float(nn)/target) * np.sqrt(a)
        #dc.SetDivisionSpacing(dx, dx, dx)
        n0 = nn
        while fv2.faces.shape[0] > target:
            polydata = fv2.toPolyData()
            dc = vtkQuadricClustering()
            dc.SetInput(polydata)
            dc.Update()
            g = dc.GetOutput()
            fv2.fromPolyData(g)
            if fv2.faces.shape[0] == n0:
                break
            else:
                n0 = fv2.faces.shape[0]
        #fv2.Simplify(target)
        m = fv2.faces.shape[0]
        c = np.zeros([m, 3])
        for k, f in enumerate(fv2.faces):
            u = (fv2.vertices[f, :]).sum(axis=0)/3
            dst = ((u - fv.vertices)**2).sum(axis=1)
            I = np.argmin(dst)
            c[k, :] = fv.vertices[I]

        dist2 = ((fv.vertices.reshape([n, 1, 3]) -
                c.reshape([1,m,3]))**2).sum(axis=2)
        idx = - np.ones(n, dtype=np.int)
        for p in range(n):
            closest = np.unravel_index(np.argmin(dist2[p, :].ravel()), [m, 1])
            idx[p] = closest[0]
        return generateDiffeons(fv, c, idx)
    else:
        raise Exception('Cannot run generateDiffeonsFromDecimation without VTK')

        


def generateDiffeons(fv, c, idx):
    a, foo = fv.computeVertexArea()
    #print idx
    nc = idx.max()
    print('Computed', nc+1, 'diffeons')
    S = np.zeros([nc+1, 3, 3])
    #C = np.zeros([nc, 3])
    for k in range(nc+1):
        I = np.flatnonzero(idx==k)
        nI = len(I)
        aI = a[I]
        ak = aI.sum()
        y = (fv.vertices[I, :] - c[k, :])
        SS = (y.reshape([nI, 3, 1]) * aI.reshape([nI, 1, 1]) * y.reshape([nI, 1, 3])).sum(axis=0)/ak
        [D,v] = LA.eig(SS)
        #print np.dot(v.T, np.dot(SS, v))
        I = np.argsort(D, axis=None)
        #print D[I]
        S[k,:,:] = D[I[1]] * (v[:,I[1]].reshape([3,1]) * v[:,I[1]].reshape([1,3])) + D[I[2]] * (v[:,I[2]].reshape([3,1])* v[:,I[2]].reshape([1,3]))
        S[k, :, :] = SS 
        #S[k,:,:] = D[I[1]] * (v[I[1], :], v[I[1], :].T) + D[I[2]] * np.dot(v[I[2], :], v[I[2], :].T)
        #[DD,v] = LA.eig(S[k, :, :])
        #print D, DD
        #S[k, :, :] = S[k, :, :] * np.sqrt(ak/(1e-10+2*np.pi * (D[1]*D[2])))
        #print np.pi * (D[1]*D[2]), ak
        #multiMatEig(S)
    return c, S, idx

        

# Saves in .vtk format
def saveDiffeons(fileName, c, S):
    with open(fileName, 'w') as fvtkout:
        fvtkout.write('# vtk DataFile Version 3.0\nSurface Data\nASCII\nDATASET UNSTRUCTURED_GRID\n') 
        fvtkout.write('\nPOINTS {0: d} float'.format(c.shape[0]))
        for ll in range(c.shape[0]):
            fvtkout.write('\n{0: f} {1: f} {2: f}'.format(c[ll,0], c[ll,1], c[ll,2]))
        fvtkout.write(('\nPOINT_DATA {0: d}').format(c.shape[0]))
        d,v = multiMatEig(S)
        fvtkout.write('\nSCALARS first_eig float 1\nLOOKUP_TABLE default')
        for ll in range(d.shape[0]):
            fvtkout.write('\n {0: .5f}'.format(d[ll,2]))
        fvtkout.write('\nSCALARS second_eig float 1\nLOOKUP_TABLE default')
        for ll in range(d.shape[0]):
            fvtkout.write('\n {0: .5f}'.format(d[ll,1]))
        fvtkout.write('\nSCALARS third_eig float 1\nLOOKUP_TABLE default')
        for ll in range(d.shape[0]):
            fvtkout.write('\n {0: .5f}'.format(d[ll,0]))

        fvtkout.write('\nVECTORS third_dir float')
        v[:, :, 0] *= np.sqrt(d[:,2]*d[:,1]).reshape([d.shape[0], 1])
        for ll in range(d.shape[0]):
            fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(v[ll, 0, 0], v[ll, 1, 0], v[ll, 2, 0]))

        fvtkout.write('\nVECTORS first_dir float')
        #print v.shape, d.shape
        v[:, :, 2] = v[:,:,2] * np.sqrt(d[:,2]).reshape([d.shape[0], 1])
        for ll in range(d.shape[0]):
            fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(v[ll, 0, 2], v[ll, 1, 2], v[ll, 2, 2]))

        fvtkout.write('\nVECTORS second_dir float')
        v[:, :, 1] *= np.sqrt(d[:,1]).reshape([d.shape[0], 1])
        for ll in range(d.shape[0]):
            fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(v[ll, 0, 1], v[ll, 1, 1], v[ll, 2, 1]))


        fvtkout.write('\nTENSORS tensors float')
        for ll in range(S.shape[0]):
            for kk in range(3):
                fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(S[ll, kk, 0], S[ll, kk, 1], S[ll, kk, 2]))
        fvtkout.write('\n')




def multiMatDet1(S, isSym=False):
    N = S.shape[0]
    dim = S.shape[1]
    if (dim==1):
        detR = S
    elif (dim == 2):
        detR = np.multiply(S[:,0, 0], S[:, 1, 1]) - np.multiply(S[:,0, 1], S[:, 1, 0])
    elif (dim==3):
        detR = (S[:, 0, 0] * S[:, 1, 1] * S[:, 2, 2] 
                -S[:, 0, 0] * S[:, 1, 2] * S[:, 2, 1]
                -S[:, 0, 1] * S[:, 1, 0] * S[:, 2, 2]
                -S[:, 0, 2] * S[:, 1, 1] * S[:, 2, 0]
                +S[:, 0, 1] * S[:, 1, 2] * S[:, 2, 0]
                +S[:, 0, 2] * S[:, 1, 0] * S[:, 2, 1])
    return detR

def multiMatEig(S, isSym=False):
    N = S.shape[0]
    dim = S.shape[1]
    d = np.zeros([N,dim])
    v = np.zeros([N,dim,dim])
    for k in range(N):
        D, V = LA.eig(S[k,:,:])
        idx = D.argsort()
        #print D[idx]
        d[k,:] = D[idx]
        v[k,:, :] = V[:,idx]
    return d,v
        
def multiMatInverse1(S, isSym=False):
    N = S.shape[0]
    dim = S.shape[1]
    if (dim==1):
        R = np.divide(1, S)
        detR = S
    elif (dim == 2):
        detR = np.multiply(S[:,0, 0], S[:, 1, 1]) - np.multiply(S[:,0, 1], S[:, 1, 0])
        R = np.zeros(S.shape)
        R[:, 0, 0] = S[:, 1, 1].copy()
        R[:, 1, 1] = S[:, 0, 0].copy()
        R[:, 0, 1] = -S[:, 0, 1]
        R[:, 1, 0] = -S[:, 1, 0]
        R = R / detR.reshape([N, 1, 1])
    elif (dim==3):
        detR = (S[:, 0, 0] * S[:, 1, 1] * S[:, 2, 2] 
                -S[:, 0, 0] * S[:, 1, 2] * S[:, 2, 1]
                -S[:, 0, 1] * S[:, 1, 0] * S[:, 2, 2]
                -S[:, 0, 2] * S[:, 1, 1] * S[:, 2, 0]
                +S[:, 0, 1] * S[:, 1, 2] * S[:, 2, 0]
                +S[:, 0, 2] * S[:, 1, 0] * S[:, 2, 1])
            #detR = np.divide(1, detR)
        R = np.zeros(S.shape)
        R[:, 0, 0] = S[:, 1, 1] * S[:, 2, 2] - S[:, 1, 2] * S[:, 2, 1]
        R[:, 1, 1] = S[:, 0, 0] * S[:, 2, 2] - S[:, 0, 2] * S[:, 2, 0]
        R[:, 2, 2] = S[:, 1, 1] * S[:, 0, 0] - S[:, 1, 0] * S[:, 0, 1]
        R[:, 0, 1] = -S[:, 0, 1] * S[:, 2, 2] + S[:, 2, 1] * S[:, 0, 2]
        R[:, 0, 2] = S[:, 0, 1] * S[:, 1, 2] - S[:, 0, 2] * S[:, 1, 1]
        R[:, 1, 2] = -S[:, 0, 0] * S[:, 1, 2] + S[:, 0, 2] * S[:, 1, 0]
        if isSym:
            R[:, 1, 0] = R[:, 0, 1].copy()
            R[:, 2, 0] = R[:, 0, 2].copy()
            R[:, 2, 1] = R[:, 1, 2].copy()
        else:
            R[:, 1, 0] = -S[:, 1, 0] * S[:, 2, 2] + S[:, 1, 2] * S[:, 2, 0]
            R[:, 2, 0] = S[:, 1, 0] * S[:, 2, 1] - S[:, 2, 0] * S[:, 1, 1]
            R[:, 2, 1] = -S[:, 0, 0] * S[:, 2, 1] + S[:, 2, 0] * S[:, 0, 1]
        R = R / detR.reshape([N, 1, 1])
    return R, detR
        
def multiMatInverse2(S, isSym=False):
    N = S.shape[0]
    M = S.shape[1]
    dim = S.shape[2] ;
    if (dim==1):
        R = np.divide(1, S)
        detR = S
    elif (dim == 2):
        R = np.zeros([N, M, dim, dim])
        detR = np.multiply(S[:, :,0, 0], S[:, :, 1, 1]) - np.multiply(S[:, :,0, 1], S[:, :, 1, 0])
        R[:, :, 0, 0] = S[:, :, 1, 1].copy()
        R[:, :, 1, 1] = S[:, :, 0, 0].copy()
        R[:, :, 0, 1] = -S[:, :, 0, 1]
        R[:, :, 1, 0] = -S[:, :, 1, 0]
        R = R / detR.reshape([N, M, 1, 1])
    elif (dim==3):
        R = np.zeros([N, M, dim, dim])
        detR = (S[:, :, 0, 0] * S[:, :, 1, 1] * S[:, :, 2, 2] 
                -S[:, :, 0, 0] * S[:, :, 1, 2] * S[:, :, 2, 1]
                -S[:, :, 0, 1] * S[:, :, 1, 0] * S[:, :, 2, 2]
                -S[:, :, 0, 2] * S[:, :, 1, 1] * S[:, :, 2, 0]
                +S[:, :, 0, 1] * S[:, :, 1, 2] * S[:, :, 2, 0]
                +S[:, :, 0, 2] * S[:, :, 1, 0] * S[:, :, 2, 1])
            #detR = np.divide(1, detR)
        R[:, :, 0, 0] = S[:, :, 1, 1] * S[:, :, 2, 2] - S[:, :, 1, 2] * S[:, :, 2, 1]
        R[:, :, 1, 1] = S[:, :, 0, 0] * S[:, :, 2, 2] - S[:, :, 0, 2] * S[:, :, 2, 0]
        R[:, :, 2, 2] = S[:, :, 1, 1] * S[:, :, 0, 0] - S[:, :, 1, 0] * S[:, :, 0, 1]
        R[:, :, 0, 1] = -S[:, :, 0, 1] * S[:, :, 2, 2] + S[:, :, 2, 1] * S[:, :, 0, 2]
        R[:, :, 0, 2] = S[:, :, 0, 1] * S[:, :, 1, 2] - S[:, :, 0, 2] * S[:, :, 1, 1]
        R[:, :, 1, 2] = -S[:, :, 0, 0] * S[:, :, 1, 2] + S[:, :, 0, 2] * S[:, :, 1, 0]
        if isSym:
            R[:, :, 1, 0] = R[:, :, 0, 1].copy()
            R[:, :, 2, 0] = R[:, :, 0, 2].copy()
            R[:, :, 2, 1] = R[:, :, 1, 2].copy()
        else:
            R[:, :, 1, 0] = -S[:, :, 1, 0] * S[:, :, 2, 2] + S[:, :, 1, 2] * S[:, :, 2, 0]
            R[:, :, 2, 0] = S[:, :, 1, 0] * S[:, :, 2, 1] - S[:, :, 2, 0] * S[:, :, 1, 1]
            R[:, :, 2, 1] = -S[:, :, 0, 0] * S[:, :, 2, 1] + S[:, :, 2, 0] * S[:, :, 0, 1]
        R = R / detR.reshape([N, M, 1, 1])
    return R, detR
        
def positiveProj(S):
    N = S.shape[0]
    dim = S.shape[1] ;
    if dim==1:
        S2 = np.maximum(S, 0)
    else:
        d, v = multiMatEig(S)
        d = np.maximum(d, 1e-10)
        S2 = np.zeros(S.shape)
        for k in range(dim):
            S2 += d[:,k].reshape([N,1,1]) * v[:,:,k].reshape([N, dim, 1]) * v[:,:,k].reshape([N, 1, dim])
    return S2


def computeProducts(c, S, sig):
    M = c.shape[0]
    dim = c.shape[1]
    sig2 = sig*sig ;

    sigEye = sig2*np.eye(dim)
    #SS = sigEye.reshape([1,dim,dim]) + S 
    SS = sigEye + S 
    detR = multiMatDet1(SS, isSym=True) 
    #SS = sigEye.reshape([1,1,dim,dim]) + S.reshape([M, 1, dim, dim]) + S.reshape([1, M, dim, dim])
    SS = sigEye + S[:, np.newaxis,...] + S
    (R2, detR2) = multiMatInverse2(SS, isSym=True)
    
    diffc = c[:, np.newaxis,:] - c
    betacc = (R2 * diffc[...,np.newaxis,:]).sum(axis=3)
    dst = (betacc * diffc).sum(axis=2)
    gcc = np.sqrt((detR[:,np.newaxis]*detR)/((sig2**dim)*detR2))*np.exp(-dst/2)
    
    return gcc


def computeProductsCurrents(c, S, sig):
    M = c.shape[0]
    dim = c.shape[1]
    sig2 = sig*sig ;

    sigEye = sig2*np.eye(dim)
    SS = sigEye + S[:, np.newaxis, ...] + S
    (R2, detR2) = multiMatInverse2(SS, isSym=True)
    
    diffc = c[:, np.newaxis,:] - c
    betacc = (R2 * diffc[..., np.newaxis, :]).sum(axis=3)
    dst = (betacc * diffc).sum(axis=2)
    gcc = (sig**dim)*np.exp(-dst/2) / (np.sqrt(detR2))
    
    return gcc


def computeProductsAsym(c0, S0, c1, S1, sig):
    M0 = c0.shape[0]
    M1 = c1.shape[0]
    dim = c0.shape[1]
    sig2 = sig*sig ;

    sigEye = sig2*np.eye(dim)
    SS = sigEye + S0 
    detR0 = multiMatDet1(SS, isSym=True) 
    SS = sigEye + S1 
    detR1 = multiMatDet1(SS, isSym=True) 
    SS = sigEye + S0[:, np.newaxis, ...] + S1
    (R2, detR2) = multiMatInverse2(SS, isSym=True)
    
    diffc = c0[:, np.newaxis, :] - c1
    betacc = (R2 * diffc[..., np.newaxis, :]).sum(axis=3)
    dst = (betacc * diffc).sum(axis=2)
    gcc = np.sqrt((detR0[:, np.sqrt]*detR1)/((sig2**dim)*detR2))*np.exp(-dst/2)
    
    return gcc


def computeProductsAsymCurrents(c, S, cc, sig):
    M = c.shape[0]
    K = cc.shape[0]
    dim = c.shape[1]
    sig2 = sig*sig ;

    sigEye = sig2*np.eye(dim)
    SS = sigEye + S 
    (R, detR) = multiMatInverse1(SS, isSym=True) 
    
    diffc = c[:, np.newaxis, :] - cc
    betacc = (R[:, np.newaxis, ...] * diffc[..., np.newaxis, :]).sum(axis=3)
    dst = (betacc * diffc).sum(axis=2)
    gcc = (sig**dim)*np.exp(-dst/2)/(np.sqrt(detR).reshape(M,1))
    
    return gcc


def gaussianDiffeonsGradientMatricesPset(c, S, x, a, pc, pS, px, sig, timeStep, withJacobian=False):
    M = c.shape[0]
    dim = c.shape[1]
    sig2 = sig*sig ;
    if not (type(withJacobian)==bool):
        J = withJacobian[0]
        pJ = withJacobian[1]
        withJacobian = True
        

    sigEye = sig2*np.eye(dim)
    SS = sigEye + S 
    (R, detR) = multiMatInverse1(SS, isSym=True) 
    SS = sigEye + S[:, np.newaxis, ...] + S
    (R2, detR2) = multiMatInverse2(SS, isSym=True)

    diffx = x[..., np.newaxis, :] - c.reshape([M, dim])
    betax = (R*diffx[..., np.newaxis, :]).sum(axis=-1)
    dst = (betax * diffx).sum(axis=-1)
    fx = np.exp(-dst/2)

    diffc = c[:, np.newaxis, :] - c
    betac = (R*diffc[..., np.newaxis, :]).sum(axis=3)
    dst = (diffc * betac).sum(axis=2)
    fc = np.exp(-dst/2)
    betacc = (R2 * diffc[..., np.newaxis, :]).sum(axis=3)
    dst = (betacc * diffc).sum(axis=2)
    gcc = np.sqrt((detR[:, np.newaxis]*detR)/((sig2**dim)*detR2))*np.exp(-dst/2)

    Dv = -((fc[..., np.newaxis]*betac)[..., np.newaxis, :]*a[..., np.newaxis]).sum(axis=1)
    IDv = np.eye(dim) + timeStep * Dv ;
    pSS = (pS[...,np.newaxis] * (IDv[..., np.newaxis] * S[:, np.newaxis, ...]).sum(axis=2)[:, np.newaxis, ...]).sum(axis=2)
    
    fS = (pSS[:, np.newaxis, ...]*betac[...,np.newaxis, :]).sum(axis=3)
    #fS = (pSS.reshape([M, 1, dim, dim])*betac.reshape([M,M,dim, 1])).sum(axis=2)
    #fS = (pS.reshape([M, 1, dim,dim])* fS.reshape([M,M,1,dim])).sum(axis=3)
    grx = (fx[...,np.newaxis] * px[...,np.newaxis,:]).sum(axis=tuple(range(x.ndim-1)))
    if withJacobian:
        grJ = - ((pJ[...,np.newaxis]*fx)[...,np.newaxis] *betax).sum(axis=tuple(range(x.ndim-1)))
    grc = np.dot(fc.T, pc)
    grS = -2 * (fc[:, :, np.newaxis] * fS).sum(axis=0)
    if withJacobian:
        return grc, grS, grx, grJ, gcc
    else:        
        return grc, grS, grx, gcc

def gaussianDiffeonsGradientMatricesNormals(c, S, b, x, xS, a, pc, pS, pb, px, pxS, sig, timeStep):
    N = b.shape[0]
    M = c.shape[0]
    dim = c.shape[1]
    sig2 = sig*sig ;

    sigEye = sig2*np.eye(dim)
    SS = sigEye.reshape([1,dim,dim]) + S 
    (R, detR) = multiMatInverse1(SS, isSym=True) 
    SS = sigEye.reshape([1,1,dim,dim]) + S.reshape([M, 1, dim, dim]) + S.reshape([1, M, dim, dim])
    (R2, detR2) = multiMatInverse2(SS, isSym=True)

    diffx = x.reshape([N, 1, dim]) - c.reshape([1, M, dim])
    betax = (R.reshape([1, M, dim, dim])*diffx.reshape([N, M, 1, dim])).sum(axis=3)
    dst = (diffx * betax).sum(axis=2)
    fx = np.exp(-dst/2)

    diffc = c.reshape([M, 1, dim]) - c.reshape([1, M, dim])
    betac = (R.reshape([1, M, dim, dim])*diffc.reshape([M, M, 1, dim])).sum(axis=3)
    dst = (diffc * betac).sum(axis=2)
    fc = np.exp(-dst/2)
    betacc = (R2 * diffc.reshape([M, M, 1, dim])).sum(axis=3)
    dst = (betacc * diffc).sum(axis=2)
    gcc = np.sqrt((detR.reshape([M,1])*detR.reshape([1,M]))/((sig2**dim)*detR2))*np.exp(-dst/2)
    
    xDv = -((fx.reshape([N,M,1])*betax).reshape([N, M, 1, dim])*a.reshape([1, M, dim, 1])).sum(axis=1)
    xIDv = np.eye(dim).reshape([1,dim,dim]) + timeStep * xDv ;
    pxSxS = (pxS.reshape([N,dim,dim,1]) * (xIDv.reshape([N,dim,dim, 1]) * xS.reshape([N, 1, dim ,dim])).sum(axis=2).reshape([N,1,dim,dim])).sum(axis=2)

    Dv = -((fc.reshape([M,M,1])*betac).reshape([M, M, 1, dim])*a.reshape([1, M, dim, 1])).sum(axis=1)
    IDv = np.eye(dim).reshape([1,dim,dim]) + timeStep * Dv ;
    pSS = (pS.reshape([M,dim,dim,1]) * (IDv.reshape([M,dim,dim, 1]) * S.reshape([M, 1, dim ,dim])).sum(axis=2).reshape([M,1,dim,dim])).sum(axis=2)
    
    fxS = (pxSxS.reshape([N, 1, dim, dim])*betax.reshape([N,M,1,dim])).sum(axis=3)
    fS = (pSS.reshape([M, 1, dim, dim])*betac.reshape([M,M,1,dim])).sum(axis=3)
    #fS = (pSS.reshape([M, 1, dim, dim])*betac.reshape([M,M,dim, 1])).sum(axis=2)
    #fS = (pS.reshape([M, 1, dim,dim])* fS.reshape([M,M,1,dim])).sum(axis=3)
    fb = fx*(pb.reshape([N,1, dim])*betax.reshape([N, M, dim])).sum(axis=2) 
    grb = np.dot(fb.T, b)
    grb -= ((fx * (pb*b).sum(axis=1).reshape([N,1])).reshape([N,M,1]) * betax).sum(axis=0)
    grc = np.dot(fc.T, pc)
    grx = np.dot(fx.T, px)
    grS = -2 * (fc.reshape([M,M,1]) * fS).sum(axis=0)
    grxS = -2 * (fx.reshape([N,M,1]) * fxS).sum(axis=0)
    return grc, grS, grb, grx, grxS, gcc

def approximateSurfaceCurrent(c, S, fv, sig):
    cc = fv.centers
    nu = fv.surfel
    g1 = computeProductsCurrents(c,S,sig)
    g2 = computeProductsAsymCurrents(c, S, cc, sig)
    b = LA.solve(g1, np.dot(g2, nu))
    n0 = surfaces.currentNorm0(fv, kfun.Kernel(name='gauss', sigma=sig))
    n1 = diffeonCurrentNormDef(c,S,b,fv,sig)
    print('Norm before approx:', n0)
    print('Diff after approx:', n0 + n1)
    print('Norm of Projection:', (b*np.dot(g1, b)).sum(), -n1)
    return b

def diffeonCurrentNormDef(c, S, b, fv, sig):
    # print 'c', c
    # print 'S', S
    # print 'b', b
    g1 = computeProductsCurrents(c,S,sig)
    g2 = computeProductsAsymCurrents(c, S, fv.centers, sig)
    obj = np.multiply(b, np.dot(g1, b) - 2*np.dot(g2, fv.surfel)).sum()
    return obj

def diffeonCurrentNorm0(fv, K):
    #print 'sigma=', sig
    #K = kfun.Kernel(name='gauss', sigma=sig)
    obj = surfaces.currentNorm0(fv, K)
    return obj


def testDiffeonCurrentNormGradient(c, S, b, fv, sig):
    obj0 = diffeonCurrentNormDef(c,S,b,fv, sig)
    (gc, gS, gb) = diffeonCurrentNormGradient(c,S,b,fv,sig)
    eps = 1e-7
    dc = np.random.randn(c.shape[0], c.shape[1])
    obj = diffeonCurrentNormDef(c+eps*dc,S,b,fv, sig)
    print('c Variation:', (obj-obj0)/eps, (gc*dc).sum())
    dS = np.random.randn(S.shape[0], S.shape[1], S.shape[2])
    dS += dS.transpose((0,2,1))
    obj = diffeonCurrentNormDef(c,S+eps*dS,b,fv, sig)
    print('S Variation:', (obj-obj0)/eps, (gS*dS).sum())
    db = np.random.randn(b.shape[0], b.shape[1])
    obj = diffeonCurrentNormDef(c,S,b+eps*db,fv, sig)
    print('b Variation:', (obj-obj0)/eps, (gb*db).sum())
    

def diffeonCurrentNormGradient(c, S, b, fv, sig):
    M = b.shape[0]
    dim = b.shape[1]
    cc = fv.centers
    nu = fv.surfel
    K = cc.shape[0]
    sig2 = sig**2

    sigEye = sig2*np.eye(dim)
    SS = sigEye.reshape([1,dim,dim]) + S 
    (R, detR) = multiMatInverse1(SS, isSym=True) 
    SS = sigEye.reshape([1,1,dim,dim]) + S.reshape([M, 1, dim, dim]) + S.reshape([1, M, dim, dim])
    (R2, detR2) = multiMatInverse2(SS, isSym=True)

    diffc = c.reshape([M, 1, dim]) - cc.reshape([1, K, dim])
    betacn = (R.reshape(M, 1, dim, dim) * diffc.reshape([M, K, 1, dim])).sum(axis=3)
    dst = (betacn * diffc).sum(axis=2)
    g2 = (sig**dim)*np.exp(-dst/2)/np.sqrt(detR).reshape(M,1)

    diffc = c.reshape([M, 1, dim]) - c.reshape([1, M, dim])
    betacc = (R2 * diffc.reshape([M, M, 1, dim])).sum(axis=3)
    dst = (betacc * diffc).sum(axis=2)
    g1 = (sig**dim)*np.exp(-dst/2) / np.sqrt(detR2)

    pb = 2*(np.dot(g1, b) - np.dot(g2, nu))
    bb = (b.reshape(M, 1, dim) * b.reshape(1,M,dim)).sum(axis=2)
    bnu = (b.reshape(M, 1, dim) * nu.reshape(1,K,dim)).sum(axis=2)
    g1bb = g1*bb
    g2bnu = g2*bnu

    pc = (-2 *( (g1bb).reshape(M, M, 1) * betacc).sum(axis=1) +
          2*( (g2bnu).reshape(M, K, 1) * betacn).sum(axis=1))

    pS = ((g1bb.reshape(M,M,1,1) *(betacc.reshape(M,M,dim,1)*betacc.reshape(M,M,1,dim) - R2)).sum(axis=1)
          - (g2bnu.reshape(M,K,1,1) *(betacn.reshape(M,K,dim,1)*betacn.reshape(M,K,1,dim)
                                      - R.reshape(M,1,dim, dim))).sum(axis=1))

    return pc,pS,pb

	
 

class Direction:
    def __init__(self):
        self.c = []
        self.S = []
        self.b = []
        self.aff = []


class gdOptimizer:
    def __init__(self, surf=None, Diffeons=None,
                 DiffeonEpsForNet=None, sigmaDist = 2.5,
                 maxIter=1000, testGradient=False):
        if surf==None:
            print('Please provide a surface')
            return
        else:
            self.fv0 = surfaces.Surface(surf=surf)

        self.saveRate = 10
        self.iter = 0
        self.gradEps = -1
        self.npt = self.fv0.vertices.shape[0]
        self.dim = self.fv0.vertices.shape[1]
        self.maxIter = maxIter
        self.testGradient = testGradient
        self.sigmaDist = sigmaDist
        self.KparDist = kfun.Kernel(name = 'gauss', sigma =
				    self.sigmaDist)

        self.x0 = self.fv0.vertices
        if Diffeons==None:
            if DiffeonEpsForNet==None:
                self.c0 = np.copy(self.x0) ;
                self.S0 = np.zeros([self.x0.shape[0], self.x0.shape[1], self.x0.shape[1]])
                self.idx = None
            else:
                (self.c0, self.S0, self.idx) = generateDiffeonsFromNet(self.fv0, DiffeonEpsForNet)
            self.b0 = approximateSurfaceCurrent(self.c0, self.S0, self.fv0, self.sigmaDist)
        else:
            (self.c0, self.S0, self.b0) = Diffeons

        self.ndf = self.c0.shape[0]
        self.dim = self.c0.shape[1]
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.ndf
        self.sw = 1e-5



    def dataTerm(self, c, S, b):
        obj = diffeonCurrentNormDef(c,S,b, self.fv0, self.sigmaDist)
        return obj
    
    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = diffeonCurrentNorm0(self.fv0, self.KparDist)
        self.obj = self.obj0 + self.dataTerm(self.c0, self.S0, self.b0)

        return self.obj

    def getVariable(self):
        return [self.c0, self.S0, self.b0]

    def updateTry(self, dir, eps, objRef=None):
        cTry = self.c0 - eps * dir.c
        STry = positiveProj(self.S0 - eps * dir.S)
        bTry = self.b0 - eps * dir.b
        objTry = self.obj0 + self.dataTerm(cTry, STry, bTry)

        if np.isnan(objTry):
            print('Warning: nan in updateTry')
            return 1e500

        if (objRef == None) | (objTry < objRef):
            self.cTry = cTry
            self.STry = STry
            self.bTry = bTry

            #print 'objTry=',objTry, dir.diff.sum()
        return objTry



    def getGradient(self, coeff=1.0):
        (pc, pS, pb) = diffeonCurrentNormGradient(self.c0, self.S0, self.b0,
                                        self.fv0, self.sigmaDist)
        dim = self.dim
        sigEye = self.sw*np.eye(dim)
        SS = sigEye.reshape([1,dim,dim]) + self.S0
        pS = (SS.reshape([self.ndf,dim, dim, 1]) * pS.reshape([self.ndf,1, dim, dim])).sum(axis=2)
        pS = (pS.reshape([self.ndf,dim, dim, 1]) * SS.reshape([self.ndf,1, dim, dim])).sum(axis=2)
        #pS = (pS + pS.transpose((0,2,1)))/2

        grd = Direction()
        grd.c = pc/coeff
        grd.S = pS/coeff
        grd.b = pb/coeff

        return grd



    def addProd(self, dir1, dir2, beta):
        dir = Direction()
        dir.c = dir1.c + beta * dir2.c
        dir.S = positiveProj(dir1.S + beta * dir2.S)
        dir.b = dir1.b + beta * dir2.b
        return dir

    def copyDir(self, dir0):
        dir = Direction()
        dir.c = np.copy(dir0.c)
        dir.S = np.copy(dir0.S)
        dir.b = np.copy(dir0.b)
        return dir


    def randomDir(self):
        dirfoo = Direction()
        dirfoo.c = np.random.randn(self.ndf, self.dim)
        dirfoo.S = np.random.randn(self.ndf, self.dim, self.dim)
        dirfoo.S = (dirfoo.S + dirfoo.S.transpose((0,2,1)))/2
        dirfoo.b = np.random.randn(self.ndf, self.dim)
        return dirfoo

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        dim = self.dim
        sigEye = self.sw*np.eye(dim)
        SS = sigEye.reshape([1,dim,dim]) + self.S0
        R, d = multiMatInverse1(SS, isSym=True)
        g1R = (R.reshape([self.ndf,dim, dim, 1]) * g1.S.reshape([self.ndf,1, dim, dim])).sum(axis=2)
        g1R = (g1R.reshape([self.ndf,dim, dim, 1]) * R.reshape([self.ndf,1, dim, dim])).sum(axis=2)
        for (ll,gr) in enumerate(g2):
            res[ll] = (g1.c*gr.c).sum() + (g1R*gr.S).sum() + (g1.b*gr.b).sum()
        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.c0 = np.copy(self.cTry)
        self.S0 = np.copy(self.STry)
        self.b0 = np.copy(self.bTry)


    def endOfIteration(self):
        #print self.obj0
        self.iter += 1
        if self.iter % 10 == 0:
            self.b0 = approximateSurfaceCurrent(self.c0, self.S0, self.fv0, self.sigmaDist)



    def optimize(self):
        if self.gradEps < 0:
            grd = self.getGradient(self.gradCoeff)
            [grd2] = self.dotProduct(grd, [grd])
            self.gradEps = max(0.001, np.sqrt(grd2) / 10000)

        print('Gradient lower bound: ', self.gradEps)
        cg.cg(self, verb = True, maxIter = self.maxIter, TestGradient=self.testGradient)
        #return self.at, self.xt

