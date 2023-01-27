import logging

import numpy as np
from scipy import linalg as LA
from .curves import Curve
from .surfaces import Surface
from numba import jit, int64
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


@jit(nopython=True, debug=True)
def ComputeIntersection_(VS, SF, ES, FES, WS, u, offset):
    #print('starting intersection')
    h = np.dot(VS, u) - offset
    #h = (VS* u[None,:]).sum(axis=1) - offset
    tol = 0
    vertices = np.zeros((ES.shape[0], 3))
    weights = np.zeros(ES.shape[0])
    intersect = -np.ones(ES.shape[0], dtype=int64)

    iv = 0
    for i in range(ES.shape[0]):
        he0 = h[ES[i,0]]
        he1 = h[ES[i,1]]
        if (he0 > tol and he1 < -tol) or (he0 < -tol and he1 > tol):
            r = -he0/(he1-he0)
            vertices[iv, :] = (1-r) * VS[ES[i,0],:] + r * VS[ES[i,1],:]
            weights[iv] = (1-r) * WS[ES[i,0]] + r * WS[ES[i,1]]
            intersect[i] = iv
            iv += 1
    vertices = vertices[:iv, :]
    weights = weights[:iv]

    edges = np.zeros((FES.shape[0], 2),dtype=int64)
    ie = 0
    for i in range(FES.shape[0]):
#        I = FS[i,:]
        J = FES[i, :]
        for k in (0,1,2):
            if intersect[J[k]]>=0 and intersect[J[(k+1)%3]]>=0:
                i1 = intersect[J[k]]
                i2 = intersect[J[(k+1)%3]]
                v = vertices[i2, :] - vertices[i1,:]
                if np.sum(np.cross(u, v)*SF[i,:]) > 0:
                    edges[ie, :] = [i1, i2]
                else:
                    edges[ie, :] = [i2, i1]
                ie += 1
    edges = edges[:ie, :]
    #print('ending intersection')
    return edges, vertices, weights

@jit(nopython=True)
def CurveGrad2Surf(curveGrad, curveGradw, VS, ES, WS, u, offset):
        h = np.dot(VS, u) - offset
        tol = 0
        grad = np.zeros(VS.shape)

        iv = 0
        for i in range(ES.shape[0]):
            he0 = h[ES[i, 0]]
            he1 = h[ES[i, 1]]
            if (he0 > tol and he1 < -tol) or (he0 < -tol and he1 > tol):
                d = he1 - he0
                r = -he0 / d
                dp = VS[ES[i, 1], :] - VS[ES[i, 0], :]
                dw = WS[ES[i, 1]] - WS[ES[i, 0]]
                g0 = curveGrad[iv, :] - ((dp*curveGrad[iv, :]).sum()/d) * u
                g0 -= (dw * curveGradw[iv]/d)*u
                grad[ES[i,0], :] += (1-r) * g0
                grad[ES[i,1], :] += r * g0
                iv += 1
        return grad

        # if hfM[i] > -tol and hfm[i].min() < tol:
        #     hi = h[i,:]
        #     if shf[i].sum() > 1-1e-10:
        #         i0 = np.argmin(hi)
        #     else:
        #         i0 = np.argmax(hi)
        #     i1 = (i0+1) % 3
        #     i2 = (i0+2) % 3
        #     h0 = hi[i0]
        #     h1 = hi[i1]
        #     h2 = hi[i2]
        #     p0 = VS[FS[i,i0], :]
        #     p1 = VS[FS[i, i1], :]
        #     p2 = VS[FS[i, i2], :]
        #     q0 = (h0*p1 - h1*p0)/(h0-h1)
        #     q1 = (h0*p2 - h2*p0)/(h0-h2)
        #     vertices[iv, :] = q0
        #     vertices[iv+1,:] = q1
        #     edges[ie, :] = [iv, iv+1]
        #     iv += 2
        #     ie += 1
    #return edges, vertices





class Hyperplane:
    def __init__(self, hyperplane=None, u = (0,0,1), offset = 0):
        if hyperplane is None:
            self.u = np.array(u, dtype=float)
            self.offset = offset
        elif type(hyperplane) is Hyperplane:
            self.u = hyperplane.u
            self.offset = hyperplane.offset
        else:
            self.u = hyperplane[0]
            self.offset = hyperplane[1]

class SurfaceSection:
    def __init__(self, curve=None, hyperplane=None, surf = None, hypLabel = -1, plot = None, isOpen = False):
        self.hyperplane = Hyperplane(hyperplane)
        if surf is None:
            self.curve = Curve(curve, isOpen=isOpen)
            self.ComputeHyperplane(curve)
        else:
            #print('starting intersection')
            self.ComputeIntersection(surf, hyperplane, plot=plot)
            #print('ending intersection')
        #self.normals = self.curve.computeUnitVertexNormals()
        self.normals = np.cross(self.hyperplane.u, self.curve.linel)
        self.normals /= np.maximum(np.sqrt((self.normals**2).sum(axis=1)), 1e-10)[:, None]
        self.area = -1
        self.outer = False
        self.hypLabel = hypLabel

    def ComputeIntersection(self, surf, hyperplane, plot = None):
        if surf.edges is None:
            #print('starting edges')
            surf.getEdges()
            #print('ending edges')
        F, V, W = ComputeIntersection_(surf.vertices, surf.surfel, surf.edges, surf.faceEdges, surf.weights,
                                    hyperplane.u, hyperplane.offset)
        self.curve = Curve([F,V])
        self.curve.updateWeights(W)
        if plot is not None:
            fig = plt.figure(plot)
            fig.clf()
            ax = Axes3D(fig)
            lim1 = self.curve.addToPlot(ax, ec='k', fc='b', lw=1)
            ax.set_xlim(lim1[0][0], lim1[0][1])
            ax.set_ylim(lim1[1][0], lim1[1][1])
            ax.set_zlim(lim1[2][0], lim1[2][1])
            fig.canvas.flush_events()

        #self.curve.removeDuplicates()
        # self.curve.orientEdges()
        self.hyperplane.u = hyperplane.u
        self.hyperplane.offset = hyperplane.offset

    def ComputeHyperplane(self, c):
        m = c.vertices.mean(axis=0)
        vm = c.vertices-m[None, :]
        S = np.dot(vm.T, vm)
        l,v = LA.eigh(S, subset_by_index = [0,1])
        if l[0] > 1e-6:
            logging.info('warning: curve not planar')
        if l[1] < 1e-6:
            logging.info('warning: curve linear')
        self.hyperplane = Hyperplane(u=v[:,0], offset=(m*v[:,0]).sum())


def Surf2SecDist(surf, s, curveDist, curveDist0 = None, plot = None, target_label=None, target_comp_info=None):
    if target_comp_info is not None:
        surf_, J = surf.extract_components(comp_info=target_comp_info)
    elif target_label is not None:
        surf_, J = surf.extract_components(comp=target_label)
    else:
        surf_ = surf
    s0 = SurfaceSection(surf=surf_, hyperplane=s.hyperplane, plot = plot)
    if s0.curve.faces.shape[0]>0:
        obj = curveDist(s0.curve, s.curve)
        n0 = s0.curve.weightedLength()
        n1 = s.curve.weightedLength()
        #logging.info(f'     Source {n0:.2f} points; target {n1:.2f}; distance: {obj:.4f}')

    else:
        obj = 0
    if curveDist0 is not None:
        obj2 = obj + curveDist0(s.curve)
    return obj

def Surf2SecGrad(surf, s, curveDistGrad, target_label = None, target_comp_info=None):
    if target_comp_info is not None:
        surf_, J = surf.extract_components(comp_info=target_comp_info)
    elif target_label is not None:
        surf_, J = surf.extract_components(comp=target_label)
    else:
        surf_ = surf
        J = None
    if surf_.edges is None:
        surf_.getEdges()
    s0 = SurfaceSection(surf=surf_, hyperplane=s.hyperplane)
    if s0.curve.faces.shape[0] > 0:
        cgrad, cgradw = curveDistGrad(s0.curve, s.curve, with_weights=True)
        grad_ = CurveGrad2Surf(cgrad, cgradw, surf_.vertices, surf_.edges, surf_.weights, s.hyperplane.u, s.hyperplane.offset)
    else:
        grad_ = np.zeros(surf_.vertices.shape)

    if target_comp_info is not None or target_label is not None:
        grad = np.zeros(surf.vertices.shape)
        grad[J, :] = grad_
    else:
        grad = grad_
    return grad


def readFromTXT(filename0, check_orientation = True, find_outer=True, merge_planes=True, forceClosed = False):
    if type(filename0) == str:
        filename0 = (filename0,)
    fv1_ = ()
    hyperplane_ = np.zeros((0,4))
    nc_ = 0
    fv1 = ()
    for filename in filename0:
        with open(filename, 'r') as f:
            s = f.readline()
            nc = int(s)
            nc_ += nc
            for i in range(nc):
                s = f.readline()
                npt = int(s)
                pts = np.zeros((npt,3))
                for j in range(npt):
                    s = f.readline()
                    pts[j,:] = s.split()
                ds = np.sqrt(((pts[1:, :] - pts[:-1,:])**2).sum(axis=1))
                u = np.sqrt(((pts[0, :] - pts[-1,:])**2).sum())
                isOpen = u > 2 * ds.max() and not forceClosed
                c = Curve(pts, isOpen=isOpen)
                fv1 += (SurfaceSection(curve=c),)
    uo = np.zeros((nc_, 4))
    nh = 0
    tol = 1e-3
    hk = np.zeros(4)
    for k,f in enumerate(fv1):
        f.area = f.curve.enclosedArea()
        found = False
        hk[:3] = f.hyperplane.u
        hk[3] = f.hyperplane.offset
        if k > 0:
            dst = np.sqrt(((hk - uo[:nh, :])**2).sum(axis=1))
            dst2 = np.sqrt(((hk + uo[:nh, :])**2).sum(axis=1))
            if dst.min() < tol:
                i = np.argmin(dst)
                f.hypLabel = i
                found = True
            elif dst2.min() < tol:
                i = np.argmin(dst2)
                f.hypLabel = i
                found = True
        if not found:
            uo[nh, :] = hk
            f.hypLabel = nh
            nh += 1
    hyperplanes = uo[:nh, :]
    if find_outer:
        J = -np.ones(nh, dtype=int)
        maxArea = np.zeros(nh)
        for k,f in enumerate(fv1):
            f.hyperplane.u = uo[f.hypLabel, :3]
            f.hyperplane.offset = uo[f.hypLabel, 3]
            f.outer = False
            if f.area > maxArea[f.hypLabel]:
                maxArea[f.hypLabel] = f.area
                J[f.hypLabel] = k
        for k in range(nh):
            fv1[J[k]].outer = True
        if check_orientation:
            eps = 1e-4
            for k, f in enumerate(fv1):
                c = Curve(curve=f.curve)
                n = np.zeros(c.vertices.shape)
                for i in range(c.faces.shape[0]):
                    n[c.faces[i, 0], :] += f.normals[i] / 2
                    n[c.faces[i, 1], :] += f.normals[i] / 2
                c.updateVertices(c.vertices + eps * n)
                a = np.fabs(c.enclosedArea())
                if (a > f.area and not f.outer) or \
                        (a < f.area and f.outer):
                    f.curve.flipFaces()
                    f.normals *= -1
    else:
        for f in fv1:
            f.outer=False

    if merge_planes:
        c = []
        found_h = np.zeros(len(fv1), dtype=bool)
        for k in range(hyperplanes.shape[0]):
            c.append([])
            for i in range(len(fv1)):
                if not found_h[i]:
                    u = min(np.fabs(hyperplanes[k, :3] - fv1[i].hyperplane.u).sum()
                            + np.fabs(hyperplanes[k, 3] - fv1[i].hyperplane.offset),
                            np.fabs(hyperplanes[k, :3] + fv1[i].hyperplane.u).sum()
                            + np.fabs(hyperplanes[k, 3] + fv1[i].hyperplane.offset))
                    if u<1e-2:
                        c[k].append(fv1[i].curve)
                        found_h[i] = True

        fv1_ = []
        for i in range(len(c)):
            cv = Curve(curve=c[i])
            h = Hyperplane(u=hyperplanes[i,:3], offset=hyperplanes[i,3])
            fv1_.append(SurfaceSection(curve=cv, hyperplane=h, hypLabel=i))

    #fv1_ += fv1
    #hyperplane_ = np.concatenate((hyperplane_, hyperplanes), axis=0)

    return fv1_, hyperplanes
