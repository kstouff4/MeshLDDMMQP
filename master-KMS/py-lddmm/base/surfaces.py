#import matplotlib
import os
import numpy as np
from numba import jit, int64
import scipy as sp
import scipy.linalg as LA
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage import measure
try:
    from vtk import vtkCellArray, vtkPoints, vtkPolyData, vtkVersion,\
        vtkLinearSubdivisionFilter, vtkQuadricDecimation,\
        vtkWindowedSincPolyDataFilter, vtkImageData, VTK_FLOAT,\
        vtkDoubleArray, vtkContourFilter, vtkPolyDataConnectivityFilter,\
        vtkCleanPolyData, vtkPolyDataReader, vtkOBJReader, vtkSTLReader,\
        vtkDecimatePro, VTK_UNSIGNED_CHAR, vtkPolyDataToImageStencil,\
        vtkImageStencil
    from vtk.util.numpy_support import vtk_to_numpy
    gotVTK = True
except ImportError:
    v2n = None
    print('could not import VTK functions')
    gotVTK = False

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import LightSource
from matplotlib import colors
from matplotlib import pyplot as plt
try:
    from pygalmesh import DomainBase, generate_surface_mesh
    gotPygalmesh=True
except ImportError:
    print('could not import Pygalmesh functions')
    gotPygalmesh = False

from . import diffeo
import scipy.linalg as spLA
from scipy.sparse import coo_matrix
import glob
import logging
# try:
#     from vtk import *
#     import vtk.util.numpy_support as v2n
#     gotVTK = True
# except ImportError:
#     v2n = None
#     print('could not import VTK functions')
#     gotVTK = False


class vtkFields:
    def __init__(self):
        self.scalars = [] 
        self.vectors = []
        self.normals = []
        self.tensors = []


@jit(nopython=True)
def get_edges_(faces):
    int64 = "int64"
    nv = faces.max() + 1
    nf = faces.shape[0]

    edg0 = np.zeros((nv,nv), dtype = int64)
    for k in range(faces.shape[0]):
        i0 = faces[k,0]
        i1 = faces[k,1]
        i2 = faces[k,2]
        edg0[min(i0, i1), max(i0, i1)] = 1
        edg0[min(i1, i2), max(i1, i2)] = 1
        edg0[min(i2, i0), max(i2, i0)] = 1

    J = np.nonzero(edg0)
    ne = J[0].shape[0]
    edges = np.zeros((ne, 2), dtype = int64)
    edges[:,0] = J[0]
    edges[:,1] = J[1]

    edgi = np.zeros((nv, nv), dtype = int64)
    for k in range(ne):
        edgi[edges[k,0], edges[k,1]] = k
    
    faceEdges = np.zeros(faces.shape, dtype=int64)
    for k in range(faces.shape[0]):
        i0 = faces[k,0]
        i1 = faces[k,1]
        i2 = faces[k,2]
        faceEdges[k, 0] = edgi[min(i0, i1), max(i0,i1)]    
        faceEdges[k, 1] = edgi[min(i1, i2), max(i1,i2)]    
        faceEdges[k, 2] = edgi[min(i2, i0), max(i2,i0)]    


    edgeFaces = - np.ones((ne,2), dtype=int64)
    for k in range(faces.shape[0]):
        i0 = faces[k,0]
        i1 = faces[k,1]
        i2 = faces[k,2]
        for f in ([i0,i1], [i1,i2], [i2, i0]):
            kk = edgi[min(f[0], f[1]), max(f[0], f[1])]
            if edgeFaces[kk, 0] >= 0:
                edgeFaces[kk,1] = k
            else:
                edgeFaces[kk,0] = k

    
    bdry = np.zeros(edges.shape[0], dtype=int64)
    for k in range(edges.shape[0]):
        if edgeFaces[k,1] < 0:
            bdry[k] = 1
    return edges, edgeFaces, faceEdges, bdry
# General surface class

#@jit(nopython=True)
def extract_components_(target_comp, nbvert, faces, component, edge_info = None):
    #labels = np.zeros(self.vertices.shape[0], dtype = int)
    #for j in range(self.faces.shape[1]):
    #   labels[self.faces[:,i]] = self.component

    int_type = int
    Jf = np.zeros(component.shape[0], dtype = int_type)
    for k in target_comp:
        for i in range(component.shape[0]):
            if component[i] == k:
                Jf[i] = 1
#        Jf = np.logical_or(Jf, component==k)
#    for i in range(component.shape[0]):
#        print(component[i])
#        if component[i] in target_comp:
#            Jf[i] = True
    #Jf = np.isin(component, target_comp)

    J = np.zeros(nbvert, dtype = int_type)
    for i in range(faces.shape[1]):
        J[faces[:,i]] = np.logical_or(J[faces[:,i]], Jf)
    J = np.where(J)[0]
    #V = vertices[J,:]
    #w = weights[J]
    newI = -np.ones(nbvert, dtype=int_type)
    newI[J] = np.arange(0, J.shape[0])
    F = np.zeros(faces.shape, dtype=int_type)
    If = np.zeros(faces.shape[0], dtype=int_type)
    ii = 0
    for i in range(faces.shape[0]):
        pos = True
        for j in range(faces.shape[1]):
            F[i,j] = newI[faces[i,j]]
            if F[i,j]< 0:
                pos = False
        if pos:
            If[ii] = i
            ii += 1
    If = If[:ii]
    #If = np.max(F, axis=1) >= 0
    F = F[If, :]
    if edge_info is not None:
        edges = edge_info[0]
        faceEdges = edge_info[1]
        edgeFaces = edge_info[2]
        newIf = -np.ones(faces.shape[0], dtype = int_type)
        newIf[If] = np.arange(If.shape[0])
        E = np.zeros(edges.shape, dtype=int_type)
        Ie = np.zeros(edges.shape[0], dtype=int_type)
        ii = 0
        for i in range(edges.shape[0]):
            pos = True
            for j in range(edges.shape[1]):
                E[i,j] = newI[edges[i,j]]
                if E[i,j]< 0:
                    pos = False
            if pos:
                Ie[ii] = i
                ii += 1
        Ie = Ie[:ii]
        E = E[Ie, :]
        #E = newI[edges]
        #Ie = np.amax(E, axis=1) >= 0
        newIe = -np.ones(edges.shape[0], dtype = int_type)
        newIe[Ie] = np.arange(Ie.shape[0])
        #E = E[Ie, :]

        FE = np.zeros(faceEdges.shape, dtype=int_type)
        Ife = np.zeros(faceEdges.shape[0], dtype=int_type)
        ii = 0
        for i in range(faceEdges.shape[0]):
            pos = True
            for j in range(faceEdges.shape[1]):
                FE[i,j] = newIe[faceEdges[i,j]]
                if FE[i,j]< 0:
                    pos = False
            if pos:
                Ife[ii] = i
                ii += 1
        Ife = Ife[:ii]
        FE = FE[Ife, :]
        #FE = newIe[faceEdges]
        #I_ = np.amax(FE, axis=1) >= 0
        #FE = FE[I_, :]
        EF = np.zeros(edgeFaces.shape, dtype=int_type)
        Ief = np.zeros(edgeFaces.shape[0], dtype=int_type)
        ii = 0
        for i in range(edgeFaces.shape[0]):
            pos = True
            for j in range(edgeFaces.shape[1]):
                EF[i,j] = newIf[edgeFaces[i,j]]
                if EF[i,j]< 0:
                    pos = False
            if pos:
                Ief[ii] = i
                ii += 1
        Ief = Ief[:ii]
        EF = EF[Ief, :]
        #EF = newIf[edgeFaces]
        #I_ = np.amax(EF, axis=1) >= 0
        #EF = EF[I_, :]
    else:
        E = None
        FE = None
        EF = None

    return F, J, E, FE, EF


class Surface:
    def __init__(self, surf=None, weights=None):
        if type(surf) in (list, tuple):
            if isinstance(surf[0], Surface):
                self.concatenate(surf)
            elif type(surf[0]) is str:
                fvl = []
                for name in surf:
                    fvl.append(Surface(surf=name))
                self.concatenate(fvl)
            else:
                self.vertices = np.copy(surf[1])
                self.faces = np.int_(np.copy(surf[0]))
                self.component = np.zeros(self.faces.shape[0], dtype=int)
                if weights is None:
                    self.weights = np.ones(self.vertices.shape[0], dtype=int)
                    self.face_weights = np.ones(self.faces.shape[0], dtype=int)
                else:
                    self.updateWeights(weights)
                self.computeCentersAreas()
        elif type(surf) is str:
            self.read(surf)
        elif issubclass(type(surf), Surface):
            self.vertices = np.copy(surf.vertices)
            self.surfel = np.copy(surf.surfel)
            self.faces = np.copy(surf.faces)
            self.centers = np.copy(surf.centers)
            self.component = np.copy(surf.component)
            if weights is None:
                self.weights = np.copy(surf.weights)
                self.face_weights = np.copy(surf.face_weights)
            else:
                self.updateWeights(weights)
        else:
            self.vertices = np.empty(0)
            self.centers = np.empty(0)
            self.faces = np.empty(0)
            self.surfel = np.empty(0)
            self.component = np.empty(0)
            self.weights = np.empty(0)
            self.face_weights = np.empty(0)

        self.edges = None
        self.edgeFaces = None
        self.faceEdges = None
        # if surf == None:
        #     if FV == None:
        #         if filename == None:
        #             self.vertices = np.empty(0)
        #             self.centers = np.empty(0)
        #             self.faces = np.empty(0)
        #             self.surfel = np.empty(0)
        #             self.component = np.empty(0)
        #         else:
        #             if type(filename) is list:
        #                 fvl = []
        #                 for name in filename:
        #                     fvl.append(Surface(filename=name))
        #                 self.concatenate(fvl)
        #             else:
        #                 self.read(filename)
        #     else:
        #         self.vertices = np.copy(FV[1])
        #         self.faces = np.int_(FV[0])
        #         self.component = np.zeros(self.faces.shape[0], dtype=int)
        #         self.computeCentersAreas()
        # else:
        #     if type(surf) is list:
        #         self.concatenate(surf)
        #     else:
        #         self.vertices = np.copy(surf.vertices)
        #         self.faces = np.copy(surf.faces)
        #         #self.surfel = np.copy(surf.surfel)
        #         #self.centers = np.copy(surf.centers)
        #         self.component = np.copy(surf.component)
        #         self.computeCentersAreas()

    def read(self, filename):
        (mainPart, ext) = os.path.splitext(filename)
        if ext in ('.byu', '.g'):
            self.readbyu(filename)
        elif ext=='.off':
            self.readOFF(filename)
        elif ext=='.vtk':
            self.readVTK(filename)
        elif ext == '.stl':
            self.readSTL(filename)
        elif ext == '.obj':
            self.readOBJ(filename)
        else:
            self.vertices = np.empty(0)
            self.centers = np.empty(0)
            self.faces = np.empty(0)
            self.component = np.empty(0)
            self.surfel = np.empty(0)
            self.weights = np.empty(0)
            self.face_weights = np.empty(0)
            raise NameError('Unknown Surface Extension: '+filename)

    # face centers and area weighted normal
    def computeCentersAreas(self):
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.centers = (xDef1 + xDef2 + xDef3) / 3
        self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)/2
        w1 = self.weights[self.faces[:, 0]]
        w2 = self.weights[self.faces[:, 1]]
        w3 = self.weights[self.faces[:, 2]]
        self.face_weights = (w1+w2+w3)/3

    def updateWeights(self, w0):
        self.weights = np.copy(w0)
        w1 = self.weights[self.faces[:, 0]]
        w2 = self.weights[self.faces[:, 1]]
        w3 = self.weights[self.faces[:, 2]]
        self.face_weights = (w1+w2+w3)/3

    # modify vertices without toplogical change
    def updateVertices(self, x0):
        self.vertices = np.copy(x0) 
        xDef1 = self.vertices[self.faces[:, 0], :]
        xDef2 = self.vertices[self.faces[:, 1], :]
        xDef3 = self.vertices[self.faces[:, 2], :]
        self.centers = (xDef1 + xDef2 + xDef3) / 3
        self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)/2

    def computeSparseMatrices(self):
        self.v2f0 = sp.sparse.csc_matrix((np.ones(self.faces.shape[0]),
                                          (range(self.faces.shape[0]),
                                           self.faces[:,0]))).transpose(copy=False)
        self.v2f1 = sp.sparse.csc_matrix((np.ones(self.faces.shape[0]),
                                          (range(self.faces.shape[0]),
                                           self.faces[:,1]))).transpose(copy=False)
        self.v2f2 = sp.sparse.csc_matrix((np.ones(self.faces.shape[0]),
                                          (range(self.faces.shape[0]),
                                           self.faces[:,2]))).transpose(copy=False)

    def computeVertexArea(self, simpleMethod = False):
        # compute areas of faces and vertices
        V = self.vertices
        F = self.faces
        nv = V.shape[0]
        nf = F.shape[0]
        AF = np.zeros(nf)
        AV = np.zeros(nv)
        if simpleMethod:
            for k in range(nf):
                x12 = V[F[k, 1], :] - V[F[k, 0], :]
                x13 = V[F[k, 2], :] - V[F[k, 0], :]
                AF[k] = np.sqrt((np.cross(x12, x13) ** 2).sum()) / 2
                AV[F[k, 0]] += AF[k] / 3
                AV[F[k, 1]] += AF[k] / 3
                AV[F[k, 2]] += AF[k] / 3
        else:
            for k in range(nf):
                # determining if face is obtuse
                x12 = V[F[k,1], :] - V[F[k,0], :]
                x13 = V[F[k,2], :] - V[F[k,0], :]
                n12 = np.sqrt((x12**2).sum())
                n13 = np.sqrt((x13**2).sum())
                c1 = (x12*x13).sum()/(n12*n13)
                x23 = V[F[k,2], :] - V[F[k,1], :]
                n23 = np.sqrt((x23**2).sum())
                #n23 = norm(x23) ;
                c2 = -(x12*x23).sum()/(n12*n23)
                c3 = (x13*x23).sum()/(n13*n23)
                AF[k] = np.sqrt((np.cross(x12, x13)**2).sum())/2
                s1 = np.sqrt(1 - c1 ** 2)
                s2 = np.sqrt(1 - c2 ** 2)
                s3 = np.sqrt(1 - c3 ** 2)
                cot1 = c1 / s1
                cot2 = c2 / s2
                cot3 = c3 / s3
                if (c1 < 0):
                    #face obtuse at vertex 1
                    u2 = (x12 ** 2).sum() / (8 * cot2)
                    u3 = (x13 ** 2).sum() / (8 * cot3)
                    u1 = AF[k] - u2 - u3
                    AV[F[k, 0]] += u1
                    AV[F[k, 1]] += u2
                    AV[F[k, 2]] += u3
                    if u1 < 0 or u2 < 0 or u3 < 0:
                        print('error')
                elif (c2 < 0):
                    #face obuse at vertex 2
                    u1 = (x12 ** 2).sum() / (8 * cot1)
                    u3 = (x23 ** 2).sum() / (8 * cot3)
                    u2 = AF[k] - u1 - u3
                    AV[F[k, 0]] += u1
                    AV[F[k, 1]] += u2
                    AV[F[k, 2]] += u3
                    if u1 < 0 or u2 < 0 or u3 < 0:
                        print('error')
                elif (c3 < 0):
                    #face obtuse at vertex 3
                    u1 = (x13 ** 2).sum() / (8 * cot1)
                    u2 = (x23 ** 2).sum() / (8 * cot2)
                    u3 = AF[k] - u1 - u2
                    AV[F[k, 0]] += u1
                    AV[F[k, 1]] += u2
                    AV[F[k, 2]] += u3
                    if u1 < 0 or u2 < 0 or u3 < 0:
                        print('error')
                else:
                    #non obtuse face
                    AV[F[k,0]] += ((x12**2).sum() * cot3 + (x13**2).sum() * cot2)/8
                    AV[F[k,1]] += ((x12**2).sum() * cot3 + (x23**2).sum() * cot1)/8
                    AV[F[k,2]] += ((x13**2).sum() * cot2 + (x23**2).sum() * cot1)/8

        for k in range(nv):
            if (np.fabs(AV[k]) <1e-10):
                logging.info('Warning: vertex {0:1} has no face; use removeIsolated'.format(k))
        #print('sum check area:', AF.sum(), AV.sum())
        return AV, AF

    def computeVertexNormals(self):
        self.computeCentersAreas() 
        normals = np.zeros(self.vertices.shape)
        F = self.faces
        for k in range(F.shape[0]):
            normals[F[k,0]] += self.surfel[k]
            normals[F[k,1]] += self.surfel[k]
            normals[F[k,2]] += self.surfel[k]
        af = np.sqrt( (normals**2).sum(axis=1))
        #logging.info('min area = %.4f'%(af.min()))
        normals /=af.reshape([self.vertices.shape[0],1])

        return normals

    def computeAreaWeightedVertexNormals(self):
        self.computeCentersAreas() 
        normals = np.zeros(self.vertices.shape)
        F = self.faces
        for k in range(F.shape[0]):
            normals[F[k,0]] += self.surfel[k]
            normals[F[k,1]] += self.surfel[k]
            normals[F[k,2]] += self.surfel[k]

        return normals
         

    # Computes edges from vertices/faces
    def getEdges(self):
        self.edges, self.edgeFaces, self.faceEdges, self.bdry = get_edges_(self.faces)


    # computes the signed distance function in a small neighborhood of a shape 
    def LocalSignedDistance(self, data, value):
        d2 = 2 * np.array(data >= value) - 1
        c2 = np.cumsum(d2, axis=0)
        for j in range(2):
            c2 = np.cumsum(c2, axis=j + 1)
        (n0, n1, n2) = c2.shape

        rad = 3
        diam = 2 * rad + 1
        (x, y, z) = np.mgrid[-rad:rad + 1, -rad:rad + 1, -rad:rad + 1]
        cube = (x ** 2 + y ** 2 + z ** 2)
        maxval = (diam) ** 3
        s = 3.0 * rad ** 2
        res = d2 * s
        u = maxval * np.ones(c2.shape)
        u[rad + 1:n0 - rad, rad + 1:n1 - rad, rad + 1:n2 - rad] = (c2[diam:n0, diam:n1, diam:n2]
                                                                   - c2[0:n0 - diam, diam:n1, diam:n2]
                                                                   - c2[diam:n0, 0:n1 - diam, diam:n2]
                                                                   - c2[diam:n0,diam:n1,0:n2 - diam]
                                                                   + c2[0:n0 - diam, 0:n1 - diam, diam:n2]
                                                                   + c2[diam:n0, 0:n1 - diam, 0:n2 - diam]
                                                                   + c2[0:n0 - diam, diam:n1,0:n2 - diam]
                                                                   - c2[0:n0 - diam, 0:n1 - diam, 0:n2 - diam])

        I = np.nonzero(np.fabs(u) < maxval)
        # print len(I[0])

        for k in range(len(I[0])):
            p = np.array((I[0][k], I[1][k], I[2][k]))
            bmin = p - rad
            bmax = p + rad + 1
            # print p, bmin, bmax
            if (d2[p[0], p[1], p[2]] > 0):
                # print u[p[0],p[1], p[2]]
                # print d2[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]].sum()
                res[p[0], p[1], p[2]] = min(cube[np.nonzero(d2[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]] < 0)]) - .25
            else:
                res[p[0], p[1], p[2]] = - min(cube[np.nonzero(d2[bmin[0]:bmax[0], bmin[1]:bmax[1], bmin[2]:bmax[2]] > 0)]) - .25

        return res

    def toPolyData(self):
        if gotVTK:
            points = vtkPoints()
            for k in range(self.vertices.shape[0]):
                points.InsertNextPoint(self.vertices[k,0], self.vertices[k,1], self.vertices[k,2])
            polys = vtkCellArray()
            for k in range(self.faces.shape[0]):
                polys.InsertNextCell(3)
                for kk in range(3):
                    polys.InsertCellPoint(self.faces[k,kk])
            polydata = vtkPolyData()
            polydata.SetPoints(points)
            polydata.SetPolys(polys)
            return polydata
        else:
            raise Exception('Cannot run toPolyData without VTK')

    def fromPolyData(self, g, scales=(1.,1.,1.)):
        npoints = int(g.GetNumberOfPoints())
        nfaces = int(g.GetNumberOfPolys())
        logging.info('Dimensions: %d %d %d' %(npoints, nfaces, g.GetNumberOfCells()))
        V = np.zeros([npoints, 3])
        for kk in range(npoints):
            V[kk, :] = np.array(g.GetPoint(kk))
            #print kk, V[kk]
            #print kk, np.array(g.GetPoint(kk))
        F = np.zeros([nfaces, 3])
        gf = 0
        for kk in range(g.GetNumberOfCells()):
            c = g.GetCell(kk)
            if(c.GetNumberOfPoints() == 3):
                for ll in range(3):
                    F[gf,ll] = c.GetPointId(ll)
                    #print kk, gf, F[gf]
                gf += 1

                #self.vertices = np.multiply(data.shape-V-1, scales)
        self.vertices = np.multiply(V, scales)
        self.faces = np.int_(F[0:gf, :])
        self.component = np.zeros(self.faces.shape[0], dtype = int)
        self.weights = np.ones(self.vertices.shape[0])
        self.computeCentersAreas()

    def subDivide(self, number=1):
        if gotVTK:
            polydata = self.toPolyData()
            subdivisionFilter = vtkLinearSubdivisionFilter()
            if vtkVersion.GetVTKMajorVersion() >= 6:
                subdivisionFilter.SetInputData(polydata)
            else:
                subdivisionFilter.SetInput(polydata)
            subdivisionFilter.SetNumberOfSubdivisions(number)
            subdivisionFilter.Update()
            self.fromPolyData(subdivisionFilter.GetOutput())
        else:
            raise Exception('Cannot run subDivide without VTK')
                        
            
    def Simplify(self, target=1000.0, deciPro=False):
        if gotVTK:
            polydata = self.toPolyData()
            red = 1 - min(np.float(target) / polydata.GetNumberOfPoints(), 1)
            if deciPro:
                dc = vtkDecimatePro()
                if vtkVersion.GetVTKMajorVersion() >= 6:
                    dc.SetInputData(polydata)
                else:
                    dc.SetInput(polydata)
                dc.SetTargetReduction(red)
                dc.PreserveTopologyOn()
                dc.Update()
            else:
                dc = vtkQuadricDecimation()
                dc.SetTargetReduction(red)
                if vtkVersion.GetVTKMajorVersion() >= 6:
                    dc.SetInputData(polydata)
                else:
                    dc.SetInput(polydata)
                dc.Update()
            g = dc.GetOutput()
            self.fromPolyData(g)
            z= self.surfVolume()
            if (z > 0):
                self.flipFaces()
                logging.info('flipping volume {0:f} {1:f}'.format(z, self.surfVolume()))
        else:
            raise Exception('Cannot run Simplify without VTK')

    def flipFaces(self):
        self.faces = self.faces[:, [0,2,1]]
        self.computeCentersAreas()



    def smooth(self, n=30, smooth=0.1):
        if gotVTK:
            g = self.toPolyData()
            smoother= vtkWindowedSincPolyDataFilter()
            if vtkVersion.GetVTKMajorVersion() >= 6:
                smoother.SetInputData(g)
            else:
                smoother.SetInput(g)
            #smoother.SetInput(g)
            smoother.SetNumberOfIterations(n)
            smoother.SetPassBand(smooth)   
            smoother.NonManifoldSmoothingOn()
            smoother.NormalizeCoordinatesOn()
            smoother.GenerateErrorScalarsOn() 
            #smoother.GenerateErrorVectorsOn()
            smoother.Update()
            g = smoother.GetOutput()
            self.fromPolyData(g)
        else:
            raise Exception('Cannot run smooth without VTK')


    def Isosurface_ski(self, data, value=0.5, step = 1):
        verts,faces,n,v = measure.marching_cubes(data, allow_degenerate=False, level=value,step_size=step)
        self.__init__(surf=(faces,verts))


    # Computes isosurfaces using vtk               
    def Isosurface(self, data, value=0.5, target=1000.0, scales = (1., 1., 1.),
                   smooth = 0.1, fill_holes = 1., orientation=1):
        if gotVTK:
            #data = self.LocalSignedDistance(data0, value)
            if isinstance(data, vtkImageData):
                img = data
            else:
                img = vtkImageData()
                img.SetDimensions(data.shape)
                img.SetOrigin(0,0,0)
                if vtkVersion.GetVTKMajorVersion() >= 6:
                    img.AllocateScalars(VTK_FLOAT,1)
                else:
                    img.SetNumberOfScalarComponents(1)
                v = vtkDoubleArray()
                v.SetNumberOfValues(data.size)
                v.SetNumberOfComponents(1)
                for ii,tmp in enumerate(np.ravel(data, order='F')):
                    v.SetValue(ii,tmp)
                    img.GetPointData().SetScalars(v)
                
            cf = vtkContourFilter()
            if vtkVersion.GetVTKMajorVersion() >= 6:
                cf.SetInputData(img)
            else:
                cf.SetInput(img)
            cf.SetValue(0,value)
            cf.SetNumberOfContours(1)
            cf.Update()
            #print cf
            connectivity = vtkPolyDataConnectivityFilter()
            connectivity.ScalarConnectivityOff()
            connectivity.SetExtractionModeToLargestRegion()
            if vtkVersion.GetVTKMajorVersion() >= 6:
                connectivity.SetInputData(cf.GetOutput())
            else:
                connectivity.SetInput(cf.GetOutput())
            connectivity.Update()
            g = connectivity.GetOutput()
    
            if smooth > 0:
                smoother= vtkWindowedSincPolyDataFilter()
                if vtkVersion.GetVTKMajorVersion() >= 6:
                    smoother.SetInputData(g)
                else:
                    smoother.SetInput(g)
                #     else:
                # smoother.SetInputConnection(contour.GetOutputPort())    
                smoother.SetNumberOfIterations(30)
                #this has little effect on the error!
                #smoother.BoundarySmoothingOff()
                #smoother.FeatureEdgeSmoothingOff()
                #smoother.SetFeatureAngle(120.0)
                smoother.SetPassBand(smooth)        #this increases the error a lot!
                smoother.NonManifoldSmoothingOn()
                #smoother.NormalizeCoordinatesOn()
                #smoother.GenerateErrorScalarsOn() 
                #smoother.GenerateErrorVectorsOn()
                smoother.Update()
                g = smoother.GetOutput()

            #dc = vtkDecimatePro()
            if target>0:
                red = 1 - min(np.float(target)/g.GetNumberOfPoints(), 1)
                #print 'Reduction: ', red
                dc = vtkQuadricDecimation()
                dc.SetTargetReduction(red)
                #dc.AttributeErrorMetricOn()
                #dc.SetDegree(10)
                #dc.SetSplitting(0)
                if vtkVersion.GetVTKMajorVersion() >= 6:
                    dc.SetInputData(g)
                else:
                    dc.SetInput(g)
                    #dc.SetInput(g)
                #print dc
                dc.Update()
                g = dc.GetOutput()
            #print 'points:', g.GetNumberOfPoints()
            cp = vtkCleanPolyData()
            if vtkVersion.GetVTKMajorVersion() >= 6:
                cp.SetInputData(g)
            else:
                cp.SetInput(g)
                #        cp.SetInput(dc.GetOutput())
            #cp.SetPointMerging(1)
            cp.ConvertPolysToLinesOn()
            cp.SetAbsoluteTolerance(1e-5)
            cp.Update()
            g = cp.GetOutput()
            self.fromPolyData(g,scales)
            z= self.surfVolume()
            if (orientation*z < 0):
                self.flipFaces()
                #print 'flipping volume', z, self.surfVolume()
                logging.info('flipping volume %.2f %.2f' % (z, self.surfVolume()))

        else:
            raise Exception('Cannot run Isosurface without VTK')
    
    # Ensures that orientation is correct
    def edgeRecover(self):
        v = self.vertices
        f = self.faces
        nv = v.shape[0]
        nf = f.shape[0]
        # faces containing each oriented edge
        edg0 = np.int_(np.zeros((nv, nv)))
        # number of edges between each vertex
        edg = np.int_(np.zeros((nv, nv)))
        # contiguous faces
        edgF = np.int_(np.zeros((nf, nf)))
        for (kf, c) in enumerate(f):
            if (edg0[c[0],c[1]] > 0):
                edg0[c[1],c[0]] = kf+1  
            else:
                edg0[c[0],c[1]] = kf+1
                
            if (edg0[c[1],c[2]] > 0):
                edg0[c[2],c[1]] = kf+1  
            else:
                edg0[c[1],c[2]] = kf+1  

            if (edg0[c[2],c[0]] > 0):
                edg0[c[0],c[2]] = kf+1  
            else:
                edg0[c[2],c[0]] = kf+1  

            edg[c[0],c[1]] += 1
            edg[c[1],c[2]] += 1
            edg[c[2],c[0]] += 1


        for kv in range(nv):
            I2 = np.nonzero(edg0[kv,:])
            for kkv in I2[0].tolist():
                if edg0[kkv,kv] > 0:
                    edgF[edg0[kkv,kv]-1,edg0[kv,kkv]-1] = kv+1

        isOriented = np.int_(np.zeros(f.shape[0]))
        isActive = np.int_(np.zeros(f.shape[0]))
        I = np.nonzero(np.squeeze(edgF[0,:]))
        # list of faces to be oriented
        # Start with face 0 and its neighbors
        activeList = [0]+I[0].tolist()
        lastOriented = 0
        isOriented[0] = True
        for k in activeList:
            isActive[k] = True 

        while lastOriented < len(activeList)-1:
            #next face to be oriented
            j = activeList[lastOriented +1]
            # find an already oriented face among all neighbors of j
            I = np.nonzero(edgF[j,:])
            foundOne = False
            for kk in I[0].tolist():
                if (foundOne==False) and (isOriented[kk]):
                    foundOne = True
                    u1 = edgF[j,kk] -1
                    u2 = edgF[kk,j] - 1
                    if not ((edg[u1,u2] == 1) and (edg[u2,u1] == 1)): 
                        # reorient face j
                        edg[f[j,0],f[j,1]] -= 1
                        edg[f[j,1],f[j,2]] -= 1
                        edg[f[j,2],f[j,0]] -= 1
                        a = f[j,1]
                        f[j,1] = f[j,2]
                        f[j,2] = a
                        edg[f[j,0],f[j,1]] += 1
                        edg[f[j,1],f[j,2]] += 1
                        edg[f[j,2],f[j,0]] += 1
                elif (not isActive[kk]):
                    activeList.append(kk)
                    isActive[kk] = True
            if foundOne:
                lastOriented = lastOriented+1
                isOriented[j] = True
                #print 'oriented face', j, lastOriented,  'out of',  nf,  ';  total active', len(activeList) 
            else:
                logging.info('Unable to orient face {0:d}'.format(j))
                return
        self.vertices = v
        self.faces = f

        z= self.surfVolume()
        if (z > 0):
            self.flipFaces()

    def removeIsolated(self):
        N = self.vertices.shape[0]
        inFace = np.int_(np.zeros(N))
        for k in range(3):
            inFace[self.faces[:,k]] = 1
        J = np.nonzero(inFace)
        self.vertices = self.vertices[J[0], :]
        logging.info('Found %d isolated vertices'%(N-J[0].shape[0]))
        Q = -np.ones(N)
        for k,j in enumerate(J[0]):
            Q[j] = k
        self.faces = np.int_(Q[self.faces])


    def removeDuplicates(self, c=0.0001, verb = False):
        c2 = c ** 2
        N0 = self.vertices.shape[0]
        w = np.zeros(N0, dtype=int)

        newv = np.zeros(self.vertices.shape)
        newweights = np.ones(self.vertices.shape[0])
        removed = np.zeros(self.vertices.shape[0], dtype=bool)
        newv[0, :] = self.vertices[0, :]
        N = 1
        for kj in range(1, N0):
            dist = ((self.vertices[kj, :] - newv[0:N, :]) ** 2).sum(axis=1)
            # print dist.shape
            J = np.nonzero(dist < c2)
            J = J[0]
            # print kj, ' ', J, len(J)
            if (len(J) > 0):
                if verb:
                    logging.info("duplicate: {0:d} {1:d}".format(kj, J[0]))
                removed[kj] = True
                w[kj] = J[0]
            else:
                w[kj] = N
                newv[N, :] = self.vertices[kj, :]
                newweights[N] = self.weights[kj]
                N = N + 1

        newv = newv[0:N, :]
        newweights = newweights[0:N]
        self.vertices = newv
        self.weights = newweights
        self.faces = w[self.faces]

        newf = np.zeros(self.faces.shape, dtype=int)
        Nf = self.faces.shape[0]
        newcomp = np.zeros(Nf)
        nj = 0
        for kj in range(Nf):
            if len(set(self.faces[kj,:]))< 3:
                if verb:
                    logging.info('Empty face: {0:d} {1:d}'.format(kj, nj))
            else:
                newf[nj, :] = self.faces[kj, :]
                newcomp[nj] = self.component[kj]
                #newc[nj] = self.component[kj]
                nj += 1
        self.faces = newf[0:nj, :]
        self.component = newcomp[0:nj]
        self.computeCentersAreas()
        return removed
        #self.component = newc[0:nj]


    def addFace(self, f, faces, edges):
        faces.append(f)
        edges[(f[0],f[1])] = len(faces)-1
        edges[(f[1],f[2])] = len(faces)-1
        edges[(f[2],f[0])] = len(faces)-1


    def split_in_4(self, faces, vert, weights, val, popped, edges, i, with_vertex = None):
        if not popped[i]:
            vali = np.array(val)[faces[i]]
            kmax = faces[i][vali.argmax()]
            kmin = faces[i][vali.argmin()]
            kmid = [j for j in faces[i] if j != kmin and j!= kmax][0]
            if val[kmid] > 0:
                kpivot = kmin
            else:
                kpivot = kmax
            if val[kmax] > 0:
                k0 = faces[i][0]
                k1 = faces[i][1]
                k2 = faces[i][2]

                if [kmin, kmid, kmax] in [[k0,k1,k2], [k1,k2,k0], [k2,k0,k1]]:
                    pos = True
                else:
                    pos = False

                # select[res.faces[i,:]] = True
                eps = 1e-10 * np.sign(val[kpivot])
                if val[kmin] < 0:
                    r0 = val[kmax]/(val[kmax] - val[kmin])
                    v0 = (1-r0) * vert[kmax] + r0 * vert[kmin]
                    knew0 = len(vert)
                    vert.append(v0)
                    w0 = (1-r0) * weights[kmax] + r0 * weights[kmin]
                    weights.append(w0)
                    val.append(0)
                    if val[kmid] < 1e-10 and val[kmid] > -1e-10:
                        if pos:
                            self.addFace([kmid, kmax, knew0], faces, edges)
                        else:
                            self.addFace([kmax, kmid, knew0], faces, edges)
                        popped += [False]
                        if pos:
                            self.addFace([kmid, knew0, kmin], faces, edges)
                        else:
                            self.addFace([knew0, kmid, kmin], faces, edges)
                        popped += [False]

                    if pos:
                        if (kmin,kmax) in edges:
                            jf = edges[(kmin, kmax)]
                            valf = np.array(val)[faces[jf]]
                            if valf.min() > 0 or valf.max() < 0:
                                jj = [j for j in faces[jf] if j not in faces[i]][0]
                                self.addFace([kmax, jj, knew0], faces, edges)
                                self.addFace([kmin, knew0, jj], faces, edges)
                                popped += [False]*2
                                popped[jf] = True
                    else:
                        if (kmax, kmin) in edges:
                            jf = edges[(kmax, kmin)]
                            valf = np.array(val)[faces[jf]]
                            if valf.min() > 0 or valf.max() < 0:
                                jj = [j for j in faces[jf] if j not in faces[i]][0]
                                self.addFace([kmin, jj, knew0], faces, edges)
                                self.addFace([kmax, knew0, jj], faces, edges)
                                popped += [False]*2
                                popped[jf] = True

                    if val[kmid] > 1e-10:
                        r1 = val[kmid] / (val[kmid]-val[kmin])
                        v1 = (1-r1) * vert[kmid] + r1 * vert[kmin]
                        knew1 = len(vert)
                        vert.append(v1)
                        w1 = (1 - r1) * weights[kmid] + r1 * weights[kmin]
                        weights.append(w1)
                        val.append(0)
                        v2 = (vert[kmid]+vert[kmax])/2
                        knew2 = len(vert)
                        vert.append(v2)
                        w2 = (weights[kmid] + weights[kmax])/2
                        weights.append(w2)
                        val.append((val[kmid]+val[kmax])/2)
                        if pos:
                            self.addFace([knew1, kmid, knew2], faces, edges)
                            self.addFace([knew2, kmax, knew0], faces, edges)
                            self.addFace([knew0, knew1, knew2], faces, edges)
                            self.addFace([knew0, kmin, knew1], faces, edges)
                            popped += [False] * 4
                            if (kmid, kmin) in edges:
                                jf = edges[(kmid, kmin)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([kmin, jj, knew1], faces, edges)
                                    self.addFace([kmid, knew1, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True
                            if (kmax, kmid) in edges:
                                jf = edges[(kmax, kmid)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([kmid, jj, knew2], faces, edges)
                                    self.addFace([kmax, knew2, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True
                        else:
                            self.addFace([kmid, knew1, knew2], faces, edges)
                            self.addFace([kmax, knew2, knew0], faces, edges)
                            self.addFace([knew1, knew0, knew2], faces, edges)
                            self.addFace([kmin, knew0, knew1], faces, edges)
                            popped += [False] * 4
                            if (kmin, kmid) in edges:
                                jf = edges[(kmin, kmid)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([jj, kmin, knew1], faces, edges)
                                    self.addFace([knew1, kmid, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True
                            if (kmid, kmax) in edges:
                                jf = edges[(kmid, kmax)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([jj, kmid, knew2], faces, edges)
                                    self.addFace([knew2, kmax, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True

                    if val[kmid] < -1e-10:
                        r1 = val[kmax] / (val[kmax]-val[kmid])
                        v1 = (1-r1) * vert[kmax] + r1 * vert[kmid]
                        knew1 = len(vert)
                        vert.append(v1)
                        w1 = (1 - r1) * weights[kmax] + r1 * weights[kmid]
                        weights.append(w1)
                        val.append(0)
                        v2 = (vert[kmid]+vert[kmin])/2
                        knew2 = len(vert)
                        vert.append(v2)
                        w2 = (weights[kmid] + weights[kmin])/2
                        weights.append(w2)
                        val.append((val[kmid]+val[kmin])/2)
                        if pos:
                            self.addFace([knew1, knew2, kmid], faces, edges)
                            self.addFace([knew2, knew0, kmin], faces, edges)
                            self.addFace([knew0, knew1, knew1], faces, edges)
                            self.addFace([knew1, kmax, knew0], faces, edges)
                            popped += [False] * 4
                            if (kmax, kmid) in edges:
                                jf = edges[(kmax, kmid)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([kmid, jj, knew1], faces, edges)
                                    self.addFace([kmax, knew1, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True
                            if (kmid, kmin) in edges:
                                jf = edges[(kmid, kmin)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([kmin, jj, knew2], faces, edges)
                                    self.addFace([kmid, knew2, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True
                        else:
                            self.addFace([knew2, knew1, kmid], faces, edges)
                            self.addFace([knew0, knew2, kmin], faces, edges)
                            self.addFace([knew1, knew0, knew1], faces, edges)
                            self.addFace([kmax, knew1, knew0], faces, edges)
                            popped += [False] * 4
                            if (kmid, kmax) in edges:
                                jf = edges[(kmid, kmax)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([jj, kmid, knew1], faces, edges)
                                    self.addFace([knew1, kmax, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True
                            if (kmin, kmid) in edges:
                                jf = edges[(kmin, kmid)]
                                valf = np.array(val)[faces[jf]]
                                if valf.min() > 0 or valf.max() < 0:
                                    jj = [j for j in faces[jf] if j not in faces[i]][0]
                                    self.addFace([jj, kmin, knew2], faces, edges)
                                    self.addFace([knew2, kmid, jj], faces, edges)
                                    popped += [False] * 2
                                    popped[jf] = True
                    popped[i] = True



    def truncate(self, ineq = (), val = None):
        res = Surface(surf=self)
        if val is None:
            val = 1e100*np.ones(res.vertices.shape[0])
        for a in ineq:
            val = np.minimum(val, (res.vertices*a[0:3]).sum(axis=1) - a[3])

        val = list(val)
        vert = list(res.vertices)
        faces = list(res.faces)
        weights = list(res.weights)
        edges = {}
        for i,f in enumerate(faces):
            edges[(f[0],f[1])] = i
            edges[(f[1], f[2])] = i
            edges[(f[2], f[0])] = i

        popped = [False]*len(faces)
        nf = len(faces)
        i = 0
        while i < nf:
            res.split_in_4(faces, vert, weights, val, popped, edges, i)
            i += 1

        npopped = 0
        nf = len(faces)
        for i in range(nf):
            if popped[i]:
                faces.pop(i-npopped)
                npopped += 1


        res = Surface(surf=(np.array(faces, dtype=int), np.array(vert)))
        res.updateWeights(np.array(weights))
        removed = res.removeDuplicates()
        val = np.array(val)[np.logical_not(removed)]
        #val = (res.vertices*a[0:3]).sum(axis=1) - a[3]
        res = res.cut(val>-1e-5)

        return res



    def cut(self, select):
        res = Surface()
        res.vertices = self.vertices[select, :]
        newindx = np.arange(self.vertices.shape[0], dtype=int)
        newindx[select] = np.arange(select.sum(), dtype=int)
        res.faces = np.zeros(self.faces.shape, dtype= int)
        res.component = np.zeros(self.faces.shape[0], dtype= int)
        j = 0
        for i in range(self.faces.shape[0]):
            if np.all(select[self.faces[i, :]]):
                res.faces[j, :] = newindx[np.copy(self.faces[i, :])]
                res.component[j] = self.component[i]
                j += 1
        res.faces = res.faces[0:j, :]
        res.component = res.component[0:j]
        res.updateWeights(self.weights[select])
        res.computeCentersAreas()
        return res

    def select_faces(self, select):
        res = Surface()
        res.faces = self.faces[select, :]
        res.component = self.component[select]
        selectv = np.zeros(self.vertices.shape[0], dtype=bool)
        for j in range(res.faces.shape[0]):
            selectv[res.faces[j,:]] = True
        res.vertices = self.vertices[selectv, :]
        newindx = - np.ones(self.vertices.shape[0], dtype=int)
        newindx[selectv] = np.arange(selectv.sum(), dtype=int)
        res.faces = newindx[res.faces]
        res.updateWeights(self.weights[selectv])
        res.computeCentersAreas()
        return res, np.nonzero(selectv)[0]

    def createFlatApproximation(self, thickness=None, M=50):
        if ~gotPygalmesh:
            raise Exception('Cannot run function without pygalmesh')

        # Compute center
        a = self.computeVertexArea()[0]
        A = a.sum()
        x0 = (self.vertices * a[:,np.newaxis]).sum(axis=0)/A

        # Inertial operator
        vm = self.vertices - x0
        J = ((vm[:, :, np.newaxis] * vm[:, np.newaxis, :])*a[:, np.newaxis, np.newaxis]).sum(axis=0)/A
        w,v = LA.eigh(J)
        emax = 3*np.sqrt(w.max())
        #J = LA.inv(J)
        #c0 = (np.dot(ftmpl.vertices, J)*ftmpl.vertices).sum()/ftmpl.vertices.shape[0]
        #dst = np.sqrt(((ftmpl.vertices - x0)**2).sum(axis=1)).mean(axis=0)
        #M = 100
        if thickness is None:
            w0 = w[0]
        else:
            w0 = (thickness/2)**2

        class Pancake(DomainBase):
            def __init__(self):
                super().__init__()

            def eval(self, x0):
                x = emax * (x0[0] - 1)
                y = emax * (x0[1] - 1)
                z = emax * (x0[2] - 1)
                I1 = np.maximum(z ** 2 / w[2] + y ** 2 / w[1] + x ** 2/w0 - 3, x ** 2/w0 - 1)
                return I1

            def get_bounding_sphere_squared_radius(self):
                return 12.0

        # d = Heart()
        d = Pancake()
        mesh = generate_surface_mesh(d, max_facet_distance=1.0, min_facet_angle=30.0,
                                               max_radius_surface_delaunay_ball=2/M)
        # [x,y,z] = np.mgrid[0:2*M+1, 0:2*M+1, 0:2*M+1] / M
        # x = emax*(x-1)
        # y = emax*(y-1)
        # z = emax*(z-1)
        # I1 = np.logical_and((z ** 2 / w[2] + y ** 2 / w[1]) < 3, x ** 2 < w0)
        # h = Surface()
        h = Surface(surf=(mesh.cells[0].data, mesh.points))
        #h.updateVertices(emax*(h.vertices-1))
        # h.Isosurface_ski(data=I1, value=.5, step=3)
        labels = np.zeros(h.vertices.shape[0], dtype=int)
        u2 = (h.vertices[:,0] - 1) * (emax)
        for j in range(labels.shape[0]):
            #u2 = x[0]  * v[0, 0] + x[1] * v[1, 0] + x[2] * v[2, 0]
            if np.fabs(u2[j]-np.sqrt(w0)) < .05:
                labels[j] = 1
            elif np.fabs(u2[j]+np.sqrt(w0)) < .05:
                labels[j] = 2

        u = np.dot(h.vertices-1, v.T)
        # u = np.zeros(hv.shape)
        # u[0,:] = x*v[0,0] + y*v[1,0] + z*v[2,0]
        # u[1,:] = x*v[0,1] + y*v[1,1] + z*v[2,1]
        # u[2,:] = x*v[0,2] + y*v[1,2] + z*v[2,2]
        #h.updateVertices(u)
        # I1 = np.logical_and((u2 ** 2 / w[2] + u1 ** 2 / w[1]) < 3, u0 ** 2 < w0)
        # h = Surface()
        # h.Isosurface_ski(data=I1, value=.5, step=3)
        logging.info(f'Vertices: {h.vertices.shape[0]}')
        #h.Isosurface(I1, value = 1, target=max(1000, self.vertices.shape[0]), scales=[1, 1, 1], smooth=0.0001)
        h.updateVertices(x0 + (u)*emax)
        # labels = np.zeros(h.vertices.shape[0], dtype=int)
        # for j in range(labels.shape[0]):
        #     x = h.vertices[j,:] - x0
        #     u2 = x[0]  * v[0, 0] + x[1] * v[1, 0] + x[2] * v[2, 0]
        #     if np.fabs(u2-np.sqrt(w0)) < emax/M:
        #         labels[j] = 1
        #     elif np.fabs(u2+np.sqrt(w0)) < emax/M:
        #         labels[j] = 2
        return h, labels, np.sqrt(w0)

    def rayTraces(self, points, ray):
        nu = np.sqrt((ray**2).sum())
        u = ray/nu
        v = np.cross(np.array([1,0,0]), u)
        nv = np.sqrt((v**2).sum())
        if nv < 1e-4:
            v = np.cross(np.array([0, 1, 0]), u)
            nv = np.sqrt((v**2).sum())
        v /= nv
        w = np.cross(v,u)
        x0 = self.vertices[self.faces[:,0], :]
        x1 = self.vertices[self.faces[:,1], :]
        x2 = self.vertices[self.faces[:,2], :]
        nf = self.faces.shape[0]
        npt = points.shape[0]
        pu = np.dot(points, u)
        pv = np.dot(points, v)
        pw = np.dot(points, w)
        c00 = np.dot(x0, u)[:, None] - pu[None, :]
        c10 = np.dot(x0, v)[:, None] - pv[None, :]
        c20 = np.dot(x0, w)[:, None] - pw[None, :]
        c01 = np.dot(x1, u)[:, None] - pu[None, :]
        c11 = np.dot(x1, v)[:, None] - pv[None, :]
        c21 = np.dot(x1, w)[:, None] - pw[None, :]
        c02 = np.dot(x2, u)[:, None] - pu[None, :]
        c12 = np.dot(x2, v)[:, None] - pv[None, :]
        c22 = np.dot(x2, w)[:, None] - pw[None, :]

        det = c00*c11*c22 - c00*c12*c21 - c01*c10*c22 - c02*c11*c20 + c01*c12*c20 + c02*c10*c21
        # com = np.zeros((3,3, nv, npt))
        # com[0, 0, ...] = c11*c22 - c12*c21
        # com[1, 1, ...] = c00*c22 - c02*c20
        # com[2, 2, ...] = c11*c00 - c10*c01
        # com[0, 1, ...] = -(c01*c22 - c21*c02)
        # com[1, 0, ...] = -(c10*c22 -c12*c20)
        # com[0, 2, ...] = c01*c12 - c02*c11
        # com[2, 0, ...] = c10*c21 - c20*c11
        # com[1, 2, ...] = -(c00*c12 - c02*c10)
        # com[2, 1, ...] = -(c00*c21 - c20*c01)
        abc = np.zeros((3, nf, npt))
        abc[0, ...] = c11*c22 - c12*c21
        abc[1, ...] = -(c10*c22 -c12*c20)
        abc[2, ...] = c10*c21 - c20*c11
        abc *= np.sign(det)
        inter = np.logical_and(abc[0,...] > 0, abc[1,...] > 0)
        inter2 = np.logical_and(abc[2,...]>0, np.fabs(det) > 1e-6)
        inter = np.logical_and(inter, inter2)
        n_inter = inter.sum(axis = 0)
        return n_inter % 2==1


    def compute3DVolumeImage(self, xmin = 0, xmax = 100, spacing = 1., origin = None):
        ln = xmax - xmin + 1
        vpoints = self.toPolyData()
        whiteImage = vtkImageData()
        whiteImage.SetSpacing(spacing ,spacing ,spacing)
        whiteImage.SetDimensions(ln ,ln ,ln)
        whiteImage.SetExtent(xmin , xmax , xmin ,xmax ,xmin,xmax)
        bounds = vpoints.GetBounds()
        if origin is None:
            origin = np.zeros(3)
            # for i in range(3):
            #     origin[i] = bounds[2*i] + spacing / 2
        whiteImage.SetOrigin(origin[0] ,origin[1] ,origin[2])
        whiteImage.AllocateScalars(VTK_UNSIGNED_CHAR ,1)
        count = whiteImage.GetNumberOfPoints()
        inval = 255
        outval = 0
        for i in range(count):
            whiteImage.GetPointData().GetScalars().SetTuple1(i ,inval)

        orig2 = np.zeros(3)
        for i in range(3):
             orig2[i] = bounds[2*i]

        pol2stenc = vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(vpoints)
        pol2stenc.SetOutputOrigin((orig2[0], orig2[1], orig2[2]))
        pol2stenc.SetOutputSpacing(spacing, spacing, spacing)
        pol2stenc.SetOutputWholeExtent(whiteImage.GetExtent())
        pol2stenc.Update()

        img2stenc = vtkImageStencil()
        img2stenc.SetInputData(whiteImage)
        img2stenc.SetStencilConnection(pol2stenc.GetOutputPort())
        img2stenc.ReverseStencilOff()
        img2stenc.SetBackgroundValue(outval)
        img2stenc.Update()

        result = img2stenc.GetOutput()

        img = vtk_to_numpy(result.GetPointData().GetScalars())
        # dims = result.GetDimensions()
        #
        # img = img.reshape(dims[2], dims[1], dims[0])
        # img = img.transpose(2, 1, 0)
        img = np.reshape(img, (ln,ln,ln)).astype(float)
        # img = np.zeros((ln,ln,ln))
        # ii = 0
        # for j in range(ln):
        #     for i in range(ln):
        #         for k in range(ln):
        #             img[j,i,k] = result.GetPointData().GetScalars().GetTuple1(ii)
        #             ii += 1

        return img, origin, orig2

    def plot(self, fig=1, ec = 'b', fc = 'r', al=.5, lw=1, azim = 100, elev = 45, setLim=True, addTo = False):
        f = plt.figure(fig)
        plt.clf()
        ax = Axes3D(f, azim=azim, elev=elev)
        self.addToPlot(ax, ec=ec, fc=fc, al=al, setLim=setLim)
        plt.axis('off')



    def addToPlot(self, ax, ec = 'b', fc = 'r', al=.5, lw=1, setLim=True):
        x = self.vertices[self.faces[:,0],:]
        y = self.vertices[self.faces[:,1],:]
        z = self.vertices[self.faces[:,2],:]
        a = np.concatenate([x,y,z], axis=1)
        poly = [ [a[i,j*3:j*3+3] for j in range(3)] for i in range(a.shape[0])]
        tri = Poly3DCollection(poly, alpha=al, linewidths=lw)
        tri.set_edgecolor(ec)
        ls = LightSource(90, 45)
        fc = np.array(colors.to_rgb(fc))
        normals = self.surfel/np.sqrt((self.surfel**2).sum(axis=1))[:,None]
        shade = ls.shade_normals(normals, fraction=1.0)
        fc = fc[None, :] * shade[:, None]
        tri.set_facecolor(fc)
        ax.add_collection3d(tri)
        xlim = [self.vertices[:,0].min(),self.vertices[:,0].max()]
        ylim = [self.vertices[:,1].min(),self.vertices[:,1].max()]
        zlim = [self.vertices[:,2].min(),self.vertices[:,2].max()]
        if setLim:
            ax.set_xlim(xlim[0], xlim[1])
            ax.set_ylim(ylim[0], ylim[1])
            ax.set_zlim(zlim[0], zlim[1])
        return [xlim, ylim, zlim]


    def distToTriangle(self, x0, T):
        n0 = (x0**2).sum()
        S = np.dot(T, T.T)
        SI = LA.inv(S)
        y0 = np.dot(S.T,x0)
        v = SI.sum(axis = 1)
        u = np.dot(SI,y0)
        v = SI.sum(axis=1)
        L = (1 - u.sum())/v.sum()
        u += L*v
        if min(u) >= 0:
            dist = n0 - (y0*u).sum() + L
            return dist

        dx0 = ((x0-T[0,:])**2).sum()
        dx1 = ((x0 - T[1, :]) ** 2).sum()
        dx2 = ((x0 - T[2, :]) ** 2).sum()
        d01 = S[0,0] - 2*S[0,1] + S[1,1]
        d02 = S[0,0] - 2*S[0,2] + S[2,2]
        d12 = S[1,1] - 2*S[1,2] + S[2,2]
        s0 = ((x0 - T[0,:])*(T[1,:]-T[0,:])).sum()
        s1 = ((x0 - T[1,:])*(T[2,:]-T[1,:])).sum()
        s2 = ((x0 - T[2,:])*(T[0,:]-T[2,:])).sum()
        dist = -1
        if s0 >= 0 and s0 <= d01:
            dist = dx0 - s0**2/d01
            if s1 >= 0 and s1 <= d12:
                dist = min(dist, dx1 - s1 ** 2 / d12)
                if s2 >= 0 and s2 <= d02:
                    dist = min(dist, dx2 - s2 ** 2 / d02)
        elif s1 >= 0 and s1 <= d12:
            dist = dx1 - s1 ** 2 / d12
            if s2 >= 0 and s2 <= d02:
                dist = min(dist, dx2 - s2 ** 2 / d02)
        elif s2 >= 0 and s2 <= d02:
            dist = dx2 - s2 ** 2 / d02

        if dist > -0.5:
            return dist

        return min(dx0, dx1, dx2)

    def laplacianMatrix(self):
        F = self.faces
        V = self.vertices
        nf = F.shape[0]
        nv = V.shape[0]

        AV, AF = self.computeVertexArea()

        # compute edges and detect boundary
        #edm = sp.lil_matrix((nv,nv))
        edm = -np.ones((nv,nv), dtype=int)
        E = np.zeros((3*nf, 2), dtype=int)
        j = 0
        for k in range(nf):
            if edm[F[k,0], F[k,1]]== -1:
                edm[F[k,0], F[k,1]] = j
                edm[F[k,1], F[k,0]] = j
                E[j, :] = [F[k,0], F[k,1]]
                j = j+1
            if (edm[F[k,1], F[k,2]]== -1):
                edm[F[k,1], F[k,2]] = j
                edm[F[k,2], F[k,1]] = j
                E[j, :] = [F[k,1], F[k,2]]
                j = j+1
            if (edm[F[k,0], F[k,2]]== -1):
                edm[F[k,2], F[k,0]] = j
                edm[F[k,0], F[k,2]] = j
                E[j, :] = [F[k,2], F[k,0]]
                j = j+1
        E = E[0:j, :]
        
        edgeFace = np.zeros([j, nf], dtype=int)
        ne = j
        #print E
        for k in range(nf):
            edgeFace[edm[F[k,0], F[k,1]], k] = 1 
            edgeFace[edm[F[k,1], F[k,2]], k] = 1 
            edgeFace[edm[F[k,2], F[k,0]], k] = 1 
    
        bEdge = np.zeros([ne, 1])
        bVert = np.zeros([nv, 1])
        edgeAngles = np.zeros([ne, 2])
        for k in range(ne):
            I = np.flatnonzero(edgeFace[k, :])
            #print 'I=', I, F[I, :], E.shape
            #print 'E[k, :]=', k, E[k, :]
            #print k, edgeFace[k, :]
            for u in range(len(I)):
                f = I[u]
                i1l = np.flatnonzero(F[f, :] == E[k,0])
                i2l = np.flatnonzero(F[f, :] == E[k,1])
                #print f, F[f, :]
                #print i1l, i2l
                i1 = i1l[0]
                i2 = i2l[0]
                s = i1+i2
                if s == 1:
                    i3 = 2
                elif s==2:
                    i3 = 1
                elif s==3:
                    i3 = 0
                x1 = V[F[f,i1], :] - V[F[f,i3], :]
                x2 = V[F[f,i2], :] - V[F[f,i3], :]
                a = (np.cross(x1, x2) * np.cross(V[F[f,1], :] - V[F[f,0], :], V[F[f, 2], :] - V[F[f, 0], :])).sum()
                b = (x1*x2).sum()
                if (a  > 0):
                    edgeAngles[k, u] = b/np.sqrt(a)
                else:
                    edgeAngles[k, u] = b/np.sqrt(-a)
            if (len(I) == 1):
                # boundary edge
                bEdge[k] = 1
                bVert[E[k,0]] = 1
                bVert[E[k,1]] = 1
                edgeAngles[k,1] = 0 
        

        # Compute Laplacian matrix
        L = np.zeros([nv, nv])

        for k in range(ne):
            L[E[k,0], E[k,1]] = (edgeAngles[k,0] + edgeAngles[k,1]) /2
            L[E[k,1], E[k,0]] = L[E[k,0], E[k,1]]

        for k in range(nv):
            L[k,k] = - L[k, :].sum()

        A = np.zeros([nv, nv])
        for k in range(nv):
            A[k, k] = AV[k]

        return L,A

    def graphLaplacianMatrix(self):
        F = self.faces
        V = self.vertices
        nf = F.shape[0]
        nv = V.shape[0]

        # compute edges and detect boundary
        #edm = sp.lil_matrix((nv,nv))
        edm = -np.ones([nv,nv])
        E = np.zeros([3*nf, 2])
        j = 0
        for k in range(nf):
            if (edm[F[k,0], F[k,1]]== -1):
                edm[F[k,0], F[k,1]] = j
                edm[F[k,1], F[k,0]] = j
                E[j, :] = [F[k,0], F[k,1]]
                j = j+1
            if (edm[F[k,1], F[k,2]]== -1):
                edm[F[k,1], F[k,2]] = j
                edm[F[k,2], F[k,1]] = j
                E[j, :] = [F[k,1], F[k,2]]
                j = j+1
            if (edm[F[k,0], F[k,2]]== -1):
                edm[F[k,2], F[k,0]] = j
                edm[F[k,0], F[k,2]] = j
                E[j, :] = [F[k,2], F[k,0]]
                j = j+1
        E = E[0:j, :]
        
        ne = j
        #print E

        # Compute Laplacian matrix
        L = np.zeros([nv, nv])

        for k in range(ne):
            L[E[k,0], E[k,1]] = 1
            L[E[k,1], E[k,0]] = 1

        for k in range(nv):
            L[k,k] = - L[k, :].sum()

        return L


    def laplacianSegmentation(self, k):
        (L, AA) =  self.laplacianMatrix()
        #print (L.shape[0]-k-1, L.shape[0]-2)
        (D, y) = spLA.eigh(L, AA, eigvals= (L.shape[0]-k, L.shape[0]-1))
        #V = real(V) ;
        #print D
        N = y.shape[0]
        d = y.shape[1]
        I = np.argsort(y.sum(axis=1))
        I0 =np.floor((N-1)*np.linspace(0, 1, num=k)).astype(int)
        #print y.shape, L.shape, N, k, d
        C = y[I0, :].copy()

        eps = 1e-20
        Cold = C.copy()
        u = ((C.reshape([k,1,d]) - y.reshape([1,N,d]))**2).sum(axis=2)
        T = u.min(axis=0).sum()/(N)
        #print T
        j=0
        while j< 5000:
            u0 = u - u.min(axis=0).reshape([1, N])
            w = np.exp(-u0/T)
            w = w / (eps + w.sum(axis=0).reshape([1,N]))
            #print w.min(), w.max()
            cost = (u*w).sum() + T*(w*np.log(w+eps)).sum()
            C = np.dot(w, y) / (eps + w.sum(axis=1).reshape([k,1]))
            #print j, 'cost0 ', cost

            u = ((C.reshape([k,1,d]) - y.reshape([1,N,d]))**2).sum(axis=2)
            cost = (u*w).sum() + T*(w*np.log(w+eps)).sum()
            err = np.sqrt(((C-Cold)**2).sum(axis=1)).sum()
            #print j, 'cost ', cost, err, T
            if ( j>100) & (err < 1e-4 ):
                break
            j = j+1
            Cold = C.copy()
            T = T*0.99

            #print k, d, C.shape
        dst = ((C.reshape([k,1,d]) - y.reshape([1,N,d]))**2).sum(axis=2)
        md = dst.min(axis=0)
        idx = np.zeros(N).astype(int)
        for j in range(N):
            I = np.flatnonzero(dst[:,j] < md[j] + 1e-10) 
            idx[j] = I[0]
        I = -np.ones(k).astype(int)
        kk=0
        for j in range(k):
            if True in (idx==j):
                I[j] = kk
                kk += 1
        idx = I[idx]
        if idx.max() < (k-1):
            logging.info('Warning: kmeans convergence with %d clusters instead of %d' %(idx.max(), k))
            #ml = w.sum(axis=1)/N
        nc = idx.max()+1
        C = np.zeros([nc, self.vertices.shape[1]])
        a, foo = self.computeVertexArea()
        for k in range(nc):
            I = np.flatnonzero(idx==k)
            nI = len(I)
            #print a.shape, nI
            aI = a[I]
            ak = aI.sum()
            C[k, :] = (self.vertices[I, :]*aI).sum(axis=0)/ak
        
        

        return idx, C



    # Computes surface volume
    def surfVolume(self):
        f = self.faces
        v = self.vertices
        z = 0
        for c in f:
            z += np.linalg.det(v[c[:], :])/6
        return z

    # Computes surface area
    def surfArea(self):
        return np.sqrt((self.surfel**2).sum(axis=1)).sum()

    # Reads from .off file
    def readOFF(self, offfile):
        with open(offfile,'r') as f:
            ln0 = readskip(f,'#')
            ln = ln0.split()
            if ln[0].lower() != 'off':
                logging.info('Not OFF format')
                return
            ln = readskip(f,'#').split()
            # read header
            npoints = int(ln[0])  # number of vertices
            nfaces = int(ln[1]) # number of faces
                                #print ln, npoints, nfaces
                        #fscanf(fbyu,'%d',1);		% number of edges
                        #%ntest = fscanf(fbyu,'%d',1);		% number of edges
            # read data
            self.vertices = np.empty([npoints, 3])
            for k in range(npoints):
                ln = readskip(f,'#').split()
                self.vertices[k, 0] = float(ln[0]) 
                self.vertices[k, 1] = float(ln[1]) 
                self.vertices[k, 2] = float(ln[2])

            self.weights = np.ones(self.vertices.shape[0])
            self.faces = np.int_(np.empty([nfaces, 3]))
            for k in range(nfaces):
                ln = readskip(f,'#').split()
                if (int(ln[0]) != 3):
                    logging.info('Reading only triangulated surfaces')
                    return
                self.faces[k, 0] = int(ln[1]) 
                self.faces[k, 1] = int(ln[2]) 
                self.faces[k, 2] = int(ln[3])

        self.computeCentersAreas()
        self.component = np.zeros(self.faces.shape[0], dtype=int)


        
    # Reads from .byu file
    def readbyu(self, byufile):
        with open(byufile,'r') as fbyu:
            ln0 = fbyu.readline()
            ln = ln0.split()
            # read header
            ncomponents = int(ln[0])	# number of components
            npoints = int(ln[1])  # number of vertices
            nfaces = int(ln[2]) # number of faces
                        #fscanf(fbyu,'%d',1);		% number of edges
                        #%ntest = fscanf(fbyu,'%d',1);		% number of edges
            for k in range(ncomponents):
                fbyu.readline() # components (ignored)
            # read data
            self.vertices = np.empty([npoints, 3])
            k=-1
            while k < npoints-1:
                ln = fbyu.readline().split()
                k=k+1
                self.vertices[k, 0] = float(ln[0]) 
                self.vertices[k, 1] = float(ln[1]) 
                self.vertices[k, 2] = float(ln[2])
                if len(ln) > 3:
                    k=k+1
                    self.vertices[k, 0] = float(ln[3])
                    self.vertices[k, 1] = float(ln[4]) 
                    self.vertices[k, 2] = float(ln[5])

            self.weights = np.ones(self.vertices.shape[0])
            self.faces = np.empty([nfaces, 3])
            ln = fbyu.readline().split()
            kf = 0
            j = 0
            while ln:
                if kf >= nfaces:
                    break
                #print nfaces, kf, ln
                for s in ln:
                    self.faces[kf,j] = int(np.fabs(int(s)))
                    j = j+1
                    if j == 3:
                        kf=kf+1
                        j=0
                ln = fbyu.readline().split()
        self.faces = np.int_(self.faces) - 1
        self.computeCentersAreas()
        # xDef1 = self.vertices[self.faces[:, 0], :]
        # xDef2 = self.vertices[self.faces[:, 1], :]
        # xDef3 = self.vertices[self.faces[:, 2], :]
        # self.centers = (xDef1 + xDef2 + xDef3) / 3
        # self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)/2
        self.component = np.zeros(self.faces.shape[0], dtype=int)

    #Saves in .byu format
    def savebyu(self, byufile):
        #FV = readbyu(byufile)
        #reads from a .byu file into matlab's face vertex structure FV

        with open(byufile,'w') as fbyu:
            # copy header
            ncomponents = 1	    # number of components
            npoints = self.vertices.shape[0] # number of vertices
            nfaces = self.faces.shape[0]		# number of faces
            nedges = 3*nfaces		# number of edges

            str = '{0: d} {1: d} {2: d} {3: d} 0\n'.format(ncomponents, npoints, nfaces,nedges)
            fbyu.write(str) 
            str = '1 {0: d}\n'.format(nfaces)
            fbyu.write(str) 


            k=-1
            while k < (npoints-1):
                k=k+1 
                str = '{0: f} {1: f} {2: f} '.format(self.vertices[k, 0], self.vertices[k, 1], self.vertices[k, 2])
                fbyu.write(str) 
                if k < (npoints-1):
                    k=k+1
                    str = '{0: f} {1: f} {2: f}\n'.format(self.vertices[k, 0], self.vertices[k, 1], self.vertices[k, 2])
                    fbyu.write(str) 
                else:
                    fbyu.write('\n')

            j = 0 
            for k in range(nfaces):
                for kk in (0,1):
                    fbyu.write('{0: d} '.format(self.faces[k,kk]+1))
                    j=j+1
                    if j==16:
                        fbyu.write('\n')
                        j=0

                fbyu.write('{0: d} '.format(-self.faces[k,2]-1))
                j=j+1
                if j==16:
                    fbyu.write('\n')
                    j=0

    def saveVTK(self, fileName, scalars = None, normals = None, tensors=None, scal_name='scalars',
                vectors=None, vect_name='vectors'):
        vf = vtkFields()
        #print scalars
        vf.scalars.append('weights')
        vf.scalars.append(self.weights)
        if not (scalars is None):
            vf.scalars.append(scal_name)
            vf.scalars.append(scalars)
        if not (vectors is None):
            vf.vectors.append(vect_name)
            vf.vectors.append(vectors)
        if not (normals is None):
            vf.normals.append('normals')
            vf.normals.append(normals)
        if not (tensors is None):
            vf.tensors.append('tensors')
            vf.tensors.append(tensors)
        self.saveVTK2(fileName, vf)

    # Saves in .vtk format
    def saveVTK2(self, fileName, vtkFields = None):
        F = self.faces
        V = self.vertices

        with open(fileName, 'w') as fvtkout:
            fvtkout.write('# vtk DataFile Version 3.0\nSurface Data\nASCII\nDATASET POLYDATA\n') 
            fvtkout.write('\nPOINTS {0: d} float'.format(V.shape[0]))
            for ll in range(V.shape[0]):
                fvtkout.write('\n{0: f} {1: f} {2: f}'.format(V[ll,0], V[ll,1], V[ll,2]))
            fvtkout.write('\nPOLYGONS {0:d} {1:d}'.format(F.shape[0], 4*F.shape[0]))
            for ll in range(F.shape[0]):
                fvtkout.write('\n3 {0: d} {1: d} {2: d}'.format(F[ll,0], F[ll,1], F[ll,2]))

            fvtkout.write(('\nCELL_DATA {0: d}').format(F.shape[0]))
            fvtkout.write('\nSCALARS labels int 1\nLOOKUP_TABLE default')
            for ll in range(F.shape[0]):
                fvtkout.write('\n {0:d}'.format(self.component[ll]))

            if vtkFields is not  None:
                wrote_pd_hdr = False
                if len(vtkFields.scalars) > 0:
                    if not wrote_pd_hdr:
                        fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
                        wrote_pd_hdr = True
                    nf = len(vtkFields.scalars)//2
                    for k in range(nf):
                        fvtkout.write('\nSCALARS '+ vtkFields.scalars[2*k] +' float 1\nLOOKUP_TABLE default')
                        for ll in range(V.shape[0]):
                            #print scalars[ll]
                            fvtkout.write('\n {0: .5f}'.format(vtkFields.scalars[2*k+1][ll]))
                if len(vtkFields.vectors) > 0:
                    if not wrote_pd_hdr:
                        fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
                        wrote_pd_hdr = True
                    nf = len(vtkFields.vectors)//2
                    for k in range(nf):
                        fvtkout.write('\nVECTORS '+ vtkFields.vectors[2*k] +' float')
                        vectors = vtkFields.vectors[2*k+1]
                        for ll in range(V.shape[0]):
                            fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(vectors[ll, 0], vectors[ll, 1], vectors[ll, 2]))
                if len(vtkFields.normals) > 0:
                    if not wrote_pd_hdr:
                        fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
                        wrote_pd_hdr = True
                    nf = len(vtkFields.normals)//2
                    for k in range(nf):
                        fvtkout.write('\nNORMALS '+ vtkFields.normals[2*k] +' float')
                        vectors = vtkFields.normals[2*k+1]
                        for ll in range(V.shape[0]):
                            fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(vectors[ll, 0], vectors[ll, 1], vectors[ll, 2]))
                if len(vtkFields.tensors) > 0:
                    if not wrote_pd_hdr:
                        fvtkout.write(('\nPOINT_DATA {0: d}').format(V.shape[0]))
                        wrote_pd_hdr = True
                    nf = len(vtkFields.tensors)//2
                    for k in range(nf):
                        fvtkout.write('\nTENSORS '+ vtkFields.tensors[2*k] +' float')
                        tensors = vtkFields.tensors[2*k+1]
                        for ll in range(V.shape[0]):
                            for kk in range(2):
                                fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(tensors[ll, kk, 0], tensors[ll, kk, 1], tensors[ll, kk, 2]))
                fvtkout.write('\n')


    # Reads .vtk file
    def readVTK(self, fileName):
        if gotVTK:
            u = vtkPolyDataReader()
            u.SetFileName(fileName)
            u.Update()
            v = u.GetOutput()
            w = v.GetPointData().GetScalars('weights')
            lab = v.GetCellData().GetScalars('labels')
            #print v
            npoints = int(v.GetNumberOfPoints())
            nfaces = int(v.GetNumberOfPolys())
            V = np.zeros([npoints, 3])
            for kk in range(npoints):
                V[kk, :] = np.array(v.GetPoint(kk))

            if lab:
                Lab = np.zeros(nfaces, dtype=int)
                for kk in range(nfaces):
                    Lab[kk] = lab.GetTuple(kk)[0]
            else:
                Lab = np.zeros(nfaces, dtype=int)

            if w:
                W = np.zeros(npoints)
                for kk in range(npoints):
                    W[kk] = w.GetTuple(kk)[0]
            else:
                W = np.ones(npoints)

            F = np.zeros([nfaces, 3])
            for kk in range(nfaces):
                c = v.GetCell(kk)
                for ll in range(3):
                    F[kk,ll] = c.GetPointId(ll)
            
            self.vertices = V
            self.weights = W
            self.faces = np.int_(F)
            self.computeCentersAreas()
            # xDef1 = self.vertices[self.faces[:, 0], :]
            # xDef2 = self.vertices[self.faces[:, 1], :]
            # xDef3 = self.vertices[self.faces[:, 2], :]
            # self.centers = (xDef1 + xDef2 + xDef3) / 3
            # self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)/2
            self.component = Lab #np.zeros(self.faces.shape[0], dtype=int)
        else:
            raise Exception('Cannot run readVTK without VTK')
    
    # Reads .vtk file
    def readFromImage(self, fileName, with_vfld = True):
        self.img = diffeo.gridScalars(fileName=fileName)
        self.img.data /= self.img.data.max() + 1e-10
        self.Isosurface(self.img.data, smooth=0.001)
        if with_vfld:
            #smoothData = cg.linearcg(lambda x: -diffeo.laplacian(x), -self.img.data, iterMax=500)
            smoothData = sp.sparse.linalg.cg(lambda x: -diffeo.laplacian(x), -self.img.data, maxiter=500)
            self.vfld = diffeo.gradient(smoothData)
    
    # Reads .vtk file
    def initFromImage(self, img):
        self.img = diffeo.gridScalars(data=img)
        self.img.data /= self.img.data.max() + 1e-10
        self.Isosurface(self.img.data)
        smoothData = sp.sparse.linalg.cg(lambda x: -diffeo.laplacian(x) ,-self.img.data ,maxiter=500)
        #smoothData = cg.linearcg(lambda x: -diffeo.laplacian(x), -self.img.data, iterMax=500)
        self.vfld = diffeo.gradient(smoothData)
    
    # Reads .obj file
    def readOBJ(self, fileName):
        if gotVTK:
            u = vtkOBJReader()
            u.SetFileName(fileName)
            u.Update()
            v = u.GetOutput()
            #print v
            npoints = int(v.GetNumberOfPoints())
            nfaces = int(v.GetNumberOfPolys())
            V = np.zeros([npoints, 3])
            for kk in range(npoints):
                V[kk, :] = np.array(v.GetPoint(kk))
    
            F = np.zeros([nfaces, 3])
            for kk in range(nfaces):
                c = v.GetCell(kk)
                for ll in range(3):
                    F[kk,ll] = c.GetPointId(ll)
            
            self.vertices = V
            self.weights = np.ones(self.vertices.shape[0])
            self.faces = np.int_(F)
            self.computeCentersAreas()
            # xDef1 = self.vertices[self.faces[:, 0], :]
            # xDef2 = self.vertices[self.faces[:, 1], :]
            # xDef3 = self.vertices[self.faces[:, 2], :]
            # self.centers = (xDef1 + xDef2 + xDef3) / 3
            # self.surfel =  np.cross(xDef2-xDef1, xDef3-xDef1)/2
            self.component = np.zeros(self.faces.shape[0], dtype=int)
        else:
            raise Exception('Cannot run readOBJ without VTK')

    # Reads .stl file
    def readSTL(self, fileName):
        if gotVTK:
            u = vtkSTLReader()
            u.SetFileName(fileName)
            u.Update()
            v = u.GetOutput()
            npoints = int(v.GetNumberOfPoints())
            nfaces = int(v.GetNumberOfPolys())
            V = np.zeros([npoints, 3])
            for kk in range(npoints):
                V[kk, :] = np.array(v.GetPoint(kk))
            F = np.zeros([nfaces, 3])
            for kk in range(nfaces):
                c = v.GetCell(kk)
                for ll in range(3):
                    F[kk, ll] = c.GetPointId(ll)

            self.vertices = V
            self.weights = np.ones(self.vertices.shape[0])
            self.faces = np.int_(F)
            # xDef1 = self.vertices[self.faces[:, 0], :]
            # xDef2 = self.vertices[self.faces[:, 1], :]
            # xDef3 = self.vertices[self.faces[:, 2], :]
            # self.centers = (xDef1 + xDef2 + xDef3) / 3
            # self.surfel = np.cross(xDef2 - xDef1, xDef3 - xDef1) / 2
            self.computeCentersAreas()
            self.component = np.zeros(self.faces.shape[0], dtype=int)
            print(f'STL file: {self.vertices.shape[0]} vertices, {self.faces.shape[0]} faces')
        else:
            raise Exception('Cannot run readSTL without VTK')

    def concatenate(self, fvl):
        nv = 0
        nf = 0
        for fv in fvl:
            nv += fv.vertices.shape[0]
            nf += fv.faces.shape[0]
        self.vertices = np.zeros([nv,3])
        self.weights = np.zeros(nv)
        self.faces = np.zeros([nf,3], dtype='int')
        self.component = np.zeros(nf, dtype='int')

        nv0 = 0
        nf0 = 0
        c = 0
        for fv in fvl:
            nv = nv0 + fv.vertices.shape[0]
            nf = nf0 + fv.faces.shape[0]
            self.vertices[nv0:nv, :] = fv.vertices
            self.weights[nv0:nv] = fv.weights
            self.faces[nf0:nf, :] = fv.faces + nv0
            self.component[nf0:nf] = fv.component + c
            nv0 = nv
            nf0 = nf
            c = self.component[:nf].max() + 1
        self.computeCentersAreas()

    def connected_components(self, split=False):
        self.getEdges()
        N = self.edges.max()+1
        A = csr_matrix((np.ones(self.edges.shape[0]), (self.edges[:,0], self.edges[:,1])), shape=(N,N))
        nc, labels = connected_components(A, directed=False)
        self.component = labels[self.faces[:,0]]
        logging.info(f'Found {nc} connected components')
        if split:
            return self.split_components(labels)

    def split_components(self, labels):
        nc = labels.max() + 1
        res = []
        for i in range(nc):
            J = np.nonzero(labels == i)[0]
            V = self.vertices[J,:]
            w = self.weights[J]
            newI = -np.ones(self.vertices.shape[0], dtype=int)
            newI[J] = np.arange(0, J.shape[0])
            F = newI[self.faces]
            I = np.amax(F, axis=1) >= 0
            F = F[I, :]
            res.append(Surface(surf=(F,V), weights=w))
        return res

    def extract_components(self, comp=None, comp_info=None):
        #labels = np.zeros(self.vertices.shape[0], dtype = int)
        #for j in range(self.faces.shape[1]):
         #   labels[self.faces[:,i]] = self.component

        #print('extracting components')
        if comp_info is not None:
            F, J, E, FE, EF = comp_info
        elif comp is not None:
            if self.edges is None:
                F, J, E, FE, EF = extract_components_(comp, self.vertices.shape[0], self.faces, self.component,
                                                            edge_info = None)
            else:
                F, J, E, FE, EF = extract_components_(comp, self.vertices.shape[0], self.faces, self.component,
                                                            edge_info = (self.edges, self.faceEdges, self.edgeFaces))
        else:
            res = Surface
            J = np.zeros(self.vertices.shape[0], dtype=bool)
            return res, J

        V = self.vertices[J,:]
        w = self.weights[J]
        res = Surface(surf=(F,V), weights=w)
        if self.edges is not None:
            res.edges = E
            res.faceEdges = FE
            res.edgeFaces = EF

        #print(f'End of extraction: vertices: {res.vertices.shape[0]} faces: {res.faces.shape[0]}')
        return res, J

    def normGrad(self, phi):
        v1 = self.vertices[self.faces[:,0],:]
        v2 = self.vertices[self.faces[:,1],:]
        v3 = self.vertices[self.faces[:,2],:]
        l1 = ((v2-v3)**2).sum(axis=1)
        l2 = ((v1-v3)**2).sum(axis=1)
        l3 = ((v1-v2)**2).sum(axis=1)
        phi1 = phi[self.faces[:,0],:]
        phi2 = phi[self.faces[:,1],:]
        phi3 = phi[self.faces[:,2],:]
        a = 4*np.sqrt((self.surfel**2).sum(axis=1))
        u = l1*((phi2-phi1)*(phi3-phi1)).sum(axis=1) + l2*((phi3-phi2)*(phi1-phi2)).sum(axis=1) + l3*((phi1-phi3)*(phi2-phi3)).sum(axis=1)
        res = (u/a).sum()
        return res
    
    def laplacian(self, phi, weighted=False):
        res = np.zeros(phi.shape)
        v1 = self.vertices[self.faces[:,0],:]
        v2 = self.vertices[self.faces[:,1],:]
        v3 = self.vertices[self.faces[:,2],:]
        l1 = (((v2-v3)**2).sum(axis=1))[...,np.newaxis]
        l2 = (((v1-v3)**2).sum(axis=1))[...,np.newaxis]
        l3 = (((v1-v2)**2).sum(axis=1))[...,np.newaxis]
        phi1 = phi[self.faces[:,0],:]
        phi2 = phi[self.faces[:,1],:]
        phi3 = phi[self.faces[:,2],:]
        a = 8*(np.sqrt((self.surfel**2).sum(axis=1)))[...,np.newaxis]
        r1 = (l1 * (phi2 + phi3-2*phi1) + (l2-l3) * (phi2-phi3))/a
        r2 = (l2 * (phi1 + phi3-2*phi2) + (l1-l3) * (phi1-phi3))/a
        r3 = (l3 * (phi1 + phi2-2*phi3) + (l2-l1) * (phi2-phi1))/a
        for k,f in enumerate(self.faces):
            res[f[0],:] += r1[k,:]
            res[f[1],:] += r2[k,:]
            res[f[2],:] += r3[k,:]
        if weighted:
            av = self.computeVertexArea()
            return res/av[0]
        else:
            return res

    def diffNormGrad(self, phi):
        res = np.zeros((self.vertices.shape[0],phi.shape[1]))
        v1 = self.vertices[self.faces[:,0],:]
        v2 = self.vertices[self.faces[:,1],:]
        v3 = self.vertices[self.faces[:,2],:]
        l1 = (((v2-v3)**2).sum(axis=1))
        l2 = (((v1-v3)**2).sum(axis=1))
        l3 = (((v1-v2)**2).sum(axis=1))
        phi1 = phi[self.faces[:,0],:]
        phi2 = phi[self.faces[:,1],:]
        phi3 = phi[self.faces[:,2],:]
        #a = ((self.surfel**2).sum(axis=1))
        a = 2*np.sqrt((self.surfel**2).sum(axis=1))
        u = l1*((phi2-phi1)*(phi3-phi1)).sum(axis=1) + l2*((phi3-phi2)*(phi1-phi2)).sum(axis=1) + l3*((phi1-phi3)*(phi2-phi3)).sum(axis=1)
        #u = (2*u/a**2)[...,np.newaxis]
        u = (u/a**3)[...,np.newaxis]
        a = a[...,np.newaxis]
        
        r1 = - u * np.cross(v2-v3,self.surfel) + 2*((v1-v3) *(((phi3-phi2)*(phi1-phi2)).sum(axis=1))[:,np.newaxis]
            + (v1-v2)*(((phi1-phi3)*(phi2-phi3)).sum(axis=1)[:,np.newaxis]))/a
        r2 = - u * np.cross(v3-v1,self.surfel) + 2*((v2-v1) *(((phi1-phi3)*(phi2-phi3)).sum(axis=1))[:,np.newaxis]
            + (v2-v3)*(((phi2-phi1)*(phi3-phi1)).sum(axis=1))[:,np.newaxis])/a
        r3 = - u * np.cross(v1-v2,self.surfel) + 2*((v3-v2) *(((phi2-phi1)*(phi3-phi1)).sum(axis=1))[:,np.newaxis]
            + (v3-v1)*(((phi3-phi2)*(phi1-phi2)).sum(axis=1)[:,np.newaxis]))/a
        for k,f in enumerate(self.faces):
            res[f[0],:] += r1[k,:]
            res[f[1],:] += r2[k,:]
            res[f[2],:] += r3[k,:]
        return res/2
    
    def meanCurvatureVector(self):
        res = np.zeros(self.vertices.shape)
        v1 = self.vertices[self.faces[:,0],:]
        v2 = self.vertices[self.faces[:,1],:]
        v3 = self.vertices[self.faces[:,2],:]
        a = np.sqrt(((self.surfel**2).sum(axis=1)))
        a = a[...,np.newaxis]
        
        r1 = - np.cross(v2-v3,self.surfel)/a
        r2 = - np.cross(v3-v1,self.surfel)/a
        r3 = - np.cross(v1-v2,self.surfel)/a
        for k,f in enumerate(self.faces):
            res[f[0],:] += r1[k,:]
            res[f[1],:] += r2[k,:]
            res[f[2],:] += r3[k,:]
        return res


# Reads several .byu files
def readMultipleByu(regexp, Nmax = 0):
    files = glob.glob(regexp)
    if Nmax > 0:
        nm = min(Nmax, len(files))
    else:
        nm = len(files)
    fv1 = []
    for k in range(nm):
        fv1.append(Surface(files[k]))
    return fv1

# saves time dependent surfaces (fixed topology)
def saveEvolution(fileName, fv0, xt):
    fv = Surface(fv0)
    for k in range(xt.shape[0]):
        fv.vertices = np.squeeze(xt[k, :, :])
        fv.savebyu(fileName+'{0: 02d}'.format(k)+'.byu')







def readskip(f, c):
    ln0 = f.readline()
    #print ln0
    while (len(ln0) > 0 and ln0[0] == c):
        ln0 = f.readline()
    return ln0

# class MultiSurface:
#     def __init__(self, pattern):
#         self.surf = []
#         files = glob.glob(pattern)
#         for f in files:
#             self.surf.append(Surface(filename=f))