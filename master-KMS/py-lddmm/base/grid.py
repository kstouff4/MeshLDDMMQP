import numpy as np
import scipy as sp
import scipy.interpolate as interp
try:
    from vtk import vtkPolyDataReader, VTK_ERROR
    gotVTK = True
except ImportError:
    print('could not import VTK functions')
    gotVTK = False
import os

# General surface class
class Grid:
    def __init__(self, gridPoints=None):
        if gridPoints == None:
            self.vertices = []
            self.faces = []
            self.nrow = 0
            self.ncol = 0
        else:
            x = gridPoints[0]
            y = gridPoints[1]
            #print x.shape, y.shape, x.size
            self.vertices = np.zeros([x.size, 2])
            self.vertices[:, 0] = x.flatten()
            self.vertices[:, 1] = y.flatten()
            n = x.shape[0]
            m = x.shape[1]
            self.nrow = n
            self.ncol = m
            self.faces = np.zeros([n*(m-1)+m*(n-1), 2])
            j = 0 
            for k in range(n):
                for l in range(m-1):
                    self.faces[j,:] = (k*m+l,k*m+l+1)
                    j += 1
            for k in range(n-1):
                for l in range(m):
                    self.faces[j,:] = (k*m+l,(k+1)*m+l)
                    j += 1
            self.faces = np.int_(self.faces)

    def copy(self, src):
        self.vertices = np.copy(src.vertices)
        self.faces = np.copy(src.faces)

    def resample(self, rate=2.):
        if type(rate) != list:
            ratec = rate
            rater = rate
        else:
            ratec = rate[1]
            rater = rate[0]

        x0 = np.reshape(self.vertices[:,0], (self.nrow, self.ncol))
        y0 = np.reshape(self.vertices[:,1], (self.nrow, self.ncol))
        v = sp.arange(0,self.nrow,1.)
        u = sp.arange(0,self.ncol,1.)
        fx = interp.interp2d(u,v,x0)
        fy = interp.interp2d(u,v,y0)
        nr = np.ceil(self.nrow/rater)
        nc = np.ceil(self.ncol/ratec)
        v = sp.arange(0,rater*nr,rater)
        u = sp.arange(0,ratec*nc,ratec)
        x = fx(u,v)
        y = fy(u,v)
        return Grid(gridPoints=(x,y))

    def skipLines(self, gap=1):
        nr = gap*np.ceil((self.nrow-1.)/gap) + 1
        nc = gap*np.ceil((self.ncol-1.)/gap) + 1
        g1 = self.resample([self.nrow/nr,self.ncol/nc])
        nr = g1.nrow
        nc = g1.ncol
        nrg = (nr-1)/gap + 1
        ncg = (nc-1)/gap + 1
        g1.faces = np.zeros([nrg*(nc-1)+ncg*(nr-1), 2], dtype=int)
        j = 0
        for k in range(0, nr, gap):
            for l in range(0, nc-1):
                g1.faces[j, :] = (k * nc + l, k * nc + l + 1)
                j += 1
        for k in range(0, nr-1):
            for l in range(0, nc, gap):
                g1.faces[j, :] = (k * nc + l, (k + 1) * nc + l)
                j += 1
        return g1

    # Saves in .vtk format
    def saveVTK(self, fileName):
        F = self.faces
        V = self.vertices

        with open(fileName, 'w') as fvtkout:
            fvtkout.write('# vtk DataFile Version 3.0\nSurface Data\nASCII\nDATASET POLYDATA\n') 
            fvtkout.write('\nPOINTS {0: d} float'.format(V.shape[0]))
            for ll in range(V.shape[0]):
                fvtkout.write('\n{0: f} {1: f} {2: f}'.format(V[ll,0], V[ll,1], 0))
            fvtkout.write('\nLINES {0:d} {1:d}'.format(F.shape[0], 3*F.shape[0]))
            for ll in range(F.shape[0]):
                fvtkout.write('\n2 {0: d} {1: d}'.format(F[ll,0], F[ll,1]))
            fvtkout.write('\n')

    # Reads .vtk file
    def readVTK(self, fileName):
        if gotVTK:
            u = vtkPolyDataReader()
            u.SetFileName(fileName)
            try:
                u.Update()
            except VTK_ERROR:
                print('error')

            v = u.GetOutput()
            npoints = int(v.GetNumberOfPoints())
            nfaces = int(v.GetNumberOfLines())
            V = np.zeros([npoints, 3])
            for kk in range(npoints):
                V[kk, :] = np.array(v.GetPoint(kk))
            V = V[:, 0:2]

            F = np.zeros([nfaces, 2], dtype=int)
            for kk in range(nfaces):
                c = v.GetCell(kk)
                for ll in range(2):
                    F[kk, ll] = c.GetPointId(ll)
            m = 0
            while F[m,0] == m:
                m += 1
            self.ncol = m+1
            self.nrow = npoints/self.ncol
            print(npoints, self.ncol*self.nrow)

            self.vertices = V
            self.faces = np.int_(F)
        else:
            raise Exception('Cannot read VTK files without VTK functions')

    def restrict(self, keepVert):
        V = np.copy(self.vertices)
        F = np.copy(self.faces)
        newInd = -np.ones(V.shape[0])
        j=0
        for k,kv in enumerate(keepVert):
            if kv:
                self.vertices[j,:] = V[k, :]
                newInd[k] = j
                j+=1
        self.vertices = self.vertices[0:j,:]
        j=0
        for k in range(F.shape[0]):
            if keepVert[F[k,0]] & keepVert[F[k,1]]:
                self.faces[j,0] = newInd[F[k,0]]
                self.faces[j,1] = newInd[F[k,1]]
                j+=1 
        self.faces = self.faces[0:j,:]

    def inPolygon(self, fv):
        nvert = self.vertices.shape[0]
        K = np.zeros(nvert)
        for k in range(nvert-1):
            pv0 = fv.vertices[fv.faces[:,0], :] - self.vertices[k,:]
            pv1 = fv.vertices[fv.faces[:,1], :] - self.vertices[k,:]
            c = (pv1[:,1]*pv0[:,0]) - (pv1[:,0]*pv0[:,1]) 
            #print c.shape
            c0 = np.sqrt((pv0**2).sum(axis=1))
            c1 = np.sqrt((pv1**2).sum(axis=1))
            c = np.divide(c, np.multiply(c0,c1)+1e-10)
            c = np.arcsin(c)
            #print c
            w = c.sum()/np.pi
            #print w /np.pi
            if abs(w) > 0.5:
                K[k] = 1
        return np.int_(K)

    def distPolygon(self, fv):
        nvert = self.vertices.shape[0]
        D = np.zeros(nvert)
        for k in range(nvert-1):
            D[k] = np.min(np.sqrt((fv.vertices[:, 0] - self.vertices[k,0])**2 +
                                  (fv.vertices[:, 1] - self.vertices[k,1])**2))
        return D

    def signedDistPolygon(self, fv):
        D = self.distPolygon(fv)
        K = self.inPolygon(fv)
        D *= 1-2*K 
        return D
