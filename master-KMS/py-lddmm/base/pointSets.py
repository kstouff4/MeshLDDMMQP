import numpy as np
from os.path import splitext
from vtk import vtkOBJReader
from scipy.spatial.distance import squareform, pdist
from numpy.random import default_rng


class PointSet:
    def __init__(self, data=None, weights = None, maxPoints=None):
        if type(data) is str:
            self.read(data, maxPoints=maxPoints)
        elif issubclass(type(data), PointSet):
            self.points = np.copy(data.points)
            if weights is None:
                self.weights = np.copy(data.weights)
            else:
                self.weights = weights
        elif isinstance(data, np.ndarray):
            self.points = data.copy()
            if weights is None:
                self.weights = np.ones(data.shape[0])
            else:
                self.weights = weights
        else:
            self.points = np.empty(0)
            self.weights = np.empty(0)

    def updatePoints(self, pts):
        self.points = np.copy(pts)

    def addToPlot(self, ax, ec = 'b', fc = 'r', al=.5, lw=1):
        x = self.points[:, 0]
        y = self.points[:, 1]
        z = self.points[:, 2]
        ax.scatter(x,y,z, alpha=al)
        xlim = [x.min(),x.max()]
        ylim = [y.min(),y.max()]
        zlim = [z.min(),z.max()]
        return [xlim, ylim, zlim]

    def set_weights(self, knn=5):
        d = squareform(pdist(self.points))
        d = np.sort(d, axis=1)
        eps = d[:, knn].mean() + 1e-10
        self.weights = np.pi * eps ** 2 / (d < eps).sum(axis=1)


    def read(self, filename, maxPoints=None):
        head, tail = splitext(filename)
        if tail in ('.obj', '.OBJ'):
            self.readOBJ(filename, maxPoints=maxPoints)
        else:
            self.points, self.weights = readVector(filename)

    def readOBJ(self, fileName, maxPoints=None):
        u = vtkOBJReader()
        u.SetFileName(fileName)
        u.Update()
        v = u.GetOutput()
        # print v
        npoints = int(v.GetNumberOfPoints())
        V = np.zeros([npoints, 3])
        for kk in range(npoints):
            V[kk, :] = np.array(v.GetPoint(kk))

        if maxPoints is not None and maxPoints < npoints:
            rng = default_rng()
            select = rng.choice(npoints, maxPoints, replace=False)
            V = V[select, :]

        self.points = V
        self.set_weights(5)


    def saveVTK(self, filename):
        self.save(filename)
    def save(self, filename):
        savePoints(filename, self.points, scalars=self.weights)

def readVector(filename):
    try:
        with open(filename, 'r') as fn:
            ln0 = fn.readline().split()
            N = int(ln0[0])
            dim = int(ln0[1])
            #print 'reading ', filename, ':', N, ' landmarks'
            v = np.zeros([N, dim])
            w = np.zeros([N,1])

            for i in range(N):
                ln0 = fn.readline().split()
                #print ln0
                for k in range(3):
                    v[i,k] = float(ln0[k])
                w[i] = ln0[3]
    except IOError:
        print('cannot open ', filename)
        raise
    return v,w




def loadlmk(filename, dim=3):
# [x, label] = loadlmk(filename, dim)
# Loads 3D landmarks from filename in .lmk format.
# Determines format version from first line in file
#   if version number indicates scaling and centering, transform coordinates...
# the optional parameter s in a 3D scaling factor

    try:
        with open(filename, 'r') as fn:
            ln0 = fn.readline()
            versionNum = 1
            versionStrs = ln0.split("-")
            if len(versionStrs) == 2:
                try:
                    versionNum = int(float(versionStrs[1]))
                except:
                    pass

            #print fn
            ln = fn.readline().split()
            #print ln0, ln
            N = int(ln[0])
            #print 'reading ', filename, ':', N, ' landmarks'
            x = np.zeros([N, dim])
            label = []

            for i in range(N):
                ln = fn.readline()
                label.append(ln) 
                ln0 = fn.readline().split()
                #print ln0
                for k in range(dim):
                    x[i,k] = float(ln0[k])
            if versionNum >= 6:
                lastLine = ''
                nextToLastLine = ''
                # read the rest of the file
                # the last two lines contain the center and the scale variables
                while 1:
                    thisLine = fn.readline()
                    if not thisLine:
                        break
                    nextToLastLine = lastLine
                    lastLine = thisLine
                    
                centers = nextToLastLine.rstrip('\r\n').split(',')
                scales = lastLine.rstrip('\r\n').split(',')
                if len(scales) == dim and len(centers) == dim:
                    if scales[0].isdigit and scales[1].isdigit and scales[2].isdigit and centers[0].isdigit \
                            and centers[1].isdigit and centers[2].isdigit:
                        x[:, 0] = x[:, 0] * float(scales[0]) + float(centers[0])
                        x[:, 1] = x[:, 1] * float(scales[1]) + float(centers[1])
                        x[:, 2] = x[:, 2] * float(scales[2]) + float(centers[2])
                
    except IOError:
        print('cannot open ', filename)
        raise
    return x, label




def  savelmk(x, filename):
# savelmk(x, filename)
# save landmarks in .lmk format.

    with open(filename, 'w') as fn:
        str = 'Landmarks-1.0\n {0: d}\n'.format(x.shape[0])
        fn.write(str)
        for i in range(x.shape[0]):
            str = '"L-{0:d}"\n'.format(i)
            fn.write(str)
            str = ''
            for k in range(x.shape[1]):
                str = str + '{0: f} '.format(x[i,k])
            str = str + '\n'
            fn.write(str)
        fn.write('1 1 \n')

        
# Saves in .vtk format
def savePoints(fileName, x, vector=None, scalars=None):
    if x.shape[1] <3:
        x = np.concatenate((x, np.zeros((x.shape[0],3-x.shape[1]))), axis=1)
    with open(fileName, 'w') as fvtkout:
        fvtkout.write('# vtk DataFile Version 3.0\nSurface Data\nASCII\nDATASET UNSTRUCTURED_GRID\n') 
        fvtkout.write('\nPOINTS {0: d} float'.format(x.shape[0]))
        for ll in range(x.shape[0]):
            fvtkout.write('\n{0: f} {1: f} {2: f}'.format(x[ll,0], x[ll,1], x[ll,2]))
        if vector is None and scalars is None:
            return
        fvtkout.write(('\nPOINT_DATA {0: d}').format(x.shape[0]))
        if scalars is not None:
            fvtkout.write('\nSCALARS scalars float 1\nLOOKUP_TABLE default')
            for ll in range(x.shape[0]):
                fvtkout.write('\n {0: .5f} '.format(scalars[ll]))

        if vector is not None:
            fvtkout.write('\nVECTORS vector float')
            for ll in range(x.shape[0]):
                fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(vector[ll, 0], vector[ll, 1], vector[ll, 2]))

        fvtkout.write('\n')

# Saves in .vtk format
def saveTrajectories(fileName, xt):
    with open(fileName, 'w') as fvtkout:
        fvtkout.write('# vtk DataFile Version 3.0\ncurves \nASCII\nDATASET POLYDATA\n')
        npt = xt.shape[0]*xt.shape[1]
        fvtkout.write('\nPOINTS {0: d} float'.format(npt))
        if xt.shape[2] == 2:
            xt = np.concatenate((xt, np.zeros([xt.shape[0],xt.shape[1], 1])))
        for t in range(xt.shape[0]):
            for ll in range(xt.shape[1]):
                fvtkout.write('\n{0: f} {1: f} {2: f}'.format(xt[t,ll,0], xt[t,ll,1], xt[t,ll,2]))
        nlines = (xt.shape[0]-1)*xt.shape[1]
        fvtkout.write('\nLINES {0:d} {1:d}'.format(nlines, 3*nlines))
        for t in range(xt.shape[0]-1):
            for ll in range(xt.shape[1]):
                fvtkout.write('\n2 {0: d} {1: d}'.format(t*xt.shape[1]+ll, (t+1)*xt.shape[1]+ll))

        fvtkout.write(('\nPOINT_DATA {0: d}').format(npt))
        fvtkout.write('\nSCALARS time int 1\nLOOKUP_TABLE default')
        for t in range(xt.shape[0]):
            for ll in range(xt.shape[1]):
                fvtkout.write(f'\n{t}')

        fvtkout.write('\n')



def epsilonNet(x, rate):
    #print 'in epsilon net'
    n = x.shape[0]
    dim = x.shape[1]
    inNet = np.zeros(n, dtype=int)
    inNet[0]=1
    net = np.nonzero(inNet)[0]
    survivors = np.ones(n, dtype=np.int)
    survivors[0] = 0 ;
    dist2 = ((x.reshape([n, 1, dim]) -
              x.reshape([1,n,dim]))**2).sum(axis=2)
    d2 = np.sort(dist2, axis=0)
    i = np.int_(1.0/rate)
    eps2 = (np.sqrt(d2[i,:]).sum()/n)**2
    #print n, d2.shape, i, np.sqrt(eps2)
    

    i1 = np.nonzero(dist2[net, :] < eps2)
    survivors[i1[1]] = 0
    i2 = np.nonzero(survivors)[0]
    while len(i2) > 0:
        closest = np.unravel_index(np.argmin(dist2[net.reshape([len(net),1]), i2.reshape([1, len(i2)])].ravel()), [len(net), len(i2)])
        inNet[i2[closest[1]]] = 1 
        net = np.nonzero(inNet)[0]
        i1 = np.nonzero(dist2[net, :] < eps2)
        survivors[i1[1]] = 0
        i2 = np.nonzero(survivors)[0]
        #print len(net), len(i2)
    idx = - np.ones(n, dtype=np.int)
    for p in range(n):
        closest = np.unravel_index(np.argmin(dist2[net, p].ravel()), [len(net), 1])
        #print 'p=', p, closest, len(net)
        idx[p] = closest[0]
        
        #print idx
    return net, idx


