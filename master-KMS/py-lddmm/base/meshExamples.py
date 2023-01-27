import numpy as np
import pygalmesh
from .curveExamples import Circle
from .surfaceExamples import Sphere
import loggingUtils
from .meshes import Mesh, select_faces__


class TwoDiscs(Mesh):
    def __init__(self, largeRadius = 10., smallRadius = 4.5, targetSize=250):
        f = Circle(radius=largeRadius, targetSize=targetSize)
        super(TwoDiscs, self).__init__(f,volumeRatio=5000)
        imagev = np.array(((self.vertices - np.array(f.center)[None, :]) ** 2).sum(axis=1) < smallRadius**2, dtype=float)
        image = np.zeros((self.faces.shape[0], 2))
        image[:, 0] = (imagev[self.faces[:, 0]] + imagev[self.faces[:, 1]] + imagev[self.faces[:, 2]]) / 3
        image[:, 1] = 1 - image[:, 0]
        self.updateImage(image)

class TwoBalls(Mesh):
    def __init__(self, largeRadius = 10., smallRadius = 4.5):
        mesh = pygalmesh.generate_mesh(
            pygalmesh.Ball([0.0, 0.0, 0.0], largeRadius),
            min_facet_angle=30.0,
            max_radius_surface_delaunay_ball=.75,
            max_facet_distance=0.025,
            max_circumradius_edge_ratio=2.0,
            max_cell_circumradius=.75,  # lambda x: abs(np.sqrt(np.dot(x, x)) - 0.5) / 5 + 0.025,
            verbose=False
        )
        super(TwoBalls, self).__init__([np.array(mesh.cells[1].data, dtype=int), np.array(mesh.points, dtype=float)])
        c0 = np.array([0,0,0])
        imagev = np.array(((self.vertices - c0[None, :]) ** 2).sum(axis=1) < smallRadius**2, dtype=float)
        image = np.zeros((self.faces.shape[0], 2))
        image[:, 0] = (imagev[self.faces[:, 0]] + imagev[self.faces[:, 1]] + imagev[self.faces[:, 2]]
                           + imagev[self.faces[:, 3]]) / 4
        image[:, 1] = 1 - image[:, 0]
        self.updateImage(image)


class MoGCircle(Mesh):
    def __init__(self, largeRadius = 10., nregions = 5, ntypes = 5, ngenes=10, density = 10., centers=None, a=1,
                 targetSize=500, cellTypes = True, typeProb = None, geneProb = None, alpha = None):
        f = Circle(radius=2*largeRadius, targetSize=targetSize)
        super(MoGCircle, self).__init__(f,volumeRatio=5000)

        if centers is None:
            self.nregions = nregions
            pts = np.random.normal(0, 1, (nregions, 2))
            pts = pts / np.sqrt((pts**2).sum(axis=1))[:,None]
            r = np.sqrt(np.random.uniform(0,1,(nregions,1)))
            self.GaussCenters = largeRadius*r*pts
        else:
            self.nregions = centers.shape[0]
            self.GaussCenters = centers

        ## prior on cell types and genes
        if typeProb is None:
            typeProb = np.random.dirichlet([a] * ntypes, nregions)
        if geneProb is None:
            geneProb = np.random.dirichlet([a] * ngenes, ntypes)

        self.label = np.zeros(self.faces.shape[0], dtype=int)
        if cellTypes:
            image = np.zeros((self.faces.shape[0], ntypes))
        else:
            image = np.zeros((self.faces.shape[0], ngenes))
        self.types = np.zeros((self.faces.shape[0], ntypes))
        if alpha is None:
            alpha = np.random.poisson(density, nregions)
        weights = np.zeros(self.faces.shape[0])
        for k in range(self.faces.shape[0]):
            distk = ((self.centers[k,:] - self.GaussCenters)**2).sum(axis=1)
            jk = np.argmin(distk)
            ## type composition of the simplex
            self.types[k,:] = np.random.dirichlet(a - 1 + typeProb[jk, :])
            self.label[k] = jk
            #weights[k] = np.random.poisson(alpha[self.label[k]]) * np.exp(-distk[jk]/(2*(largeRadius/nregions)**2))
            weights[k] = np.random.poisson(alpha[self.label[k]]) * np.exp(-distk[jk]/(2*(largeRadius/10)**2))
            if not cellTypes:
                for t in range(ntypes):
                    image[k, :] += np.random.choice(np.floor(ngenes*self.types[k,t]), p=geneProb[jk, :])

        # for k in range(self.faces.shape[0]):
        #     weights[k] = np.random.poisson(alpha[self.label[k]])
        self.typeProb = typeProb
        self.geneProb = geneProb
        self.alpha = alpha
        self.updateWeights(weights)
        if cellTypes:
            self.updateImage(self.types)
        else:
            self.updateImage(image)

        newv, newf, newg2, keepface = select_faces__(weights[:, None], self.vertices, self.faces,
                                                      threshold=.1*density)
        wgts = self.weights[keepface]
        newg = np.copy(self.image[keepface, :])
        self.vertices = newv
        self.faces = newf
        self.computeCentersVolumesNormals()
        self.updateWeights(wgts)
        self.image = newg

