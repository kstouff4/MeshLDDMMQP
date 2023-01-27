from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import os
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import matplotlib.pyplot as plt
import numpy as np
import pygalmesh
from base import surfaces, surfaceExamples
from base import loggingUtils
from base.surfaces import Surface
from base.kernelFunctions import Kernel
from base.affineRegistration import rigidRegistration
from base.surfaceMatching import SurfaceMatching, SurfaceMatchingParam
import pykeops
pykeops.clean_pykeops()
plt.ion()

model = 'Balls'
typeCost = 'LDDMM'

def compute(model):
    loggingUtils.setup_default_logging('../Output', stdOutput = True)
    if typeCost == 'LDDMM':
        sigmaKernel = 5
        internalCost = None
        internalWeight = 0.
        regweight = 1.
    else:
        sigmaKernel = 0.5
        internalWeight = 100.
        regweight = 0.1
        internalCost = 'elastic'

    sigmaDist = 10.
    sigmaError = 1.
    #internalCost = 'h1'
    landmarks = None
    if model=='Balls':
        s2 = np.sqrt(2)
        class Ball1(pygalmesh.DomainBase):
            def __init__(self):
                super().__init__()

            def eval(self, x):
                return (.06 - ((x[0] - .50) ** 2 + 0.5 * x[1] ** 4 + x[2] ** 2))

            def get_bounding_sphere_squared_radius(self):
                return 4.0

        class Ball2(pygalmesh.DomainBase):
            def __init__(self):
                super().__init__()

            def eval(self, x):
                I1 = np.maximum(0.05 - (x[0] - .7) ** 2 - 0.5 * x[1] ** 2 - x[2] ** 2,
                                0.02 - (x[0] - .50) ** 2 - 0.5 * x[1] ** 2 - x[2] ** 2)
                return I1

            def get_bounding_sphere_squared_radius(self):
                return 4.0

        # d = Heart()
        d = Ball1()
        mesh = pygalmesh.generate_surface_mesh(d, max_facet_distance=0.05, min_facet_angle=30.0,
                                               max_radius_surface_delaunay_ball=0.05, verbose=False)
        fv1 = Surface(surf=(mesh.cells[0].data, mesh.points))
        fv1.updateVertices(fv1.vertices*100)

        d = Ball2()
        mesh = pygalmesh.generate_surface_mesh(d, max_facet_distance=0.05, min_facet_angle=30.0,
                                               max_radius_surface_delaunay_ball=0.05, verbose=False)
        fv2 = Surface(surf=(mesh.cells[0].data, mesh.points))
        fv2.updateVertices(fv2.vertices*100)

        # M=100
        # targSize = 1000
        # [x,y,z] = np.mgrid[0:2*M, 0:2*M, 0:2*M]/float(M)
        # y = y-1
        # z = z-1
        #
        # I1 = .06 - ((x-.50)**2 + 0.5*y**4 + z**2)
        # fv1 = Surface()
        # fv1.Isosurface(I1, value = 0, target=targSize, scales=[1, 1, 1], smooth=0.01)
        #
        # #return fv1
        #
        # u = (z + y)/s2
        # v = (z - y)/s2
        # I1 = np.maximum(0.05 - (x-.7)**2 - 0.5*y**2 - z**2, 0.02 - (x-.50)**2 - 0.5*y**2 - z**2)
        # fv2 = Surface()
        # fv2.Isosurface(I1, value = 0, target=targSize, scales=[1, 1, 1], smooth=0.01)

        ftemp = fv1
        ftarg = fv2
        #landmarks = (fv1.vertices[0:5, :], fv2.vertices[0:5, :], 1)
        #internalCost = None
    elif model=='Hearts':
        [x,y,z] = np.mgrid[0:200, 0:200, 0:200]/100.
        ay = np.fabs(y-1)
        az = np.fabs(z-1)
        ax = np.fabs(x-0.5)
        s2 = np.sqrt(2)
        c1 = np.sqrt(0.06)
        c2 = np.sqrt(0.045)
        c3 = 0.1

        I1 = np.minimum(c1**2 - (ax**2 + 0.5*ay**2 + az**2), np.minimum((ax**2 + 0.5*ay**2 + az**2)-c2**2, 1+c3-y)) 
        fv1 = Surface()
        fv1.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)

        p = 1.75
        s1 = 1.2
        s2 = 1.4
        I1 = np.minimum(c1**p/s1 - ((ax**p + 0.5*ay**p + az**p)), np.minimum((s2*ax**p + s2*0.5*ay**p + s2*az**p)-c2**p/s1, 1+c3/s1-y))  
        fv3 = Surface()
        fv3.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)
        
        fv3.vertices[:,1] += 15 - 15/s1

        ftemp = fv1
        ftarg = fv3
    elif model=='KCrane':
        ftemp = surfaces.Surface(surf='../testData/Surfaces/KCrane/blub_triangulated_reduced.obj')
        ftarg = surfaces.Surface(surf='../testData/Surfaces/KCrane/spot_triangulated_reduced.obj')
        R0, T0 = rigidRegistration(surfaces = (ftarg.vertices, ftemp.vertices),  rotWeight=0., verb=False, temperature=10., annealing=True)
        ftarg.updateVertices(np.dot(ftarg.vertices, R0.T) + T0)
        sigmaKernel = 0.5
        sigmaDist = 5.
        sigmaError = 0.01
        internalWeight = 10.
    elif model=='snake':
        M=100
        [x,y,z] = np.mgrid[0:2*M, 0:2*M, 0:2*M]/float(M)
        x = x-1
        y = y-1
        z = z-1
        t = np.arange(-0.5, 0.5, 0.01)

        r = .3
        c = .95
        delta = 0.05
        h = 0.25
        f1 = np.zeros((t.shape[0],3))
        f1[:,0] = r*np.cos(2*np.pi*c*t) -r
        f1[:,1] = r*np.sin(2*np.pi*c*t)
        fig = plt.figure(4)
        f1[:,2] = h*t
        fig.clf()
        ax = fig.gca(projection='3d')
        ax.plot(f1[:,0], f1[:,1], f1[:,2])

        f2 = np.zeros((t.shape[0],3))
        f2[:,0] = r*np.cos(2*np.pi*c*t)-r
        f2[:,1] = r*np.sin(2*np.pi*c*t)
        f2[:,2] = -h*t
        ax.plot(f2[:,0], f2[:,1], f2[:,2])
#        ax.axis('equal')
        plt.pause((0.1))

        dst = (x[..., np.newaxis] - f1[:,0])**2 + (y[..., np.newaxis] - f1[:,1])**2 + (z[..., np.newaxis] - f1[:,2])**2
        dst = np.min(dst, axis=3)
        ftarg = Surface()
        ftarg.Isosurface((dst < delta**2), value=0.5)
        dst = (x[..., np.newaxis] - f2[:,0])**2 + (y[..., np.newaxis] - f2[:,1])**2 + (z[..., np.newaxis] - f2[:,2])**2
        dst = np.min(dst, axis=3)
        ftemp = Surface()
        ftemp.Isosurface((dst < delta**2), value=0.5)
        sigmaKernel = np.array([1,5,10])
        sigmaDist = 10.
        sigmaError = .1
        internalWeight = 5.
        internalCost = None
    else:
        return

    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = sigmaKernel)

    sm = SurfaceMatchingParam(timeStep=0.1, algorithm='bfgs', KparDiff=K1, KparDist=('gauss', sigmaDist),
                              sigmaError=sigmaError, errorType='varifold', internalCost=internalCost)
    sm.KparDiff.pk_dtype = 'float64'
    sm.KparDist.pk_dtype = 'float64'
    f = SurfaceMatching(Template=ftemp, Target=ftarg, outputDir='../Output/surfaceMatchingTest/'+model +'_'+typeCost,
                        param=sm, regWeight = regweight, Landmarks = landmarks,
                        unreduced=False, unreducedWeight= 0.,
                        #subsampleTargetSize = 500,
                        internalWeight=internalWeight, maxIter=20, affine= 'none', rotWeight=.01, transWeight = .01,
                        scaleWeight=10., affineWeight=100.)
    f.testGradient = True
    f.optimizeMatching()
    for k in range(20):
        f.unreducedWeight += 0.1 * f.ds
        f.reset = True
        f.optimizeMatching()
    plt.ioff()
    plt.show()

    return f


compute(model)
