import logging
from base import surfaces
from base import conjugateGradient as cg ,kernelFunctions as kfun ,pointEvolution as evol ,loggingUtils ,bfgs
from base.surfaceMatchingNormalExtremities import SurfaceMatching
from base.surfaceMatching import SurfaceMatchingParam
from base.affineBasis import *
from base import examples


#
def run(opt=None):
    # outputDir = '/cis/home/younes/Development/Results/ERC_Normals_ADNI_014_S_4058/'
    fvTop = None
    fvBottom = None
    if opt is not None:
        outputDir = '/Users/younes/Development/Results/BOK/'+opt
        loggingUtils.setup_default_logging(outputDir, fileName='info', stdOutput=True)

        if opt == 'Corner':
            fvBottom = surfaces.Surface(filename='/Users/younes/Development/Data/NormalData/cornerTemplate.byu')
            fvTop = surfaces.Surface(filename='/Users/younes/Development/Data/NormalData/cornerTarget.byu')
        if opt == 'cat':
            fvBottom = surfaces.Surface(filename='/Users/younes/Development/Data/fromKwame/cat/Template.vtk')
            fvBottom.smooth(n=100)
            fvBottom.Simplify(target=2500)
            fvTop = surfaces.Surface(filename='/Users/younes/Development/Data/fromKwame/cat/Target.vtk')
            fvTop.smooth(n=100)
            fvTop.Simplify(target=2500)
        elif opt == 'SueData':
            probDir = '/cis/home/younes/MorphingData/SUE/023_S_4035_L_mo00_ERC_and_TEC/'
            fvTop = surfaces.Surface(filename=probDir + 'Template.vtk')
            fvBottom = surfaces.Surface(filename=probDir + 'Target.vtk')
        elif opt == 'heart':
            [x, y, z] = np.mgrid[0:200, 0:200, 0:200] / 100.
            ay = np.fabs(y - 1)
            az = np.fabs(z - 1)
            ax = np.fabs(x - 0.5)
            s2 = np.sqrt(2)
            c1 = np.sqrt(0.06)
            c2 = np.sqrt(0.045)
            c3 = 0.1

            # return fv1

            # s1 = 1.375
            # s2 = 2
            s1 = 1.1
            s2 = 1.2
            p = 1.75
            I1 = np.minimum(c1 ** p / s1 - ((ax ** p + 0.5 * ay ** p + az ** p)),
                            np.minimum((s2 * ax ** p + s2 * 0.5 * ay ** p + s2 * az ** p) - c2 ** p / s1,
                                       1 + c3 / s1 - y))
            fvBottom = surfaces.Surface()
            fvBottom.Isosurface(I1, value=0, target=1000, scales=[1, 1, 1], smooth=0.01)

            fvBottom.vertices[:, 1] += 15 - 15 / s1
            # I1 = np.minimum(c1 ** 2 - (ax ** 2 + 0.5 * ay ** 2 + az ** 2),
            #                 np.minimum((ax ** 2 + 0.5 * ay ** 2 + az ** 2) - c2 ** 2, 1 + c3 - y))
            # fvTop = surfaces.Surface()
            # fvTop.Isosurface(I1, value=0, target=1000, scales=[1, 1, 1], smooth=0.01)

        elif opt == 'ellipses':
            [x, y, z] = np.mgrid[0:200, 0:200, 0:200] / 100.
            ay = np.fabs(y - 1)
            az = np.fabs(z - 1)
            ax = np.fabs(x - 0.5)
            s2 = np.sqrt(2)
            c1 = np.sqrt(0.06)
            c2 = np.sqrt(0.035)
            c3 = 0.1

            I1 = c1 ** 2 - (ax ** 2 + 0.5 * ay ** 2 + az ** 2)
            I2 = -(ax ** 2 + 0.5 * ay ** 2 + az ** 2) + c2 ** 2
            #1 + c3 - y
            fvTop = surfaces.Surface()
            #I1 = np.logical_and(I1 > 0, np.logical_and(ay>-0.001, ay < 0.201))-0.5
            fvTop.Isosurface(I1, value=0, target=5000, scales=[1, 1, 1], smooth=-0.01)
            fvTop = fvTop.truncate((np.array([0,1,0,85]), np.array([0,-1,0,-100])))
            #fvTop.smooth()
            #select = np.logical_and(fvTop.vertices[:,1] > 75, fvTop.vertices[:,1] < 90)
            #fvTop = fvTop.cut(select)

            fvBottom = surfaces.Surface()
            #I2 = np.logical_and(I2 > 0, np.logical_and(ay>-0.01, ay < 0.21))-0.5
            fvBottom.Isosurface(I2, value=0, target=5000, scales=[1, 1, 1], smooth=-0.01)
            fvBottom = fvBottom.truncate((np.array([0,1,0,85]), np.array([0,-1,0,-100])))
            #select = np.logical_and(fvBottom.vertices[:,1] > 75, fvBottom.vertices[:,1] < 90)
            #fvBottom = fvBottom.cut(select)

            #fvBottom.vertices[:, 1] += 15 - 15 / s1
        elif opt == 'flowers':
            fvBottom,fvTop = examples.flowers()
        elif opt == 'waves':
            fvBottom,fvTop = examples.waves(w1=(2,0.25), d=20,delta=5)

        #fvTop = surfaces.Surface(filename='/cis/home/younes/MorphingData/Tilaksurfaces/Separated_Cuts/DH1MiddleOuter.byu')
        #fvBottom = surfaces.Surface(filename='/cis/home/younes/MorphingData/Tilaksurfaces/Separated_Cuts/DH1MiddleInner.byu')
        # outputDir = '/cis/home/younes/Development/Results/tilakAW1Superior'

        #fvBottom = surfaces.Surface(
        #    filename='/cis/home/younes/MorphingData/SueExamples/bottom_041_S_4720_L_mo00_ERC_and_TEC.byu')
        # fv0 = surfaces.Surface(filename='/cis/home/younes/MorphingData/Tilaksurfaces/Separated_Cuts/NK1Inner.byu')
        # fv1 = surfaces.Surface(filename='/cis/home/younes/MorphingData/Tilaksurfaces/Separated_Cuts/NK1Outer.byu')
        # fv0 = surfaces.Surface(filename='/Users/younes/Development/Data/ALLIE/Template.vtk')
        # fv1 = surfaces.Surface(filename='/Users/younes/Development/Data/ALLIE/Target.vtk')
        # fv1 = surfaces.Surface(filename='/cis/home/younes/MorphingData/Tilaksurfaces/Separated_Cuts/NK1Outer.byu')
        # # fv1 = surfaces.Surface(filename='/cis/home/younes/MorphingData/surfaces/chelsea/bottom_surface_smooth.byu')
        # fv0 = surfaces.Surface(filename='/cis/home/younes/MorphingData/surfaces/chelsea/top_surface_smooth.byu')

        fvBottom.removeIsolated()
        fvBottom.edgeRecover()
        # fvTop.subDivide(1)

        K1 = kfun.Kernel(name='laplacian', sigma=np.array([2.5]), order=3)

        if opt=='heart':
            K1 = kfun.Kernel(name='laplacian', sigma=.1, order=3)
            sm = SurfaceMatchingParam(timeStep=0.1, algorithm='bfgs', KparDiff=K1, sigmaDist=.1,
                                                      sigmaError=.1, errorType='currentMagnitude', internalCost=None)

            fTemp = surfaces.Surface(surf=fvBottom)
            nu = fTemp.computeVertexNormals()
            fTemp.updateVertices(fvBottom.vertices + 1e-5 * nu)
            s1 = fTemp.surfVolume()
            fTemp.updateVertices(fvBottom.vertices - 1e-5 * nu)
            s2 = fTemp.surfVolume()
            if s1 < s2:
                fvBottom.flipFaces()
                logging.info('Flipping orientation of bottom shape.')

            f = SurfaceMatching(Template=fvBottom, outputDir=outputDir, param=sm, regWeight=1.,
                                saveTrajectories=True, symmetric=False, pplot=True,
                                affine='none', testGradient=True, internalWeight=1000., affineWeight=1e3, maxIter_cg=50,
                                maxIter_al=50, mu=1e-5)
        else:

            fvTop.removeIsolated()
            fvTop.edgeRecover()
            # K2 = kfun.Kernel(sigma = 2.5)
            # print fv0.normGrad(fv0.vertices)
            # print fv0.normGrad(fv0.vertices)
            sm = SurfaceMatchingParam(timeStep=0.05, algorithm='bfgs', KparDiff=K1, sigmaDist=1.,
                                                      sigmaError=.1, errorType='varifold', internalCost='h1')

            fTemp = surfaces.Surface(surf=fvBottom)
            nu = fTemp.computeVertexNormals()
            fTemp.updateVertices(fvBottom.vertices+1e-5*nu)
            d1 = surfaces.haussdorffDist(fTemp, fvTop)
            fTemp.updateVertices(fvBottom.vertices-1e-5*nu)
            d2 = surfaces.haussdorffDist(fTemp, fvTop)
            if d2 < d1:
                fvBottom.flipFaces()
                logging.info('Flipping orientation of bottom shape.')

            f = SurfaceMatching(Template=fvBottom, Target=fvTop, outputDir=outputDir, param=sm, regWeight=1.,
                                saveTrajectories=True, symmetric=False, pplot=True,
                                affine='none', testGradient=True, internalWeight=100., affineWeight=1e3, maxIter_cg=50,
                                maxIter_al=50, mu=1e-4)
        f.optimizeMatching()


if __name__ == "__main__":
    run(opt='cat')
