from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import argparse
import logging
from base import loggingUtils
from base import surfaces
from base.kernelFunctions import *
from base import surfaceMatching
from base import affineRegistration



def runLongitudinalSurface(template, targetList, minL=3, atrophy=False, splines=False, resultDir='.'):
    if atrophy:
        pass
    elif splines:
        import secondOrderMatching as match
    else:
        pass

    

    if len(template) > 0:
        fv0 = surfaces.Surface(surf=template)
        z = fv0.surfVolume()
        if z < 0:
            fv0.flipFaces()
    else:
        return
    K1 = Kernel(name='laplacian', sigma = 2.5, order=4)
    sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, KparDist=('gauss', 1.5),
                                              sigmaError=1., errorType='varifold')

    outputDir = resultDir
    info_outputDir = outputDir
    loggingUtils.setup_default_logging(info_outputDir, fileName='info', stdOutput=True)

    logging.info(targetList[0])
    fv = []
    for fn in targetList:
        try:
            #fv += [surfaces.Surface(filename=fn+'.byu')]
            fv1 = surfaces.Surface(surf=fn)
            z = fv1.surfVolume()
            if z < 0:
                fv1.flipFaces()
            fv += [fv1]
            logging.info(fn)
        except NameError as e:
            print(e)
    logging.info(outputDir)
        ## Reversing order to test bias
        #fv.reverse()
    if len(template) == 0:
        fv0 = surfaces.Surface(surf=fv[0])

    coeff = [1.0, 1.1, 0.9, 1.0]

    for k,fs in enumerate(fv):
        fs.updateVertices(fs.vertices*coeff[k])
        R0, T0 = affineRegistration.rigidRegistration(surfaces = (fs.vertices, fv0.vertices),  verb=False,
                                                      temperature=10., annealing=True, translationOnly=True)
        fs.updateVertices(np.dot(fs.vertices, R0.T) + T0)


    try:
        if atrophy:
            f = match.SurfaceMatching(Template=fv0, Targets=fv, outputDir=outputDir, param=sm, regWeight=.1,
                                      affine='euclidean', testGradient=True, affineWeight=.1,  maxIter_cg=50,
                                      maxIter_al=50, mu=0.0001)
        elif splines:
            f = match.SurfaceMatching(Template=fv0, Targets=fv, outputDir=outputDir, param=sm, regWeight=.1,
                                      affine='none', typeRegression='geodesic',
                                      testGradient=False, affineWeight=.1,  maxIter=1000)
        else:
            f = match.SurfaceMatching(Template=fv0, Targets=fv, outputDir=outputDir, param=sm, regWeight=.1,
                                    affine='none', testGradient=False, affineWeight=.1,  maxIter=1000)
    except NameError:
        print('exception')

    try:
        f.optimizeMatching()
    except NameError:
        print('Exception')
 

template = '/Users/younes/Development/Data/sculptris/AtrophyLargeNoise/baseline.vtk'
targetList = []
for k in range(1,11,3):
    targetList.append('/Users/younes/Development/Data/sculptris/AtrophyLargeNoise/followUp'+str(k)+'.vtk')
resultDir = '/Users/younes/Development/Results/TimeSeriesSimulationLDDMM'


runLongitudinalSurface(template, targetList, atrophy=False, splines=False, resultDir=resultDir)
