#import numpy as np
#import logging
from base.surfaces import Surface
from base import gaussianDiffeons as gd
from base import conjugateGradient as cg, diffeo, kernelFunctions as kfun, pointEvolution as evol, bfgs, loggingUtils
from base.gaussianDiffeonsSurfaceMatching import *
from base import examples

def compute(createsurfaces=True):
    if createsurfaces:
        fv1, fv2 = examples.bumps2(centers1 = ([-0.1,-0.3], [.3,0.25], [-0.1,.3], [.25,-.3]),
          scale1 = (.5, .5, .5, .5),
          weights1 = (.3, .3, .3, .3),
          centers2 = ([.3,-.3], [-.3, .3], [.3, .3], [-.3,-.3]),
          scale2 = (.3,.3,.3, .3),
          weights2 = (.5,.5,.5, .5),
          d=25)

        # fv1.saveVTK('/Users/younes/Development/Results/Diffeons/fv1alt.vtk')
        # fv2.saveVTK('/Users/younes/Development/Results/Diffeons/fv2alt.vtk')
    else:
        if False:
            #path = '/Users/younes/Development/project/ncbc/data/template/PDS-II/AllScan1_PDSII/shape_analysis/hippocampus/'
            path = '/Volumes/CIS/project/ncbc/data/template/PDS-II/AllScan1_PDSII/shape_analysis/hippocampus/'
            #sub1 = '0186193_1_6'
            #sub2 = '1449400_1_L'
            sub2 = 'LU027_R_sumNCBC20100628'
            fv1 = surfaces.Surface(surf =path + '5_population_template_qc/newTemplate.byu')
            v1 = fv1.surfVolume()
            #f0.append(surfaces.Surface(surf = path+'amygdala/biocardAmyg 2/'+sub1+'_amyg_L.byu'))
            fv2 = surfaces.Surface(surf =path + '2_qc_flipped_registered/' + sub2 + '_registered.byu')
            v2 = fv2.surfVolume()
            if (v2*v1 < 0):
                fv2.flipFaces()
        else:
            #f1.append(surfaces.Surface(surf = path+'amygdala/biocardAmyg 2/'+sub2+'_amyg_L.byu'))
            fv1 = Surface(surf='/Users/younes/Development/Results/Diffeons/fv1Alt.vtk')
            fv2  = Surface(surf='/Users/younes/Development/Results/Diffeons/fv2Alt.vtk')

        #return fv1, fv2

    ## Object kernel
    r0 = 100./fv1.vertices.shape[0]
    T0 = 100
    withDiffeons=True

    sm = SurfaceMatchingParam(timeStep=0.1, sigmaKernel=5., sigmaDist=5., sigmaError=1.,algorithm='bfgs',
                              #errorType='diffeonCurrent')
                              errorType='varifold')

    if withDiffeons:
        gdOpt = gd.gdOptimizer(surf=fv1, sigmaDist = .5, DiffeonEpsForNet = r0, testGradient=True, maxIter=100)
        gdOpt.optimize()
        outputDir = '/Users/younes/Development/Results/Diffeons/BallsAlt50_500_d'
        f = SurfaceMatching(Template=fv1, Target=fv2, outputDir=outputDir,param=sm, testGradient=False,
        Diffeons = (gdOpt.c0, gdOpt.S0, gdOpt.idx),
        subsampleTargetSize = 500,
        #DecimationTarget=100,
                            #DiffeonEpsForNet = r0,
                            #DiffeonSegmentationRatio=r0,
                            maxIter=10000, affine='none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)
    else:
        outputDir = '/Users/younes/Development/Results/Diffeons/Scale100_250_0'
        f = SurfaceMatching(Template=fv1, Target=fv2, outputDir=outputDir,param=sm, testGradient=False,
                            subsampleTargetSize = 500,
                            #zeroVar=True,
                            #DecimationTarget=T0,
                            #DiffeonEpsForNet = r0,
                            #DiffeonSegmentationRatio=r0,
                            maxIter=10000, affine='none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)

    loggingUtils.setup_default_logging(outputDir, fileName='info', stdOutput=True)
    f.optimizeMatching()
    return f
    #f.maxIter = 200
    f.param.sigmaError=1.0
    f.setOutputDir('/Users/younes/Development/Results/Diffeons2/Scale2')
    f.restart(DecimationTarget = 2*T0)
    # f.restart(DiffeonEpsForNet = 2*r0)
    #f.restart(DiffeonSegmentationRatio = 0.025)
    #f.maxIter = 300
    f.param.sigmaError=1.
    f.setOutputDir('/Users/younes/Development/Results/Diffeons2/Scale3')
    f.restart(DecimationTarget = 3*T0)
    #f.restart(DiffeonEpsForNet = 4*r0)
    #f.restart(DiffeonSegmentationRatio = 0.05)

    return f

if __name__=="__main__":
    compute(True)
