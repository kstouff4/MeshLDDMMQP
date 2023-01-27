import os.path
import argparse
import logging
from base import loggingUtils
import base.surfaces as surfaces
from base.kernelFunctions import Kernel
from base import surfaceMatching
from base import secondOrderMatching as match
#import secondOrderMatching as match

def compute(tmpl, targetDir, outputDir, display=True, geodesic=False, rescale=False):

    #outputDir = '/Users/younes/Development/Results/biocardTS/spline'
    #outputDir = '/cis/home/younes/MorphingData/twoBallsStitched'
    #outputDir = '/Users/younes/Development/Results/tight_stitched_rigid2_10'
    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info', stdOutput = display)
    else:
        loggingUtils.setup_default_logging()

    #sub = '2729611'
    fv = []
    fv0 = surfaces.Surface(filename=tmpl)
    j = 0 
    print(targetDir+'/imageOutput_time_{0:d}_channel_0.vtk'.format(j))
    while os.path.exists(targetDir+'/imageOutput_time_{0:d}_channel_0.vtk'.format(j)):
        fv = fv + [targetDir + '/imageOutput_time_{0:d}_channel_0.vtk'.format(j) ]
        j += 1

    #print fv
    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = 2.5, order=4)
    sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, KparDist=('gauss', 2.5),
                                              sigmaError=.1, errorType='L2Norm')
    if geodesic:
        logging.info('Running Geodesic Regression')
        f = match.SurfaceMatching(Template=fv0, fileTarg=fv, outputDir=outputDir, param=sm, regWeight=.1, typeRegression='geodesic',
                                  affine='euclidean', rescaleTemplate=rescale, testGradient=False, rotWeight=1.)
    else:
        logging.info('Running Spline Regression')
        f = match.SurfaceMatching(Template=fv0, fileTarg=fv, outputDir=outputDir, param=sm, regWeight=.1, typeRegression='splines2',
                                  affine='euclidean', rescaleTemplate=rescale, testGradient=False, rotWeight=1.)
 
    #, affine='none', rotWeight=0.1))
    f.optimizeMatching()


    return f

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='runs second order longitudinal surface registration')
    parser.add_argument('sub', metavar='sub', type = str, help='subject directory')
    parser.add_argument('--display', action = 'store_true', dest = 'display', default = False, help='To also print on standard output')
    parser.add_argument('--geodesic', action = 'store_true', dest = 'geodesic', default = False, help='Geodesic Regression')
    parser.add_argument('--rescale', action = 'store_true', dest = 'rescale', default = False, help='rescale template to baseline volume')
    args = parser.parse_args()
    #print args.sub
    if args.geodesic:
        outDir = '/cis/home/younes/Development/Results/L2TimeSeriesGeodesicRegression/'
    else:
        outDir = '/cis/home/younes/Development/Results/L2TimeSeriesSplines/'
    compute('/cis/home/younes/MorphingData/TimeseriesResults/estimatedTemplate.byu', 
            '/cis/home/younes/MorphingData/TimeseriesResults/' + args.sub, 
            outDir + args.sub, display=args.display, geodesic = args.geodesic, rescale=args.rescale)

