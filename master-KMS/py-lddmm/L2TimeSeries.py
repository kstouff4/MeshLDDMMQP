import os.path
import argparse
from base import loggingUtils
import base.surfaces as surfaces
from base.kernelFunctions import *
from base import surfaceMatching
#import secondOrderMatching as match

def compute(tmpl, targetDir, outputDir, display=True, Atrophy=False, rescale=False):
    if Atrophy:
        import surfaceTimeSeriesAtrophy as match
    else:
        import surfaceTimeSeries as match

    #outputDir = '/Users/younes/Development/Results/biocardTS/spline'
    #outputDir = '/Users/younes/Development/Results/L2TimeSeriesAtrophy'
    #outputDir = '/cis/home/younes/MorphingData/twoBallsStitched'
    #outputDir = '/Users/younes/Development/Results/tight_stitched_rigid2_10'
    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info', stdOutput=display)
    else:
        loggingUtils.setup_default_logging()

    fv = []
    fv0 = surfaces.Surface(filename=tmpl)
    j = 0 
    print(targetDir+'/imageOutput_time_{0:d}_channel_0.vtk'.format(j))
    while os.path.exists(targetDir+'/imageOutput_time_{0:d}_channel_0.vtk'.format(j)):
        fv = fv + [targetDir + '/imageOutput_time_{0:d}_channel_0.vtk'.format(j) ]
        j += 1

    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = 2.5, order=4)
    sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=2.5, sigmaError=.1, errorType='L2Norm')
    if Atrophy:
        f = match.SurfaceMatching(Template=fv0, fileTarg=fv, outputDir=outputDir, param=sm, regWeight=.1,
                                affine='euclidean', rescaleTemplate=rescale, testGradient=False, rotWeight=1.,  maxIter_cg=100, mu=0.0001)
    else:
       f = match.SurfaceMatching(Template=fv0, fileTarg=fv, outputDir=outputDir, param=sm, regWeight=1.,
                                affine='euclidean', rescaleTemplate=rescale, testGradient=False, affineWeight=10,  maxIter=1000)
 
    #, affine='none', rotWeight=0.1))
    f.optimizeMatching()


    return f

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='runs first order longitudinal surface registration')
    parser.add_argument('sub', metavar='sub', type = str, help='subject directory')
    parser.add_argument('--display', action = 'store_true', dest = 'display', default = False, help='To also print on standard output')
    parser.add_argument('--atrophy', action = 'store_true', dest = 'atrophy', default = False, help='Atrophy Constraint')
    parser.add_argument('--rescale', action = 'store_true', dest = 'rescale', default = False, help='rescale template to baseline volume')
    args = parser.parse_args()
    #print args.sub
    compute('/cis/home/younes/MorphingData/TimeseriesResults/estimatedTemplate.byu', 
            '/cis/home/younes/MorphingData/TimeseriesResults/' + args.sub, 
            '/cis/home/younes/Development/Results/L2TimeSeriesAtrophy/'+ args.sub, display=args.display, Atrophy=args.atrophy, rescale=args.rescale)

