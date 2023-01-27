from Common import loggingUtils
from Surfaces.surfaces import *
from Common.kernelFunctions import *
#import surfaceTimeSeries as match
from Surfaces import secondOrderMatching as match, surfaceMatching


def compute():

    outputDir = '/Users/younes/Development/Results/threeBallsSplineAndMomentum'
    #outputDir = '/cis/home/younes/MorphingData/twoBallsStitched'
    #outputDir = '/Users/younes/Development/Results/tight_stitched_rigid2_10'
    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info')
    else:
        loggingUtils.setup_default_logging()

    Tg = 2000
    npt = 100.0
    ## Build Two colliding ellipses
    [x,y,z] = np.mgrid[0:2*npt, 0:2*npt, 0:2*npt]/npt
    y = y-1
    z = z-1
    x = x-1
    s2 = np.sqrt(2)
    
    I1 = .06 - ((x)**2 + 0.5*((y)**2) + (z)**2)  
    fv1 = Surface()
    fv1.Isosurface(I1, value = 0, target=Tg, scales=[1, 1, 1])

    I1 = .08 - ((x+.05)**2 + 0.3*(y-0.05)**2 + (z)**2)  
    fv2 = Surface() ;
    fv2.Isosurface(I1, value = 0, target=Tg, scales=[1, 1, 1])

    I1 = .06 - (1.25*(x)**2 + 0.5*(y-0.25)**2 + 0.75*(z)**2)  
    fv3 = Surface() ;
    fv3.Isosurface(I1, value = 0, target=Tg, scales=[1, 1, 1])


    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = 50.0, order=4)


    sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=50., sigmaError=10., errorType='measure')
    f = (match.SurfaceMatching(Template=fv1, Targets=(fv2,fv3), outputDir=outputDir, param=sm, typeRegression='both',
                          testGradient=False, maxIter=1000))
    #, affine='none', rotWeight=0.1))
    f.optimizeMatching()


    return f

if __name__=="__main__":
    compute()

