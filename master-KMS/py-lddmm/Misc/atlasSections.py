from Common import loggingUtils
from Curves.curves import *
from Common.kernelFunctions import *
from Curves.curveMultiPhase import *

def compute():
    outputDir='/Users/younes/Development/Results/atlasSections49'
    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info')
    else:
        loggingUtils.setup_default_logging()
    ## Build Two colliding ellipses
    fv01 = Curve(filename='/cis/home/younes/MorphingData/Mostofskycurves/atlas04_curve01.txt')
    fv02 = Curve(filename='/cis/home/younes/MorphingData/Mostofskycurves/atlas04_curve02.txt')
    fv03 = Curve(filename='/cis/home/younes/MorphingData/Mostofskycurves/atlas04_curve03.txt')
    fv11 = Curve(filename='/cis/home/younes/MorphingData/Mostofskycurves/atlas09_curve01.txt')
    fv12 = Curve(filename='/cis/home/younes/MorphingData/Mostofskycurves/atlas09_curve02.txt')
    fv13 = Curve(filename='/cis/home/younes/MorphingData/Mostofskycurves/atlas09_curve03.txt')

    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = 10.0)
    ## Background kernel
    K2 = Kernel(name='laplacian', sigma = 1.0)

    sm = CurveMatchingParam(timeStep=0.1, KparDiff=K1, KparDiffOut=K2, sigmaDist=2., sigmaError=.1, errorType='measure')
    f = (CurveMatching(Template=(fv01,fv02,fv03), Target=(fv11,fv12,fv13), outputDir=outputDir,
                       param=sm, mu=1.,regWeightOut=1., testGradient=False,
                       typeConstraint='stitched', maxIter_cg=10000, maxIter_al=100, affine='none', rotWeight=10))
    f.optimizeMatching()


    return f


if __name__=="__main__":
    compute()
