from Curves.curves import *
from Common.kernelFunctions import *
from Curves.curveMultiPhase import *
from Common import loggingUtils


def compute():
    outputDir='/Users/younes/Development/Results/curves_Stitched_0'
    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info')
    else:
        loggingUtils.setup_default_logging()
    ## Build Two colliding ellipses
    t = np.arange(0, 2*np.pi, 0.02)

    p = np.zeros([len(t), 2])
    p[:,0] = -1 + (1 - 0.25* np.cos(2*t)) * np.cos(t)
    p[:,1] = -1 + (1 - 0.25* np.cos(2*t)) * np.sin(t)
    fv1 = Curve(pointSet = p) ;

    p[:,0] = 1. + 0.75 * (1 - 0.15* np.cos(6*t)) * np.cos(t)
    p[:,1] = 1 + 0.75*(1 - 0.25* np.cos(6*t)) * np.sin(t)
    #p[:,0] = 1 +  np.cos(t)
    #p[:,1] = 1 + 0.75 * np.sin(t)
    fv2 = Curve(pointSet = p) ;

    p[:,0] = -1.0 + 0.75 * np.cos(t)
    p[:,1] = np.sin(t)
    fv3 = Curve(pointSet = p) ;

    p[:,0] = 1.0 + (1 - 0.25* np.cos(6*t)) * np.cos(t)
    p[:,1] =  (1 - 0.25* np.cos(6*t)) * np.sin(t)
    fv4 = Curve(pointSet = p) ;

    print fv1.vertices.shape[0], fv2.vertices.shape[0], fv3.vertices.shape[0], fv4.vertices.shape[0] 

    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = 1)
    ## Background kernel
    K2 = Kernel(name='laplacian', sigma = .1)

    sm = CurveMatchingParam(timeStep=0.1, KparDiff=K1, KparDiffOut=K2, sigmaDist=0.5, sigmaError=.01, errorType='current')
    f = (CurveMatching(Template=(fv1,fv2), Target=(fv3,fv4), outputDir='/Users/younes/Development/Results/curves_Stitched_0',param=sm, mu=.001,regWeightOut=1., testGradient=False,
                       typeConstraint='sliding', maxIter_cg=10000, maxIter_al=100, affine='none', rotWeight=10))
    f.optimizeMatching()


    return f

if __name__=="__main__":
    compute()
