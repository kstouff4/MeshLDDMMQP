import matplotlib
matplotlib.use("TKAgg")
import numpy as np
from base import examples
from base import loggingUtils
from base.curves import Curve
from base.kernelFunctions import Kernel
from base.curveMultiPhase import CurveMultiPhaseParam, CurveMultiPhase

def compute(name = 'TwoCircles'):

    outputDir='/Users/younes/Development/Results/curveMatching3'
    
    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info')
    else:
        loggingUtils.setup_default_logging()

    if name == 'TwoCircles':
        ## Build Two colliding ellipses
        [x,y] = np.mgrid[0:200, 0:200]/100.
        y = y-1
        s2 = np.sqrt(2)

        I1 = .06 - ((x-.30)**2 + 0.5*y**2)
        fv1 = Curve()
        fv1.Isocontour(I1, value = 0, target=750, scales=[1, 1])
        #return

        I1 = .06 - ((x-1.70)**2 + 0.5*y**2)
        fv2 = Curve()
        fv2.Isocontour(I1, value=0, target=750, scales=[1, 1])

        I1 = 0.16 - ((x-.7)**2 + (y+0.25)**2)
        fv3 = Curve()
        fv3.Isocontour(I1, value = 0, target=750, scales=[1, 1])

        I1 = 0.16 - ((x-1.3)**2 + (y-0.25)**2)
        fv4 = Curve()
        fv4.Isocontour(I1, value=0, target=750, scales=[1, 1])
    elif name == 'Peanuts':
        [x,y] = np.mgrid[0:200, 0:200]/100.
        y = y-1
        x = x-1
        fv1 = examples.peanut(r = (0.5, 1), tau=(-.25,-.4))
        fv2 = examples.peanut(r = (1,0.5), tau=(0.25,.4))
        fv3 = examples.peanut(r = (1, 0.5), tau=(.25,-.4))
        fv4 = examples.peanut(r = (0.5,1), tau=(-.25,.4))



    #print fv1.vertices.shape[0], fv2.vertices.shape[0], fv3.vertices.shape[0], fv4.vertices.shape[0] 

    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = 50.0)
    ## Background kernel
    K2 = Kernel(name='laplacian', sigma = 5.0)

    sm = CurveMultiPhaseParam(timeStep=0.1, KparDiff=K1, KparDiffOut=K2, sigmaDist=20., sigmaError=.1,
                              errorType='measure')
    f = (CurveMultiPhase(Template=(fv1,fv2), Target=(fv3,fv4), outputDir=outputDir,
                       param=sm, mu=.01,regWeightOut=1., testGradient=False,
                       typeConstraint='sliding', maxIter_cg=10000, maxIter_al=100, affine='none', rotWeight=10))
    f.optimizeMatching()


    return f


if __name__=="__main__":
    compute('Peanuts')
