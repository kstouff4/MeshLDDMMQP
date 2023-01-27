from Common import loggingUtils
from Surfaces.surfaces import *
from Common.kernelFunctions import *
from Surfaces.surfaceMultiPhase import *

def compute():

    outputDir = '/Users/younes/Development/Results/twoBallsSliding2ndTry'
    #outputDir = '/cis/home/younes/MorphingData/twoBallsStitched'
    #outputDir = '/Users/younes/Development/Results/tight_stitched_rigid2_10'
    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info')
    else:
        loggingUtils.setup_default_logging()


    if False:
        Tg = 2000
        npt = 100.0
        ## Build Two colliding ellipses
        [x,y,z] = np.mgrid[0:2*npt, 0:2*npt, 0:2*npt]/npt
        y = y-1
        z = z-1
        x = x-1
        s2 = np.sqrt(2)

        I1 = .06 - ((x+.2)**2 + 0.5*(y-0.25)**2 + (z)**2)  
        fv1 = Surface() ;
        fv1.Isosurface(I1, value = 0, target=Tg, scales=[1, 1, 1])

        I1 = .06 - ((x-.2)**2 + 0.5*(y+0.25)**2 + (z)**2) 
        fv2 = Surface() ;
        fv2.Isosurface(I1, value=0, target=Tg, scales=[1, 1, 1])
    
        u = (z + y)/s2
        v = (z - y)/s2
        I1 = .095 - ((x+.25)**2 + (v)**2 + 0.5*(u+.25)**2) 
        fv3 = Surface() ;
        fv3.Isosurface(I1, value = 0, target=Tg, scales=[1, 1, 1])

        u = (z + y)/s2
        v = (z - y)/s2
        I1 = .095 - ((x-.25)**2 + (v)**2 + 0.5*(u-.25)**2) 
        fv4 = Surface() ;
        fv4.Isosurface(I1, value=0, target=Tg, scales=[1, 1, 1])
    else:
        fv1 = Surface(filename='/Users/younes/Development/Data/surfaces/fshpere1.obj')
        fv2 = Surface(filename='/Users/younes/Development/Data/surfaces/fshpere2.obj')
        fv3 = Surface(filename='/Users/younes/Development/Data/surfaces/fshpere1b.obj')
        fv4 = Surface(filename='/Users/younes/Development/Data/surfaces/fshpere2b.obj')
        fv1.vertices *= 100
        fv2.vertices *= 100
        fv3.vertices *= 100
        fv4.vertices *= 100

    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = 50.0, order=4)
    ## Background kernel
    K2 = Kernel(name='laplacian', sigma = 10.0, order=2)


    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, KparDiffOut=K2, sigmaDist=50., sigmaError=10., errorType='current')
    f = (SurfaceMatching(Template=(fv1,fv2), Target=(fv3,fv4), outputDir=outputDir, param=sm, mu=.001,regWeightOut=1.,
                          testGradient=False, typeConstraint='sliding', maxIter_cg=1000, maxIter_al=100, affine='none', rotWeight=0.1))
    f.optimizeMatching()


    return f

if __name__=="__main__":
    compute()

