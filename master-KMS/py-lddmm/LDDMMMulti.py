import argparse
from base import loggingUtils
from base.surfaces import *
from base.kernelFunctions import *
from base.surfaceMultiPhase import *

def compute(args=None, noArgs=True):
    if noArgs:
        outputDir = '/Users/younes/Development/Results/multiShapes'
        #outputDir = '/cis/home/younes/MorphingData/twoBallsStitched'
        #outputDir = '/Users/younes/Development/Results/tight_stitched_rigid2_10'
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
            fTmpl = (fv1, fv2)
            fTarg = (fv3, fv4)
        ## Object kernel
        K1 = Kernel(name='laplacian', sigma = 50.0, order=4)
        ## Background kernel
        K2 = Kernel(name='laplacian', sigma = 10.0, order=2)
        typeConstraint = 'stitched'
        sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, KparDiffOut=K2,
                                  sigmaDist=50., sigmaError=10., errorType='varifold')
        mu = 0.1 ;
    else:
        fTmpl = []
        for name in args.template:
            fTmpl.append(Surface(filename=name))
        for f in fTmpl:
            f.vertices *= args.scaleFactor
        fTarg = []
        for name in args.target:
            fTarg.append(Surface(filename=name))
        for f in fTarg:
            f.vertices *= args.scaleFactor
        ## Object kernel
        K1 = Kernel(name='laplacian', sigma = args.sigmaKernelIn, order=4)
        ## Background kernel
        K2 = Kernel(name='laplacian', sigma = args.sigmaKernelOut, order=2)
        sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, KparDiffOut=K2,
                                  sigmaDist=args.sigmaDist, sigmaError=args.sigmaError, errorType=args.typeError)
        outputDir = args.dirOut
        loggingUtils.setup_default_logging(fileName=outputDir + args.logFile, stdOutput = args.stdOutput)
        if args.sliding:
            typeConstraint = 'slidingV2'
        else:
            typeConstraint = 'stitched'
        mu = args.mu
        
    f = (SurfaceMatching(Template=fTmpl, Target=fTarg, outputDir=outputDir, param=sm, mu=mu,regWeightOut=1.,
                          testGradient=False, typeConstraint=typeConstraint, maxIter_cg=1000, maxIter_al=100, affine='none', rotWeight=0.1))
    f.optimizeMatching()


    return f


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='registration with multiple shapes')
    parser.add_argument('--template', metavar='template', nargs='+', type = str, help='templates', default=None)
    parser.add_argument('--target', metavar='target', nargs='+', type = str, help='targets', default=None)
    parser.add_argument('--sigmaKernelShape', metavar='sigmaKernelShape', type=float,
                        dest='sigmaKernelIn', default = 100, help='kernel width (shapes)') 
    parser.add_argument('--typeKernelShape', metavar='typeKernelShape', type=str,
                        dest='typeKernelIn', default = 'laplacian', help='kernel type (shapes)') 
    parser.add_argument('--sigmaKernelBkg', metavar='sigmaKernelBkg', type=float,
                        dest='sigmaKernelOut', default = 10, help='kernel width (background)') 
    parser.add_argument('--typeKernelBkg', metavar='typeKernelBkg', type=str,
                        dest='typeKernelOut', default = 'laplacian', help='kernel type (background)') 
    parser.add_argument('--sigmaDist', metavar='sigmaDist', type=float, dest='sigmaDist',
                        default = 50, help='kernel width (error term); (default = 50)') 
    parser.add_argument('--sigmaError', metavar='sigmaError', type=float, dest='sigmaError',
                        default = 10, help='weight (error term); (default = 10)') 
    parser.add_argument('--scaleFactor', metavar='scaleFactor', type=float, dest='scaleFactor',
                        default = 1, help='scale factor for all surfaces') 
    parser.add_argument('--dirOut', metavar = 'dirOut', type = str, dest = 'dirOut', default = None, help='Output directory')
    parser.add_argument('--logFile', metavar = 'logFile', type = str, dest = 'logFile', default = 'info.txt', help='Output log file')
    parser.add_argument('--stdout', action = 'store_true', dest = 'stdOutput', default = False, help='To also print on standard output')
    parser.add_argument('--typeError', metavar='typeError', type=str,
                        dest='typeError', default = 'varifold', help='error term: measure, current or varifold') 
    parser.add_argument('--sliding', action = 'store_true', dest = 'sliding', default = False, help='To use sliding constraint')
    parser.add_argument('--mu', metavar='mu', type=float, dest='mu',
                        default = 0.1, help='augmented lagrangian initial weight; (default = 0.1)') 

    args = parser.parse_args()
    if args.target == None or args.template == None:
        print('Error: At least one template and one target are required')
        print('use -h option for help')
        exit()

    compute(args, noArgs=False)

