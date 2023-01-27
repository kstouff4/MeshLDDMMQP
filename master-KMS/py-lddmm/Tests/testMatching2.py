from Surfaces.surfaces import *
from Common.kernelFunctions import *
from Surfaces.surfaceMatching import *
from affineRegistration import *


def compute():

    fv0 = Surface(filename='/Users/younes/Dropbox/ADNI_toyExample/test/amgy_length7_sub101.byu')
    fv1 = Surface(filename='/Users/younes/Dropbox/ADNI_toyExample/test/amgy_length5_sub343.byu')

    ## Object kernel
    K1 = Kernel(name='gauss', sigma = 3.5)

    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=2.5, sigmaError=1., errorType='current')
    R0, T0 = rigidRegistration(surfaces = (fv1.vertices, fv0.vertices),  verb=False, temperature=10., annealing=True)
    fv1.vertices = np.dot(fv1.vertices, R0.T) + T0
    f = SurfaceMatching(Template=fv1, Target=fv0, outputDir='/Users/younes/Development/Results/Surface',param=sm, testGradient=True,
                         maxIter=1000, affine= 'none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)

    f.optimizeMatching()


    return f

if __name__=="__main__":
    compute()
