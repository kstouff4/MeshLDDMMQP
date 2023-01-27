import matplotlib
matplotlib.use("TKAgg")
from base.surfaces import Surface
from base.kernelFunctions import *
from base.surfaceWithIsometries import *
from base import loggingUtils


def branches(angles, lengths = .75, radii=.05, center = [0,0,0],
             ImageSize = 100, npt = 100):

    [x,y,z] = np.mgrid[-ImageSize:ImageSize+1, -ImageSize:ImageSize+1, -ImageSize:ImageSize+1]/float(ImageSize)

    if (type(lengths) is int) | (type(lengths) is float):
        lengths = np.tile(lengths, len(angles))
    if (type(radii) is int) | (type(radii) is float):
        radii = np.tile(radii, len(angles))

    t = np.mgrid[0:npt+1]/float(npt)
    img = np.zeros(x.shape)
    dst = np.zeros(x.shape)

    for kk,th in enumerate(angles):
        #print kk, th
        u = [np.cos(th[0])*np.sin(th[1]), np.sin(th[0])*np.sin(th[1]), np.cos(th[1])]
        s = (x-center[0])*u[0] + (y-center[1])*u[1] + (z-center[2])*u[2]
        dst = np.sqrt((x-center[0] - s*u[0])**2 + (y-center[1] - s*u[1])**2 + (z-center[2] - s*u[2])**2)
        end = center + lengths[kk]*np.array([np.cos(th[0])*np.sin(th[1]), np.sin(th[0])*np.sin(th[1]), np.cos(th[1])])
        dst1 = (x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2
        dst2 = (x-end[0])**2 + (y-end[1])**2 + (z-end[2])**2
        (I1, I2, I3) = np.nonzero(s<0)
        dst[I1, I2, I3] = np.sqrt(dst1[I1, I2, I3])
        (I1, I2, I3) = np.nonzero(s>lengths[kk])
        dst[I1, I2, I3] = np.sqrt(dst2[I1, I2, I3])
        (I1, I2, I3) = np.nonzero(dst <= radii[kk])
        #print dst.min(), dst.max()
        img[I1, I2, I3] = 1
    fv = Surface()
    print('computing isosurface')
    fv.Isosurface(img, value = 0.5, target=1000, scales=[1, 1, 1])
    return fv

def compute():

    pi = np.pi
    fv1 = branches([[pi/8,pi/4], [pi/4, pi/2.5], [-pi/4, -3*pi/4]])
    fv2 = branches([[pi/8,pi/4], [pi/2.5, pi/4], [-pi/2, -0.7*pi]])
    K1 = Kernel(name='laplacian', sigma = 2.0, order=3)

    sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.05, KparDiff=K1, sigmaDist=20, sigmaError=1.,
                                              errorType='varifold', internalCost='h1')
    dirOut = '/Users/younes/Development/Results/IsometriesShortNorm1'
    loggingUtils.setup_default_logging(dirOut, fileName='info.txt',
                                       stdOutput=True)
    f = SurfaceWithIsometries(Template=fv1, Target=fv2, outputDir=dirOut, centerRadius = [100., 100., 100., 30.],
                              param=sm, mu=.001, testGradient=True, maxIter_cg=100, maxIter_al=100,
                              affine='none', rotWeight=1.,
                              transWeight=1., internalWeight=50.)
    print(f.gradCoeff)
    f.optimizeMatching()


    return f

if __name__ == "__main__":
    compute()


