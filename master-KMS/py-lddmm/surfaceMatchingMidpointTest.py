import numpy as np
import os
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
    print(os.environ)
import matplotlib.pyplot as plt
from base import surfaces
from base import loggingUtils
from base.surfaceExamples import Sphere, Torus, Ellipse
from base.kernelFunctions import Kernel
from base.affineRegistration import rigidRegistration
from base.surfaceMatchingMidpoint import SurfaceMatchingMidpoint
from base.surfaceMatching import SurfaceMatchingParam, SurfaceMatching
plt.ion()
loggingUtils.setup_default_logging('', stdOutput = True)
sigmaKernel = 2.5
sigmaDist = 2.
sigmaError = 1.
regweight = 1.
internalWeight = 1.
internalCost = 'None'
## Object kernel

c = np.zeros(3)
d = np.array([11, 0, 0])
# ftemp = Sphere(c, 10)
# ftarg = Torus(c, 10, 4)
ftemp = Ellipse(c, (20, 10, 10))
ftarg = surfaces.Surface(surf= (Sphere(c-d, 10), Sphere(c+d, 10)))

K1 = Kernel(name='laplacian', sigma = sigmaKernel)
vfun = (lambda u: 1+u, lambda u: np.ones(u.shape))

sm = SurfaceMatchingParam(timeStep=0.1, algorithm='cg', KparDiff=K1, KparDist=('gauss', sigmaDist), sigmaError=sigmaError,
                          errorType='varifold', internalCost=internalCost)
f = SurfaceMatchingMidpoint(Template=ftemp, Target=ftarg, outputDir='/cis/home/younes/Development/Results/TopChange/Midpoint_Balls',
                            param=sm, testGradient=False, regWeight=regweight,
                            internalWeight=internalWeight, maxIter=200, affine= 'none', rotWeight=.01,
                            transWeight = .01, scaleWeight=10., affineWeight=100.)

f.optimizeMatching()
sm = SurfaceMatchingParam(timeStep=0.05, algorithm='cg', KparDiff=K1, KparDist=('gauss', sigmaDist), sigmaError=sigmaError,
                          errorType='varifold', vfun=vfun, internalCost=internalCost)
f = SurfaceMatching(Template=ftemp, Target=ftarg, outputDir='/cis/home/younes/Development/Results/TopChange/Endpoint_Balls',
                            param=sm, testGradient=True, regWeight=regweight,
                            internalWeight=internalWeight, maxIter=200, affine= 'none', rotWeight=.01,
                            transWeight = .01, scaleWeight=10., affineWeight=100.)

f.optimizeMatching()
plt.ioff()
plt.show()

