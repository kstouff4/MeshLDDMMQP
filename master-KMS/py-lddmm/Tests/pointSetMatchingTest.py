from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
import os
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from base import pointSets
from base import loggingUtils
from base.pointSets import PointSet
from base.kernelFunctions import Kernel
from base.affineRegistration import rigidRegistration
from base.pointSetMatching import PointSetMatching, PointSetMatchingParam
import pykeops
#pykeops.clean_pykeops()
plt.ion()

model = 'Gaussian'
dim = 20
true_dim = 3
N = 1000


A = np.random.normal(0,1, size=(dim, dim))
R = expm(A - A.T)
fv0 = np.zeros((N, dim))
fv0[:, :true_dim+1] = np.random.normal(0,1,size=(N, true_dim+1))
fv0 /= np.sqrt((fv0**2).sum(axis=1))[:,None]
fv0 = PointSet(data=np.dot(fv0, R))
fv1pts = np.zeros((N,dim))
fv1pts[:, :true_dim+1] = np.random.normal(0,1,size=(N, true_dim+1))
fv1pts /= np.sqrt((fv1pts**2).sum(axis=1))[:,None]
fv1 = PointSet(data=np.dot(fv1pts, R))

R2, T2 = rigidRegistration((fv1pts, fv1.points))


loggingUtils.setup_default_logging('../Output', stdOutput = True)
sigmaKernel = .5
sigmaDist = 2.
sigmaError = .01

## Object kernel
K1 = Kernel(name='gauss', sigma = sigmaKernel)

sm = PointSetMatchingParam(timeStep=0.1, KparDiff=K1, KparDist=('gauss', sigmaDist),
                           sigmaError=sigmaError, errorType='measure', algorithm='cg')
sm.KparDiff.pk_dtype = 'float64'
sm.KparDist.pk_dtype = 'float64'
f = PointSetMatching(Template=fv0, Target=fv1, outputDir='../Output/pointSetMatchingTest/'+model,param=sm,
                    testGradient=True, maxIter=1000, affine= 'none', rotWeight=.01, transWeight = .01,
                    scaleWeight=10., affineWeight=100.)

f.optimizeMatching()
plt.ioff()
plt.show()

