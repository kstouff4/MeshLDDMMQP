from sys import path as sys_path
sys_path.append('..')
from base import loggingUtils
from base.imageMatchingMetamorphosis import Metamorphosis
from base.imageMatchingBase import ImageMatchingParam
import matplotlib
matplotlib.use("QT5Agg")
import matplotlib.pyplot as plt
import pyfftw

pyfftw.config.NUM_THREADS = -1

loggingUtils.setup_default_logging('', stdOutput = True)

ftemp = '../testData/Images/2D/faces/s23/5.pgm'
ftarg = '../testData/Images/2D/faces/s30/10.pgm'

sm = ImageMatchingParam(dim=2, timeStep=0.05, algorithm='cg', sigmaKernel = 5, order=3, sigmaSmooth = .5,
                        kernelSize=24, typeKernel='laplacian', sigmaError=10., rescaleFactor=1, padWidth = 10,
                        affineAlign = 'euclidean')
f = Metamorphosis(Template=ftemp, Target=ftarg, outputDir='../Output/imageMatchingTestMetamorphosis',param=sm,
                    testGradient=False, regWeight = 1., maxIter=1000)

f.optimizeMatching()
plt.ioff()
plt.show()

