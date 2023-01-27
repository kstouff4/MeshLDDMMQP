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
from base import loggingUtils
from base.surfaces import Surface
from base.kernelFunctions import Kernel
from base.surfaceTemplate import SurfaceTemplateParam, SurfaceTemplate
import glob

project = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus'
home = '/cis/home/younes'
#project = '/Users/younes/Development/Data/biocard'
#home = '/Users/younes'
files = glob.glob(project+'/2_qc_flipped_registered/*L_reg.byu')
print(len(files))
fv1 = []
for k in range(len(files)):
    if files[k].split('_')[-5] == '1':
        fv1.append(Surface(surf = files[k]))
print(len(fv1))

K1 = Kernel(name='gauss', sigma = 6.5)
K0 = Kernel(name='laplacian', sigma = 2.5)
Kdist = Kernel(name='gauss', sigma = 2.5)

loggingUtils.setup_default_logging(home + '/Development/Results/surfaceTemplateBiocard', fileName='info.txt', stdOutput = True)
sm = SurfaceTemplateParam(timeStep=0.1, KparDiff=K1, KparDiff0 = K0, KparDist=Kdist, sigmaError=1., errorType='varifold',
                          internalCost='elastic')
f = SurfaceTemplate(HyperTmpl=None, Targets=fv1, outputDir=home + '/Development/Results/surfaceTemplateBiocard',param=sm, testGradient=True,
                    lambdaPrior = 0.1, sgd = 5, internalWeight = 50, 
                    maxIter=1000, affine='none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)
f.computeTemplate()


