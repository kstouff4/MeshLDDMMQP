from sys import path as sys_path
sys_path.append('..')
sys_path.append('../base')
from base.surfaceExamples import Sphere, Heart
import os
from base.surfaces import Surface
from base.surfaceSection import Hyperplane, SurfaceSection
from base.surfaceMatching import SurfaceMatchingParam
from base.surfaceToSectionsMatching import SurfaceToSectionsMatching
from base import loggingUtils
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from base.kernelFunctions import Kernel
#plt.ion()


loggingUtils.setup_default_logging('', stdOutput=True)
sigmaKernel = .5
sigmaDist = .1
sigmaError = .1
internalWeight = .5
regweight = 1.
internalCost = None #'h1'


# fv1 = Heart(zoom=100)
# fv0 = Heart(p=1.75, scales=(1.1, 1.5), zoom = 100)
# m = fv1.vertices[:,1].min()
# M = fv1.vertices[:,1].max()
# h = Hyperplane()
# target = ()
# for t in range(1,10):
#     ss = SurfaceSection(surf=fv1, hyperplane = Hyperplane(u=(0, 1, 0), offset = m + 0.1*t*(M-m)))
#     target += (ss,)
#
# m = fv1.vertices[:,2].mean()
# ss = SurfaceSection(surf=fv1, hyperplane = Hyperplane(u=(0, 0, 1), offset = m))
# target += (ss,)
# m = fv1.vertices[:,0].mean()
# ss = SurfaceSection(surf=fv1, hyperplane = Hyperplane(u=(1, 0, 0), offset = m))
# target += (ss,)

errorType = 'measure'
folder = '/Users/younes/Johns Hopkins/Tag MRI Research - Data/WeightedSurfaces/'
#plane = ['Hor1', 'Hor5', 'Hor6']
plane = ['HorAll']
#fv0 = Surface(surf='/Users/younes/Johns Hopkins/Data/Cardiac/Sample_SAX_LAX_murine/Mice_pre_ED_SurfTemplate_final_template.vtk')
#fv0 = Surface(surf='/Users/younes/Development/Data/Cardiac/Data_tagplane_taglines/tagplaneDataHor1/tagplaneHor1.vtk')

fv0 = []
p_lax = 'SA50_43R_20180102_PRE-2018-01-05_JW_Tagplane_LAX_ALL'
p_sax = 'SA50_43R_20180102_PRE-2018-01-05_JW_Tagplane_SAX_ALL'
fv0_ = Surface(surf=folder + p_lax + '/TagplaneLong.vtk')
fv0_ = fv0_.connected_components(split=True)
fv0 += fv0_
fv0_ = Surface(surf=folder + p_sax + '/TagplaneShort.vtk')
fv0_ = fv0_.connected_components(split=True)
fv0 += fv0_

# for p in plane:
#     fv0_ =Surface(surf=folder + 'tagplaneData'+p + '/tagplane' + p + '.vtk')
#     # weights = np.fabs(fv0_.vertices[:, 2])
#     # weights /= weights.max()
#     # fv0_.updateWeights(weights)
#     # fv0_.weights = np.ones(fv0_.weights.shape)
#     fv0_ = fv0_.connected_components(split = True)
#     fv0 += fv0_
#fv0.flipFaces()
ntarg = 15
outputDir = '/Users/younes/Development/Results/Sections/MouseTime_temp_all'
if not os.access(outputDir, os.W_OK):
    if os.access(outputDir, os.F_OK):
        print('Cannot save in ' + outputDir)
    else:
        os.makedirs(outputDir)

for ti in range(1, 15):
    target = ()
    target += (folder + p_lax + f'/curves_phase{ti:02d}.txt',)
    target += (folder + p_sax + f'/curves_phase{ti:02d}.txt',)
    # for p in plane:
    #     target += (folder + 'tagplaneData' + p +  f'/curves_phase{ti:02d}.txt',)
    # target = '/Users/younes/Development/Data/Cardiac/Sample_SAX_LAX_murine/' +\
    #          f'SA51_D00_Corrected_Orientation_20201220T18/SA51_extracted_SAX_LAX_time{ti+1:02d}Rigid_flipped.txt'
    #
    # fig = plt.figure(13)
    # ax = Axes3D(fig)
    # lim1 = fv0.addToPlot(ax, ec='k', fc='r', al=0.1)]
    # ax.set_xlim(lim1[0][0], lim1[0][1])
    # ax.set_ylim(lim1[1][0], lim1[1][1])
    # ax.set_zlim(lim1[2][0], lim1[2][1])
    # colors = ('b', 'm', 'g', 'r', 'y', 'k')
    #
    # for k,ss in enumerate(target):
    #     ss.curve.addToPlot(ax, ec=colors[k%6], lw=5)
    # fig.canvas.flush_events()
    # plt.pause(1000)
    #
    # exit()


    K1 = Kernel(name='laplacian', sigma=sigmaKernel)

    sm = SurfaceMatchingParam(timeStep=0.1, algorithm='cg', KparDiff=K1, KparDist=('gauss', sigmaDist), sigmaError=sigmaError,
                              errorType=errorType, internalCost=internalCost)
    f = SurfaceToSectionsMatching(Template=fv0, Target= target,
                        outputDir=f'/Users/younes/Development/Results/Sections/MouseTime{ti:02d}_temp', param=sm,
                        testGradient=True, regWeight=regweight,
                        # subsampleTargetSize = 500,
                        internalWeight=internalWeight, maxIter=50, affine='translation', rotWeight=10., transWeight=10.,
                        scaleWeight=100., affineWeight=100., select_planes=None)

    f.optimizeMatching()
    fv0 = f.fvDef
    fv0.saveVTK(outputDir + f'/deformed_template{ti:02d}.vtk')
plt.ioff()
plt.show()

