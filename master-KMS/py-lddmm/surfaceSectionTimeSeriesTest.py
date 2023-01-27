from base.surfaceExamples import Sphere, Heart
from base.surfaces import Surface
from base.surfaceSection import Hyperplane, SurfaceSection
from base.surfaceMatching import SurfaceMatchingParam
from base.surfaceToSectionsTimeSeries import SurfaceToSectionsTimeSeries
from base import loggingUtils
import matplotlib
matplotlib.use("qt5agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from base.kernelFunctions import Kernel
#plt.ion()


loggingUtils.setup_default_logging('', stdOutput=True)
sigmaKernel = .5
sigmaDist = 1.
sigmaError = .5
internalWeight = 1.
regweight = 1.
internalCost = 'h1'


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

fv0 = Surface(surf='/Users/younes/Development/Data/Cardiac/Sample_SAX_LAX_murine/Mice_pre_ED_SurfTemplate_final_template.vtk')
#fv0.flipFaces()
target = []
for ti in range(15):
    target.append('/Users/younes/Development/Data/Cardiac/Sample_SAX_LAX_murine/' +\
             f'SA51_D00_Corrected_Orientation_20201220T18/SA51_extracted_SAX_LAX_time{ti+1:02d}Rigid_flipped.txt')
    #
    # fig = plt.figure(13)
    # ax = Axes3D(fig)
    # lim1 = fv0.addToPlot(ax, ec='k', fc='r', al=0.1)
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
t = 1.
times = ()
for ti in range(15):
    times += (t,)
    t += 0.5

sm = SurfaceMatchingParam(timeStep=0.1, algorithm='cg', KparDiff=K1, sigmaDist=sigmaDist, sigmaError=sigmaError,
                          errorType='current', internalCost=internalCost)
f = SurfaceToSectionsTimeSeries(Template=fv0, Target= target,
                    outputDir=f'/Users/younes/Development/Results/Sections/MouseTimeSeries', param=sm,
                    testGradient=False, regWeight=regweight,
                    # subsampleTargetSize = 500,
                    internalWeight=internalWeight, maxIter=1000, affine='none', rotWeight=10., transWeight=10.,
                    scaleWeight=100., affineWeight=100., select_planes=None, times=times)

f.optimizeMatching()
plt.ioff()
plt.show()

