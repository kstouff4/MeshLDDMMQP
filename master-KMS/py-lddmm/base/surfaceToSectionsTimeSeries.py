import numpy as np
import numpy.linalg as la
from functools import partial
import logging
from .surfaces import Surface
from .curves import Curve
from .curves import measureNorm0, measureNormDef, measureNormGradient
from .curves import currentNorm0, currentNormDef, currentNormGradient
from .curves import varifoldNorm0, varifoldNormDef, varifoldNormGradient
from .surfaceMatching import SurfaceMatchingParam
from .surfaceTimeSeries import SurfaceTimeMatching
from .surfaceSection import SurfaceSection, Surf2SecDist, Surf2SecGrad, Hyperplane, readTargetFromTXT




## Main class for surface matching
#        Template: surface class (from surface.py); if not specified, opens fileTemp
#        Target: surface class (from surface.py); if not specified, opens fileTarg
#        param: surfaceMatchingParam
#        verb: verbose printing
#        regWeight: multiplicative constant on object regularization
#        affineWeight: multiplicative constant on affine regularization
#        rotWeight: multiplicative constant on affine regularization (supercedes affineWeight)
#        scaleWeight: multiplicative constant on scale regularization (supercedes affineWeight)
#        transWeight: multiplicative constant on translation regularization (supercedes affineWeight)
#        testGradient: evaluates gradient accuracy over random direction (debug)
#        outputDir: where results are saved
#        saveFile: generic name for saved surfaces
#        affine: 'affine', 'similitude', 'euclidean', 'translation' or 'none'
#        maxIter: max iterations in conjugate gradient
class SurfaceToSectionsTimeSeries(SurfaceTimeMatching):
    def __init__(self, Template=None, Target=None, param=None, times= None, maxIter=1000,
                 regWeight = 1.0, affineWeight = 1.0, internalWeight=1.0, verb=True,
                 subsampleTargetSize=-1, affineOnly = False, rescaleTemplate=False,
                 rotWeight = None, scaleWeight = None, transWeight = None, symmetric = False,
                 testGradient=True, saveFile = 'evolution', select_planes = None,
                 saveTrajectories = False, affine = 'none', outputDir = '.',pplot=True):

        self.rescaleTemplate = rescaleTemplate
        self.times = np.array(times)
        self.nTarg = len(Target)
        if param is None:
            self.param = SurfaceMatchingParam()
        else:
            self.param = param

        if self.param.algorithm == 'cg':
             self.euclideanGradient = False
        else:
            self.euclideanGradient = True

        self.setOutputDir(outputDir)
        self.fv0 = None
        self.fv1 = None
        self.fvInit = None
        self.dim = 0
        self.fun_obj = None
        self.fun_obj0 = None
        self.fun_objGrad = None
        self.obj0 = 0
        self.coeffAff = 10
        self.obj = 0
        self.xt = None
        self.hyperplanes = None

        self.set_fun(self.param.errorType)
        self.set_template_and_target(Template, Target, subsampleTargetSize, select_planes)



        self.set_parameters(maxIter=maxIter, regWeight = regWeight, affineWeight = affineWeight,
                            internalWeight=internalWeight, verb=verb, affineOnly = affineOnly,
                            rotWeight = rotWeight, scaleWeight = scaleWeight, transWeight = transWeight,
                            symmetric = symmetric, testGradient=testGradient, saveFile = saveFile,
                            saveTrajectories = saveTrajectories, affine = affine)

        # self.saveFileList = []
        # for kk in range(self.Tsize+1):
        #     self.saveFileList.append(saveFile + f'{kk:03d}')

        self.initialize_variables()
        # self.saveFileList = []
        # for kk in range(self.Tsize+1):
        #     self.saveFileList.append(saveFile + f'{kk:03d}')
        self.gradCoeff = self.x0.shape[0]

        self.pplot = pplot
        self.colors = ('b', 'm', 'g', 'r', 'y', 'k')
        if self.pplot:
            self.initial_plot()
        self.saveRate = 10
        self.forceLineSearch = False

    def set_fun(self, errorType):
        self.param.errorType = errorType
        if errorType == 'current':
            #print('Running Current Matching')
            self.fun_obj0 = partial(currentNorm0, KparDist=self.param.KparDist, weight=1.)
            self.fun_obj = partial(currentNormDef, KparDist=self.param.KparDist, weight=1.)
            self.fun_objGrad = partial(currentNormGradient, KparDist=self.param.KparDist, weight=1.)
        elif errorType=='measure':
            #print('Running Measure Matching')
            self.fun_obj0 = partial(measureNorm0, KparDist=self.param.KparDist)
            self.fun_obj = partial(measureNormDef,KparDist=self.param.KparDist)
            self.fun_objGrad = partial(measureNormGradient,KparDist=self.param.KparDist)
        elif errorType=='varifold':
            self.fun_obj0 = partial(varifoldNorm0, KparDist=self.param.KparDist, weight=1.)
            self.fun_obj = partial(varifoldNormDef, KparDist=self.param.KparDist, weight=1.)
            self.fun_objGrad = partial(varifoldNormGradient, KparDist=self.param.KparDist, weight=1.)
        else:
            print('Unknown error Type: ', self.param.errorType)


    def set_template_and_target(self, Template, Target, subsampleTargetSize=-1, select_planes=None):
        if Template is None:
            logging.error('Please provide a template surface')
            return
        else:
            self.fv0 = Surface(surf=Template)


        self.fv1 =[]
        self.hyperplanes = []
        if type(Target) in (list, tuple):
            for t in Target:
                if type(t) in (list, tuple):
                    self.fv1.append(t)
                elif type(t) == str:
                    fv, hyperplanes = readTargetFromTXT(t)
                    self.fv1.append(fv)
                    self.hyperplanes.append(hyperplanes)
        else:
            logging.error('Target must be a list or tuple of SurfaceSection')
            return
        if select_planes is not None:
            self.selectPlanes((select_planes))
        self.fvInit = Surface(surf=self.fv0)
        self.fix_orientation()
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        if self.fv1:
            for k, fv in enumerate(self.fv1):
                outCurve = ()
                for i, f0 in enumerate(fv):
                    fc = Curve(curve=f0.curve)
                    outCurve += (fc,)
                fc = Curve(curve=outCurve)
                fc.saveVTK(self.outputDir + f'/TargetCurves{k:03d}.vtk')
        self.dim = self.fv0.vertices.shape[1]

    def selectPlanes(self, hs):
        fv1 = []
        for fv in self.fv1:
            f0 = ()
            for k,f in enumerate(fv):
                if f.hypLabel in hs:
                    f0 += (f,)
            fv1.append(f0)
        self.fv1 = fv1


    def fix_orientation(self):
        self.fv0ori = 1
        self.fv1ori = 1



    def initial_plot(self):
        pass
        # fig = plt.figure(3)
        # ax = Axes3D(fig)
        # lim1 = self.addSurfaceToPlot(self.fv0, ax, ec='k', fc='r', al=0.2)
        # for k,f in enumerate(self.fv1):
        #     lim0 = self.addCurveToPlot(f, ax, ec=self.colors[k%len(self.colors)], fc='b', lw=5)
        #     for i in range(3):
        #         lim1[i][0] = min(lim0[i][0], lim1[i][0])
        #         lim1[i][1] = max(lim0[i][1], lim1[i][1])
        # ax.set_xlim(lim1[0][0], lim1[0][1])
        # ax.set_ylim(lim1[1][0], lim1[1][1])
        # ax.set_zlim(lim1[2][0], lim1[2][1])
        # fig.canvas.flush_events()

    def dataTerm(self, _fvDef, fv1 = None, _fvInit = None):
        if fv1 is None:
            fv1 = self.fv1
        obj = 0
        for k,s in enumerate(_fvDef):
            for f in fv1[k]:
                obj += Surf2SecDist(s, f, self.fun_obj, curveDist0=self.fun_obj0)  # plot=101+k)
        obj /= self.param.sigmaError**2
        return obj


    def objectiveFun(self):
        if self.obj is None:
            self.obj0 = 0
            for s in self.fv1:
                for f in s:
                    self.obj0 += self.fun_obj0(f.curve) / (self.param.sigmaError**2)
            (self.obj, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            for k in range(self.nTarg):
                self.fvDef[k].updateVertices(np.squeeze(self.xt[self.jumpIndex[k], :, :]))
            self.obj += self.obj0 + self.dataTerm(self.fvDef)
        return self.obj


    def endPointGradient(self, endPoint=None):
        if endPoint is None:
            endPoint = self.fvDef
        px = []
        for k in range(self.nTarg):
            targGradient = np.zeros(endPoint[k].vertices.shape)
            for f in self.fv1[k]:
                targGradient -= Surf2SecGrad(endPoint[k], f, self.fun_objGrad)
            targGradient /= self.param.sigmaError**2
            px.append(targGradient)
        return px


    def saveCorrectedTarget(self, X0, X1):
        for k, fv in enumerate(self.fv1):
            U = la.inv(X0[self.jumpIndex[k]])
            b = X1[self.jumpIndex[k], ...]
            outCurve = ()
            for i, f0 in enumerate(fv):
                fc = Curve(curve=f0.curve)
                yyt = np.dot(fc.vertices - b, U.T)
                fc.updateVertices(yyt)
                outCurve += (fc,)
            fc = Curve(curve=outCurve)
            fc.saveVTK(self.outputDir + f'/TargetCurveCorrected{k:03d}.vtk')

    def plotAtIteration(self):
        pass


    def endOfIteration(self):
        super().endOfIteration()
        if self.iter % self.saveRate == 0:
            for k0 in range(self.nTarg):
                outCurve = ()
                normals = np.zeros((0,3))
                for k in range(self.hyperplanes[k0].shape[0]):
                    h = Hyperplane(u=self.hyperplanes[k0][k,:3], offset=self.hyperplanes[k0][k,3])
                    f2 = SurfaceSection(hyperplane=h, surf=self.fvDef[k0])
                    outCurve += (f2.curve,)
                    normals = np.concatenate((normals, f2.normals), axis=0)
                c = Curve(outCurve)
                c.saveVTK(self.outputDir+f'/DeformedCurves{k0:02d}.vtk', cell_normals=normals)



    # def optimizeMatching(self):
    #     #print 'dataterm', self.dataTerm(self.fvDef)
    #     #print 'obj fun', self.objectiveFun(), self.obj0
    #     self.coeffAff = self.coeffAff2
    #     grd = self.getGradient(self.gradCoeff)
    #     [grd2] = self.dotProduct(grd, [grd])
    #
    #     self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
    #     self.epsMax = 5.
    #     logging.info('Gradient lower bound: %f' %(self.gradEps))
    #     self.coeffAff = self.coeffAff1
    #     if self.param.algorithm == 'cg':
    #         cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=0.1)
    #     elif self.param.algorithm == 'bfgs':
    #         bfgs.bfgs(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=1.,
    #                   Wolfe=self.param.wolfe, memory=50)
    #     #return self.at, self.xt

