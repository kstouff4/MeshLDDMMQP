import os
import time
from copy import deepcopy
import numpy as np
import numpy.linalg as la
import logging
import h5py
import glob
from warnings import warn
from . import conjugateGradient as cg, kernelFunctions as kfun, pointEvolution as evol, bfgs, sgd
from . import surfaces, surface_distances as sd
from . import pointSets, pointset_distances as psd
from . import matchingParam
from .affineBasis import AffineBasis, getExponential, gradExponential
from . import pointEvolution as evol
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from numpy.random import default_rng
rng = default_rng()


## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
class SurfaceMatchingParam(matchingParam.MatchingParam):
    def __init__(self, timeStep = .1, algorithm='cg', Wolfe=True, KparDiff = None, KparDist = None,
                 sigmaError = 1.0, errorType = 'measure', vfun = None, internalCost = None):
        super().__init__(timeStep=timeStep, algorithm = algorithm, Wolfe=Wolfe,
                         KparDiff = KparDiff, KparDist = KparDist, sigmaError=sigmaError,
                         errorType = errorType, vfun=vfun)
        self.sigmaError = sigmaError
        self.internalCost = internalCost

class Direction(dict):
    def __init__(self):
        super(Direction, self).__init__()
        self['diff'] =None
        self['aff'] = None
        self['initx'] = None
        self['pts'] = None


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
class SurfaceMatching(object):
    def __init__(self, Template=None, Target=None, Landmarks = None,
                 param=None, maxIter=1000, passenger = None,
                 regWeight = 1.0, affineWeight = 1.0, internalWeight=1.0, mode='normal',
                 unreduced=False, unreducedWeight = 1.0,
                 subsampleTargetSize=-1, affineOnly = False, testGradient = None,
                 rotWeight = None, scaleWeight = None, transWeight = None, symmetric = False,
                 saveFile = 'evolution',
                 saveTrajectories = False, affine = 'none', outputDir = '.',pplot=False):
        if param is None:
            self.param = SurfaceMatchingParam()
        else:
            self.param = param

        if testGradient is not None:
            logging.warning('Warning: testGradient has no effect on SurfaceMatching Initialization and is deprecated.\n'+
                 '        Use testGradient = False | True after initialization to set this parameter.')

        self.setDotProduct(unreduced)

        if self.param.algorithm == 'sgd':
            self.sgd = True
        else:
            self.sgd = False

        self.fv0 = None
        self.fv1 = None
        self.fvInit = None
        self.dim = 0
        self.fun_obj = None
        self.fun_obj0 = None
        self.fun_objGrad = None
        self.obj0 = 0
        self.coeffAff = 1
        self.obj = 0
        self.xt = None
        self.Kdiff_dtype = self.param.KparDiff.pk_dtype
        self.Kdist_dtype = self.param.KparDist.pk_dtype
        if self.param.algorithm == 'sgd':
            self.unreduced = True
        else:
            self.unreduced = unreduced
        self.unreducedWeight = unreducedWeight
        if Target is not None and (issubclass(type(Target), pointSets.PointSet) or
                                   (type(Target) in (tuple, list) and issubclass(type(Target[0]), pointSets.PointSet))):
            self.param.errorType = 'PointSet'

        self.setOutputDir(outputDir)
        self.set_template_and_target(Template, Target, subsampleTargetSize)
        self.unreducedWeight *=  1000.0 / self.fv0.vertices.shape[0]
        #if self.unreduced:
        if self.param.algorithm == 'sgd':
            self.ds = 1.
        else:
            self.ds = self.fv0.surfArea() /  self.fv0.vertices.shape[0]
        #else:
        #    self.ds = 1.

        self.set_landmarks(Landmarks)
        self.set_fun(self.param.errorType, vfun=self.param.vfun)
        self.set_parameters(maxIter=maxIter, regWeight = regWeight, affineWeight = affineWeight,
                            internalWeight=internalWeight, mode=mode,  affineOnly = affineOnly,
                            rotWeight = rotWeight, scaleWeight = scaleWeight, transWeight = transWeight,
                            symmetric = symmetric, saveFile = saveFile,
                            saveTrajectories = saveTrajectories, affine = affine)
        self.initialize_variables()
        self.gradCoeff = self.x0.shape[0]
        self.set_passenger(passenger)
        self.pplot = pplot
        if self.pplot:
            self.initial_plot()
        if self.param.algorithm == 'sgd':
            self.set_sgd()


    def setDotProduct(self, unreduced=False):
        if self.param.algorithm == 'cg' and not unreduced:
             self.euclideanGradient = False
             self.dotProduct = self.dotProduct_Riemannian
        else:
            self.euclideanGradient = True
            self.dotProduct = self.dotProduct_euclidean

    def set_passenger(self, passenger):
        self.passenger = passenger
        if isinstance(self.passenger, surfaces.Surface):
            self.passenger_points = self.passenger.vertices
        elif self.passenger is not None:
            self.passenger_points = self.passenger
        else:
            self.passenger_points = None
        self.passengerDef = deepcopy(self.passenger)



    def set_parameters(self, maxIter=1000,
                 regWeight = 1.0, affineWeight = 1.0, internalWeight=1.0, mode = 'normal',
                 affineOnly = False,
                 rotWeight = None, scaleWeight = None, transWeight = None, symmetric = False,
                 saveFile = 'evolution',
                 saveTrajectories = False, affine = 'none'):
        self.saveRate = 10
        self.gradEps = -1
        self.randomInit = False
        self.iter = 0
        self.maxIter = maxIter
        if mode in ('normal', 'debug'):
            self.verb = True
            if mode == 'debug':
                self.testGradient = True
            else:
                self.testGradient = False
        else:
            self.verb = False
        self.mode = mode
        self.saveTrajectories = saveTrajectories
        self.symmetric = symmetric
        self.internalWeight = internalWeight
        self.regweight = regWeight
        self.reset = True

        self.affineOnly = affineOnly
        self.affine = affine
        self.affB = AffineBasis(self.dim, affine)
        self.affineDim = self.affB.affineDim
        self.affineBasis = self.affB.basis
        self.affineWeight = affineWeight * np.ones([self.affineDim, 1])
        if (len(self.affB.rotComp) > 0) and (rotWeight is not None):
            self.affineWeight[self.affB.rotComp] = rotWeight
        if (len(self.affB.simComp) > 0) and (scaleWeight is not None):
            self.affineWeight[self.affB.simComp] = scaleWeight
        if (len(self.affB.transComp) > 0) and (transWeight is not None):
            self.affineWeight[self.affB.transComp] = transWeight

        if self.param.internalCost == 'h1':
            self.internalCost = sd.normGrad
            self.internalCostGrad = sd.diffNormGrad
        elif self.param.internalCost == 'elastic':
            self.internalCost = sd.elasticNorm
            self.internalCostGrad = sd.diffElasticNorm
        else:
            if self.param.internalCost is not None:
                logging.info(f'unknown {self.internalCost:.04f}')
            self.internalCost = None



        self.obj = None
        self.objTry = None
        self.saveFile = saveFile
        self.coeffAff1 = 1.
        if self.param.algorithm == 'cg':
            self.coeffAff2 = 100.
        else:
            self.coeffAff2 = 1.
        self.coeffAff = self.coeffAff1
        self.coeffInitx = .1
        self.affBurnIn = 25
        self.forceLineSearch = False
        self.saveEPDiffTrajectories = False
        self.varCounter = 0
        self.trajCounter = 0
        self.unreducedResetRate = 50


    def set_sgd(self, control=100, template=100, target=100):
        self.weightSubset = 0.
        self.sgdEpsInit = 1e-4

        self.sgdNormalization = 'sdev'
        self.sgdBurnIn = 10000
        self.sgdMeanSelectControl = control
        self.sgdMeanSelectTemplate = template
        self.sgdMeanSelectTarget = target
        self.probSelectControl = min(1.0, self.sgdMeanSelectControl / self.fv0.vertices.shape[0])
        self.probSelectFaceTemplate = min(1.0, self.sgdMeanSelectTemplate / self.fv0.faces.shape[0])
        self.probSelectFaceTarget = min(1.0, self.sgdMeanSelectTarget / self.fv1.faces.shape[0])
        self.probSelectVertexTemplate = np.ones(self.fv0.vertices.shape[0])
        nf = np.zeros(self.fv0.vertices.shape[0])
        for k in range(self.fv0.faces.shape[0]):
            for j in range(3):
                self.probSelectVertexTemplate[self.fv0.faces[k,j]] *= \
                    1 - self.sgdMeanSelectTemplate/(self.fv0.faces.shape[0] - nf[self.fv0.faces[k,j]])
                nf[self.fv0.faces[k,j]] += 1

        self.probSelectVertexTemplate = 1 - self.probSelectVertexTemplate
        self.stateSubset = None


    def set_template_and_target(self, Template, Target, subsampleTargetSize=-1, misc=None):
        if Template is None:
            logging.error('Please provide a template surface')
            return
        else:
            self.fv0 = surfaces.Surface(surf=Template)

        if self.param.errorType != 'currentMagnitude':
            if Target is None:
                logging.error('Please provide a target surface')
                return
            else:
                if self.param.errorType == 'L2Norm':
                    self.fv1 = surfaces.Surface()
                    self.fv1.readFromImage(Target)
                elif self.param.errorType == 'PointSet':
                    self.fv1 = pointSets.PointSet(data=Target)
                else:
                    self.fv1 = surfaces.Surface(surf=Target)
        else:
            self.fv1 = None
        self.fvInit = surfaces.Surface(surf=self.fv0)
        self.fix_orientation()
        if subsampleTargetSize > 0:
            self.fvInit.Simplify(subsampleTargetSize)
            logging.info('simplified template %d' %(self.fv0.vertices.shape[0]))
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        if self.fv1:
            self.fv1.saveVTK(self.outputDir+'/Target.vtk')
        self.dim = self.fv0.vertices.shape[1]

    def set_landmarks(self, landmarks):
        if landmarks is None:
            self.match_landmarks = False
            self.tmpl_lmk = None
            self.targ_lmk = None
            self.def_lmk = None
            self.wlmk = 0
            return

        self.match_landmarks = True
        tmpl_lmk, targ_lmk, self.wlmk = landmarks
        self.tmpl_lmk = pointSets.PointSet(data=tmpl_lmk)
        self.targ_lmk = pointSets.PointSet(data=targ_lmk)

    def fix_orientation(self, fv1=None):
        if fv1 is None:
            fv1 = self.fv1
        if issubclass(type(fv1), surfaces.Surface):
            self.fv0.getEdges()
            fv1.getEdges()
            self.closed = self.fv0.bdry.max() == 0 and fv1.bdry.max() == 0
            if self.closed:
                v0 = self.fv0.surfVolume()
                if self.param.errorType == 'L2Norm' and v0 < 0:
                    self.fv0.flipFaces()
                    v0 = -v0
                v1 = fv1.surfVolume()
                if v0*v1 < 0:
                    fv1.flipFaces()
            if self.closed:
                z= self.fvInit.surfVolume()
                if z < 0:
                    self.fv0ori = -1
                else:
                    self.fv0ori = 1

                z= fv1.surfVolume()
                if z < 0:
                    self.fv1ori = -1
                else:
                    self.fv1ori = 1
            else:
                self.fv0ori = 1
                self.fv1ori = 1
        else:
            self.fv0ori = 1
            self.fv1ori = 1
        #self.fv0Fine = surfaces.Surface(surf=self.fv0)
        logging.info('orientation: {0:d}'.format(self.fv0ori))


    def initialize_variables(self):
        self.Tsize = int(round(1.0/self.param.timeStep))
        self.nvert = self.fvInit.vertices.shape[0]
        if self.match_landmarks:
            self.x0 = np.concatenate((self.fvInit.vertices, self.tmpl_lmk.points), axis=0)
            self.nlmk = self.tmpl_lmk.points.shape[0]
        else:
            self.x0 = np.copy(self.fvInit.vertices)
            self.nlmk = 0
        if self.symmetric:
            self.x0try = np.copy(self.x0)
        else:
            self.x0try = None
        self.fvDef = surfaces.Surface(surf=self.fvInit)
        if self.match_landmarks:
            self.def_lmk = pointSets.PointSet(data=self.tmpl_lmk)
        self.npt = self.x0.shape[0]

        self.at = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        if self.randomInit:
            self.at = np.random.normal(0, 1, self.at.shape)
        self.atTry = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])


        if self.param.algorithm == 'sgd':
            self.SGDSelectionPts = [None, None]
            # self.SGDSelectionCost = [None, None]

        if self.affineDim > 0:
            self.Afft = np.zeros([self.Tsize, self.affineDim])
            self.AfftTry = np.zeros([self.Tsize, self.affineDim])
        else:
            self.Afft = None
            self.AfftTry = None
        self.xt = np.tile(self.x0, [self.Tsize+1, 1, 1])
        self.xtTry = np.copy(self.xt)
        if self.unreduced:
            # v = self.fv1.vertices.mean(axis=0) - self.fv0.vertices.mean(axis=0)
            # t = np.linspace(0, 1, self.Tsize)
            # self.ct = self.x0[None, :, :] + t[:, None, None] * v[None, None, :]
            self.ct = np.tile(self.x0, [self.Tsize, 1, 1])
            if self.randomInit:
                self.ct += np.random.normal(0, 1, self.ct.shape)
            self.ctTry = np.copy(self.ct)
        else:
            self.ct = None
            self.ctTry = None



        self.v = np.zeros([self.Tsize+1, self.npt, self.dim])
        self.saveFileList = []
        for kk in range(self.Tsize+1):
            self.saveFileList.append(self.saveFile + f'{kk:03d}')


    def initial_plot(self):
        fig = plt.figure(3)
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        lim1 = self.addSurfaceToPlot(self.fv0, ax, ec='k', fc='r', setLim=False)
        if self.fv1:
            lim0 = self.addSurfaceToPlot(self.fv1, ax, ec='k', fc='b', setLim=False)
        else:
            lim0 = lim1
        ax.set_xlim(min(lim0[0][0], lim1[0][0]), max(lim0[0][1], lim1[0][1]))
        ax.set_ylim(min(lim0[1][0], lim1[1][0]), max(lim0[1][1], lim1[1][1]))
        ax.set_zlim(min(lim0[2][0], lim1[2][0]), max(lim0[2][1], lim1[2][1]))
        if self.match_landmarks:
            ax.scatter3D(self.tmpl_lmk.points[:,0], self.tmpl_lmk.points[:,1], self.tmpl_lmk.points[:,2], color='r')
            ax.scatter3D(self.targ_lmk.points[:, 0], self.targ_lmk.points[:, 1], self.targ_lmk.points[:, 2], color='b')
        fig.canvas.flush_events()

    def set_fun(self, errorType, vfun = None):
        self.param.errorType = errorType
        if errorType == 'current':
            #print('Running Current Matching')
            self.fun_obj0 = partial(sd.currentNorm0, KparDist=self.param.KparDist, weight=1.)
            self.fun_obj = partial(sd.currentNormDef, KparDist=self.param.KparDist, weight=1.)
            self.fun_objGrad = partial(sd.currentNormGradient, KparDist=self.param.KparDist, weight=1.)
        elif errorType == 'currentMagnitude':
            #print('Running Current Matching')
            self.fun_obj0 = lambda fv1 : 0
            self.fun_obj = partial(sd.currentMagnitude, KparDist=self.param.KparDist)
            self.fun_objGrad = partial(sd.currentMagnitudeGradient, KparDist=self.param.KparDist)
            # self.fun_obj0 = curves.currentNorm0
            # self.fun_obj = curves.currentNormDef
            # self.fun_objGrad = curves.currentNormGradient
        elif errorType=='measure':
            #print('Running Measure Matching')
            self.fun_obj0 = partial(sd.measureNorm0, KparDist=self.param.KparDist)
            self.fun_obj = partial(sd.measureNormDef,KparDist=self.param.KparDist)
            self.fun_objGrad = partial(sd.measureNormGradient,KparDist=self.param.KparDist)
        elif errorType=='varifold':
            self.fun_obj0 = partial(sd.varifoldNorm0, KparDist=self.param.KparDist, fun=vfun,
                                    dtype=self.param.KparDist.pk_dtype)
            self.fun_obj = partial(sd.varifoldNormDef, KparDist=self.param.KparDist, fun=vfun,
                                   dtype=self.param.KparDist.pk_dtype)
            self.fun_objGrad = partial(sd.varifoldNormGradient, KparDist=self.param.KparDist, fun=vfun,
                                       dtype=self.param.KparDist.pk_dtype)
        elif errorType == 'L2Norm':
            self.fun_obj0 = None
            self.fun_obj = None
            self.fun_objGrad = None
        elif errorType == 'PointSet':
            self.fun_obj0 = partial(sd.measureNormPS0, KparDist=self.param.KparDist)
            self.fun_obj = partial(sd.measureNormPSDef, KparDist=self.param.KparDist)
            self.fun_objGrad = partial(sd.measureNormPSGradient, KparDist=self.param.KparDist)
        else:
            logging.info(f'Unknown error Type:  {self.param.errorType}')

        if self.match_landmarks:
            self.lmk_obj0 = psd.L2Norm0
            self.lmk_obj = psd.L2NormDef
            self.lmk_objGrad = psd.L2NormGradient
        else:
            self.lmk_obj0 = None
            self.lmk_obj = None
            self.lmk_objGrad = None


    def addSurfaceToPlot(self, fv1, ax, ec = 'b', fc = 'r', al=.5, lw=1, setLim=False):
        return fv1.addToPlot(ax, ec = ec, fc = fc, al=al, lw=lw)

    def setOutputDir(self, outputDir, clean=True):
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                logging.error('Cannot save in ' + outputDir)
                return
            else:
                os.makedirs(outputDir)

        if clean:
            fileList = glob.glob(outputDir + '/*.vtk')
            for f in fileList:
                os.remove(f)


    def dataTerm(self, _fvDef, fv1 = None, _fvInit = None, _lmk_def = None, lmk1 = None):
        if fv1 is None:
            fv1 = self.fv1
        if self.param.errorType == 'L2Norm':
            obj = sd.L2Norm(_fvDef, fv1.vfld) / (self.param.sigmaError ** 2)
        else:
            obj = self.fun_obj(_fvDef, fv1) / (self.param.sigmaError**2)
            if _fvInit is not None:
                obj += self.fun_obj(_fvInit, self.fv0) / (self.param.sigmaError**2)

        if self.match_landmarks:
            if _lmk_def is None:
                logging.error('Missing deformed landmarks')
            if lmk1 is None:
                lmk1 = self.targ_lmk.points
            obj += self.wlmk * self.lmk_obj(_lmk_def.points, lmk1)
        #print 'dataterm = ', obj + self.obj0
        return obj

    def  objectiveFunDef(self, control, Afft=None, kernel = None, withTrajectory = True, withJacobian=False,
                         fv0 = None, regWeight = None):
        if fv0 is None:
            fv0 = self.fv0
        if self.match_landmarks:
            x0 = np.concatenate((fv0.vertices, self.tmpl_lmk.points), axis=0)
        else:
            x0 = fv0.vertices
        if kernel is None:
            kernel = self.param.KparDiff
        #print 'x0 fun def', x0.sum()
        if regWeight is None:
            regWeight = self.regweight
        if np.isscalar(regWeight):
            regWeight_ = np.zeros(self.Tsize)
            regWeight_[:] = regWeight
        else:
            regWeight_ = regWeight
        timeStep = 1.0/self.Tsize
        if Afft is not None:
            dim2 = self.dim**2
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, Afft[t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2+self.dim]
        else:
            A = None

        if self.unreduced:
            ct = control[0]
            at = control[1]
        else:
            ct = None
            at = control

        if withJacobian:
            if self.unreduced:
                xt,Jt  = evol.landmarkSemiReducedEvolutionEuler(x0, ct, at*self.ds, kernel, affine=A, withJacobian=True)
            else:
                xt,Jt  = evol.landmarkDirectEvolutionEuler(x0, at*self.ds, kernel, affine=A, withJacobian=True)
        else:
            Jt = None
            if self.unreduced:
                xt = evol.landmarkSemiReducedEvolutionEuler(x0, ct, at*self.ds, kernel, affine=A)
            else:
                xt  = evol.landmarkDirectEvolutionEuler(x0, at*self.ds, kernel, affine=A)
        #print xt[-1, :, :]
        #print obj
        obj=0
        obj1 = 0
        obj2 = 0
        obj3 = 0
        foo = surfaces.Surface(surf=fv0)
        for t in range(self.Tsize):
            z = xt[t, :, :]
            # if self.unreduced:
            #     z2 = (z + xt[t+1, :, :])/2
            # else:
            #     z2 = None
            a = at[t, :, :]
            if self.unreduced:
                c = ct[t,:,:]
            else:
                c = None
            #rzz = kfun.kernelMatrix(param.KparDiff, z)
            if self.unreduced:
                ca = kernel.applyK(c,a)
                ra = kernel.applyK(c, a, firstVar=z)
                obj += regWeight_[t] * timeStep * (a * ca).sum() * self.ds**2
                obj3 += self.unreducedWeight * timeStep * ((c - z)**2).sum()
            else:
                ra = kernel.applyK(z, a)
                obj += regWeight_[t]*timeStep*(a*ra).sum() * self.ds**2
            if hasattr(self, 'v'):
                self.v[t, :] = ra * self.ds
            if self.internalCost:
                foo.updateVertices(z[:self.nvert, :])
                obj1 += self.internalWeight*self.internalCost(foo, ra*self.ds)*timeStep

            if self.affineDim > 0:
                obj2 +=  timeStep * np.multiply(self.affineWeight.reshape(Afft[t].shape), Afft[t]**2).sum()
            #print xt.sum(), at.sum(), obj
        if self.mode == 'debug':
            logging.info(f'LDDMM: {obj:.4f}, unreduced penalty: {obj3:.4f}, internal cost: {obj1:.4f}, Affine cost: {obj2:.4f}')
        obj += obj1 + obj2 + obj3
        if withJacobian:
            return obj, xt, Jt
        elif withTrajectory:
            return obj, xt
        else:
            return obj


    def objectiveFun(self):
        if self.obj is None:
            if self.param.errorType == 'L2Norm':
                self.obj0 = sd.L2Norm0(self.fv1) / (self.param.sigmaError ** 2)
            else:
                self.obj0 = self.fun_obj0(self.fv1) / (self.param.sigmaError**2)
            if self.symmetric:
                self.obj0 += self.fun_obj0(self.fv0) / (self.param.sigmaError**2)
            if self.match_landmarks:
                self.obj0 += self.wlmk * self.lmk_obj0(self.targ_lmk) / (self.param.sigmaError**2)
            if self.unreduced:
                (self.obj, self.xt) = self.objectiveFunDef([self.ct, self.at], self.Afft, withTrajectory=True)
            else:
                (self.obj, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            #foo = surfaces.Surface(surf=self.fvDef)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :self.nvert, :]))
            if self.match_landmarks:
                self.def_lmk.points = self.xt[-1, self.nvert:, :]
            if self.symmetric:
                self.fvInit.updateVertices(np.squeeze(self.x0[:self.nvert, :]))
                self.obj += self.obj0 + self.dataTerm(self.fvDef, self.fvInit, _lmk_def=self.def_lmk)
            else:
                self.obj += self.obj0 + self.dataTerm(self.fvDef, _lmk_def=self.def_lmk)
            #print self.obj0,  self.dataTerm(self.fvDef)

        return self.obj


    def Direction(self):
        return Direction()

    def update(self, dr, eps):
        self.at -= eps * dr['diff']
        if self.unreduced:
            self.ct -= eps * dr['pts']
        if self.symmetric:
            self.x0 -= eps * dr['initx']
        if self.affineDim > 0:
            self.Afft -= eps*dr['aff']

    def getVariable(self):
        if self.unreduced:
            if self.symmetric:
                return [self.ct, self.at, self.Afft, self.x0]
            else:
                return [self.ct, self.at, self.Afft, None]
        else:
            if self.symmetric:
                return [self.at, self.Afft, self.x0]
            else:
                return [self.at, self.Afft, None]

    def updateTry(self, dr, eps, objRef=None):
        objTry = self.obj0
        atTry = self.at - eps * dr['diff']
        if self.unreduced:
            ctTry = self.ct - eps*dr['pts']
        else:
            ctTry = None
        if self.affineDim > 0:
            AfftTry = self.Afft - eps * dr['aff']
        else:
            AfftTry = self.Afft

        fv0 = surfaces.Surface(surf=self.fv0)
        if self.symmetric:
            x0Try = self.x0 - eps * dr['initx']
            fv0.updateVertices(x0Try)
        else:
            x0Try = None

        if self.unreduced:
            foo = self.objectiveFunDef([ctTry, atTry], AfftTry, fv0 = fv0, withTrajectory=True)
        else:
            foo = self.objectiveFunDef(atTry, AfftTry, fv0 = fv0, withTrajectory=True)
        objTry += foo[0]
        xtTry = foo[1]

        ff = surfaces.Surface(surf=self.fvDef)
        ff.updateVertices(np.squeeze(foo[1][-1, :self.nvert, :]))
        if self.match_landmarks:
            pp = pointSets.PointSet(data=self.def_lmk)
            pp.updatePoints(np.squeeze(foo[1][-1, self.nvert:, :]))
        else:
            pp = None
        if self.symmetric:
            ffI = surfaces.Surface(surf=self.fvInit)
            ffI.updateVertices(x0Try)
            objTry += self.dataTerm(ff, ffI, _lmk_def=pp)
        else:
            objTry += self.dataTerm(ff, _lmk_def=pp)
        if np.isnan(objTry):
            logging.info('Warning: nan in updateTry')
            return 1e500

        if (objRef is None) or (objTry < objRef):
            self.atTry = atTry
            self.objTry = objTry
            self.AfftTry = AfftTry
            self.xtTry = xtTry
            if self.symmetric:
                self.x0Try = x0Try
            if self.unreduced:
                self.ctTry = ctTry
            #print 'objTry=',objTry, dir.diff.sum()

        return objTry


    def testEndpointGradient(self):
        # c0 = self.dataTerm(self.fvDef, _lmk_def=self.def_lmk)
        dff = np.random.normal(size=self.fvDef.vertices.shape)
        if self.match_landmarks:
            dpp = np.random.normal(size=self.def_lmk.points.shape)
            dall = np.concatenate((dff, dpp), axis=0)
        else:
            dall = dff
            dpp = None
        c = []
        eps0 = 1e-6
        for eps in [-eps0, eps0]:
            ff = surfaces.Surface(surf=self.fvDef)
            ff.updateVertices(ff.vertices+eps*dff)
            if self.match_landmarks:
                pp = pointSets.PointSet(data=self.def_lmk)
                pp.updatePoints(pp.points + eps * dpp)
            else:
                pp = None
            c.append(self.dataTerm(ff, _lmk_def=pp))
        grd = self.endPointGradient()
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c[1]-c[0])/(2*eps), (grd*dall).sum()) )



    def endPointGradient(self, endPoint=None):
        if endPoint is None:
            endPoint = self.fvDef
            endPoint_lmk = self.def_lmk
        elif self.match_landmarks:
            endPoint_lmk = endPoint[1]
            endPoint = endPoint[0]
        else:
            endPoint_lmk = None
        if self.param.errorType == 'L2Norm':
            px = sd.L2NormGradient(endPoint, self.fv1.vfld)
        else:
            if self.fv1:
                px = self.fun_objGrad(endPoint, self.fv1)
            else:
                px = self.fun_objGrad(endPoint)
        if self.match_landmarks:
            pxl = self.wlmk*self.lmk_objGrad(endPoint_lmk.points, self.targ_lmk.points)
            px = np.concatenate((px, pxl), axis=0)
        return px / self.param.sigmaError**2

    def initPointGradient(self):
        px = self.fun_objGrad(self.fvInit, self.fv0, self.param.KparDist)
        return px / self.param.sigmaError**2
    
    
    def hamiltonianCovector(self, px1, KparDiff, regWeight, affine = None, fv0 = None, control = None):
        if fv0 is None:
            fv0 = self.fvInit
        if control is None:
            at = self.at
            if self.unreduced:
                ct = self.ct
            else:
                ct = None
            current_at = True
            if self.varCounter == self.trajCounter:
                computeTraj = False
            else:
                computeTraj = True
        else:
            if self.unreduced:
                ct = control[0]
                at = control[1]
            else:
                ct = None
                at = control
            current_at = False
            computeTraj = True
        if self.match_landmarks:
            x0 = np.concatenate((fv0.vertices, self.tmpl_lmk.points), axis=0)
        else:
            x0 = fv0.vertices
        N = x0.shape[0]
        dim = x0.shape[1]
        T = at.shape[0]
        timeStep = 1.0/T
        if computeTraj:
            if self.unreduced:
                xt = evol.landmarkSemiReducedEvolutionEuler(x0, ct, at*self.ds, KparDiff, affine=affine)
            else:
                xt = evol.landmarkDirectEvolutionEuler(x0, at*self.ds, KparDiff, affine=affine)
            if current_at:
                self.trajCounter = self.varCounter
                self.xt = xt
        else:
            xt = self.xt

        if not(affine is None):
            A0 = affine[0]
            A = np.zeros([T,dim,dim])
            for t in range(A0.shape[0]):
                A[t,:,:] = getExponential(timeStep*A0[t])
        else:
            A = None

        pxt = np.zeros([T, N, dim])
        pxt[T-1, :, :] = px1
        # if self.unreduced:
        #     pxt[T-1, :, :] -= self.unreducedWeight * ((xt[T, :, :] + xt[T-1, :, :])/2 - ct[T-1, :, :])*timeStep
        foo = surfaces.Surface(surf=fv0)
        for t in range(1, T):
            px = pxt[T-t, :, :]
            z = xt[T-t, :, :]
            #if self.unreduced:
            #     if t < T-1:
            #         z2 = (xt[T-t-1, :, :] + 2*z + xt[T-t+1, :, :])/2
            #     else:
            #         z2 = (xt[1, :, :] + xt[0, :, :])/2
            # else:
            #     z2 = None
            a = at[T-t, :, :]
            if self.unreduced:
                c = np.squeeze(ct[T - t, :, :])
                # if t < T - 1:
                #     c2 = ct[T - t - 1, :, :] + c
                # else:
                #     c2 = c
                v = KparDiff.applyK(c,a, firstVar=z)*self.ds
            else:
                c = None
                c2 = None
                v = KparDiff.applyK(z,a)*self.ds

            foo.updateVertices(z)
            if self.internalCost:
                grd = self.internalCostGrad(foo, v)
                Lv =  grd[0]
                DLv = self.internalWeight*grd[1]
                if self.unreduced:
                    zpx = KparDiff.applyDiffKT(c, px - self.internalWeight*Lv, a*self.ds,
                                               lddmm=False, firstVar=z) - DLv - 2*self.unreducedWeight * (z-c)
                else:
                    zpx = KparDiff.applyDiffKT(z, px, a*self.ds, regweight=self.regweight, lddmm=True,
                                               extra_term=-self.internalWeight * Lv) - DLv
            else:
                if self.unreduced:
                    zpx = KparDiff.applyDiffKT(c, px, a*self.ds, lddmm=False, firstVar=z) \
                        - 2*self.unreducedWeight * (z-c)
                else:
                    zpx = KparDiff.applyDiffKT(z, px, a*self.ds, regweight=self.regweight, lddmm=True)

            if not (affine is None):
                pxt[T-t-1, :, :] = np.dot(px, A[T-t]) + timeStep * zpx
            else:
                pxt[T-t-1, :, :] = px + timeStep * zpx
        return pxt, xt


    # def gradientFromHamiltonian(self, at, xt, pxt, fv0, affine, kernel, regWeight):
    #     dat = np.zeros(at.shape)
    #     timeStep = 1.0/at.shape[0]
    #     foo = surfaces.Surface(surf=fv0)
    #     nvert = foo.vertices.shape[0]
    #     if not (affine is None):
    #         A = affine[0]
    #         dA = np.zeros(affine[0].shape)
    #         db = np.zeros(affine[1].shape)
    #     for k in range(at.shape[0]):
    #         z = np.squeeze(xt[k,...])
    #         foo.updateVertices(z[:nvert, :])
    #         a = np.squeeze(at[k, :, :])
    #         px = np.squeeze(pxt[k+1, :, :])
    #         #print 'testgr', (2*a-px).sum()
    #         if not self.affineOnly:
    #             v = kernel.applyK(z,a)
    #             if self.internalCost:
    #                 Lv = self.internalCostGrad(foo, v, variables='phi')
    #                 #Lv = -foo.laplacian(v)
    #                 dat[k, :, :] = 2*regWeight*a-px + self.internalWeight * Lv
    #             else:
    #                 dat[k, :, :] = 2*regWeight*a-px
    #
    #         if not (affine is None):
    #             dA[k] = gradExponential(A[k]*timeStep, px, xt[k]) #.reshape([self.dim**2, 1])/timeStep
    #             db[k] = pxt[k+1].sum(axis=0) #.reshape([self.dim,1])
    #     if affine is None:
    #         return dat, xt, pxt
    #     else:
    #         return dat, dA, db, xt, pxt


    def hamiltonianGradient(self, px1, kernel = None, affine=None, regWeight=None, fv0=None, control=None):
        if regWeight is None:
            regWeight = self.regweight
        if fv0 is None:
            fv0 = self.fvInit
        x0 = fv0.vertices
        if control is None:
            if self.unreduced:
                control = [self.ct, self.at]
                ct = self.ct
                at = self.at
            else:
                control = self.at
                at = self.at
                ct = None
        else:
            if self.unreduced:
                ct = control[0]
                at = control[1]
            else:
                ct = None
                at = control
        if kernel is None:
            kernel  = self.param.KparDiff
        # if not self.internalCost:
        #     return evol.landmarkHamiltonianGradient(x0, at, px1, kernel, regWeight, affine=affine,
        #                                             getCovector=True)
        #
        foo = surfaces.Surface(surf=fv0)
        foo.updateVertices(x0)
        (pxt, xt) = self.hamiltonianCovector(px1, kernel, regWeight, fv0=foo, control = control, affine=affine)

        dat = np.zeros(at.shape)
        if self.unreduced:
            dct = np.zeros(ct.shape)
        else:
            dct = None
        timeStep = 1.0/at.shape[0]
        foo = surfaces.Surface(surf=fv0)
        nvert = foo.vertices.shape[0]
        if not (affine is None):
            A = affine[0]
            dA = np.zeros(affine[0].shape)
            db = np.zeros(affine[1].shape)
        for t in range(at.shape[0]):
            z = xt[t,:,:]
            # if self.unreduced:
            #     z2 = (z+xt[t+1,:,:])/2
            # else:
            #     z2 = None
            foo.updateVertices(z[:nvert, :])
            a = at[t, :, :]
            if self.unreduced:
                c = ct[t,:,:]
            else:
                c = None
            px = pxt[t, :, :]
            #print 'testgr', (2*a-px).sum()
            if not self.affineOnly:
                if self.unreduced:
                    dat[t, :, :] = 2 * regWeight * kernel.applyK(c, a) * self.ds**2 - kernel.applyK(z, px, firstVar=c) * self.ds
                    #if k > 0:
                    dct[t, :, :] = 2 * regWeight * kernel.applyDiffKT(c, a, a) * self.ds**2 \
                                   - kernel.applyDiffKT(z, a, px, firstVar=c) * self.ds \
                                    + 2 * self.unreducedWeight * (c-z)
                    v = kernel.applyK(c, a, firstVar=z)*self.ds
                else:
                    dat[t, :, :] = 2 * regWeight * a * self.ds**2 - px * self.ds
                    v = kernel.applyK(z,a)*self.ds
                if self.internalCost:
                    Lv = self.internalCostGrad(foo, v, variables='phi')
                    #Lv = -foo.laplacian(v)
                    if self.unreduced:
                        dat[t, :, :] += self.internalWeight * kernel.applyK(z, Lv, firstVar=c) * self.ds
                        #if k> 0:
                        dct[t, :, :] += self.internalWeight * kernel.applyDiffKT(z, a, Lv, firstVar=c)*self.ds
                    else:
                        dat[t, :, :] += self.internalWeight * Lv * self.ds

                if not self.unreduced and self.euclideanGradient:
                    dat[t, :, :] = kernel.applyK(z, dat[t, :, :])

            if not (affine is None):
                dA[t] = gradExponential(A[t]*timeStep, px, xt[t, :, :]) #.reshape([self.dim**2, 1])/timeStep
                db[t] = px.sum(axis=0) #.reshape([self.dim,1])

        if self.unreduced:
            if self.mode == 'debug':
                logging.info('gradient', np.fabs(dct).max(), np.fabs(dat).max())
            output = [dct]
        else:
            output = []
        if affine is None:
            return output + [dat, xt, pxt]
        else:
            return output + [dat, dA, db, xt, pxt]

        #
        # return self.gradientFromHamiltonian(control, xt, pxt, fv0, affine, kernel, regWeight)

    def endPointGradientSGD(self):
        if self.sgdMeanSelectTemplate >= self.fv0.faces.shape[0]:
            I0_ = np.arange(self.fv0.faces.shape[0])
            p0 = 1.
            sqp0 = 1.
        else:
            I0_ = rng.choice(self.fv0.faces.shape[0], self.sgdMeanSelectTemplate, replace=False)
            p0 = self.sgdMeanSelectTemplate / self.fv0.faces.shape[0]
            sqp0 = np.sqrt(self.sgdMeanSelectTemplate * (self.sgdMeanSelectTemplate - 1)
                           / (self.fv0.faces.shape[0] * (self.fv0.faces.shape[0] - 1)))

        if self.sgdMeanSelectTarget > self.fv1.faces.shape[0]:
            I1_ = np.arange(self.fv1.faces.shape[0])
            p1 = p0 / sqp0
        else:
            I1_ = rng.choice(self.fv1.faces.shape[0], self.sgdMeanSelectTarget, replace=False)
            p1 = (self.sgdMeanSelectTarget / self.fv1.faces.shape[0]) * p0 / sqp0

        select0 = np.zeros(self.fv0.faces.shape[0], dtype=bool)
        select0[I0_] = True
        fv0, I0 = self.fv0.select_faces(select0)
        self.stateSubset = I0
        xt = evol.landmarkSemiReducedEvolutionEuler(fv0.vertices, self.ct, self.at, self.param.KparDiff,
                                                    affine=self.Afft)
        endPoint = surfaces.Surface(surf=fv0)
        endPoint.updateVertices(xt[-1, :, :])
        endPoint.face_weights /= sqp0
        # endPoint.updateWeights(endPoint.weights / sqp0)

        select1 = np.zeros(self.fv1.faces.shape[0], dtype=bool)
        select1[I1_] = True
        fv1, I1 = self.fv1.select_faces(select1)
        # endPoint.saveVTK('foo.vtk')
        fv1.face_weights /= p1
        #        fv1.updateWeights(fv1.weights / p1)
        # self.SGDSelectionCost = [I0, I1]

        if self.param.errorType == 'L2Norm':
            px_ = sd.L2NormGradient(endPoint, self.fv1.vfld)
        else:
            px_ = self.fun_objGrad(endPoint, fv1)
            ## Correction for diagonal term
            if self.sgdMeanSelectTemplate < self.fv0.faces.shape[0]:
                s0 = (1/(sqp0**2) - 1/p0) #(1 / sqp0 - sqp0 / p0)  #  # (sqp0 **2 /p0-1) * p0/sqp0
                if self.param.errorType == 'varifold':
                    s1 = 2.
                else:
                    s1 = 1.

                pc = np.zeros(fv0.vertices.shape)
                xDef0 = endPoint.vertices[fv0.faces[:, 0], :]
                xDef1 = endPoint.vertices[fv0.faces[:, 1], :]
                xDef2 = endPoint.vertices[fv0.faces[:, 2], :]
                nu = np.cross(xDef1 - xDef0, xDef2 - xDef0)
                dz0 = np.cross(xDef1 - xDef2, nu)
                dz1 = np.cross(xDef2 - xDef0, nu)
                dz2 = np.cross(xDef0 - xDef1, nu)
                for k in range(fv0.faces.shape[0]):
                    pc[fv0.faces[k, 0], :] += dz0[k, :]
                    pc[fv0.faces[k, 1], :] += dz1[k, :]
                    pc[fv0.faces[k, 2], :] += dz2[k, :]
                px_ -= s1 * s0 * pc / 2

        # if self.match_landmarks:
        #     pxl = self.wlmk*self.lmk_objGrad(endPoint_lmk.points, self.targ_lmk.points)
        #     px = np.concatenate((px, pxl), axis=0)

        # px = np.zeros(self.fvDef.vertices.shape)
        # px[I0] = px_
        self.xt[:, I0, :] = xt
        return px_ / self.param.sigmaError ** 2, xt

    def checkSGDEndpointGradient(self):
        endPoint = surfaces.Surface(surf=self.fv0)
        xt = evol.landmarkSemiReducedEvolutionEuler(self.fv0.vertices, self.ct, self.at, self.param.KparDiff,
                                                    affine=self.Afft)
        endPoint.updateVertices(xt[-1, :, :])

        pxTrue = self.endPointGradient(endPoint=endPoint)
        px = np.zeros(pxTrue.shape)
        nsim = 25
        for k in range(nsim):
            px += self.endPointGradientSGD()[0]

        px /= nsim
        diff = ((px - pxTrue) ** 2).mean()
        logging.info(f'check SGD gradient: {diff:.4f}')

    def getGradientSGD(self, coeff=1.0):
        #self.checkSGDEndpointGradient()
        A = self.affB.getTransforms(self.Afft)
        px1, xt = self.endPointGradientSGD()
        # I0 = self.SGDSelectionCost[0]
        #x0[self.stateSubset, :] = xt[-1, :, :]
        if self.sgdMeanSelectControl <= self.ct.shape[1]:
            J0 = rng.choice(self.ct.shape[1], self.sgdMeanSelectControl, replace=False)
            #J1 = rng.choice(self.ct.shape[1], self.sgdMeanSelectControl, replace=False)
        else:
            J0 = np.arange(self.ct.shape[1])
            #J1 = np.arange(self.ct.shape[1])
        foo = evol.landmarkSemiReducedHamiltonianGradient(self.x0, self.ct, self.at, -px1, self.param.KparDiff,
                                                          self.regweight, getCovector = True, affine = A,
                                                          weightSubset=self.unreducedWeight,
                                                          controlSubset = J0, stateSubset=self.stateSubset,
                                                          controlProb=self.probSelectControl,
                                                          stateProb=self.probSelectVertexTemplate,
                                                          forwardTraj=xt)
        dim2 = self.dim**2
        grd = Direction()
        grd['pts'] = foo[0] / (coeff*self.Tsize)
        grd['diff'] = foo[1] / (coeff*self.Tsize)
        if self.affineDim > 0:
            grd['aff'] = np.zeros(self.Afft.shape)
            dA = foo[2]
            db = foo[3]
            grd.aff = 2*self.affineWeight.reshape([1, self.affineDim])*self.Afft
            #grd.aff = 2 * self.Afft
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               #grd.aff[t] -=  np.divide(dAff.reshape(grd.aff[t].shape), self.affineWeight.reshape(grd.aff[t].shape))
               grd['aff'][t] -=  dAff.reshape(grd['aff'][t].shape)
            grd['aff'] /= (self.coeffAff*coeff*self.Tsize)
        else:
            grd['aff'] = None
            #            dAfft[:,0:self.dim**2]/=100
        return grd



    def getGradient(self, coeff=1.0, update=None):
        if self.param.algorithm == 'sgd':
            return self.getGradientSGD(coeff=coeff)

        if update is None:
            control = None
            Afft = self.Afft
            if self.match_landmarks:
                endPoint = (self.fvDef, self.def_lmk)
            else:
                endPoint = self.fvDef
            A = self.affB.getTransforms(self.Afft)
            xt = self.xt
        else:
            if update[0]['aff'] is not None:
                Afft = self.Afft - update[1]*update[0]['aff']
                A = self.affB.getTransforms(Afft)
            else:
                A = None

            if self.unreduced:
                at = self.at - update[1] * update[0]['diff']
                ct = self.ct - update[1] * update[0]['pts']
                control = [ct, at]
                xt = evol.landmarkSemiReducedEvolutionEuler(self.x0, ct, at*self.ds, self.param.KparDiff, affine=A)
            else:
                control = self.at - update[1] * update[0]['diff']
                xt = evol.landmarkDirectEvolutionEuler(self.x0, control*self.ds, self.param.KparDiff, affine=A)


            if self.match_landmarks:
                endPoint0 = surfaces.Surface(surf=self.fv0)
                endPoint0.updateVertices(xt[-1, :self.nvert, :])
                endPoint1 = pointSets.PointSet(data=xt[-1, self.nvert:,:])
                endPoint = (endPoint0, endPoint1)
            else:
                endPoint = surfaces.Surface(surf=self.fv0)
                endPoint.updateVertices(xt[-1, :, :])


        px1 = -self.endPointGradient(endPoint=endPoint)
        dim2 = self.dim**2
        foo = self.hamiltonianGradient(px1, control=control, affine=A)
        grd = Direction()
        # if self.euclideanGradient:
        #     grd['diff'] = np.zeros(foo[0].shape)
        #     for t in range(self.Tsize):
        #         z = xt[t, :, :]
        #         grd['diff'][t,:,:] = self.param.KparDiff.applyK(z, foo[0][t, :,:])/(coeff*self.Tsize)
        # else:
        if self.unreduced:
            grd['pts'] = foo[0]/(coeff*self.Tsize)
            grd['diff'] = foo[1] / (coeff * self.Tsize)
        else:
            grd['diff'] = foo[0]/(coeff*self.Tsize)
        if self.affineDim > 0:
            grd['aff'] = np.zeros(self.Afft.shape)
            dA = foo[1]
            db = foo[2]
            grd['aff'] = 2*self.affineWeight.reshape([1, self.affineDim])*Afft
            #grd.aff = 2 * self.Afft
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               #grd.aff[t] -=  np.divide(dAff.reshape(grd.aff[t].shape), self.affineWeight.reshape(grd.aff[t].shape))
               grd['aff'][t] -=  dAff.reshape(grd['aff'][t].shape)
            grd['aff'] /= (self.coeffAff*coeff*self.Tsize)
            #            dAfft[:,0:self.dim**2]/=100
        if self.symmetric:
            grd['initx'] = (self.initPointGradient() - foo[-1][0,...])/(self.coeffInitx * coeff)
        return grd



    def addProd(self, dir1, dir2, beta):
        dr = Direction()
        for k in dir1.keys():
            if k != 'aff' and dir1[k] is not None:
                dr[k] = dir1[k] + beta * dir2[k]
        if self.affineDim > 0:
            dr['aff'] = dir1['aff'] + beta * dir2['aff']
        return dr

    def prod(self, dir1, beta):
        dr = Direction()
        for k in dir1.keys():
            if k != 'aff' and dir1[k] is not None:
                dr[k] = beta * dir1[k]
        if self.affineDim > 0:
            dr['aff'] = beta * dir1['aff']
        return dr

    def copyDir(self, dir0):
        return deepcopy(dir0)


    def randomDir(self):
        dirfoo = Direction()
        if self.affineOnly:
            dirfoo['diff'] = np.zeros((self.Tsize, self.npt, self.dim))
        else:
            dirfoo['diff'] = np.random.randn(self.Tsize, self.npt, self.dim)
        if self.unreduced:
            dirfoo['pts'] = np.random.normal(0, 1, size=self.ct.shape)
            dirfoo['pts'][0, :, :] = 0
        if self.symmetric:
            dirfoo['initx'] = np.random.randn(self.npt, self.dim)
        dirfoo['aff'] = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    def dotProduct_Riemannian(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            z = np.squeeze(self.xt[t, :, :])
            gg = np.squeeze(g1['diff'][t, :, :])
            u = self.param.KparDiff.applyK(z, gg)
            #uu = np.multiply(g1.aff[t], self.affineWeight.reshape(g1.aff[t].shape))
            if self.affineDim > 0:
                uu = g1['aff'][t]
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr['diff'][t, :, :])
                res[ll]  = res[ll] + (ggOld*u).sum()
                if self.affineDim > 0:
                    res[ll] += (uu*gr['aff'][t]).sum() * self.coeffAff
                ll = ll + 1

        if self.symmetric:
            for ll,gr in enumerate(g2):
                res[ll] += (g1['initx'] * gr['initx']).sum() * self.coeffInitx

        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        for k in g1.keys():
            if g1[k] is not None:
                for ll,gr in enumerate(g2):
                    res[ll] += (g1[k]*gr[k]).sum()
        return res

        # for t in range(self.Tsize):
        #     u = np.squeeze(g1.diff[t, :, :])
        #     if self.affineDim > 0:
        #         uu = g1.aff[t]
        #         # uu = (g1.aff[t]*self.affineWeight.reshape(g1.aff[t].shape))
        #     else:
        #         uu = 0
        #     ll = 0
        #     for gr in g2:
        #         ggOld = np.squeeze(gr.diff[t, :, :])
        #         res[ll]  += (ggOld*u).sum()
        #         if self.affineDim > 0:
        #             res[ll] += (uu*gr.aff[t]).sum()
        #             #                    +np.multiply(g1[1][t, dim2:dim2+self.dim], gr[1][t, dim2:dim2+self.dim]).sum())
        #         ll = ll + 1
        # if self.symmetric:
        #     for ll,gr in enumerate(g2):
        #         res[ll] += (g1.initx * gr.initx).sum() * self.coeffInitx
        # return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.at = np.copy(self.atTry)
        if self.unreduced:
            self.ct = np.copy(self.ctTry)
        if self.affineDim > 0:
            self.Afft = np.copy(self.AfftTry)
        self.xt = np.copy(self.xtTry)
        self.varCounter += 1
        self.trajCounter = self.varCounter
        if self.symmetric:
            self.x0 = np.copy(self.x0Try)
        #print self.at

    def saveCorrectedTarget(self, X0, X1):
        U = la.inv(X0[-1])
        f = surfaces.Surface(surf=self.fv1)
        yyt = np.dot(f.vertices - X1[-1,...], U)
        f.updateVertices(yyt)
        f.saveVTK(self.outputDir + '/TargetCorrected.vtk')
        if self.match_landmarks:
            p = pointSets.PointSet(data=self.targ_lmk)
            yyt = np.dot(p.points - X1[-1,...], U)
            p.updatePoints(yyt)
            p.saveVTK(self.outputDir + '/TargetLandmarkCorrected.vtk')


    def saveCorrectedEvolution(self, fv0, xt, at, Afft, fileName='evolution', Jacobian=None):
        f = surfaces.Surface(surf=fv0)
        if self.match_landmarks:
            p = pointSets.PointSet(data=self.tmpl_lmk)
        else:
            p = None
        X = self.affB.integrateFlow(Afft)
        displ = np.zeros(xt.shape[1])
        dt = 1.0 / self.Tsize
        fn = []
        if type(fileName) is str:
            for kk in range(self.Tsize + 1):
                fn.append(fileName + f'_corrected{kk:03d}')
        else:
            fn = fileName
        vt = None
        for t in range(self.Tsize + 1):
            U = la.inv(X[0][t])
            yyt = np.dot(self.xt[t, ...] - X[1][t, ...], U.T)
            zt = np.dot(xt[t, ...] - X[1][t, ...], U.T)
            if t < self.Tsize:
                atCorr = np.dot(at[t, ...], U.T)
                vt = self.param.KparDiff.applyK(yyt, atCorr, firstVar=zt)
            f.updateVertices(yyt[:self.nvert, :])
            if self.match_landmarks:
                p.updatePoints(yyt[self.nvert:, :])
                p.saveVTK(self.outputDir + '/' + fn[t] + '_lmk.vtk')
            vf = surfaces.vtkFields()
            if Jacobian is not None:
                vf.scalars.append('Jacobian')
                vf.scalars.append(np.exp(Jacobian[t, :self.nvert, 0]))
            vf.scalars.append('displacement')
            vf.scalars.append(displ[:self.nvert])
            vf.vectors.append('velocity')
            vf.vectors.append(vt[:self.nvert, :])
            nu = self.fv0ori * f.computeVertexNormals()
            f.saveVTK2(self.outputDir + '/' + fn[t] + '.vtk', vf)
            displ += dt * (vt * nu).sum(axis=1)
        self.saveCorrectedTarget(X[0], X[1])

    def saveEvolution(self, fv0, xt, Jacobian=None, passenger = None, fileName='evolution', velocity = None,
                      orientation= None, with_area_displacement=False):
        if velocity is None:
            velocity = self.v
        if orientation is None:
            orientation = self.fv0ori
        fn = []
        if type(fileName) is str:
            for kk in range(self.Tsize + 1):
                fn.append(fileName + f'{kk:03d}')
        else:
            fn = fileName


        fvDef = surfaces.Surface(surf=fv0)
        AV0 = fvDef.computeVertexArea()
        nu = orientation * fv0.computeVertexNormals()
        nvert = fv0.vertices.shape[0]
        npt = xt.shape[1]
        v = velocity[0, :nvert, :]
        displ = np.zeros(nvert)
        area_displ = np.zeros((self.Tsize + 1, npt))
        dt = 1.0 / self.Tsize
        for kk in range(self.Tsize + 1):
            fvDef.updateVertices(np.squeeze(xt[kk, :nvert, :]))
            AV = fvDef.computeVertexArea()
            AV = (AV[0] / AV0[0])
            vf = surfaces.vtkFields()
            if Jacobian is not None:
                vf.scalars.append('Jacobian')
                vf.scalars.append(np.exp(Jacobian[kk, :nvert, 0]))
                vf.scalars.append('Jacobian_T')
                vf.scalars.append(AV)
                vf.scalars.append('Jacobian_N')
                vf.scalars.append(np.exp(Jacobian[kk, :nvert, 0]) / AV)
            vf.scalars.append('displacement')
            vf.scalars.append(displ)
            if kk < self.Tsize:
                nu = orientation * fvDef.computeVertexNormals()
                v = velocity[kk, :nvert, :]
                kkm = kk
            else:
                kkm = kk - 1
            vf.vectors.append('velocity')
            vf.vectors.append(velocity[kkm, :nvert])
            if with_area_displacement and kk > 0:
                area_displ[kk, :] = area_displ[kk - 1, :] + dt * ((AV + 1) * (v * nu).sum(axis=1))[np.newaxis, :]
            fvDef.saveVTK2(self.outputDir + '/' + fn[kk] + '.vtk', vf)
            displ += dt * (v * nu).sum(axis=1)
            if passenger is not None and passenger[0] is not None:
                if isinstance(passenger[0], surfaces.Surface):
                    fvp = surfaces.Surface(surf=passenger[0])
                    fvp.updateVertices(passenger[1][kk,...])
                    fvp.saveVTK(self.outputDir+'/'+fn[kk]+'_passenger.vtk')
                else:
                    pointSets.savePoints(self.outputDir+'/'+fn[kk]+'_passenger.vtk', passenger[1][kk,...])
            if self.match_landmarks:
                pp = pointSets.PointSet(data=xt[kk,nvert:,:])
                pp.saveVTK(self.outputDir+'/'+fn[kk]+'_lmk.vtk')

    def saveEPDiff(self, fv0, at, fileName='evolution'):
        if self.match_landmarks:
            x0 = np.concatenate((fv0.vertices, self.tmpl_lmk), axis=0)
        else:
            x0 = fv0.vertices
        xtEPDiff, atEPdiff = evol.landmarkEPDiff(at.shape[0], x0,
                                                 np.squeeze(at[0, :, :]), self.param.KparDiff)
        fvDef = surfaces.Surface(surf=fv0)
        nvert = fv0.vertices.shape[0]
        fvDef.updateVertices(np.squeeze(xtEPDiff[-1, :nvert, :]))
        fvDef.saveVTK(self.outputDir + '/' + fileName + 'EPDiff.vtk')
        return xtEPDiff, atEPdiff

    def updateEndPoint(self, xt):
        self.fvDef.updateVertices(np.squeeze(xt[-1, :self.nvert, :]))
        if self.match_landmarks:
            self.def_lmk.updatePoints(xt[-1, self.nvert:, :])

    def plotAtIteration(self):
        fig = plt.figure(4)
        # fig.clf()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        lim0 = self.addSurfaceToPlot(self.fv1, ax, ec='k', fc='b')
        lim1 = self.addSurfaceToPlot(self.fvDef, ax, ec='k', fc='r')
        ax.set_xlim(min(lim0[0][0], lim1[0][0]), max(lim0[0][1], lim1[0][1]))
        ax.set_ylim(min(lim0[1][0], lim1[1][0]), max(lim0[1][1], lim1[1][1]))
        ax.set_zlim(min(lim0[2][0], lim1[2][0]), max(lim0[2][1], lim1[2][1]))
        if self.match_landmarks:
            ax.scatter3D(self.def_lmk.points[:,0], self.def_lmk.points[:,1], self.def_lmk.points[:,2], color='r')
            ax.scatter3D(self.targ_lmk.points[:, 0], self.targ_lmk.points[:, 1], self.targ_lmk.points[:, 2], color='b')
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        time.sleep(0.5)

    def endOfIterationSGD(self, forceSave=False):
        if forceSave or self.iter % self.saveRate == 0:
            # self.xt = evol.landmarkSemiReducedEvolutionEuler(self.x0, self.ct, self.at, self.param.KparDiff,
            #                                                  affine=self.Afft)
            pointSets.saveTrajectories(self.outputDir + '/' + self.saveFile + 'curvesSGDState.vtk', self.xt)
            pointSets.saveTrajectories(self.outputDir + '/' + self.saveFile + 'curvesSGDControl.vtk', self.ct)
            #self.updateEndPoint(self.xt)
            #self.ct = np.copy(self.xt[:-1, :, :])
            self.saveEvolution(self.fv0, self.xt)

        if self.iter % self.unreducedResetRate == 0:
            logging.info('Resetting trajectories')
            self.ct = np.copy(self.xt[:-1, :, :])
            # f.at = np.zeros(f.at.shape)
            self.ctTry = np.copy(self.ct)

        #else:
            #self.updateEndPoint(self.xt)

    def startOfIteration(self):
        if self.param.algorithm != 'sgd':
            if self.reset:
                self.param.KparDiff.pk_dtype = 'float64'
                self.param.KparDist.pk_dtype = 'float64'

    def endOfIteration(self, forceSave=False):
        self.iter += 1
        if self.param.algorithm == 'sgd':
            self.endOfIterationSGD(forceSave=forceSave)
            return


        if self.testGradient:
            self.testEndpointGradient()
        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2

        dim2 = self.dim ** 2
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2 + self.dim]
        if forceSave or self.iter % self.saveRate == 0:
            logging.info('Saving surfaces...')
            if self.passenger_points is None:
                if self.unreduced:
                    self.xt, Jt = evol.landmarkSemiReducedEvolutionEuler(self.x0, self.ct, self.at*self.ds,
                                                                         self.param.KparDiff, affine=A,
                                                                         withJacobian=True)
                else:
                    self.xt, Jt = evol.landmarkDirectEvolutionEuler(self.x0, self.at*self.ds, self.param.KparDiff, affine=A,
                                                                 withJacobian=True)
                yt = None
            else:
                if self.unreduced:
                    self.xt, Jt = evol.landmarkSemiReducedEvolutionEuler(self.x0, self.ct, self.at*self.ds,
                                                                         self.param.KparDiff, affine=A,
                                                                         withPointSet=self.passenger_points,
                                                                         withJacobian=True)
                else:
                    self.xt, yt, Jt = evol.landmarkDirectEvolutionEuler(self.x0, self.at*self.ds, self.param.KparDiff, affine=A,
                                                                 withPointSet=self.passenger_points, withJacobian=True)
                if isinstance(self.passenger, surfaces.Surface):
                    self.passengerDef.updateVertices(yt[-1,...])
                else:
                    self.passengerDef = deepcopy(yt[-1,...])
            self.trajCounter = self.varCounter

            if self.saveEPDiffTrajectories and not self.internalCost and self.affineDim <= 0:
                xtEPDiff, atEPdiff = self.saveEPDiff(self.fvInit, self.at*self.ds, fileName=self.saveFile)
                logging.info('EPDiff difference %f' % (np.fabs(self.xt[-1, :, :] - xtEPDiff[-1, :, :]).sum()))

            if self.saveTrajectories:
                pointSets.saveTrajectories(self.outputDir + '/' + self.saveFile + 'curves.vtk', self.xt)


            self.updateEndPoint(self.xt)
            self.fvInit.updateVertices(self.x0[:self.nvert, :])

            if self.affine == 'euclidean' or self.affine == 'translation':
                self.saveCorrectedEvolution(self.fvInit, self.xt, self.at, self.Afft, fileName=self.saveFileList,
                                            Jacobian=Jt)
            self.saveEvolution(self.fvInit, self.xt, Jacobian=Jt, fileName=self.saveFileList,
                               passenger = (self.passenger, yt))
            if self.unreduced:
                pointSets.saveTrajectories(self.outputDir + '/' + self.saveFile + 'curvesSGDState.vtk', self.xt)
                pointSets.saveTrajectories(self.outputDir + '/' + self.saveFile + 'curvesSGDControl.vtk', self.ct)
            self.saveHdf5(fileName=self.outputDir + '/output.h5')
        else:
            if self.varCounter != self.trajCounter:
                if self.unreduced:
                    self.xt = evol.landmarkSemiReducedEvolutionEuler(self.x0, self.ct, self.at,
                                                                         self.param.KparDiff, affine=A)
                else:
                    self.xt = evol.landmarkDirectEvolutionEuler(self.x0, self.at, self.param.KparDiff, affine=A)
                self.trajCounter = self.varCounter
            self.updateEndPoint(self.xt)
            self.fvInit.updateVertices(self.x0[:self.nvert, :])

        if self.unreduced and self.iter % self.unreducedResetRate == 0:
            logging.info('Resetting trajectories')
            self.ct = np.copy(self.xt[:-1, :, :])
            # f.at = np.zeros(f.at.shape)
            self.ctTry = np.copy(self.ct)
            self.reset = True
        if self.pplot:
            self.plotAtIteration()

        self.param.KparDiff.pk_dtype = self.Kdiff_dtype
        self.param.KparDist.pk_dtype = self.Kdist_dtype


    def saveHdf5(self, fileName):
        fout = h5py.File(fileName, 'w')
        LDDMMResult = fout.create_group('LDDMM Results')
        parameters = LDDMMResult.create_group('parameters')
        parameters.create_dataset('Time steps', data=self.Tsize)
        parameters.create_dataset('Deformation Kernel type', data = self.param.KparDiff.name)
        parameters.create_dataset('Deformation Kernel width', data = self.param.KparDiff.sigma)
        parameters.create_dataset('Deformation Kernel order', data = self.param.KparDiff.order)
        parameters.create_dataset('Spatial Varifold Kernel type', data = self.param.KparDist.name)
        parameters.create_dataset('Spatial Varifold width', data = self.param.KparDist.sigma)
        parameters.create_dataset('Spatial Varifold order', data = self.param.KparDist.order)
        template = LDDMMResult.create_group('template')
        template.create_dataset('vertices', data=self.fv0.vertices)
        template.create_dataset('faces', data=self.fv0.faces)
        target = LDDMMResult.create_group('target')
        target.create_dataset('vertices', data=self.fv1.vertices)
        target.create_dataset('faces', data=self.fv1.faces)
        deformedTemplate = LDDMMResult.create_group('deformedTemplate')
        deformedTemplate.create_dataset('vertices', data=self.fvDef.vertices)
        variables = LDDMMResult.create_group('variables')
        variables.create_dataset('alpha', data=self.at)
        if self.Afft is not None:
            variables.create_dataset('affine', data=self.Afft)
        else:
            variables.create_dataset('affine', data='None')
        descriptors = LDDMMResult.create_group('descriptors')

        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        dim2 = self.dim**2
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2 + self.dim]
        (xt, Jt) = evol.landmarkDirectEvolutionEuler(self.x0, self.at, self.param.KparDiff, affine=A,
                                                     withJacobian=True)

        AV0 = self.fv0.computeVertexArea()
        AV = self.fvDef.computeVertexArea()[0]/AV0[0]
        descriptors.create_dataset('Jacobian', data=Jt[-1,:])
        descriptors.create_dataset('Surface Jacobian', data=AV)
        descriptors.create_dataset('Displacement', data=xt[-1,...]-xt[0,...])

        fout.close()


    def endOfProcedure(self):
        if self.iter % self.saveRate != 0:
            self.endOfIteration(forceSave=True)

    def optimizeMatching(self):
        #print 'dataterm', self.dataTerm(self.fvDef)
        #print 'obj fun', self.objectiveFun(), self.obj0
        if self.unreduced:
            print(f'Unreduced weight: {self.unreducedWeight:0.4f}')

        if self.param.algorithm in ('cg', 'bfgs'):
            self.coeffAff = self.coeffAff2
            grd = self.getGradient(self.gradCoeff)
            [grd2] = self.dotProduct(grd, [grd])

            if self.gradEps < 0:
                self.gradEps = max(1e-5, np.sqrt(grd2) / 10000)
            self.epsMax = 5.
            logging.info(f'Gradient lower bound: {self.gradEps:.5f}')
            self.coeffAff = self.coeffAff1
            if self.param.algorithm == 'cg':
                cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=0.1,
                      Wolfe=self.param.wolfe)
            elif self.param.algorithm == 'bfgs':
                bfgs.bfgs(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=1.,
                          Wolfe=self.param.wolfe, memory=50)
        elif self.param.algorithm == 'sgd':
            logging.info('Running stochastic gradient descent')
            sgd.sgd(self, verb=self.verb, maxIter=self.maxIter, burnIn=self.sgdBurnIn, epsInit=self.sgdEpsInit, normalization = self.sgdNormalization)

        #return self.at, self.xt