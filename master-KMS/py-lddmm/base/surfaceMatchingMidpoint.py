import os
import numpy as np
import numpy.linalg as la
import logging
import h5py
import glob
from . import conjugateGradient as cg, kernelFunctions as kfun, pointEvolution as evol, bfgs
from . import surfaces
from . import surface_distances
from . import pointSets
from .surfaceMatching import SurfaceMatchingParam, SurfaceMatching
from .affineBasis import AffineBasis, getExponential, gradExponential
from functools import partial
import matplotlib
matplotlib.use("QT5Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class Direction:
    def __init__(self):
        self.diff0 = []
        self.diff1 = []
        self.aff = []


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
class SurfaceMatchingMidpoint(SurfaceMatching):
    def __init__(self, Template=None, Target=None, fileTempl=None, fileTarg=None, param=None, maxIter=1000,
                 regWeight = 1.0, affineWeight = 1.0, internalWeight=1.0, verb=True,
                 affineOnly = False,
                 rotWeight = None, scaleWeight = None, transWeight = None, symmetric = False,
                 testGradient=True, saveFile = 'evolution',
                 saveTrajectories = False, affine = 'none', outputDir = '.',pplot=True):
        if param == None:
            self.param = SurfaceMatchingParam()
        else:
            self.param = param

        if self.param.algorithm == 'cg':
            self.euclideanGradient = False
        else:
            self.euclideanGradient = True

        self.setOutputDir(outputDir)
        self.set_fun(self.param.errorType,vfun=self.param.vfun)

        self.set_template_and_target(Template, Target)
        self.match_landmarks = False

        self.set_parameters(maxIter=maxIter, regWeight=regWeight, affineWeight=affineWeight,
                            internalWeight=internalWeight, verb=verb, affineOnly=affineOnly,
                            rotWeight=rotWeight, scaleWeight=scaleWeight, transWeight=transWeight,
                            symmetric=symmetric, testGradient=testGradient, saveFile=saveFile,
                            saveTrajectories=saveTrajectories, affine=affine)

        self.initialize_variables()
        self.gradCoeff = self.x0.shape[0]

        self.pplot = pplot
        if self.pplot:
            self.initial_plot()


    def initialize_variables(self):
        self.Tsize = int(round(1.0/self.param.timeStep))
        self.x0 = np.copy(self.fv0.vertices)
        self.fvDef0 = surfaces.Surface(surf=self.fv0)
        self.x1 = np.copy(self.fv1.vertices)
        self.fvDef1 = surfaces.Surface(surf=self.fv1)
        #self.nvert = self.fvInit.vertices.shape[0]
        self.npt0 = self.fv0.vertices.shape[0]
        self.npt1 = self.fv1.vertices.shape[0]


        self.at0 = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        if self.randomInit:
            self.at0 = np.random.normal(0, 1, self.at0.shape)
        self.at0Try = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.at1 = np.zeros([self.Tsize, self.x1.shape[0], self.x1.shape[1]])
        if self.randomInit:
            self.at1 = np.random.normal(0, 1, self.at1.shape)
        self.at1Try = np.zeros([self.Tsize, self.x1.shape[0], self.x1.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.AfftTry = np.zeros([self.Tsize, self.affineDim])
        self.xt0 = np.tile(self.x0, [self.Tsize+1, 1, 1])
        self.xt1 = np.tile(self.x1, [self.Tsize+1, 1, 1])
        self.v0 = np.zeros([self.Tsize+1, self.npt0, self.dim])
        self.v1 = np.zeros([self.Tsize+1, self.npt1, self.dim])


    def set_fun(self, errorType, vfun=None):
        self.param.errorType = errorType
        if errorType == 'current':
            #print('Running Current Matching')
            self.fun_obj = partial(surface_distances.currentNorm, KparDist=self.param.KparDist, weight=1.)
            self.fun_objGrad = partial(surface_distances.currentNormGradient, KparDist=self.param.KparDist, weight=1.)
        elif errorType=='measure':
            #print('Running Measure Matching')
            self.fun_obj = partial(surface_distances.measureNorm,KparDist=self.param.KparDist)
            self.fun_objGrad = partial(surface_distances.measureNormGradient,KparDist=self.param.KparDist)
        elif errorType=='varifold':
            self.fun_obj = partial(surface_distances.varifoldNorm, KparDist=self.param.KparDist, fun=vfun)
            self.fun_objGrad = partial(surface_distances.varifoldNormGradient, KparDist=self.param.KparDist, fun=vfun)
        else:
            print('Unknown error Type: ', self.param.errorType)


    def dataTerm(self, _fvDef0, _fvdef1):
        obj = self.fun_obj(_fvDef0, _fvdef1) / (self.param.sigmaError**2)
        return obj

    def  objectiveFunDef2(self, at0, at1, Afft, kernel = None, withTrajectory = False,
                          withJacobian=False, regWeight = None):
        res0 = self.objectiveFunDef(at0, Afft, kernel=kernel, withTrajectory=withTrajectory,
                                    withJacobian=withJacobian, regWeight=regWeight, fv0=self.fv0)
        res1 = self.objectiveFunDef(at1, kernel=kernel, withTrajectory=withTrajectory,
                                    withJacobian=withJacobian, regWeight=regWeight, fv0=self.fv1)
        if withJacobian:
            return res0[0]+res1[0], [res0[1], res1[1]], [res0[2], res1[2]]
        elif withTrajectory:
            return res0[0]+res1[0], [res0[1], res1[1]]
        else:
            return res0+res1


    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = 0 #self.fun_obj0(self.fv1) / (self.param.sigmaError**2)
            (self.obj, xt) = self.objectiveFunDef2(self.at0, self.at1, self.Afft, withTrajectory=True, regWeight=self.regweight)
            self.xt0 = xt[0]
            self.xt1 = xt[1]
            self.fvDef0.updateVertices(np.squeeze(self.xt0[-1, :, :]))
            self.fvDef1.updateVertices(np.squeeze(self.xt1[-1, :, :]))
            self.obj += self.obj0 + self.dataTerm(self.fvDef0, self.fvDef1)
        return self.obj

    def getVariable(self):
        return [self.at0, self.at1, self.Afft]

    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        at0Try = self.at0 - eps * dir.diff0
        at1Try = self.at1 - eps * dir.diff1
        if self.affineDim > 0:
            AfftTry = self.Afft - eps * dir.aff
        else:
            AfftTry = self.Afft

        foo = self.objectiveFunDef2(at0Try, at1Try, AfftTry, withTrajectory=True, regWeight=self.regweight)
        objTry += foo[0]

        ff0 = surfaces.Surface(surf=self.fvDef0)
        ff0.updateVertices(np.squeeze(foo[1][0][-1, :, :]))
        ff1 = surfaces.Surface(surf=self.fvDef1)
        ff1.updateVertices(np.squeeze(foo[1][1][-1, :, :]))
        objTry += self.dataTerm(ff0, ff1)
        if np.isnan(objTry):
            logging.info('Warning: nan in updateTry')
            return 1e500

        if (objRef == None) or (objTry < objRef):
            self.at0Try = at0Try
            self.at1Try = at1Try
            self.objTry = objTry
            self.AfftTry = AfftTry

        return objTry


    def testEndpointGradient(self):
        c0 = self.dataTerm(self.fvDef0, self.fvDef1)
        ff0 = surfaces.Surface(surf=self.fvDef0)
        ff1 = surfaces.Surface(surf=self.fvDef1)
        dff0 = np.random.normal(size=ff0.vertices.shape)
        dff1 = np.random.normal(size=ff1.vertices.shape)
        eps = 1e-6
        ff0.updateVertices(ff0.vertices+eps*dff0)
        ff1.updateVertices(ff1.vertices+eps*dff1)
        c1 = self.dataTerm(ff0, ff1)
        grd = self.endPointGradient()
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c1-c0)/eps,
                                                                      (grd[0]*dff0).sum()
                                                                      + (grd[1]*dff1).sum()) )

    def endPointGradient(self, endPoint=None):
        if endPoint is None:
            endPoint0 = self.fvDef0
            endPoint1 = self.fvDef1
        else:
            endPoint0 = endPoint[0]
            endPoint1 = endPoint[1]
        px0 = self.fun_objGrad(endPoint0, endPoint1)
        px1 = self.fun_objGrad(endPoint1, endPoint0)
        return px0 / self.param.sigmaError**2, px1 / self.param.sigmaError**2


    # def hamiltonianCovector2(self, x0, x1, at0, at1, px01, px11, KparDiff, regWeight, affine = None):
    #     pxt0, xt0 = self.hamiltonianCovector(at0, px01, KparDiff, regWeight, affine = affine)
    #     pxt1, xt1 = self.hamiltonianCovector(at1, px11, KparDiff, regWeight)
    #     return [pxt0, pxt1], [xt0, xt1]

    def hamiltonianGradient2(self, px01, px11, kernel = None, affine=None, regWeight=None,
                             at0=None, at1=None):
        if affine is None:
            dat0, xt0, pxt0 = self.hamiltonianGradient(px01, kernel=kernel, regWeight=regWeight,
                                                       fv0= self.fv0, at=at0)
        else:
            dat0, dA, db, xt0, pxt0 = self.hamiltonianGradient(px01, kernel = kernel, affine=affine,
                                                               regWeight=regWeight, fv0=self.fv0, at=at0)
        dat1, xt1, pxt1 = self.hamiltonianGradient(px11, kernel = kernel, regWeight=regWeight,
                                                   fv0=self.fv1, at=at1)

        if affine is None:
            return [dat0, dat1], [xt0, xt1], [pxt0, pxt1]
        else:
            return [dat0, dat1], dA, db, [xt0, xt1], [pxt0, pxt1]


    def getGradient(self, coeff=1.0, update=None):
        if update is None:
            at0 = self.at0
            endPoint0 = self.fvDef0
            at1 = self.at1
            endPoint1 = self.fvDef1
            A = self.affB.getTransforms(self.Afft)
        else:
            A = self.affB.getTransforms(self.Afft - update[1]*update[0].aff)
            at0 = self.at0 - update[1] *update[0].diff0
            xt0 = evol.landmarkDirectEvolutionEuler(self.x0, at0, self.param.KparDiff, affine=A)
            endPoint0 = surfaces.Surface(surf=self.fv0)
            endPoint0.updateVertices(xt0[-1, :, :])
            at1 = self.at1 - update[1] *update[0].diff1
            xt1 = evol.landmarkDirectEvolutionEuler(self.x1, at1, self.param.KparDiff)
            endPoint1 = surfaces.Surface(surf=self.fv1)
            endPoint1.updateVertices(xt1[-1, :, :])


        px1 = self.endPointGradient(endPoint=[endPoint0, endPoint1])
        px01 = -px1[0]
        px11 = -px1[1]
        foo = self.hamiltonianGradient2(px01, px11, at0=at0, at1=at1, affine=A, regWeight=self.regweight)
        grd = Direction()
        if self.euclideanGradient:
            grd.diff0 = np.zeros(foo[0][0].shape)
            grd.diff1 = np.zeros(foo[0][1].shape)
            for t in range(self.Tsize):
                z = self.xt0[t, :, :]
                grd.diff0[t,:,:] = self.param.KparDiff.applyK(z, foo[0][0][t, :,:])/(coeff*self.Tsize)
                z = self.xt1[t, :, :]
                grd.diff1[t,:,:] = self.param.KparDiff.applyK(z, foo[0][1][t, :,:])/(coeff*self.Tsize)
        else:
            grd.diff0 = foo[0][0]/(coeff*self.Tsize)
            grd.diff1 = foo[0][1] / (coeff * self.Tsize)
        grd.aff = np.zeros(self.Afft.shape)
        if self.affineDim > 0:
            dim2 = self.dim ** 2
            dA = foo[1]
            db = foo[2]
            grd.aff = 2*self.affineWeight.reshape([1, self.affineDim])*self.Afft
            #grd.aff = 2 * self.Afft
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               grd.aff[t] -=  dAff.reshape(grd.aff[t].shape)
            grd.aff /= (self.coeffAff*coeff*self.Tsize)
        return grd



    def addProd(self, dir1, dir2, beta):
        dir = Direction()
        dir.diff0 = dir1.diff0 + beta * dir2.diff0
        dir.diff1 = dir1.diff1 + beta * dir2.diff1
        dir.aff = dir1.aff + beta * dir2.aff
        return dir

    def prod(self, dir1, beta):
        dir = Direction()
        dir.diff0 = beta * dir1.diff0
        dir.diff1 = beta * dir1.diff1
        dir.aff = beta * dir1.aff
        return dir

    def copyDir(self, dir0):
        dir = Direction()
        dir.diff0 = np.copy(dir0.diff0)
        dir.diff1 = np.copy(dir0.diff1)
        dir.aff = np.copy(dir0.aff)
        return dir


    def randomDir(self):
        dirfoo = Direction()
        if self.affineOnly:
            dirfoo.diff0 = np.zeros((self.Tsize, self.npt0, self.dim))
            dirfoo.diff1 = np.zeros((self.Tsize, self.npt1, self.dim))
        else:
            dirfoo.diff0 = np.random.randn(self.Tsize, self.npt0, self.dim)
            dirfoo.diff1 = np.random.randn(self.Tsize, self.npt1, self.dim)
        dirfoo.aff = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            z0 = np.squeeze(self.xt0[t, :, :])
            gg0 = np.squeeze(g1.diff0[t, :, :])
            z1 = np.squeeze(self.xt1[t, :, :])
            gg1 = np.squeeze(g1.diff1[t, :, :])
            u0 = self.param.KparDiff.applyK(z0, gg0)
            u1 = self.param.KparDiff.applyK(z1, gg1)
            #uu = np.multiply(g1.aff[t], self.affineWeight.reshape(g1.aff[t].shape))
            uu = g1.aff[t]
            ll = 0
            for gr in g2:
                ggOld0 = np.squeeze(gr.diff0[t, :, :])
                ggOld1 = np.squeeze(gr.diff1[t, :, :])
                res[ll]  += (ggOld0*u0).sum() + (ggOld1*u1).sum()
                if self.affineDim > 0:
                    res[ll] += (uu*gr.aff[t]).sum() * self.coeffAff
                ll = ll + 1
        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            u0 = np.squeeze(g1.diff0[t, :, :])
            u1 = np.squeeze(g1.diff1[t, :, :])
            uu = (g1.aff[t]*self.affineWeight.reshape(g1.aff[t].shape))
            ll = 0
            for gr in g2:
                ggOld0 = np.squeeze(gr.diff0[t, :, :])
                ggOld1 = np.squeeze(gr.diff1[t, :, :])
                res[ll]  += (ggOld0*u0).sum() + (ggOld1*u1).sum()
                if self.affineDim > 0:
                    res[ll] += (uu*gr.aff[t]).sum()
                ll = ll + 1
        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.at0 = np.copy(self.at0Try)
        self.at1 = np.copy(self.at1Try)
        self.Afft = np.copy(self.AfftTry)


    def endOfIteration(self):
        self.iter += 1
        if self.testGradient:
            self.testEndpointGradient()
        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2

        if (self.iter % self.saveRate == 0) :
            logging.info('Saving surfaces...')
            (obj1, xt) = self.objectiveFunDef2(self.at0, self.at1, self.Afft, withTrajectory=True, regWeight=self.regweight)
            self.xt0 = xt[0]
            self.xt1 = xt[1]

            if self.saveTrajectories:
                pointSets.saveTrajectories(self.outputDir + '/' + self.saveFile + 'curves0.vtk', self.xt0)
                pointSets.saveTrajectories(self.outputDir + '/' + self.saveFile + 'curves1.vtk', self.xt1)

            self.fvDef0.updateVertices(np.squeeze(self.xt0[-1, :, :]))
            self.fvDef1.updateVertices(np.squeeze(self.xt1[-1, :, :]))
            dim2 = self.dim**2
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, self.Afft[t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2+self.dim]

            (xt0, Jt0)  = evol.landmarkDirectEvolutionEuler(self.x0, self.at0, self.param.KparDiff, affine=A,
                                                              withJacobian=True)
            (xt1, Jt1)  = evol.landmarkDirectEvolutionEuler(self.x1, self.at1, self.param.KparDiff,
                                                              withJacobian=True)
            if self.affine=='euclidean' or self.affine=='translation':
                self.saveCorrectedEvolution(self.fv0, xt0, self.Afft, self.at0, fileName=self.saveFile +'0',
                                            Jacobian=Jt0)
            self.saveEvolution(self.fv0, xt0, Jacobian=Jt0, fileName=self.saveFile+'_0_', velocity=self.v0)
            self.saveEvolution(self.fv1, xt1, Jacobian=Jt1, fileName=self.saveFile+'_1_', velocity=self.v1,
                               orientation=1)
        else:
            (obj1, xt) = self.objectiveFunDef2(self.at0, self.at1, self.Afft, withTrajectory=True, regWeight=self.regweight)
            self.xt0 = xt[0]
            self.xt1 = xt[1]
            self.fvDef0.updateVertices(np.squeeze(self.xt0[-1, :, :]))
            self.fvDef1.updateVertices(np.squeeze(self.xt1[-1, :, :]))

        if self.pplot:
            fig=plt.figure(4)
            #fig.clf()
            ax = Axes3D(fig)
            lim0 = self.addSurfaceToPlot(self.fvDef0, ax, ec = 'k', fc = 'b')
            lim1 = self.addSurfaceToPlot(self.fvDef1, ax, ec='k', fc='r')
            ax.set_xlim(min(lim0[0][0],lim1[0][0]), max(lim0[0][1],lim1[0][1]))
            ax.set_ylim(min(lim0[1][0],lim1[1][0]), max(lim0[1][1],lim1[1][1]))
            ax.set_zlim(min(lim0[2][0],lim1[2][0]), max(lim0[2][1],lim1[2][1]))
            fig.canvas.flush_events()
            #plt.axis('equal')
            #plt.pause(0.1)


    def endOfProcedure(self):
        self.endOfIteration()

    def optimizeMatching(self):
        self.coeffAff = self.coeffAff2
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
        self.epsMax = 5.
        logging.info('Gradient lower bound: %f' %(self.gradEps))
        self.coeffAff = self.coeffAff1
        if self.param.algorithm == 'cg':
            cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=0.1)
        elif self.param.algorithm == 'bfgs':
            bfgs.bfgs(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=1.,
                      Wolfe=self.param.wolfe, memory=50)

