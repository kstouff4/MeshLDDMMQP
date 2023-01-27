import os
from copy import deepcopy
import numpy as np
import scipy.linalg as la
import logging
from . import matchingParam
from . import conjugateGradient as cg, kernelFunctions as kfun, pointEvolution as evol, loggingUtils, bfgs
from .pointSets import PointSet
from . import pointSets, pointset_distances as psd
from .affineBasis import AffineBasis, getExponential
import matplotlib


## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
class PointSetMatchingParam(matchingParam.MatchingParam):
    def __init__(self, timeStep = .1, algorithm='cg', Wolfe=True, KparDiff = None, KparDist = None,
                 sigmaError = 1.0, errorType = 'measure'):
        super().__init__(timeStep=timeStep, algorithm = algorithm, Wolfe=Wolfe,
                         KparDiff = KparDiff, KparDist = KparDist, sigmaError=sigmaError,
                         errorType = errorType)
        self.sigmaError = sigmaError


class Direction:
    def __init__(self):
        self.diff = []
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
class PointSetMatching(object):

    def __init__(self, Template=None, Target=None, param=None, maxIter=1000,
                 regWeight = 1.0, affineWeight = 1.0, verb=True,
                 rotWeight = None, scaleWeight = None, transWeight = None,
                 testGradient=True, saveFile = 'evolution', pplot = True,
                 saveTrajectories = False, affine = 'none', outputDir = '.'):
        if param is None:
            self.param = PointSetMatchingParam()
        else:
            self.param = param

        if self.param.algorithm == 'cg':
             self.euclideanGradient = False
        else:
            self.euclideanGradient = True

        self.fv0 = None
        self.fv1 = None
        self.fvInit = None
        self.dim = 0
        self.fun_obj = None
        self.fun_obj0 = None
        self.fun_objGrad = None
        self.obj0 = 0
        self.coeffAff = 1
        self.obj = None
        self.objDef = 0
        self.objData = 0
        self.objTry = None
        self.objTryDef = 0
        self.objTryData = 0
        self.xt = None
        self.setOutputDir(outputDir)
        self.set_template_and_target(Template, Target)
        self.set_fun(self.param.errorType)

        self.reset = False
        self.Kdiff_dtype = self.param.KparDiff.pk_dtype
        self.Kdist_dtype = self.param.KparDist.pk_dtype

            #print np.fabs(self.fv1.surfel-self.fv0.surfel).max()
        self.set_parameters(maxIter=maxIter, regWeight = regWeight, affineWeight = affineWeight,
                            verb=verb, rotWeight = rotWeight, scaleWeight = scaleWeight,
                            transWeight = transWeight, testGradient=testGradient, saveFile = saveFile,
                            saveTrajectories = saveTrajectories, affine = affine)
        self.initialize_variables()
        self.gradCoeff = self.x0.shape[0]
        self.pplot = pplot


    def initialize_variables(self):
        self.x0 = np.copy(self.fv0.points)
        self.fvDef = deepcopy(self.fv0)
        self.npt = self.x0.shape[0]
        # self.u = np.zeros((self.dim, 1))
        # self.u[0:2] = 1/np.sqrt(2.)

        self.Tsize = int(round(1.0/self.param.timeStep))
        self.at = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.atTry = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.AfftTry = np.zeros([self.Tsize, self.affineDim])
        self.xt = np.tile(self.x0, [self.Tsize+1, 1, 1])
        self.v = np.zeros([self.Tsize+1, self.npt, self.dim])


    def set_template_and_target(self, Template, Target):
        if Template is None:
            logging.error('Please provide a template surface')
            return
        else:
            if isinstance(Template, PointSet):
                self.fv0 = deepcopy(Template)
            else:
                self.fv0 = pointSets.loadlmk(Template)[0]

        if Target is None:
            logging.error('Please provide a target surface')
            return
        else:
            if isinstance(Target, PointSet):
                self.fv1 = deepcopy(Target)
            else:
                self.fv1 = pointSets.loadlmk(Template)[0]

        self.fv0.save(self.outputDir + '/Template.vtk')
        self.fv1.save(self.outputDir + '/Target.vtk')
        self.dim = self.fv0.points.shape[1]

    def set_fun(self, errorType):
        self.param.errorType = errorType
        if errorType == 'L2':
            self.fun_obj0 = psd.L2Norm0
            self.fun_obj = psd.L2NormDef
            self.fun_objGrad = psd.L2NormGradient
        elif errorType == 'measure':
            self.fun_obj0 = psd.measureNorm0
            self.fun_obj = psd.measureNormDef
            self.fun_objGrad = psd.measureNormGradient
        else:
            logging.error('Unknown error Type: ' + self.param.errorType)

    def set_parameters(self, maxIter=1000,
                 regWeight = 1.0, affineWeight = 1.0, internalWeight=1.0, verb=True,
                 affineOnly = False,
                 rotWeight = None, scaleWeight = None, transWeight = None, symmetric = False,
                 testGradient=True, saveFile = 'evolution',
                 saveTrajectories = False, affine = 'none'):
        self.saveRate = 10
        self.gradEps = -1
        self.randomInit = False
        self.iter = 0
        self.maxIter = maxIter
        self.verb = verb
        self.saveTrajectories = saveTrajectories
        self.symmetric = symmetric
        self.testGradient = testGradient
        self.internalWeight = internalWeight
        self.regweight = regWeight

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



    def setOutputDir(self, outputDir):
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                logging.error('Cannot save in ' + outputDir)
                return
            else:
                os.makedirs(outputDir)


    def dataTerm(self, _fvDef, _fvInit = None):
        # if self.param.errorType == 'classification':
        #     obj = pointSets.LogisticScoreL2(_fvDef, self.fv1, self.u, w=self.wTr, intercept=self.intercept, l1Cost=self.l1Cost) \
        #           / (self.param.sigmaError**2)
        #     #obj = pointSets.LogisticScore(_fvDef, self.fv1, self.u) / (self.param.sigmaError**2)
        if self.param.errorType == 'measure':
            obj = self.fun_obj(_fvDef, self.fv1, self.param.KparDist) / (self.param.sigmaError ** 2)
        else:
            obj = self.fun_obj(_fvDef, self.fv1) / (self.param.sigmaError**2)
        return obj

    def  objectiveFunDef(self, at, Afft, kernel = None, withTrajectory = False, withJacobian=False, x0 = None, regWeight = None):
        if x0 is None:
            x0 = self.x0
        if kernel is None:
            kernel = self.param.KparDiff
        #print 'x0 fun def', x0.sum()
        if regWeight is None:
            regWeight = self.regweight
        timeStep = 1.0/self.Tsize
        dim2 = self.dim**2
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        if withJacobian:
            (xt,Jt)  = evol.landmarkDirectEvolutionEuler(x0, at, kernel, affine=A, withJacobian=True)
        else:
            xt  = evol.landmarkDirectEvolutionEuler(x0, at, kernel, affine=A)

        obj=0
        obj1 = 0 
        for t in range(self.Tsize):
            z = np.squeeze(xt[t, :, :])
            a = np.squeeze(at[t, :, :])
            ra = kernel.applyK(z, a)
            if hasattr(self, 'v'):  
                self.v[t, :] = ra
            obj = obj + regWeight*timeStep*(a*ra).sum()

            if self.affineDim > 0:
                obj1 +=  timeStep * (self.affineWeight.reshape(Afft[t].shape) * Afft[t]**2).sum()
            #print xt.sum(), at.sum(), obj
        #print obj, obj+obj1
        obj += obj1
        if withJacobian:
            return obj, xt, Jt
        elif withTrajectory:
            return obj, xt
        else:
            return obj

    def makeTryInstance(self, pts):
        ff = PointSet(data=pts, weights=self.fv0.weights)
        return ff

    def objectiveFun(self):
        if self.obj == None:
            if self.param.errorType == 'measure':
                self.obj0 = self.fun_obj0(self.fv1, self.param.KparDist) / (self.param.sigmaError ** 2)
            else:
                self.obj0 = self.fun_obj0(self.fv1) / (self.param.sigmaError**2)
            (self.objDef, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.fvDef.points = np.copy(np.squeeze(self.xt[-1, :, :]))
            self.objData = self.dataTerm(self.fvDef)
            self.obj = self.obj0 + self.objData + self.objDef
        return self.obj

    def getVariable(self):
        return [self.at, self.Afft]

    def updateTry(self, dr, eps, objRef=None):
        objTry = self.obj0
        atTry = self.at - eps * dr.diff
        if self.affineDim > 0:
            AfftTry = self.Afft - eps * dr.aff
        else:
            AfftTry = self.Afft

        foo = self.objectiveFunDef(atTry, AfftTry, withTrajectory=True)
        objTryDef = foo[0]
        ff = self.makeTryInstance(np.squeeze(foo[1][-1, :, :]))
        objTryData = self.dataTerm(ff)
        objTry = self.obj0 + objTryData + objTryDef

        if np.isnan(objTry):
            logging.info('Warning: nan in updateTry')
            return 1e500

        if (objRef is None) or (objTry < objRef):
            self.atTry = atTry
            self.objTry = objTry
            self.objTryData = objTryData
            self.objTryDef = objTryDef
            self.AfftTry = AfftTry
            #print 'objTry=',objTry, dir.diff.sum()

        return objTry


    def testEndpointGradient(self):
        c0 = self.dataTerm(self.fvDef)
        ff = deepcopy(self.fvDef)
        dff = np.random.normal(size=ff.points.shape)
        eps = 1e-6
        ff.points += eps*dff
        c1 = self.dataTerm(ff)
        grd = self.endPointGradient()
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c1-c0)/eps, (grd*dff).sum()) )

    def endPointGradient(self, endPoint= None):
        if endPoint is None:
            endPoint = self.fvDef
        if self.param.errorType == 'measure':
            px = self.fun_objGrad(endPoint, self.fv1, self.param.KparDist)
        else:
            px = self.fun_objGrad(endPoint, self.fv1)
        return px / self.param.sigmaError**2

    
    # def hamiltonianCovector(self, x0, at, px1, KparDiff, regWeight, affine = None):
    #     N = x0.shape[0]
    #     dim = x0.shape[1]
    #     M = at.shape[0]
    #     timeStep = 1.0/M
    #     xt = evol.landmarkDirectEvolutionEuler(x0, at, KparDiff, affine=affine)
    #     if not(affine is None):
    #         A0 = affine[0]
    #         A = np.zeros([M,dim,dim])
    #         for k in range(A0.shape[0]):
    #             A[k,...] = getExponential(timeStep*A0[k])
    #     else:
    #         A = np.zeros([M,dim,dim])
    #         for k in range(M):
    #             A[k,...] = np.eye(dim)
    #
    #     pxt = np.zeros([M+1, N, dim])
    #     pxt[M, :, :] = px1
    #     foo = np.copy(self.fv0)
    #     for t in range(M):
    #         px = np.squeeze(pxt[M-t, :, :])
    #         z = np.squeeze(xt[M-t-1, :, :])
    #         a = np.squeeze(at[M-t-1, :, :])
    #         foo = np.copy(z)
    #         v = KparDiff.applyK(z,a)
    #         a1 = np.concatenate((px[np.newaxis,...], a[np.newaxis,...], -2*regWeight*a[np.newaxis,...]))
    #         a2 = np.concatenate((a[np.newaxis,...], px[np.newaxis,...], a[np.newaxis,...]))
    #         zpx = KparDiff.applyDiffKT(z, a1, a2)
    #
    #         if not (affine is None):
    #             pxt[M-t-1, :, :] = np.dot(px, A[M-t-1]) + timeStep * zpx
    #         else:
    #             pxt[M-t-1, :, :] = px + timeStep * zpx
    #     return pxt, xt
    
    def hamiltonianGradient(self, px1, kernel = None, affine=None, regWeight=None, x0=None, at=None):
        if regWeight is None:
            regWeight = self.regweight
        if x0 is None:
            x0 = self.x0
        if at is None:
            at = self.at
        if kernel is None:
            kernel  = self.param.KparDiff
        return evol.landmarkHamiltonianGradient(x0, at, px1, kernel, regWeight, affine=affine,
                                                getCovector=True)
                                                    
    def setUpdate(self, update):
        at = self.at - update[1] * update[0].diff

        Afft = self.Afft - update[1] * update[0].aff
        if len(update[0].aff) > 0:
            A = self.affB.getTransforms(Afft)
        else:
            A = None
        xt = evol.landmarkDirectEvolutionEuler(self.x0, at, self.param.KparDiff, affine=A)
        endPoint = PointSet(data=self.fv0)
        endPoint.updatePoints(xt[-1, :, :])

        return at, Afft, xt, endPoint

    def getGradient(self, coeff=1.0, update=None):
        if update is None:
            at = None
            Afft = self.Afft
            endPoint = self.fvDef
            A = self.affB.getTransforms(self.Afft)
            xt = self.xt
        else:
            at, Afft, xt, endPoint = self.setUpdate(update)

        dim2 = self.dim**2
        px1 = -self.endPointGradient(endPoint=endPoint)
        foo = self.hamiltonianGradient(px1, at=at, affine=A)
        grd = Direction()
        if self.euclideanGradient:
            grd.diff = np.zeros(foo[0].shape)
            for t in range(self.Tsize):
                z = xt[t, :, :]
                grd.diff[t,:,:] = self.param.KparDiff.applyK(z, foo[0][t, :,:])/(coeff*self.Tsize)
        else:
            grd.diff = foo[0]/(coeff*self.Tsize)
        grd.aff = np.zeros(self.Afft.shape)
        if self.affineDim > 0:
            dA = foo[1]
            db = foo[2]
            grd.aff = 2*self.affineWeight.reshape([1, self.affineDim])*Afft
            #grd.aff = 2 * self.Afft
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               #grd.aff[t] -=  np.divide(dAff.reshape(grd.aff[t].shape), self.affineWeight.reshape(grd.aff[t].shape))
               grd.aff[t] -=  dAff.reshape(grd.aff[t].shape)
            grd.aff /= (self.coeffAff*coeff*self.Tsize)
            #            dAfft[:,0:self.dim**2]/=100
        return grd

    def startOfIteration(self):
        if self.reset:
            logging.info('Switching to 64 bits')
            self.param.KparDiff.pk_dtype = 'float64'
            self.param.KparDist.pk_dtype = 'float64'


    def addProd(self, dir1, dir2, beta):
        dir = Direction()
        dir.diff = dir1.diff + beta * dir2.diff
        dir.aff = dir1.aff + beta * dir2.aff
        return dir

    def prod(self, dir1, beta):
        dir = Direction()
        dir.diff = beta * dir1.diff
        dir.aff = beta * dir1.aff
        return dir

    def copyDir(self, dir0):
        dir = Direction()
        dir.diff = np.copy(dir0.diff)
        dir.aff = np.copy(dir0.aff)
        return dir


    def randomDir(self):
        dirfoo = Direction()
        dirfoo.diff = np.random.randn(self.Tsize, self.npt, self.dim)
        dirfoo.aff = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            z = np.squeeze(self.xt[t, :, :])
            gg = np.squeeze(g1.diff[t, :, :])
            u = self.param.KparDiff.applyK(z, gg)
            #uu = np.multiply(g1.aff[t], self.affineWeight.reshape(g1.aff[t].shape))
            uu = g1.aff[t]
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr.diff[t, :, :])
                res[ll]  = res[ll] + np.multiply(ggOld,u).sum()
                if self.affineDim > 0:
                    #print np.multiply(np.multiply(g1[1][t], gr[1][t]), self.affineWeight).shape
                    #res[ll] += np.multiply(uu, gr.aff[t]).sum() * self.coeffAff
                    res[ll] += np.multiply(uu, gr.aff[t]).sum() * self.coeffAff
                    #                    +np.multiply(g1[1][t, dim2:dim2+self.dim], gr[1][t, dim2:dim2+self.dim]).sum())
                ll = ll + 1
        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            u = np.squeeze(g1.diff[t, :, :])
            if self.affineDim > 0:
                uu = g1.aff[t]
            else:
                uu = 0
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr.diff[t, :, :])
                res[ll]  += (ggOld*u).sum()
                if self.affineDim > 0:
                    res[ll] += (uu*gr.aff[t]).sum()
                ll = ll + 1
        return res


    def acceptVarTry(self):
        self.obj = self.objTry
        self.objDef = self.objTryDef
        self.objData = self.objTryData
        self.at = np.copy(self.atTry)
        self.Afft = np.copy(self.AfftTry)
        #print self.at

    def endOfIteration(self, endP=False):
        self.iter += 1
        if self.testGradient:
            self.testEndpointGradient()

        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2
        if (self.iter % self.saveRate == 0 or endP) :
            logging.info('Saving Points...')
            (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)

            self.fvDef.points = np.copy(np.squeeze(self.xt[-1, :, :]))
            dim2 = self.dim**2
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, self.Afft[t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2+self.dim]
            (xt, Jt)  = evol.landmarkDirectEvolutionEuler(self.x0, self.at, self.param.KparDiff, affine=A,
                                                              withJacobian=True)
            if self.affine=='euclidean' or self.affine=='translation':
                X = self.affB.integrateFlow(self.Afft)
                displ = np.zeros(self.x0.shape[0])
                dt = 1.0 /self.Tsize
                for t in range(self.Tsize+1):
                    U = la.inv(X[0][t])
                    yyt = np.dot(self.xt[t,...] - X[1][t, ...], U.T)
                    f = np.copy(yyt)
                    # vf = surfaces.vtkFields() ;
                    # vf.scalars.append('Jacobian') ;
                    # vf.scalars.append(np.exp(Jt[t, :]))
                    # vf.scalars.append('displacement')
                    # vf.scalars.append(displ)
                    # vf.vectors.append('velocity') ;
                    # vf.vectors.append(vt)
                    # nu = self.fv0ori*f.computeVertexNormals()
                    pointSets.savelmk(f, self.outputDir + '/' + self.saveFile + 'Corrected' + str(t) + '.lmk')
                f = np.copy(self.fv1)
                yyt = np.dot(f - X[1][-1, ...], U.T)
                f = np.copy(yyt)
                pointSets.savePoints(self.outputDir + '/TargetCorrected.vtk', f)
            for kk in range(self.Tsize+1):
                fvDef = PointSet(data=np.squeeze(xt[kk, :, :]), weights=self.fv0.weights)
                fvDef.save(self.outputDir + '/' + self.saveFile + str(kk) + '.vtk')
        (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
        self.fvDef.points = np.copy(np.squeeze(self.xt[-1, :, :]))
        self.param.KparDiff.pk_dtype = self.Kdiff_dtype
        self.param.KparDist.pk_dtype = self.Kdist_dtype


    def optimizeMatching(self):
        #print 'dataterm', self.dataTerm(self.fvDef)
        #print 'obj fun', self.objectiveFun(), self.obj0
        self.coeffAff = self.coeffAff2
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
        logging.info('Gradient lower bound: %f' %(self.gradEps))
        self.coeffAff = self.coeffAff1
        #self.restartRate = self.relearnRate
        if self.param.algorithm == 'cg':
            cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=0.1,)
        elif self.param.algorithm == 'bfgs':
            bfgs.bfgs(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=1.,
                      Wolfe=self.param.wolfe, memory=50)
        #bfgs.bfgs(self, verb = self.verb, maxIter = self.maxIter,TestGradient=self.testGradient, epsInit=0.1)
        #return self.at, self.xt




