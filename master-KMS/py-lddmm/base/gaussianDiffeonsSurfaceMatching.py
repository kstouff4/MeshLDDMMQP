import numpy.linalg as LA
import numpy as np
from .bfgs import bfgs
from . import gaussianDiffeons as gd
from . import conjugateGradient as cg, kernelFunctions as kfun, surfaces
from . import pointEvolution as evol
from . import surfaceMatching
from .surfaceMatching import Direction
from .affineBasis import AffineBasis

## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normalization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
class SurfaceMatchingParam(surfaceMatching.SurfaceMatchingParam):
    def __init__(self, timeStep = .1, sigmaKernel = 6.5, algorithm='bfgs',
                 sigmaDist=2.5, sigmaError=1.0, errorType='measure'):
        super().__init__(timeStep = timeStep, sigmaKernel =  sigmaKernel, errorType=errorType,
                       algorithm=algorithm, sigmaDist = sigmaDist, sigmaError = sigmaError, typeKernel ='gauss')
        self.KparDiff = kfun.Kernel(name = self.typeKernel, sigma = self.sigmaKernel)
        self.KparDist = kfun.Kernel(name = 'gauss', sigma = self.sigmaDist)
        self.errorType = errorType
        if errorType == 'current':
            self.fun_obj0 = surfaces.currentNorm0
            self.fun_obj = surfaces.currentNormDef
            self.fun_objGrad = surfaces.currentNormGradient
        elif errorType == 'measure':
            self.fun_obj0 = surfaces.measureNorm0
            self.fun_obj = surfaces.measureNormDef
            self.fun_objGrad = surfaces.measureNormGradient
        elif errorType == 'varifold':
            self.fun_obj0 = surfaces.varifoldNorm0
            self.fun_obj = surfaces.varifoldNormDef
            self.fun_objGrad = surfaces.varifoldNormGradient
        elif errorType=='diffeonCurrent':
            self.fun_obj0 = gd.diffeonCurrentNorm0
            self.fun_obj = gd.diffeonCurrentNormDef
            self.fun_objGrad = gd.diffeonCurrentNormGradient
        else:
            print('Unknown error Type: ', self.errorType)
	
 

# class Direction:
#     def __init__(self):
#         self.diff = []
#         self.aff = []


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
class SurfaceMatching(surfaceMatching.SurfaceMatching):
    def __init__(self, Template=None, Target=None, Diffeons=None, EpsilonNet=None, DecimationTarget=None,
                 subsampleTargetSize = -1, 
                 DiffeonEpsForNet=None, DiffeonSegmentationRatio=None, zeroVar=False, fileTempl=None,
                 fileTarg=None, param=None, maxIter=1000, regWeight = 1.0, affineWeight = 1.0, verb=True,
                 rotWeight = None, scaleWeight = None, transWeight = None, testGradient=False, saveFile = 'evolution', affine = 'none', outputDir = '.'):
        super().__init__(Template=Template, Target=Target, subsampleTargetSize=subsampleTargetSize,fileTempl=fileTempl,fileTarg=fileTarg,param=param,
                       maxIter=maxIter,regWeight=regWeight,affineWeight=affineWeight,verb=verb,rotWeight=rotWeight,scaleWeight=scaleWeight, transWeight=transWeight,
                       testGradient=testGradient, saveFile=saveFile,affine=affine,outputDir=outputDir)
        # if Template==None:
        #     if fileTempl==None:
        #         print('Please provide a template surface')
        #         return
        #     else:
        #         self.fv0 = surfaces.Surface(filename=fileTempl)
        # else:
        #     self.fv0 = surfaces.Surface(surf=Template)
        # if Target==None:
        #     if fileTarg==None:
        #         print('Please provide a target surface')
        #         return
        #     else:
        #         self.fv1 = surfaces.Surface(filename=fileTarg)
        # else:
        #     self.fv1 = surfaces.Surface(surf=Target)

        self.fv0Fine = surfaces.Surface(surf=self.fv0)
        self.saveRate = 10
        # self.iter = 0
        # self.gradEps = -1
        # self.npt = self.fv0.vertices.shape[0]
        # self.dim = self.fv0.vertices.shape[1]
        # self.setOutputDir(outputDir)
        # self.maxIter = maxIter
        # self.verb = verb
        # self.testGradient = testGradient
        # self.regweight = regWeight
        # self.affine = affine
        # affB = AffineBasis(self.dim, affine)
        # self.affineDim = affB.affineDim
        # self.affineBasis = affB.basis
        # self.affineWeight = affineWeight * np.ones([self.affineDim, 1])
        # if (len(affB.rotComp) > 0) & (rotWeight != None):
        #     self.affineWeight[affB.rotComp] = rotWeight
        # if (len(affB.simComp) > 0) & (scaleWeight != None):
        #     self.affineWeight[affB.simComp] = scaleWeight
        # if (len(affB.transComp) > 0) & (transWeight != None):
        #     self.affineWeight[affB.transComp] = transWeight
        #
        # if param==None:
        #     self.param = SurfaceMatchingParam()
        # else:
        #     self.param = param
        # self.x0 = self.fv0.vertices
        if Diffeons==None:
            if EpsilonNet==None:
                if DecimationTarget==None:
                    if DiffeonEpsForNet==None:
                        if DiffeonSegmentationRatio==None:
                            self.c0 = np.copy(self.x0) ;
                            self.S0 = np.zeros([self.x0.shape[0], self.x0.shape[1], self.x0.shape[1]])
                            self.idx = None
                        else:
                            (self.c0, self.S0, self.idx) = gd.generateDiffeonsFromSegmentation(self.fv0, DiffeonSegmentationRatio)
                            #self.S0 *= self.param.sigmaKernel**2;
                    else:
                        (self.c0, self.S0, self.idx) = gd.generateDiffeonsFromNet(self.fv0, DiffeonEpsForNet)
                else:
                    (self.c0, self.S0, self.idx) = gd.generateDiffeonsFromDecimation(self.fv0, DecimationTarget)
            else:
                (self.c0, self.S0, self.idx) = gd.generateDiffeons(self.fv0, EpsilonNet[0], EpsilonNet[1])
        else:
            (self.c0, self.S0, self.idx) = Diffeons
        if zeroVar:
            self.S0 = np.zeros(self.S0.shape)

            #print self.S0
        if (subsampleTargetSize > 0):
            self.fv0.Simplify(subsampleTargetSize)
            v0 = self.fv0.surfVolume()
            v1 = self.fv0Fine.surfVolume()
            if (v0*v1 < 0):
                self.fv0.flipFaces()
            if self.param.errorType == 'diffeonCurrent':
                n = self.fv0Fine.vertices.shape[0]
                m = self.fv0.vertices.shape[0]
                dist2 = ((self.fv0Fine.vertices.reshape([n, 1, 3]) -
                          self.fv0.vertices.reshape([1,m,3]))**2).sum(axis=2)
                idx = - np.ones(n, dtype=np.int)
                for p in range(n):
                    closest = np.unravel_index(np.argmin(dist2[p, :].ravel()), [m, 1])
                    idx[p] = closest[0]
                (x0, xS0, idx) = gd.generateDiffeons(self.fv0Fine, self.fv0.vertices, idx)
                b0 = gd.approximateSurfaceCurrent(x0, xS0, self.fv0Fine, self.param.KparDist.sigma)
                gdOpt = gd.gdOptimizer(surf=self.fv0Fine, sigmaDist = self.param.KparDist.sigma, Diffeons = (x0, xS0, b0) , testGradient=False, maxIter=50)
                gdOpt.optimize()
                self.x0 = gdOpt.c0
                self.xS0 = gdOpt.S0
                self.b0 = gdOpt.b0
            else:
                self.x0 = self.fv0.vertices
            print('simplified template', self.fv0.vertices.shape[0])
        #self.fvDef = surfaces.Surface(surf=self.fv0)
        self.ndf = self.c0.shape[0]
        #self.Tsize = int(round(1.0/self.param.timeStep))
        self.at = np.zeros([self.Tsize, self.c0.shape[0], self.x0.shape[1]])
        self.atTry = np.zeros([self.Tsize, self.c0.shape[0], self.x0.shape[1]])
        # self.Afft = np.zeros([self.Tsize, self.affineDim])
        # self.AfftTry = np.zeros([self.Tsize, self.affineDim])
        # self.xt = np.tile(self.x0, [self.Tsize+1, 1, 1])
        self.ct = np.tile(self.c0, [self.Tsize+1, 1, 1])
        self.St = np.tile(self.S0, [self.Tsize+1, 1, 1, 1])
        print('error type:', self.param.errorType)
        if self.param.errorType =='diffeonCurrent':
            self.xSt = np.tile(self.xS0, [self.Tsize+1, 1, 1, 1])
            self.bt = np.tile(self.b0, [self.Tsize+1, 1, 1])
            self.dcurr = True
        else:
            self.dcurr=False

        # self.obj = None
        # self.objTry = None
        self.gradCoeff = self.ndf
        # self.saveFile = saveFile
        # self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        # self.fv1.saveVTK(self.outputDir+'/Target.vtk')


    def  objectiveFunDef(self, at, Afft, withTrajectory = False, withJacobian=False, initial = None):
        if initial is None:
            x0 = self.x0
            c0 = self.c0
            S0 = self.S0
            if self.dcurr:
                b0 = self.b0
                xS0 = self.xS0
        else:
            x0 = self.x0
            if self.dcurr:
                (c0, S0, b0, xS0) = initial
            else:
                (c0, S0) = initial
        param = self.param
        timeStep = 1.0/self.Tsize
        dim2 = self.dim**2
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        if withJacobian:
            if self.dcurr:
                (ct, St, bt, xt, xSt, Jt)  = evol.gaussianDiffeonsEvolutionEuler(c0, S0, at, param.sigmaKernel, affine=A, withJacobian=True, withNormals=b0, withDiffeonSet=(x0, xS0))
            else:
                (ct, St, xt, Jt)  = evol.gaussianDiffeonsEvolutionEuler(c0, S0, at, param.sigmaKernel, affine=A, withPointSet = x0, withJacobian=True)
        else:
            if self.dcurr:
                (ct, St, bt, xt, xSt)  = evol.gaussianDiffeonsEvolutionEuler(c0, S0, at, param.sigmaKernel, affine=A, withNormals=b0, withDiffeonSet=(x0, xS0))
            else:
                (ct, St, xt)  = evol.gaussianDiffeonsEvolutionEuler(c0, S0, at, param.sigmaKernel, affine=A, withPointSet = x0)

        #print xt[-1, :, :]
        #print obj
        obj=0
        #print St.shape
        for t in range(self.Tsize):
            c = np.squeeze(ct[t, :, :])
            S = np.squeeze(St[t, :, :, :])
            a = np.squeeze(at[t, :, :])
            #rzz = kfun.kernelMatrix(param.KparDiff, z)
            rcc = gd.computeProducts(c, S, param.sigmaKernel)
            obj = obj + self.regweight*timeStep*np.multiply(a, np.dot(rcc,a)).sum()
            if self.affineDim > 0:
                obj +=  timeStep * np.multiply(self.affineWeight.reshape(Afft[t].shape), Afft[t]**2).sum()
            #print xt.sum(), at.sum(), obj
        if withJacobian:
            if self.dcurr:
                return obj, ct, St, bt, xt, xSt, Jt
            else:
                return obj, ct, St, xt, Jt
        elif withTrajectory:
            if self.dcurr:
                return obj, ct, St, bt, xt, xSt
            else:
                return obj, ct, St, xt
        else:
            return obj

    def dataTerm(self, _data):
        if self.dcurr:
            obj = self.param.fun_obj(_data[0], _data[1], _data[2], self.fv1, self.param.KparDist.sigma) / (self.param.sigmaError**2)
        else:
            obj = self.param.fun_obj(_data, self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
        return obj
    
    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = self.param.fun_obj0(self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
            if self.dcurr:
                (self.obj, self.ct, self.St, self.bt, self.xt, self.xSt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
                data = (self.xt[-1,:,:], self.xSt[-1,:,:,:], self.bt[-1,:,:])
                self.obj += self.obj0 + self.dataTerm(data)
            else:
                (self.obj, self.ct, self.St, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
                self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
                self.obj += self.obj0 + self.dataTerm(self.fvDef)

        return self.obj

    def getVariable(self):
        return [self.at, self.Afft]

    def updateTry(self, dir, eps, objRef=None):
        atTry = self.at - eps * dir.diff
        if self.affineDim > 0:
            AfftTry = self.Afft - eps * dir.aff
        else:
            AfftTry = self.Afft
        if self.dcurr:
            objTry, ct, St, bt, xt, xSt = self.objectiveFunDef(atTry, AfftTry, withTrajectory=True)
            data = (xt[-1,:,:], xSt[-1,:,:,:], bt[-1,:,:])
            objTry += self.obj0 + self.dataTerm(data)
        else:
            objTry, ct, St, xt = self.objectiveFunDef(atTry, AfftTry, withTrajectory=True)
            ff = surfaces.Surface(surf=self.fvDef)
            ff.updateVertices(np.squeeze(xt[-1, :, :]))
            objTry += self.obj0 + self.dataTerm(ff)

        if np.isnan(objTry):
            print('Warning: nan in updateTry')
            return 1e500

        if (objRef == None) | (objTry < objRef):
            self.atTry = atTry
            self.objTry = objTry
            self.AfftTry = AfftTry

            #print 'objTry=',objTry, dir.diff.sum()
        return objTry



    def endPointGradient(self, endpoint=None):
        if endpoint is not None:
            if self.dcurr:
                endpoint = [self.xt[-1, :, :], self.xSt[-1, :, :, :], self.bt[-1, :, :]]
            else:
                endpoint = self.fvDef
        if self.dcurr:
            #print self.bt.shape
            (px, pxS, pb) = self.param.fun_objGrad(endpoint[0], endpoint[1], endpoint[2],
                                                   self.fv1, self.param.KparDist.sigma)
            pc = np.zeros(self.c0.shape)
            pS = np.zeros(self.S0.shape)
            #gd.testDiffeonCurrentNormGradient(self.ct[-1, :, :], self.St[-1, :, :, :], self.bt[-1, :, :],
            #                               self.fv1, self.param.KparDist.sigma)
            px = px / self.param.sigmaError**2
            pxS = pxS / self.param.sigmaError**2
            pb = pb / self.param.sigmaError**2
            return (pc, pS, pb, px, pxS)
        else:
            px = self.param.fun_objGrad(endpoint, self.fv1, self.param.KparDist)/ self.param.sigmaError**2
            pc = np.zeros(self.c0.shape)
            pS = np.zeros(self.S0.shape)
            return (pc, pS, px) 


    def getGradient(self, coeff=1.0, update=None):
        if update is None:
            at = self.at
            endPoint = self.fvDef
            A = self.affB.getTransforms(self.Afft)
        else:
            A = self.affB.getTransforms(self.Afft - update[1]*update[0].aff)
            at = self.at - update[1] *update[0].diff
            #xt = evol.landmarkDirectEvolutionEuler(self.x0, at, self.param.KparDiff, affine=A)
            if self.dcurr:
                ct ,St ,bt ,xt ,xSt = evol.gaussianDiffeonsEvolutionEuler(self.c0 ,self.S0 ,at ,self.param.sigmaKernel ,affine=A,
                                                                       withNormals=self.b0 ,withDiffeonSet=(self.x0 ,self.xS0))
                endPoint = [xt[-1, :, :], xSt[-1, :, :, :], bt[-1, :, :]]
            else:
                ct ,St ,xt = evol.gaussianDiffeonsEvolutionEuler(self.c0 ,self.S0 ,at ,self.param.sigmaKernel ,
                                                                 affine=A ,withPointSet=self.x0)
                endPoint = surfaces.Surface(surf=self.fv0)
                endPoint.updateVertices(xt[-1, :, :])

        # A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        # dim2 = self.dim**2
        # if self.affineDim > 0:
        #     for t in range(self.Tsize):
        #         AB = np.dot(self.affineBasis, self.Afft[t])
        #         A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
        #         A[1][t] = AB[dim2:dim2+self.dim]

        if self.dcurr:
            (pc1, pS1, pb1, px1, pxS1) = self.endPointGradient(endpoint=endPoint)
            foo = evol.gaussianDiffeonsGradientNormals(self.c0, self.S0, self.b0, self.x0, self.xS0,
                                                       at, -pc1, -pS1, -pb1, -px1, -pxS1, self.param.sigmaKernel, self.regweight, affine=A, euclidean=self.euclideanGradient)
        else:
            (pc1, pS1, px1) = self.endPointGradient(endpoint=endPoint)
            foo = evol.gaussianDiffeonsGradientPset(self.c0, self.S0, self.x0,
                                                    at, -pc1, -pS1, -px1, self.param.sigmaKernel, self.regweight, affine=A, euclidean=self.euclideanGradient)

        grd = Direction()
        grd.diff = foo[0]/(coeff*self.Tsize)
        grd.aff = np.zeros(self.Afft.shape)
        if self.affineDim > 0:
            dim2 = self.dim**2
            dA = foo[1]
            db = foo[2]
            grd.aff = 2 * self.Afft
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               grd.aff[t] -=  np.divide(dAff.reshape(grd.aff[t].shape), self.affineWeight.reshape(grd.aff[t].shape))
            grd.aff /= (coeff*self.Tsize)
        return grd



    # def addProd(self, dir1, dir2, beta):
    #     dir = Direction()
    #     dir.diff = dir1.diff + beta * dir2.diff
    #     dir.aff = dir1.aff + beta * dir2.aff
    #     return dir
    #
    # def copyDir(self, dir0):
    #     dir = Direction()
    #     dir.diff = np.copy(dir0.diff)
    #     dir.aff = np.copy(dir0.aff)
    #     return dir
    #
    #
    def randomDir(self):
        dirfoo = Direction()
        dirfoo.diff = np.random.randn(self.Tsize, self.ndf, self.dim)
        dirfoo.aff = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            c = np.squeeze(self.ct[t, :, :])
            S = np.squeeze(self.St[t, :, :, :])
            gg = np.squeeze(g1.diff[t, :, :])
            rcc = gd.computeProducts(c, S, self.param.sigmaKernel)
            (L, W) = LA.eigh(rcc)
            rcc += (L.max()/1000)*np.eye(rcc.shape[0])
            u = np.dot(rcc, gg)
            uu = np.multiply(g1.aff[t], self.affineWeight.reshape(g1.aff[t].shape))
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr.diff[t, :, :])
                res[ll]  = res[ll] + np.multiply(ggOld,u).sum()
                if self.affineDim > 0:
                    res[ll] += np.multiply(uu, gr.aff[t]).sum()
                    #                    +np.multiply(g1[1][t, dim2:dim2+self.dim], gr[1][t, dim2:dim2+self.dim]).sum())
                ll = ll + 1

        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            gg = np.squeeze(g1.diff[t, :, :])
            uu = np.multiply(g1.aff[t], self.affineWeight.reshape(g1.aff[t].shape))
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr.diff[t, :, :])
                res[ll]  = res[ll] + np.multiply(ggOld,gg).sum()
                if self.affineDim > 0:
                    res[ll] += np.multiply(uu, gr.aff[t]).sum()
                ll = ll + 1

        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.at = np.copy(self.atTry)
        self.Afft = np.copy(self.AfftTry)
        #print self.at
    def saveB(self, fileName, c, b):
        with open(fileName, 'w') as fvtkout:
            fvtkout.write('# vtk DataFile Version 3.0\nSurface Data\nASCII\nDATASET UNSTRUCTURED_GRID\n') 
            fvtkout.write('\nPOINTS {0: d} float'.format(c.shape[0]))
            for ll in range(c.shape[0]):
                fvtkout.write('\n{0: f} {1: f} {2: f}'.format(c[ll,0], c[ll,1], c[ll,2]))
            fvtkout.write(('\nPOINT_DATA {0: d}').format(c.shape[0]))

            fvtkout.write('\nVECTORS bt float')
            for ll in range(c.shape[0]):
                fvtkout.write('\n {0: .5f} {1: .5f} {2: .5f}'.format(b[ll, 0], b[ll, 1], b[ll, 2]))
            fvtkout.write('\n')


    def endOfIteration(self):
        #print self.obj0
        self.iter += 1
        if (self.iter % self.saveRate == 0) :
            if self.dcurr:
                print('saving...')
                (obj1, self.ct, self.St, self.bt, self.xt, self.xSt, Jt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True, withJacobian=True)
                dim2 = self.dim**2
                A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
                if self.affineDim > 0:
                    for t in range(self.Tsize):
                        AB = np.dot(self.affineBasis, self.Afft[t])
                        A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                        A[1][t] = AB[dim2:dim2+self.dim]
                (ct, St, xt, Jt)  = evol.gaussianDiffeonsEvolutionEuler(self.c0, self.S0, self.at, self.param.sigmaKernel, affine=A,
                                                                        withPointSet = self.fv0Fine.vertices, withJacobian=True)
                fvDef = surfaces.Surface(surf=self.fv0Fine)
                for kk in range(self.Tsize+1):
                    self.fvDef.updateVertices(np.squeeze(self.xt[kk, :, :]))
                    fvDef.updateVertices(np.squeeze(xt[kk, :, :]))
                    # foo = (gd.diffeonCurrentNormDef(self.xt[kk], self.xSt[kk], self.bt[kk], self.fvDef, self.param.KparDist.sigma)
                    #        + gd.diffeonCurrentNorm0(self.fvDef, self.param.KparDist))/ (self.param.sigmaError**2)
                    # print foo
                    fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = Jt[kk, :], scal_name='Jacobian')
                    gd.saveDiffeons(self.outputDir +'/'+ self.saveFile+'Diffeons'+str(kk)+'.vtk', self.ct[kk,:,:], self.St[kk,:,:,:])
                    self.saveB(self.outputDir +'/'+ self.saveFile+'Bt'+str(kk)+'.vtk', self.xt[kk,:,:], self.bt[kk,:,:])
            else:
                (obj1, self.ct, self.St, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
                self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
                dim2 = self.dim**2
                A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
                if self.affineDim > 0:
                    for t in range(self.Tsize):
                        AB = np.dot(self.affineBasis, self.Afft[t])
                        A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                        A[1][t] = AB[dim2:dim2+self.dim]
                (ct, St, xt, Jt)  = evol.gaussianDiffeonsEvolutionEuler(self.c0, self.S0, self.at, self.param.sigmaKernel, affine=A,
                                                                        withPointSet = self.fv0Fine.vertices, withJacobian=True)
                fvDef = surfaces.Surface(surf=self.fv0Fine)
                for kk in range(self.Tsize+1):
                    fvDef.updateVertices(np.squeeze(xt[kk, :, :]))
                    fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = Jt[kk, :], scal_name='Jacobian')
                        #self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = self.idx, scal_name='Labels')
                    gd.saveDiffeons(self.outputDir +'/'+ self.saveFile+'Diffeons'+str(kk)+'.vtk', self.ct[kk,:,:], self.St[kk,:,:,:])

            #print self.bt

        else:
            if self.dcurr:
                (obj1, self.ct, self.St, self.bt, self.xt, self.xSt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            else:
                (obj1, self.ct, self.St, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
                self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))

    def restart(self, EpsilonNet=None, DiffeonEpsForNet=None, DiffeonSegmentationRatio=None, DecimationTarget=None):
        if EpsilonNet is None:
            if DecimationTarget is None:
                if DiffeonEpsForNet is None:
                    if DiffeonSegmentationRatio is None:
                        c0 = np.copy(self.x0)
                        S0 = np.zeros([self.x0.shape[0], self.x0.shape[1], self.x0.shape[1]])
                        #net = range(c0.shape[0])
                        idx = range(c0.shape[0])
                    else:
                        (c0, S0, idx) = gd.generateDiffeonsFromSegmentation(self.fv0, DiffeonSegmentationRatio)
                        #self.S0 *= self.param.sigmaKernel**2;
                else:
                    (c0, S0, idx) = gd.generateDiffeonsFromNet(self.fv0, DiffeonEpsForNet)
            else:
                (c0, S0, idx) = gd.generateDiffeonsFromDecimation(self.fv0, DecimationTarget)
        else:
            net = EpsilonNet[2]
            (c0, S0, idx) = gd.generateDiffeons(self.fv0, EpsilonNet[0], EpsilonNet[1])

        (ctPrev, StPrev, ct, St) = evol.gaussianDiffeonsEvolutionEuler(self.c0, self.S0, self.at, self.param.sigmaKernel, withDiffeonSet=(c0, S0))
        at = np.zeros([self.Tsize, c0.shape[0], self.x0.shape[1]])
        #fvDef = surfaces.Surface(surf=self.fvDef)
        for t in range(self.Tsize):
            g1 = gd.computeProducts(np.squeeze(ct[t,:,:]),np.squeeze(St[t,:,:]), self.param.sigmaKernel) 
            g2 = gd.computeProductsAsym(np.squeeze(ct[t,:,:]),np.squeeze(St[t,:,:]), np.squeeze(ctPrev[t,:,:]),np.squeeze(StPrev[t,:,:]), self.param.sigmaKernel)
            g2a = np.dot(g2, np.squeeze(self.at[t, :, :]))
            at[t, :, :] = LA.solve(g1, g2a)
            g0 = gd.computeProducts(np.squeeze(ctPrev[t,:,:]),np.squeeze(StPrev[t,:,:]), self.param.sigmaKernel)
            n0 = np.multiply(self.at[t, :, :], np.dot(g0, self.at[t, :, :])).sum()
            n1 = np.multiply(at[t, :, :], np.dot(g1, at[t, :, :])).sum()
            print('norms: ', n0, n1)
            # fvDef.updateVertices(np.squeeze(self.xt[t, :, :]))
            # (AV, AF) = fvDef.computeVertexArea()
            # weights = np.zeros([c0.shape[0], self.c0.shape[0]])
            # diffArea = np.zeros(self.c0.shape[0])
            # diffArea2 = np.zeros(c0.shape[0])
            # for k in range(self.npt):
            #     diffArea[self.idx[k]] += AV[k] 
            #     diffArea2[idx[k]] += AV[k]
            #     weights[idx[k], self.idx[k]] += AV[k]
            # weights /= diffArea.reshape([1, self.c0.shape[0]])
            # at[t] = np.dot(weights, self.at[t, :, :])
        self.c0 = c0
        self.idx = idx
        self.S0 = S0
        self.at = at
        self.ndf = self.c0.shape[0]
        self.ct = np.tile(self.c0, [self.Tsize+1, 1, 1])
        self.St = np.tile(self.S0, [self.Tsize+1, 1, 1, 1])
        if self.dcurr:
            self.b0 = gd.approximateSurfaceCurrent(self.c0, self.S0, self.fv0, self.param.KparDist.sigma)
            #print self.b0.shape
            self.bt = np.tile(self.b0, [self.Tsize+1, 1, 1])
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.ndf
        self.optimizeMatching()


    def optimizeMatching(self):
        obj0 = self.param.fun_obj0(self.fv1, self.param.KparDist) # / (self.param.sigmaError**2)
        if self.dcurr:
            (obj, self.ct, self.St, self.bt, self.xt, self.xSt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            data = (self.xt[-1,:,:], self.xSt[-1,:,:,:], self.bt[-1,:,:])
            print('objDef = ', obj, 'dataterm = ',  obj0 + self.dataTerm(data)* (self.param.sigmaError**2))
            print(obj0 + surfaces.currentNormDef(self.fv0, self.fv1, self.param.KparDist))
        else:
            (obj, self.ct, self.St, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            print('objDef = ', obj, 'dataterm = ',  obj0 + self.dataTerm(self.fvDef))

        if self.gradEps < 0:
            grd = self.getGradient(self.gradCoeff)
            [grd2] = self.dotProduct(grd, [grd])

            self.gradEps = max(0.001, np.sqrt(grd2) / 10000)

        print('Gradient lower bound: ', self.gradEps)
        if self.param.algorithm == 'cg':
            cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient)
        else:
            bfgs(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient)
        #return self.at, self.xt

