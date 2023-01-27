import logging
import os
import numpy.linalg as la
from . import surfaces, surface_distances as sd
from . import pointSets
from . import conjugateGradient as cg, pointEvolution as evol
from .affineBasis import *
from .surfaceMatching import SurfaceMatching, Direction



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
class SurfaceTimeMatching(SurfaceMatching):
    def __init__(self, Template=None, Target=None, param=None, times = None,
                 maxIter=1000, regWeight = 1.0, affineWeight = 1.0, mode='normal', affine = 'none',
                  rotWeight = None, scaleWeight = None, transWeight = None, internalWeight=1., volumeWeight = None,
                  rescaleTemplate=False, subsampleTargetSize=-1, testGradient=True,  saveFile = 'evolution', outputDir = '.'):

        self.rescaleTemplate = rescaleTemplate
        if times is None:
            self.times = None
        else:
            self.times = np.array(times)
        self.nTarg = len(Target)
        super().__init__(Template=Template, Target=Target, param=param, maxIter=maxIter,
                 regWeight = regWeight, affineWeight = affineWeight, internalWeight=internalWeight, mode=mode,
                 subsampleTargetSize=subsampleTargetSize, affineOnly = False,
                 rotWeight = rotWeight, scaleWeight = scaleWeight, transWeight = transWeight, symmetric = False,
                 testGradient=testGradient, saveFile = saveFile,
                 saveTrajectories = False, affine = affine, outputDir = outputDir,pplot=False)

        self.volumeWeight = volumeWeight
        self.ds = 1.
        #self.saveRate = 1

        # if self.affine=='euclidean' or self.affine=='translation':
        #     self.saveCorrected = True
        # else:
        #     self.saveCorrected = False


        # self.fv0Fine = surfaces.Surface(surf=self.fv0)
        # if (subsampleTargetSize > 0):
        #     self.fv0.Simplify(subsampleTargetSize)
        #     print('simplified template', self.fv0.vertices.shape[0])
        # v0 = self.fv0.surfVolume()
        # #print 'v0', v0
        # if self.param.errorType == 'L2Norm' and v0 < 0:
        #     #print 'flip'
        #     self.fv0.flipFaces()
        #     v0 = -v0
        # for s in self.fv1:
        #     v1 = s.surfVolume()
        #     #print 'v1', v1
        #     if (v0*v1 < 0):
        #         #print 'flip1'
        #         s.flipFaces()

    def initialize_variables(self):
        self.Tsize = int(round(1.0 / self.param.timeStep))
        self.nvert = self.fvInit.vertices.shape[0]
        if self.match_landmarks:
            self.x0 = np.concatenate((self.fvInit.vertices, self.tmpl_lmk.points), axis=0)
            self.nlmk = self.tmpl_lmk.points.shape[0]
        else:
            self.x0 = np.copy(self.fvInit.vertices)
            self.nlmk = 0
        self.x0try = np.copy(self.x0)
        self.npt = self.x0.shape[0]
        if self.times is None:
            self.times = 1+np.arange(self.nTarg)
        self.Tsize = int(round(self.times[-1]/self.param.timeStep))
        self.jumpIndex = np.round(self.times/self.param.timeStep).astype(int)
        self.isjump = np.zeros(self.Tsize+1, dtype=bool)
        for k in self.jumpIndex:
            self.isjump[k] = True


        self.at = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        if self.randomInit:
            self.at = np.random.normal(0, 1, self.at.shape)
        self.atTry = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.AfftTry = np.zeros([self.Tsize, self.affineDim])
        self.xt = np.tile(self.x0, [self.Tsize + 1, 1, 1])
        self.xtTry = np.copy(self.xt)
        self.v = np.zeros([self.Tsize + 1, self.npt, self.dim])

        self.fvDef = []
        for k in range(self.nTarg):
            self.fvDef.append(surfaces.Surface(surf=self.fv0))
        self.def_lmk = []
        if self.match_landmarks:
            for k in range(self.nTarg):
                self.def_lmk.append(pointSets.PointSet(data=self.tmpl_lmk))
        self.a0 = np.zeros([self.x0.shape[0], self.x0.shape[1]])

        self.regweight_ = np.ones(self.Tsize)
        self.regweight_[range(self.jumpIndex[0])] = self.regweight
        self.saveFileList = []
        for kk in range(self.Tsize+1):
            if kk < self.jumpIndex[0]:
                self.saveFileList.append(self.saveFile + f'fromTemplate{kk:03d}')
            else:
                self.saveFileList.append(self.saveFile + f'{kk-self.jumpIndex[0]:03d}')

    def set_template_and_target(self, Template, Target, subsampleTargetSize=-1):
        if Template is None:
            logging.error('Please provide a template surface')
            return
        else:
            self.fv0 = surfaces.Surface(surf=Template)

        if Target is None:
            logging.error('Please provide a list of target surfaces')
            return
        else:
            self.fv1 = []
            if self.param.errorType == 'L2Norm':
                for f in Target:
                    fv1 = surfaces.Surface()
                    fv1.readFromImage(f)
                    self.fv1.append(fv1)
            elif self.param.errorType == 'PointSet':
                for f in Target:
                    self.fv1.append(pointSets.PointSet(data=f))
            else:
                for f in Target:
                    self.fv1.append(surfaces.Surface(surf=f))


        if self.rescaleTemplate:
            f0 = np.fabs(self.fv0.surfVolume())
            f1 = np.fabs(self.fv1[0].surfVolume())
            self.fv0.updateVertices(self.fv0.vertices * (f1/f0)**(1./3))
            m0 = np.mean(self.fv0.vertices, axis = 0)
            m1 = np.mean(self.fv1[0].vertices, axis = 0)
            self.fv0.updateVertices(self.fv0.vertices + (m1-m0))


        self.fvInit = surfaces.Surface(surf=self.fv0)
        self.fix_orientation()
        if subsampleTargetSize > 0:
            self.fvInit.Simplify(subsampleTargetSize)
            logging.info('simplified template %d' %(self.fv0.vertices.shape[0]))
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        for k,f in enumerate(self.fv1):
            f.saveVTK(self.outputDir+f'/Target{k:03d}.vtk')
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
        self.targ_lmk = []
        for l in targ_lmk:
            self.targ_lmk.append(pointSets.PointSet(data=l))

    def initial_plot(self):
        pass

    def fix_orientation(self, fv1 = None):
        if fv1 is None:
            fv1 = self.fv1
        if fv1 and issubclass(type(fv1[0]), surfaces.Surface):
            self.fv0.getEdges()
            self.closed = self.fv0.bdry.max() == 0
            if self.closed:
                v0 = self.fv0.surfVolume()
                if self.param.errorType == 'L2Norm' and v0 < 0:
                    self.fv0.flipFaces()
                    v0 = -v0
                z = self.fvInit.surfVolume()
                if z < 0:
                    self.fv0ori = -1
                else:
                    self.fv0ori = 1
            else:
                self.fv0ori = 1
                v0 = 0
            self.fv1ori = np.zeros(len(self.fv1), dtype=int)
            for k,f in enumerate(fv1):
                f.getEdges()
                closed = self.closed and f.bdry.max() == 0
                if closed:
                    v1 = f.surfVolume()
                    if v0*v1 < 0:
                        f.flipFaces()
                    z= f.surfVolume()
                    if z < 0:
                        self.fv1ori[k] = -1
                    else:
                        self.fv1ori[k] = 1
                else:
                    self.fv1ori[k] = 1
        else:
            self.fv0ori = 1
            self.fv1ori = None
        #self.fv0Fine = surfaces.Surface(surf=self.fv0)
        logging.info('orientation: {0:d}'.format(self.fv0ori))


    def dataTerm(self, _fvDef, fv1 = None, fvInit = None, _lmk_def = None, lmk1 = None):
        obj = 0
        if fv1 is None:
            fv1 = self.fv1
        if self.match_landmarks:
            if _lmk_def is None:
                _lmk_def = self.def_lmk
            if lmk1 is None:
                lmk1 = self.targ_lmk
            for k, s in enumerate(_fvDef):
                obj += super().dataTerm(s, fv1=fv1[k], _lmk_def=_lmk_def[k], lmk1=lmk1[k].points)
        else:
            for k,s in enumerate(_fvDef):
                obj += super().dataTerm(s, fv1 = fv1[k])
                if self.volumeWeight:
                    obj += self.volumeWeight * (s.surfVolume() - fv1[k].surfVolume()) ** 2
        return obj

    def objectiveFun(self):
        if self.obj is None:
            (self.obj, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.obj0 = 0
            for k in range(self.nTarg):
                if self.param.errorType == 'L2Norm':
                    self.obj0 += sd.L2Norm0(self.fv1[k]) / (self.param.sigmaError ** 2)
                else:
                    self.obj0 += self.fun_obj0(self.fv1[k]) / (self.param.sigmaError**2)
                if self.match_landmarks:
                    self.obj0 += self.wlmk * self.lmk_obj0(self.targ_lmk[k]) / (self.param.sigmaError ** 2)
                #foo = surfaces.Surface(surf=self.fvDef[k])
                self.fvDef[k].updateVertices(np.squeeze(self.xt[self.jumpIndex[k], :self.nvert, :]))
                if self.match_landmarks:
                    self.def_lmk.points = self.xt[self.jumpIndex[k], self.nvert:, :]
                #foo.computeCentersAreas()
            self.obj += self.obj0 + self.dataTerm(self.fvDef, _lmk_def=self.def_lmk)
            #print self.obj0,  self.dataTerm(self.fvDef)
        return self.obj

    # def getVariable(self):
    #     return (self.at, self.Afft)
    #
    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        atTry = self.at - eps * dir['diff']
        if self.affineDim > 0:
            AfftTry = self.Afft - eps * dir['aff']
        else:
            AfftTry = self.Afft
        foo = self.objectiveFunDef(atTry, AfftTry, withTrajectory=True)
        objTry += foo[0]
        xtTry = foo[1]

        ff = []
        for k in range(self.nTarg):
            ff.append(surfaces.Surface(surf=self.fvDef[k]))
            ff[k].updateVertices(np.squeeze(foo[1][self.jumpIndex[k], :self.nvert, :]))
        if self.match_landmarks:
            pp = []
            for k in range(self.nTarg):
                pp = pointSets.PointSet(data=self.def_lmk[k])
                pp.updatePoints(np.squeeze(foo[1][self.jumpIndex[k], self.nvert:, :]))
        else:
            pp = None
        # objTry += self.dataTerm(ff)
        objTry += self.dataTerm(ff, _lmk_def=pp)
        if np.isnan(objTry):
            print('Warning: nan in updateTry')
            return 1e500

        if (objRef is None) or (objTry < objRef):
            self.atTry = atTry
            self.AfftTry = AfftTry
            self.objTry = objTry
            self.xtTry = xtTry
            #print 'objTry=',objTry, dir.diff.sum()

        return objTry

    def testEndpointGradient(self):
        c0 = self.dataTerm(self.fvDef)
        eps0 = 1e-8
        f2 = [[],[]]
        p2 = [[],[]]
        grd = self.endPointGradient()
        grdscp = 0
        c = [0, 0]
        for k,f in enumerate(self.fvDef):
            dff = np.random.normal(size=f.vertices.shape)
            if self.match_landmarks:
                dpp = np.random.normal(size=self.def_lmk[k].points.shape)
                dall = np.concatenate((dff, dpp), axis=0)
            else:
                dall = dff
                dpp = None
            grdscp += (grd[k]*dall).sum()
            eps = [-eps0, eps0]
            for j in range(2):
                if self.match_landmarks:
                    pp = pointSets.PointSet(data=self.def_lmk[k])
                    pp.updatePoints(pp.points + eps[j] * dpp)
                else:
                    pp = None
                ff = surfaces.Surface(surf=f)
                ff.updateVertices(ff.vertices+eps[j]*dff)
                f2[j].append(ff)
                p2[j].append(pp)
        #print(f2)
        c0 = self.dataTerm(f2[0], _lmk_def=p2[0])
        c1 = self.dataTerm(f2[1], _lmk_def=p2[1])
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c1-c0)/(2*eps0), -grdscp) )


    # def endPointGradient(self, endPoint=None):
    #     if endPoint is None:
    #         endPoint = self.fvDef
    #     if self.param.errorType == 'L2Norm':
    #         px = surfaces.L2NormGradient(endPoint, self.fv1.vfld)
    #     else:
    #         if self.fv1:
    #             px = self.fun_objGrad(endPoint, self.fv1)
    #         else:
    #             px = self.fun_objGrad(endPoint)
    #     return px / self.param.sigmaError**2

    def endPointGradient(self, endPoint=None):
        if endPoint is None:
            endPoint = self.fvDef
            endPoint_lmk = self.def_lmk
        elif self.match_landmarks:
            endPoint_lmk = endPoint[1]
            endPoint = endPoint[0]
        else:
            endPoint_lmk = None
        px = []
        for k in range(self.nTarg):
            if self.param.errorType == 'L2Norm':
                targGradient = -sd.L2NormGradient(endPoint[k], self.fv1[k].vfld)/(self.param.sigmaError**2)
            else:
                if self.fv1:
                    targGradient = -self.fun_objGrad(endPoint[k], self.fv1[k])/(self.param.sigmaError**2)
                else:
                    targGradient = -self.fun_objGrad(endPoint[k])/(self.param.sigmaError**2)
            if self.volumeWeight:
                targGradient -= (2./3) * self.volumeWeight*(endPoint[k].surfVolume() - self.fv1[k].surfVolume()) * self.fvDef[k].computeAreaWeightedVertexNormals()
            if self.match_landmarks:
                pxl = self.wlmk * self.lmk_objGrad(endPoint_lmk[k].points, self.targ_lmk[k].points)
                targGradient = np.concatenate((targGradient, pxl), axis=0)
            px.append(targGradient)
        #print "px", (px[0]**2).sum()
        return px

    def hamiltonianCovector(self, px1, KparDiff, regweight, affine=None, fv0 = None, control=None):
        if fv0 is None:
            fv0 = self.fvInit
        if control is None:
            at = self.at
            current_at = True
            if self.varCounter == self.trajCounter:
                computeTraj = False
            else:
                computeTraj = True
        else:
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
        nTarg = len(px1)
        timeStep = 1.0 / T
        if computeTraj:
            xt = evol.landmarkDirectEvolutionEuler(x0, at, KparDiff, affine=affine)
            if current_at:
                self.trajCounter = self.varCounter
                self.xt = xt
        else:
            xt = self.xt
        pxt = np.zeros([T, N, dim])
        pxt[T-1, :, :] = px1[nTarg - 1]
        jk = nTarg - 2
        if not(affine is None):
            A0 = affine[0]
            A = np.zeros([T,dim,dim])
            for k in range(A0.shape[0]):
                A[k,...] = getExponential(timeStep*A0[k])
        else:
            A = np.zeros([T,dim,dim])
            for k in range(T):
                A[k,...] = np.eye(dim)

        foo = surfaces.Surface(surf=fv0)
        for t in range(1,T):
            px = np.squeeze(pxt[T - t, :, :])
            z = np.squeeze(xt[T - t, :, :])
            a = np.squeeze(at[T - t, :, :])
            foo.updateVertices(z)
            v = KparDiff.applyK(z,a)
            if self.internalCost:
                grd = self.internalCostGrad(foo, v)
                Lv =  grd[0]
                DLv = self.internalWeight*grd[1]
                zpx = self.param.KparDiff.applyDiffKT(z, px, a, regweight=self.regweight, lddmm=True,
                                                      extra_term = -self.internalWeight*Lv) - DLv
            else:
                zpx = self.param.KparDiff.applyDiffKT(z, px, a, regweight=self.regweight, lddmm=True)
            if not (affine is None):
                pxt[T-t-1, :, :] = np.dot(px, A[T-t]) + timeStep * zpx
            else:
                pxt[T-t-1, :, :] = px + timeStep * zpx
            if (t < T - 1) and self.isjump[T - t]:
                pxt[T - t - 1, :, :] += px1[jk]
                jk -= 1
            # print 'zpx', np.fabs(zpx).sum(), np.fabs(px).sum(), z.sum()
            # print 'pxt', np.fabs((pxt)[M-t-2]).sum()

        return pxt, xt

    def getGradient(self, coeff=1.0, update = None):
        if update is None:
            control = None
            Afft = self.Afft
            if self.match_landmarks:
                endPoint = (self.fvDef, self.def_lmk)
            else:
                endPoint = self.fvDef
            A = self.affB.getTransforms(self.Afft)
            # at = self.at
            # endPoint = self.fvDef
            # A = self.affB.getTransforms(self.Afft)
        else:
            control = self.at - update[1] * update[0]['diff']
            if len(update[0].aff) > 0:
                A = self.affB.getTransforms(self.Afft - update[1]*update[0]['aff'])
            else:
                A = None
            xt = evol.landmarkDirectEvolutionEuler(self.x0, control, self.param.KparDiff, affine=A)
            endPoint = []
            fvDef = surfaces.Surface(surf=self.fv0)
            for k in range(self.nTarg):
                if self.match_landmarks:
                    endPoint0 = surfaces.Surface(surf=self.fv0)
                    endPoint0.updateVertices(xt[self.jumpIndex[k], :self.nvert, :])
                    endPoint1 = pointSets.PointSet(data=xt[self.jumpIndex[k], self.nvert:, :])
                    endPoint.append((endPoint0, endPoint1))
                else:
                    fvDef.updateVertices(np.squeeze(xt[self.jumpIndex[k], :, :]))
                    endPoint.append(fvDef)

        px1 = self.endPointGradient(endPoint=endPoint)
            
        dim2 = self.dim**2

        foo = self.hamiltonianGradient(px1, affine=A, control=control)
        grd = Direction()
        # if self.euclideanGradient:
        #     grd['diff'] = np.zeros(foo[0].shape)
        #     for t in range(self.Tsize):
        #         z = self.xt[t, :, :]
        #         grd['diff'][t,:,:] = self.param.KparDiff.applyK(z, foo[0][t, :,:])/(coeff*self.Tsize)
        # else:
        grd['diff'] = foo[0]/(coeff*self.Tsize)
        grd['aff'] = np.zeros(self.Afft.shape)
        # if self.affineBurnIn:
        #     grd.diff *= 0
        if self.affineDim > 0 and self.iter < self.affBurnIn:
            dA = foo[1]
            db = foo[2]
            grd['aff'] = 2*np.multiply(self.affineWeight.reshape([1, self.affineDim]), self.Afft)
            #grd.aff = 2 * self.Afft
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               #grd.aff[t] -=  np.divide(dAff.reshape(grd.aff[t].shape), self.affineWeight.reshape(grd.aff[t].shape))
               grd['aff'][t] -=  dAff.reshape(grd['aff'][t].shape)
            grd['aff'] /= (self.coeffAff*coeff*self.Tsize)
        return grd


    def saveCorrectedTarget(self, X0, X1):
        for k, fv in enumerate(self.fv1):
            U = la.inv(X0[self.jumpIndex[k]])
            if self.param.errorType == "PointSet":
                f = pointSets.PointSet(data=fv)
                yyt = np.dot(f.points - X1[self.jumpIndex[k], ...], U.T)
                f.points = yyt
            else:
                f = surfaces.Surface(surf=fv)
                yyt = np.dot(f.vertices - X1[self.jumpIndex[k], ...], U.T)
                f.updateVertices(yyt)
                if self.match_landmarks:
                    p = pointSets.PointSet(data=self.targ_lmk[k])
                    yyt = np.dot(p.points - X1[-1, ...], U)
                    p.updatePoints(yyt)
                    p.saveVTK(self.outputDir + f'/Target{k:02d}LandmarkCorrected.vtk')
            f.saveVTK(self.outputDir + f'/Target{k:02d}Corrected.vtk')

    def saveHdf5(self, fileName):
        pass

    def updateEndPoint(self, xt):
        for k in range(self.nTarg):
            self.fvDef[k].updateVertices(np.squeeze(xt[self.jumpIndex[k], :self.nvert, :]))
            if self.match_landmarks:
                self.def_lmk[k].updatePoints(xt[self.jumpIndex[k], self.nvert:, :])

    # def endOfIteration(self):
    #     self.iter += 1
    #     if self.iter >= self.affBurnIn:
    #         self.coeffAff = self.coeffAff2
    #         # self.affineBurnIn = False
    #     if (self.iter % self.saveRate == 0):
    #         logging.info('Saving surfaces...')
    #         (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
    #         for k in range(self.nTarg):
    #             self.fvDef[k].updateVertices(np.squeeze(self.xt[self.jumpIndex[k], :, :]))
    #         dim2 = self.dim**2
    #         A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
    #         if self.affineDim > 0:
    #             for t in range(self.Tsize):
    #                 AB = np.dot(self.affineBasis, self.Afft[t])
    #                 A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
    #                 A[1][t] = AB[dim2:dim2+self.dim]
    #
    #         (xt, ft, Jt)  = evol.landmarkDirectEvolutionEuler(self.x0, self.at, self.param.KparDiff, affine=A,
    #                                                        withPointSet = self.fv0Fine.vertices, withJacobian=True)
    #
    #         if self.affine == 'euclidean' or self.affine == 'translation':
    #             self.saveCorrectedEvolution(self.fvInit, xt, self.at, self.Afft, fileName=self.saveFile,
    #                                         Jacobian=Jt)
    #         if self.saveCorrected:
    #             f = surfaces.Surface(surf=self.fv0)
    #             X = self.affB.integrateFlow(self.Afft)
    #             displ = np.zeros(self.x0.shape[0])
    #             dt = 1.0 /self.Tsize
    #             for t in range(self.Tsize+1):
    #                 U = la.inv(X[0][t])
    #                 yyt = np.dot(xt[t,...] - X[1][t, ...], U.T)
    #                 zt = np.dot(ft[t,...] - X[1][t, ...], U.T)
    #                 if t < self.Tsize:
    #                     at = np.dot(self.at[t,...], U.T)
    #                     vt = self.param.KparDiff.applyK(yyt, at, firstVar=zt)
    #                 f.updateVertices(zt)
    #                 vf = surfaces.vtkFields()
    #                 vf.scalars.append('Jacobian')
    #                 vf.scalars.append(np.exp(Jt[t, :])-1)
    #                 vf.scalars.append('displacement')
    #                 vf.scalars.append(displ)
    #                 vf.vectors.append('velocity')
    #                 vf.vectors.append(vt)
    #                 nu = self.fv0ori*f.computeVertexNormals()
    #                 if t >= self.jumpIndex[0]:
    #                     displ += dt * (vt*nu).sum(axis=1)
    #                 f.saveVTK2(self.outputDir +'/'+self.saveFile+'Corrected'+str(t)+'.vtk', vf)
    #
    #             for k,fv in enumerate(self.fv1):
    #                 f = surfaces.Surface(surf=fv)
    #                 U = la.inv(X[0][self.jumpIndex[k]])
    #                 yyt = np.dot(f.vertices - X[1][self.jumpIndex[k], ...], U.T)
    #                 f.updateVertices(yyt)
    #                 f.saveVTK(self.outputDir +'/Target'+str(k)+'Corrected.vtk')
    #
    #         fvDef = surfaces.Surface(surf=self.fv0)
    #         AV0 = fvDef.computeVertexArea()
    #         nu = self.fv0ori*self.fv0.computeVertexNormals()
    #         #v = self.v[0,...]
    #         displ = np.zeros(self.npt)
    #         dt = 1.0 /self.Tsize
    #         v = self.param.KparDiff.applyK(ft[0,...], self.at[0,...], firstVar=self.xt[0,...])
    #         for kk in range(self.Tsize+1):
    #             fvDef.updateVertices(np.squeeze(ft[kk, :, :]))
    #             AV = fvDef.computeVertexArea()
    #             AV = (AV[0]/AV0[0])-1
    #             vf = surfaces.vtkFields()
    #             vf.scalars.append('Jacobian')
    #             vf.scalars.append(np.exp(Jt[kk, :])-1)
    #             vf.scalars.append('Jacobian_T')
    #             vf.scalars.append(AV)
    #             vf.scalars.append('Jacobian_N')
    #             vf.scalars.append(np.exp(Jt[kk, :])/(AV+1)-1)
    #             vf.scalars.append('displacement')
    #             vf.scalars.append(displ)
    #             if kk < self.Tsize:
    #                 nu = self.fv0ori*fvDef.computeVertexNormals()
    #                 v = self.param.KparDiff.applyK(ft[kk,...], self.at[kk,...], firstVar=self.xt[kk,...])
    #                 #v = self.v[kk,...]
    #                 kkm = kk
    #             else:
    #                 kkm = kk-1
    #             if kk >= self.jumpIndex[0]:
    #                 displ += dt * (v*nu).sum(axis=1)
    #             vf.vectors.append('velocity')
    #             vf.vectors.append(self.v[kkm,:])
    #             fvDef.saveVTK2(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', vf)
    #     else:
    #         (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
    #

    # def optimizeMatching(self):
    #     #print 'dataterm', self.dataTerm(self.fvDef)
    #     #print 'obj fun', self.objectiveFun(), self.obj0
    #     grd = self.getGradient(self.gradCoeff)
    #     [grd2] = self.dotProduct(grd, [grd])
    #
    #     self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
    #     logging.info('Gradient lower bound: %f'  %(self.gradEps))
    #     #print 'x0:', self.x0
    #     #print 'y0:', self.y0
    #     self.cgBurnIn = self.affBurnIn
    #
    #     cg.cg(self, verb = self.verb, maxIter = self.maxIter,TestGradient=self.testGradient, epsInit=0.1)
    #     #return self.at, self.xt

