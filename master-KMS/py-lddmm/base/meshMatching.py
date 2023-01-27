import os
from copy import deepcopy
import numpy as np
import h5py
import scipy.linalg as la
import logging
from . import matchingParam
from . import conjugateGradient as cg, kernelFunctions as kfun, pointEvolution as evol, loggingUtils, bfgs
from .pointSets import PointSet
from . import meshes, mesh_distances as msd
from .affineBasis import AffineBasis, getExponential
from . import pointSetMatching
import matplotlib

## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
class MeshMatchingParam(matchingParam.MatchingParam):
    def __init__(self, timeStep = .1, algorithm='cg', Wolfe=True, KparDiff = None, KparDist = None,
                 KparIm = None, sigmaError = 1.0):
        super().__init__(timeStep=timeStep, algorithm = algorithm, Wolfe=Wolfe,
                         KparDiff = KparDiff, KparDist = KparDist, sigmaError=sigmaError)
        self.typeKIm = 'gauss'
        self.sigmaKIm = 1.
        self.orderKIm = 3
        if type(KparIm) in (list,tuple):
            self.typeKIm = KparIm[0]
            self.sigmaKIm = KparIm[1]
            if len(KparIm) > 2:
                self.orderKIm = KparIm[2]
            self.KparIm = kfun.Kernel(name = self.typeKIm, sigma = self.sigmaKIm,
                                      order=self.orderKIm)
        else:
            self.KparIm = KparIm
        self.sigmaError = sigmaError


class Direction:
    def __init__(self):
        self.diff = []
        self.aff = []


## Main class for image varifold matching
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
class MeshMatching(pointSetMatching.PointSetMatching):
    def __init__(self, Template, Target, param=None, maxIter=1000,
                 regWeight = 1.0, affineWeight = 1.0, verb=True,
                 rotWeight = None, scaleWeight = None, transWeight = None,
                 testGradient=True, saveFile = 'evolution',
                 saveTrajectories = False, affine = 'none', outputDir = '.',pplot=True):

        if param is None:
            self.param = MeshMatchingParam()
        else:
            self.param = param

        pointSetMatching.PointSetMatching.__init__(self, Template=Template, Target=Target, param=param, maxIter=maxIter,
                 regWeight=regWeight, affineWeight=affineWeight, verb=verb,
                 rotWeight=rotWeight, scaleWeight=scaleWeight, transWeight=transWeight,
                 testGradient=testGradient, saveFile=saveFile,
                 saveTrajectories=saveTrajectories, affine=affine, outputDir=outputDir, pplot=pplot)

        self.Kim_dtype = self.param.KparIm.pk_dtype


    def initialize_variables(self):
        self.x0 = np.copy(self.fv0.vertices)
        self.fvDef = deepcopy(self.fv0)
        self.npt = self.x0.shape[0]

        self.Tsize = int(round(1.0/self.param.timeStep))
        self.at = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.atTry = np.zeros([self.Tsize, self.x0.shape[0], self.x0.shape[1]])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.AfftTry = np.zeros([self.Tsize, self.affineDim])
        self.xt = np.tile(self.x0, [self.Tsize+1, 1, 1])
        self.v = np.zeros([self.Tsize+1, self.npt, self.dim])


    def set_template_and_target(self, Template, Target):
        self.fv0 = meshes.Mesh(mesh=Template)
        self.fv1 = meshes.Mesh(mesh=Target)

        self.fv0.save(self.outputDir + '/Template.vtk')
        self.fv1.save(self.outputDir + '/Target.vtk')
        self.dim = self.fv0.vertices.shape[1]

    def set_fun(self, errorType):
        self.param.errorType = errorType
        self.fun_obj0 = msd.varifoldNorm0
        self.fun_obj = msd.varifoldNormDef
        self.fun_objGrad = msd.varifoldNormGradient


    def dataTerm(self, _fvDef, _fvInit = None):
        # logging.info('dataTerm ' + self.param.KparIm.name)
        # if self.param.errorType == 'classification':
        #     obj = pointSets.LogisticScoreL2(_fvDef, self.fv1, self.u, w=self.wTr, intercept=self.intercept, l1Cost=self.l1Cost) \
        #           / (self.param.sigmaError**2)
        #     #obj = pointSets.LogisticScore(_fvDef, self.fv1, self.u) / (self.param.sigmaError**2)
        obj = self.fun_obj(_fvDef, self.fv1, self.param.KparDist, self.param.KparIm) / (self.param.sigmaError ** 2)
        return obj


    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = self.fun_obj0(self.fv1, self.param.KparDist, self.param.KparIm) / (self.param.sigmaError ** 2)
            (self.objDef, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.objData = self.dataTerm(self.fvDef)
            self.obj = self.obj0 + self.objData + self.objDef
        return self.obj


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
        parameters.create_dataset('Image Varifold Kernel type', data = self.param.KparIm.name)
        parameters.create_dataset('Image Varifold width', data = self.param.KparIm.sigma)
        parameters.create_dataset('Image Varifold order', data = self.param.KparIm.order)
        template = LDDMMResult.create_group('template')
        template.create_dataset('vertices', data=self.fv0.vertices)
        template.create_dataset('faces', data=self.fv0.faces)
        template.create_dataset('image', data=self.fv0.image)
        target = LDDMMResult.create_group('target')
        target.create_dataset('vertices', data=self.fv1.vertices)
        target.create_dataset('faces', data=self.fv1.faces)
        target.create_dataset('image', data=self.fv1.image)
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

        AV0 = self.fv0.computeVertexVolume()
        AV = self.fvDef.computeVertexVolume()/AV0
        descriptors.create_dataset('Jacobian', data=Jt[-1,:])
        descriptors.create_dataset('Surface Jacobian', data=AV)
        descriptors.create_dataset('Displacement', data=xt[-1,...]-xt[0,...])

        fout.close()


    def makeTryInstance(self, pts):
        ff = meshes.Mesh(mesh=self.fvDef)
        ff.updateVertices(pts)
        return ff



    def endPointGradient(self, endPoint= None):
        if endPoint is None:
            endPoint = self.fvDef
        px = self.fun_objGrad(endPoint, self.fv1, self.param.KparDist, self.param.KparIm)
        return px / self.param.sigmaError**2
    
    def testEndpointGradient(self):
        c0 = self.dataTerm(self.fvDef)
        ff = deepcopy(self.fvDef)
        dff = np.random.normal(size=ff.vertices.shape)
        eps = 1e-6
        ff.updateVertices( ff.vertices + eps*dff)
        c1 = self.dataTerm(ff)
        grd = self.endPointGradient()
        logging.info("test endpoint gradient: {0:.5f} {1:.5f}".format((c1-c0)/eps, (grd*dff).sum()) )

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
    #     for t in range(M):
    #         px = np.squeeze(pxt[M-t, :, :])
    #         z = np.squeeze(xt[M-t-1, :, :])
    #         a = np.squeeze(at[M-t-1, :, :])
    #         a1 = np.concatenate((px[np.newaxis,...], a[np.newaxis,...], -2*regWeight*a[np.newaxis,...]))
    #         a2 = np.concatenate((a[np.newaxis,...], px[np.newaxis,...], a[np.newaxis,...]))
    #         zpx = KparDiff.applyDiffKT(z, px, a, lddmm=True)
    #
    #
    #         if not (affine is None):
    #             pxt[M-t-1, :, :] = np.dot(px, A[M-t-1]) + timeStep * zpx
    #         else:
    #             pxt[M-t-1, :, :] = px + timeStep * zpx
    #     return pxt, xt

    def setUpdate(self, update):
        at = self.at - update[1] * update[0].diff

        Afft = self.Afft - update[1] * update[0].aff
        if len(update[0].aff) > 0:
            A = self.affB.getTransforms(Afft)
        else:
            A = None
        xt = evol.landmarkDirectEvolutionEuler(self.x0, at, self.param.KparDiff, affine=A)
        endPoint = meshes.Mesh(mesh=self.fv0)
        endPoint.updateVertices(xt[-1, :, :])

        return at, Afft, xt, endPoint

    # def getGradient(self, coeff=1.0, update=None):
    #     if update is None:
    #         at = None
    #         Afft = self.Afft
    #         endPoint = self.fvDef
    #         A = self.affB.getTransforms(self.Afft)
    #         xt = self.xt
    #     else:
    #         at = self.at - update[1] * update[0].diff
    #         Afft = self.Afft - update[1]*update[0].aff
    #         if len(update[0].aff) > 0:
    #             A = self.affB.getTransforms(Afft)
    #         else:
    #             A = None
    #         xt = evol.landmarkDirectEvolutionEuler(self.x0, at, self.param.KparDiff, affine=A)
    #         endPoint = meshes.Mesh(mesh=self.fv0)
    #         endPoint.updateVertices(xt[-1, :, :])
    #
    #     dim2 = self.dim**2
    #     px1 = -self.endPointGradient(endPoint=endPoint)
    #     foo = self.hamiltonianGradient(px1, at=at, affine=A)
    #     grd = Direction()
    #     if self.euclideanGradient:
    #         grd.diff = np.zeros(foo[0].shape)
    #         for t in range(self.Tsize):
    #             z = xt[t, :, :]
    #             grd.diff[t,:,:] = self.param.KparDiff.applyK(z, foo[0][t, :,:])/(coeff*self.Tsize)
    #     else:
    #         grd.diff = foo[0]/(coeff*self.Tsize)
    #     grd.aff = np.zeros(self.Afft.shape)
    #     if self.affineDim > 0:
    #         dA = foo[1]
    #         db = foo[2]
    #         grd.aff = 2*self.affineWeight.reshape([1, self.affineDim])*Afft
    #         #grd.aff = 2 * self.Afft
    #         for t in range(self.Tsize):
    #            dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
    #            #grd.aff[t] -=  np.divide(dAff.reshape(grd.aff[t].shape), self.affineWeight.reshape(grd.aff[t].shape))
    #            grd.aff[t] -=  dAff.reshape(grd.aff[t].shape)
    #         grd.aff /= (self.coeffAff*coeff*self.Tsize)
    #         #            dAfft[:,0:self.dim**2]/=100
    #     return grd

    def startOfIteration(self):
        if self.reset:
            logging.info('Switching to 64 bits')
            self.param.KparDiff.pk_dtype = 'float64'
            self.param.KparDist.pk_dtype = 'float64'
            self.param.KparIm.pk_dtype = 'float64'



    def endOfIteration(self, endP=False):
        self.iter += 1
        if self.testGradient:
            self.testEndpointGradient()

        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2
        if (self.iter % self.saveRate == 0 or endP) :
            logging.info('Saving Points...')
            (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)

            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            dim2 = self.dim**2
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, self.Afft[t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2+self.dim]
            (xt, Jt)  = evol.landmarkDirectEvolutionEuler(self.x0, self.at, self.param.KparDiff, affine=A,
                                                              withJacobian=True)
            # if self.affine=='euclidean' or self.affine=='translation':
            #     X = self.affB.integrateFlow(self.Afft)
            #     displ = np.zeros(self.x0.shape[0])
            #     dt = 1.0 /self.Tsize
            #     for t in range(self.Tsize+1):
            #         U = la.inv(X[0][t])
            #         yyt = np.dot(self.xt[t,...] - X[1][t, ...], U.T)
            #         f = np.copy(yyt)
            #         # vf = surfaces.vtkFields() ;
            #         # vf.scalars.append('Jacobian') ;
            #         # vf.scalars.append(np.exp(Jt[t, :]))
            #         # vf.scalars.append('displacement')
            #         # vf.scalars.append(displ)
            #         # vf.vectors.append('velocity') ;
            #         # vf.vectors.append(vt)
            #         # nu = self.fv0ori*f.computeVertexNormals()
            #         pointSets.savelmk(f, self.outputDir + '/' + self.saveFile + 'Corrected' + str(t) + '.lmk')
            #     f = np.copy(self.fv1)
            #     yyt = np.dot(f - X[1][-1, ...], U.T)
            #     f = np.copy(yyt)
            #     pointSets.savePoints(self.outputDir + '/TargetCorrected.vtk', f)
            for kk in range(self.Tsize+1):
                fvDef = meshes.Mesh(mesh=self.fvDef)
                fvDef.updateVertices(xt[kk, :, :])
                fvDef.save(self.outputDir + '/' + self.saveFile + str(kk) + '.vtk')

            self.saveHdf5(fileName=self.outputDir + '/output.h5')

        (obj1, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
        self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]), checkOrientation=True)
        self.param.KparDiff.pk_dtype = self.Kdiff_dtype
        self.param.KparDist.pk_dtype = self.Kdist_dtype
        self.param.KparIm.pk_dtype = self.Kim_dtype
        logging.info(f'Objective function components: Def={self.objDef:.04f} Data={self.objData+ self.obj0:0.4f}')

    def endOfProcedure(self):
        self.endOfIteration(endP=True)

    # def optimizeMatching(self):
    #     #print 'dataterm', self.dataTerm(self.fvDef)
    #     #print 'obj fun', self.objectiveFun(), self.obj0
    #     self.coeffAff = self.coeffAff2
    #     grd = self.getGradient(self.gradCoeff)
    #     [grd2] = self.dotProduct(grd, [grd])
    #
    #     self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
    #     logging.info('Gradient lower bound: %f' %(self.gradEps))
    #     self.coeffAff = self.coeffAff1
    #     #self.restartRate = self.relearnRate
    #     if self.param.algorithm == 'cg':
    #         cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=0.1,)
    #     elif self.param.algorithm == 'bfgs':
    #         bfgs.bfgs(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=1.,
    #                   Wolfe=self.param.wolfe, memory=50)
    #     #bfgs.bfgs(self, verb = self.verb, maxIter = self.maxIter,TestGradient=self.testGradient, epsInit=0.1)
    #     #return self.at, self.xt
    #
    #
