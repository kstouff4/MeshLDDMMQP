import os
import numpy.linalg as LA
import scipy.ndimage as Img
import scipy.stats.mstats as stats
from . import gaussianDiffeons as gd
from . import conjugateGradient as cg, diffeo, kernelFunctions as kfun, pointEvolution as evol, bfgs
from .affineBasis import *
from PIL import Image

## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normalization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'


def ImageMatchingDist(gr, J, im0, im1):
    #print gr.shape
    #print gr[10:20, 10:20, 0]
    #imdef0 = Img.interpolation.map_coordinates(im1.data, gr.transpose(range(-1, gr.ndim-1)), order=1, mode='nearest')
    imdef = diffeo.multilinInterp(im1.data, gr.transpose(range(-1, gr.ndim - 1)))
    #print 'diff:', (imdef-imdef0).max()
    # print im1.data.min(), im1.data.max()
    # print imdef.min(), imdef.max()
    res = (((im0.data - imdef)**2)*np.exp(J)).sum()
    return res

def ImageMatchingGradient(gr, J, im0, im1, gradient=None):
    imdef = diffeo.multilinInterp(im1.data, gr.transpose(range(-1, gr.ndim - 1)))
    #imdef = Img.interpolation.map_coordinates(im1.data, gr.transpose(range(-1, gr.ndim-1)), order=1, mode='nearest')
    # if not (gradient==None):
    #     gradIm1 = gradient
    # else:
    #     gradIm1 = diffeo.gradient(im1.data, im1.resol)
    # gradDef = np.zeros(gradIm1.shape)
    # for k in range(gradIm1.shape[0]):
    #     gradDef[k,...] = diffeo.multilinInterp(gradIm1[k, ...], gr.transpose(range(-1, gr.ndim-1)))
        #gradDef[k,...] = Img.interpolation.map_coordinates(gradIm1[k, ...], gr.transpose(range(-1, gr.ndim-1)), order=1, mode='nearest')
    gradDef = diffeo.multilinInterpGradient(im1.data, gr.transpose(range(-1, gr.ndim - 1)))
    
    expJ = np.exp(J)
    pgr = ((-2*(im0.data-imdef)*expJ)*gradDef).transpose(np.append(range(1, gr.ndim), 0))
    pJ =  ((im0.data - imdef)**2)*expJ
    return pgr, pJ



class ImageMatchingParam:
    def __init__(self, timeStep = .1, algorithm = 'bfgs', Wolfe=False, sigmaKernel = 6.5, sigmaError=1.0, dimension=2, errorType='L2', KparDiff = None, typeKernel='gauss'):
        self.timeStep = timeStep
        self.sigmaKernel = sigmaKernel
        self.sigmaError = sigmaError
        self.typeKernel = typeKernel
        self.errorType = errorType
        self.dimension = dimension
        self.algorithm = algorithm
        self.wolfe = Wolfe
        if errorType == 'L2':
            self.fun_obj = ImageMatchingDist
            self.fun_objGrad = ImageMatchingGradient
        else:
            print('Unknown error Type: ', self.errorType)
        if KparDiff == None:
            self.KparDiff = kfun.Kernel(name = self.typeKernel, sigma = self.sigmaKernel)
        else:
            self.KparDiff = KparDiff



class Direction:
    def __init__(self):
        self.diff = []
        self.aff = []


## Main class for image matching
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
class ImageMatching:
    def __init__(self, Template=None, Target=None, Diffeons=None, EpsilonNet=None, DecimationTarget=1,
                 subsampleTemplate = 1, targetMargin=10, templateMargin=0,
                 DiffeonEpsForNet=None, DiffeonSegmentationRatio=None, zeroVar=False, fileTempl=None,
                 fileTarg=None, param=None, maxIter=1000, regWeight = 1.0, affineWeight = 1.0, verb=True,
                 rotWeight = None, scaleWeight = None, transWeight = None, testGradient=False, saveFile = 'evolution', affine = 'none', outputDir = '.'):
        if Template==None:
            if fileTempl==None:
                print('Please provide a template image')
                return
            else:
                self.im0 = diffeo.gridScalars(filename=fileTempl)
        else:
            self.im0 = diffeo.gridScalars(grid=Template)
            #print self.im0.data.shape, Template.data.shape
        if Target==None:
            if fileTarg==None:
                print('Please provide a target image')
                return
            else:
                self.im1 = diffeo.gridScalars(filename=fileTarg)
        else:
            self.im1 = diffeo.gridScalars(grid=Target)

            #zoom = np.minimum(np.array(self.im0.data.shape).astype(float)/np.array(self.im1.data.shape), np.ones(self.im0.data.ndim))
        zoom = np.array(self.im0.data.shape).astype(float)/np.array(self.im1.data.shape)
        self.im1.data = Img.interpolation.zoom(self.im1.data, zoom, order=0)

        # Include template in bigger image
        if templateMargin >0:
            I = range(-templateMargin, self.im0.data.shape[0]+templateMargin)
            for k in range(1, self.im0.data.ndim):
                I = (I, range(-templateMargin, self.im0.data.shape[k]+templateMargin))
            self.gr1 = np.array(np.meshgrid(*I, indexing='ij'))
            #print self.im1.data.shape
            self.im0.data = diffeo.multilinInterp(self.im0.data, self.gr1)

        # Include target in bigger image
        I = range(-targetMargin, self.im1.data.shape[0]+targetMargin)
        for k in range(1, self.im1.data.ndim):
            I = (I, range(-targetMargin, self.im1.data.shape[k]+targetMargin))
        self.gr1 = np.array(np.meshgrid(*I, indexing='ij'))
        #print self.im1.data.shape
        self.im1.data = diffeo.multilinInterp(self.im1.data, self.gr1)
        self.gr1 = self.gr1.transpose(np.append(range(1,self.gr1.ndim), 0)) +targetMargin
        #print self.im1.data.shape
        self.im0Fine = diffeo.gridScalars(grid=self.im0)
        self.saveRate = 5
        self.iter = 0
        self.gradEps = -1
        self.dim = self.im0.data.ndim
        self.setOutputDir(outputDir)
        self.maxIter = maxIter
        self.verb = verb
        self.testGradient = testGradient
        self.regweight = regWeight
        self.affine = affine
        affB = AffineBasis(self.dim, affine)
        self.affineDim = affB.affineDim
        self.affineBasis = affB.basis
        self.affineWeight = affineWeight * np.ones([self.affineDim, 1])
        if (len(affB.rotComp) > 0) & (rotWeight != None):
            self.affineWeight[affB.rotComp] = rotWeight
        if (len(affB.simComp) > 0) & (scaleWeight != None):
            self.affineWeight[affB.simComp] = scaleWeight
        if (len(affB.transComp) > 0) & (transWeight != None):
            self.affineWeight[affB.transComp] = transWeight

        if param is None:
            self.param = ImageMatchingParam()
        else:
            self.param = param
        self.dim = self.param.dimension
        #self.x0 = self.fv0.vertices
        if Diffeons is None:
            gradIm0 = np.sqrt((diffeo.gradient(self.im0.data, self.im0.resol) ** 2).sum(axis=0))
            m0 = stats.mquantiles(gradIm0, 0.75)/10. + 1e-5
            if DecimationTarget==None:
                DecimationTarget = 1
            gradIm0 = Img.filters.maximum_filter(gradIm0, DecimationTarget)
            I = range(templateMargin, self.im0.data.shape[0]-templateMargin, DecimationTarget)
            for k in range(1, self.im0.data.ndim):
                I = (I, range(templateMargin, self.im0.data.shape[k]-templateMargin, DecimationTarget))
            u = np.meshgrid(*I, indexing='ij')
            self.c0 = np.zeros([u[0].size, self.dim])
            for k in range(self.dim):
                self.c0[:,k] = u[k].flatten()
            # if self.dim == 1:
            #     self.c0 = range(0, self.im0.data.shape[0], DecimationTarget)
            # elif self.dim == 2:
            #     u = np.mgrid[0:self.im0.data.shape[0]:DecimationTarget, 0:self.im0.data.shape[1]:DecimationTarget]
            #     self.c0 = np.zeros([u[0].size, self.dim])
            #     self.c0[:,0] = u[0].flatten()
            #     self.c0[:,1] = u[1].flatten()
            # elif self.dim == 3:
            #     u = np.mgrid[0:self.im0.data.shape[0]:DecimationTarget, 0:self.im0.data.shape[1]:DecimationTarget, 0:self.im0.data.shape[2]:DecimationTarget]
            #     self.c0 = np.zeros([u[0].size, self.dim])
            #     self.c0[:,0] = u[0].flatten()
            #     self.c0[:,1] = u[1].flatten()
            #     self.c0[:,2] = u[2].flatten()
            gradIm0 = diffeo.multilinInterp(gradIm0, self.c0.T)
            #print gradIm0
            jj = 0
            for kk in range(self.c0.shape[0]):
                if gradIm0[kk] > m0:
                    self.c0[jj, :] = self.c0[kk, :]
                    jj += 1
            self.c0 = self.c0[0:jj, :]
            print('keeping ', jj, ' diffeons')
                #print self.im0.resol
            self.c0 = targetMargin - templateMargin + self.im0.origin + self.c0 * self.im0.resol
            self.S0 = np.tile( (DecimationTarget*np.diag(self.im0.resol)/2)**2, [self.c0.shape[0], 1, 1])
        else:
            (self.c0, self.S0, self.idx) = Diffeons

        if zeroVar:
            self.S0 = np.zeros(self.S0.shape)

            #print self.c0
            #print self.S0
        if subsampleTemplate == None:
            subsampleTemplate = 1
        self.im0.resol *= subsampleTemplate
        self.im0.data = Img.filters.median_filter(self.im0.data, size=subsampleTemplate)
        I = range(0, self.im0.data.shape[0], subsampleTemplate)
        II = range(0, self.im0.data.shape[0])
        for k in range(1, self.im0.data.ndim):
            I = (I, range(0, self.im0.data.shape[k], subsampleTemplate))
            II = (II, range(0, self.im0.data.shape[k]))
        self.gr0= np.array(np.meshgrid(*I, indexing='ij'))
        self.gr0Fine= np.array(np.meshgrid(*II, indexing='ij'))
        self.im0.data = diffeo.multilinInterp(self.im0.data, self.gr0)
        self.gr0 = self.gr0.transpose(np.append(range(1,self.gr0.ndim), 0))
        self.gr0Fine = self.gr0Fine.transpose(np.append(range(1,self.gr0Fine.ndim), 0))
        #print self.gr0.shape
        # if self.dim == 1:
        #     self.im0.data = self.im0.data[0:self.im0.data.shape[0]:subsampleTemplate]
        #     self.gr0 = range(self.im0.data.shape[0])
        # elif self.dim == 2:
        #     self.im0.data = self.im0.data[0:self.im0.data.shape[0]:subsampleTemplate, 0:self.im0.data.shape[1]:subsampleTemplate]
        #     self.gr0 = np.mgrid[0:self.im0.data.shape[0], 0:self.im0.data.shape[1]].transpose((1, 2,0))
        # elif self.dim == 3:
        #     self.im0.data = self.im0.data[0:self.im0.data.shape[0]:subsampleTemplate, 0:self.im0.data.shape[1]:subsampleTemplate, 0:self.im0.data.shape[2]:subsampleTemplate]
        #     self.gr0 = np.mgrid[0:self.im0.data.shape[0], 0:self.im0.data.shape[1], 0:self.im0.data.shape[2]].transpose((1,2, 3, 0))
        self.gr0 = targetMargin-templateMargin+self.im1.origin + self.gr0 * self.im1.resol 
        self.gr0Fine = targetMargin-templateMargin+self.im1.origin + self.gr0Fine * self.im1.resol 
        self.J0 = np.log(self.im0.resol.prod()) * np.ones(self.im0.data.shape) 
        self.ndf = self.c0.shape[0]
        self.Tsize = int(round(1.0/self.param.timeStep))
        self.at = np.zeros([self.Tsize, self.c0.shape[0], self.dim])
        self.atTry = np.zeros([self.Tsize, self.c0.shape[0], self.dim])
        self.Afft = np.zeros([self.Tsize, self.affineDim])
        self.AfftTry = np.zeros([self.Tsize, self.affineDim])
        self.imt = np.tile(self.im0, np.insert(np.ones(self.dim, dtype=int), 0, self.Tsize+1))
        self.Jt = np.tile(self.J0, np.insert(np.ones(self.dim, dtype=int), 0, self.Tsize+1))
        self.grt = np.tile(self.gr0, np.insert(np.ones(self.dim+1, dtype=int), 0, self.Tsize+1))
        self.ct = np.tile(self.c0, [self.Tsize+1, 1, 1])
        self.St = np.tile(self.S0, [self.Tsize+1, 1, 1, 1])
        print('error type:', self.param.errorType)
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.ndf
        self.saveFile = saveFile
        self.gradIm1 = diffeo.gradient(self.im1.data, self.im1.resol)
        self.im0.saveImg(self.outputDir+'/Template.png', normalize=True)
        self.im1.saveImg(self.outputDir+'/Target.png', normalize=True)

    def setOutputDir(self, outputDir):
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print('Cannot save in ' + outputDir)
                return
            else:
                os.makedirs(outputDir)


    def  objectiveFunDef(self, at, Afft, withTrajectory = False, initial = None):
        if initial == None:
            c0 = self.c0
            S0 = self.S0
            gr0 = self.gr0
            J0 = self.J0
        else:
            gr0 = self.gr0
            J0 = self.J0
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
        (ct, St, grt, Jt)  = evol.gaussianDiffeonsEvolutionEuler(c0, S0, at, param.sigmaKernel, affine=A, withPointSet= gr0, withJacobian=J0)

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
        if withTrajectory:
            return obj, ct, St, grt, Jt
        else:
            return obj

    def dataTerm(self, x1, J1):
        obj = self.param.fun_obj(x1, J1, self.im0, self.im1) / (self.param.sigmaError**2)
        return obj

    def objectiveFun(self):
        if self.obj == None:
            (self.obj, self.ct, self.St, self.grt, self.Jt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            #data = (self.xt[-1,:,:], self.Jt[-1,:])
            self.obj += self.dataTerm(self.grt[-1,...], self.Jt[-1,...])

        return self.obj

    def getVariable(self):
        return [self.at, self.Afft]

    def updateTry(self, dir, eps, objRef=None):
        atTry = self.at - eps * dir.diff
        if self.affineDim > 0:
            AfftTry = self.Afft - eps * dir.aff
        else:
            AfftTry = self.Afft
        objTry, ct, St, grt, Jt = self.objectiveFunDef(atTry, AfftTry, withTrajectory=True)
        #data = (grt[-1,:,:], Jt[-1,:])
        objTry += self.dataTerm(grt[-1,...], Jt[-1,...])

        if np.isnan(objTry):
            print('Warning: nan in updateTry')
            return 1e500

        if (objRef == None) | (objTry < objRef):
            self.atTry = atTry
            self.objTry = objTry
            self.AfftTry = AfftTry

            #print 'objTry=',objTry, dir.diff.sum()
        return objTry



    def endPointGradient(self):
        (pg, pJ) = self.param.fun_objGrad(self.grt[-1, ...], self.Jt[-1, ...], self.im0, self.im1, gradient=self.gradIm1)
        pc = np.zeros(self.c0.shape)
        pS = np.zeros(self.S0.shape)
        #gd.testDiffeonCurrentNormGradient(self.ct[-1, :, :], self.St[-1, :, :, :], self.bt[-1, :, :],
        #                               self.fv1, self.param.KparDist.sigma)
        pg = pg / self.param.sigmaError**2
        pJ = pJ / self.param.sigmaError**2
        return (pc, pS, pg, pJ)


    def getGradient(self, coeff=1.0):
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        dim2 = self.dim**2
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, self.Afft[t])
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]

        (pc1, pS1, pg1, pJ1) = self.endPointGradient()
        foo = evol.gaussianDiffeonsGradientPset(self.c0, self.S0, self.gr0, self.at, -pc1, -pS1, -pg1, self.param.sigmaKernel, self.regweight,
                                                affine=A, withJacobian = (self.J0, -pJ1))

        grd = Direction()
        grd.diff = foo[0]/(coeff*self.Tsize)
        grd.aff = np.zeros(self.Afft.shape)
        if self.affineDim > 0:
            dA = foo[1]
            db = foo[2]
            grd.aff = 2 * self.Afft
            for t in range(self.Tsize):
               dAff = np.dot(self.affineBasis.T, np.vstack([dA[t].reshape([dim2,1]), db[t].reshape([self.dim, 1])]))
               grd.aff[t] -=  np.divide(dAff.reshape(grd.aff[t].shape), self.affineWeight.reshape(grd.aff[t].shape))
            grd.aff /= (coeff*self.Tsize)
        return grd



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
        ddir = Direction()
        ddir.diff = np.copy(dir0.diff)
        ddir.aff = np.copy(dir0.aff)
        return ddir


    def randomDir(self):
        dirfoo = Direction()
        dirfoo.diff = np.random.randn(self.Tsize, self.ndf, self.dim)
        dirfoo.aff = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    # def getBGFSDir(Var, oldVar, grd, grdOld):
    #     s = (Var[0] - oldVar[0]).unravel()
    #     y = (grd.diff - grdOld.diff).unravel()
    #     if skipBGFS==0:
    #         rho = max(0, (s*y).sum())
    #     else:
    #         rho = 0
    #     Imsy = np.eye((s.shape[0], s.shape[0])) - rho*np.dot(s, y.T)
    #     H0 = np.dot(Imsy, np.dot(H0, Imsy)) + rho * np.dot(s, s.T)
    #     dir0.diff = (np.dot(H0, grd.diff.unravel()))

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        dim2 = self.dim**2
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
                ll = ll + 1

        return res

    def dotProduct_euclidean(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            u = np.squeeze(g1.diff[t, :, :])
            uu = (g1.aff[t]*self.affineWeight.reshape(g1.aff[t].shape))
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
        self.at = np.copy(self.atTry)
        self.Afft = np.copy(self.AfftTry)
        #print self.at

    # def startOfIteration(self):
    #     u0 = self.dataTerm(self.grt[-1, ...], self.Jt[-1, ...])
    #     (pg, pJ) = self.param.fun_objGrad(self.grt[-1,...], self.Jt[-1, ...], self.im0, self.im1, gradient=self.gradIm1)
    #     eps = 1e-8
    #     dg = np.random.normal(size=self.grt[-1,...].shape)
    #     dJ = np.random.normal(size=self.Jt[-1,...].shape)
    #     ug = self.dataTerm(self.grt[-1, ...]+eps*dg, self.Jt[-1, ...])
    #     uJ = self.dataTerm(self.grt[-1, ...], self.Jt[-1, ...]+eps*dJ)
    #     print 'Test end point gradient: grid ', (ug-u0)/eps, (pg*dg).sum(), 'jacobian', (uJ-u0)/eps, (pJ*dJ).sum() 


    def endOfIteration(self):
        #print self.obj0
        self.iter += 1
        if (self.iter % self.saveRate) == 0:
            print('saving...')
            (obj1, self.ct, self.St, self.grt, self.Jt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            dim2 = self.dim**2
            A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, self.Afft[t])
                    A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[1][t] = AB[dim2:dim2+self.dim]
            (ct, St, grt)  = evol.gaussianDiffeonsEvolutionEuler(self.c0, self.S0, self.at, self.param.sigmaKernel, affine=A, withPointSet= self.gr0Fine)
            # (ct, St, grt, Jt)  = evol.gaussianDiffeonsEvolutionEuler(self.c0, self.S0, self.at, self.param.sigmaKernel, affine=A,
            #                                                         withPointSet = self.fv0Fine.vertices, withJacobian=True)
            imDef = diffeo.gridScalars(grid = self.im1)
            for kk in range(self.Tsize+1):
                imDef.data = diffeo.multilinInterp(self.im1.data, grt[kk, ...].transpose(range(-1, self.grt.ndim - 2)))
                imDef.saveImg(self.outputDir +'/'+ self.saveFile+str(kk)+'.png', normalize=True)
                if self.dim==3:
                    gd.saveDiffeons(self.outputDir +'/'+ self.saveFile+'Diffeons'+str(kk)+'.vtk', self.ct[kk,...], self.St[kk,...])
                elif self.dim==2:
                    (R, detR) = gd.multiMatInverse1(self.St[kk,...], isSym=True)
                    diffx = self.gr1[..., np.newaxis, :] - self.ct[kk, ...]
                    betax = (R*diffx[..., np.newaxis, :]).sum(axis=-1)
                    dst = (betax * diffx).sum(axis=-1)
                    diffIm = np.minimum((255*(1-dst)*(dst < 1)).astype(float).sum(axis=-1), 255)
                    out = Image.fromarray(diffIm.astype(np.uint8))
                    out.save(self.outputDir +'/'+ self.saveFile+'Diffeons'+str(kk)+'.png')
        else:
            (obj1, self.ct, self.St, self.grt, self.Jt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)



    def optimizeMatching(self):
        # obj0 = self.param.fun_obj0(self.fv1, self.param.KparDist) # / (self.param.sigmaError**2)
        # if self.dcurr:
        #     (obj, self.ct, self.St, self.bt, self.xt, self.xSt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
        #     data = (self.xt[-1,:,:], self.xSt[-1,:,:,:], self.bt[-1,:,:])
        #     print 'objDef = ', obj, 'dataterm = ',  obj0 + self.dataTerm(data)* (self.param.sigmaError**2)
        #     print obj0 + surfaces.currentNormDef(self.fv0, self.fv1, self.param.KparDist)
        # else:
        #     (obj, self.ct, self.St, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
        #     self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
        #     print 'objDef = ', obj, 'dataterm = ',  obj0 + self.dataTerm(self.fvDef)

        if self.gradEps < 0:
            grd = self.getGradient(self.gradCoeff)
            [grd2] = self.dotProduct(grd, [grd])

            self.gradEps = max(0.001, np.sqrt(grd2) / 10000)

        print('Gradient lower bound: ', self.gradEps)
        if self.param.algorithm == 'cg':
            cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=0.01)
        elif self.param.algorithm == 'bfgs':
            bfgs.bfgs(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=1.,
                      Wolfe=self.param.wolfe, memory=25)
        #return self.at, self.xt

