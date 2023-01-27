import numpy.linalg as la
import logging
from . import surfaces
#import pointEvolution_fort as evol_omp
from . import conjugateGradient as cg, kernelFunctions as kfun, pointEvolution as evol
from . import surfaceMatching
from .affineBasis import *


## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDiffOut: background kernel: if not specified, use typeKernel with width sigmaKernelOut
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
class SurfaceMatchingParamAtrophy(surfaceMatching.SurfaceMatchingParam):
    def __init__(self, timeStep = .1, KparDiff = None, KparDist = None, KparDiffOut = None, sigmaKernel = 6.5,
                 sigmaKernelOut=6.5, sigmaDist=2.5, sigmaError=1.0, typeKernel='gauss', errorType='varifold'):
        surfaceMatching.SurfaceMatchingParam.__init__(self, timeStep = timeStep, KparDiff = KparDiff, KparDist=KparDist,
                                                      sigmaKernel =  sigmaKernel, sigmaDist = sigmaDist,
                                                      sigmaError = sigmaError,
                                                      typeKernel = typeKernel, errorType=errorType)
        self.sigmaKernelOut = sigmaKernelOut
        if KparDiffOut == None:
            self.KparDiffOut = kfun.Kernel(name = self.typeKernel, sigma = self.sigmaKernelOut)
        else:
            self.KparDiffOut = KparDiffOut


## Main class for surface matching
#        Template: sequence of surface classes (from surface.py); if not specified, opens files in fileTemp
#        Target: sequence of surface classes (from surface.py); if not specified, opens files in fileTarg
#        par: surfaceMatchingParam
#        verb: verbose printing
#        regWeight: multiplicative constant on object regularization
#        regWeightOut: multiplicative constant on background regularization
#        affineWeight: multiplicative constant on affine regularization
#        testGradient: evaluates gradient accuracy over random direction (debug)
#        mu: initial value for quadratic penalty normalization
#        outputDir: where results are saved
#        saveFile: generic name for saved surfaces
#        affine: 'affine', 'euclidean' or 'none'
#        maxIter_cg: max iterations in conjugate gradient
#        maxIter_al: max interation for augmented lagrangian

class SurfaceMatchingAtrophy(surfaceMatching.SurfaceMatching):
    def __init__(self, Template=None, Target=None, fileTempl=None, fileTarg=None, param=None, verb=True, regWeight=1.0,
                 affineWeight = 1.0,
                  testGradient=False, mu = 0.1, outputDir='.', saveFile = 'evolution', affine='none', volumeOnly=False,
                  rotWeight = None, scaleWeight = None, transWeight = None, symmetric=False, maxIter_cg=1000, maxIter_al=100):
        super(SurfaceMatchingAtrophy, self).__init__(Template, Target, fileTempl, fileTarg, param, maxIter_cg, regWeight, affineWeight,
                                                              verb, -1, rotWeight, scaleWeight, transWeight, symmetric, testGradient,
                                                              saveFile, False, affine, outputDir)


        self.volumeOnly = volumeOnly
        self.maxIter_cg = maxIter_cg
        self.maxIter_al = maxIter_al
        self.iter = 0

        if self.volumeOnly:
            self.cval = np.zeros(self.Tsize+1)
            self.cstr = np.zeros(self.Tsize+1)
            self.lmb = np.zeros(self.Tsize+1)
        else:
            self.cval = np.zeros([self.Tsize+1, self.npt])
            self.cstr = np.zeros([self.Tsize+1, self.npt])
            self.lmb = np.zeros([self.Tsize+1, self.npt])
        self.nu = np.zeros([self.Tsize+1, self.npt, self.dim])
        
        self.mu = mu
        self.useKernelDotProduct = True
        self.dotProduct = self.kernelDotProduct
        #self.useKernelDotProduct = False
        #self.dotProduct = self.standardDotProduct


    def constraintTerm(self, xt, at, Afft):
        timeStep = 1.0/self.Tsize
        obj=0
        cval = np.zeros(self.cval.shape)
        for t in range(self.Tsize):
            a = at[t]
            x = xt[t]
            # if self.affineDim > 0:
            #     AB = np.dot(self.affineBasis, Afft[t]) 
            #     A = AB[0:dim2].reshape([self.dim, self.dim])
            #     b = AB[dim2:dim2+self.dim]
            # else:
            #     A = np.zeros([self.dim, self.dim])
            #     b = np.zeros(self.dim)
            nu = np.zeros(x.shape)
            fk = self.fv0.faces
            xDef0 = x[fk[:, 0], :]
            xDef1 = x[fk[:, 1], :]
            xDef2 = x[fk[:, 2], :]
            nf =  np.cross(xDef1-xDef0, xDef2-xDef0)
            for kk,j in enumerate(fk[:,0]):
                nu[j, :] += nf[kk,:]
            for kk,j in enumerate(fk[:,1]):
                nu[j, :] += nf[kk,:]
            for kk,j in enumerate(fk[:,2]):
                nu[j, :] += nf[kk,:]
            if not self.volumeOnly:
                nu /= np.sqrt((nu**2).sum(axis=1)).reshape([nu.shape[0], 1])
            nu *= self.fv0ori


            #r = self.param.KparDiff.applyK(x, a) + np.dot(x, A.T) + b
            r = self.param.KparDiff.applyK(x, a) 
            self.nu[t,...] = nu
            self.v[t,...] = r
            if self.volumeOnly:
                cstr = (nu*r).sum()
            else:
                cstr = np.squeeze((nu*r).sum(axis=1))
            self.cstr[t,...] = np.maximum(cstr, 0)
            cval[t,...] = np.maximum(cstr - self.lmb[t,...]*self.mu, 0)
            obj += 0.5*timeStep * (cval[t,...]**2).sum()/self.mu

        #print 'cstr', obj
        return obj,cval

    def constraintTermGrad(self, xt, at, Afft):
        lmb = np.zeros(self.cval.shape)
        dxcval = np.zeros(xt.shape)
        dacval = np.zeros(at.shape)
        dAffcval = np.zeros(Afft.shape)
        dim2 = self.dim**2
        for t in range(self.Tsize):
            a = at[t]
            x = xt[t]
            # if self.affineDim > 0:
            #     AB = np.dot(self.affineBasis, Afft[t]) 
            #     A = AB[0:dim2].reshape([self.dim, self.dim])
            #     b = AB[dim2:dim2+self.dim]
            # else:
            #     A = np.zeros([self.dim, self.dim])
            #     b = np.zeros(self.dim)
            fk = self.fv0.faces
            nu = np.zeros(x.shape)
            xDef0 = x[fk[:, 0], :]
            xDef1 = x[fk[:, 1], :]
            xDef2 = x[fk[:, 2], :]
            nf =  np.cross(xDef1-xDef0, xDef2-xDef0)
            for kk,j in enumerate(fk[:,0]):
                nu[j, :] += nf[kk,:]
            for kk,j in enumerate(fk[:,1]):
                nu[j, :] += nf[kk,:]
            for kk,j in enumerate(fk[:,2]):
                nu[j, :] += nf[kk,:]
            normNu = np.sqrt((nu**2).sum(axis=1))
            if not self.volumeOnly:
                nu /= normNu.reshape([nu.shape[0], 1])
            nu *= self.fv0ori
            #r2 = self.param.KparDiffOut.applyK(zB, aB)

            #dv = self.param.KparDiff.applyK(x, a) + np.dot(x, A.T) + b
            vt = self.param.KparDiff.applyK(x, a)
            if self.volumeOnly:
                lmb[t] = -np.maximum((nu*vt).sum() -self.lmb[t]*self.mu, 0)/self.mu
                lnu = lmb[t]*nu
                lv = vt * lmb[t]
            else:
                lmb[t, :] = -np.maximum(np.multiply(nu, vt).sum(axis=1) -self.lmb[t,:]*self.mu, 0)/self.mu
                lnu = np.multiply(nu, lmb[t, :].reshape([self.npt, 1]))
                lv = vt * lmb[t,:,np.newaxis]
            #lnu = np.multiply(nu, np.mat(lmb[t, npt:npt1]).T)
            #print lnu.shape
            dxcval[t] = self.param.KparDiff.applyDiffKT(x, a, lnu)
            dxcval[t] += self.param.KparDiff.applyDiffKT(x, lnu, a)
            #dxcval[t] += np.dot(lnu, A)
            if self.useKernelDotProduct:
                dacval[t] = np.copy(lnu)
            else:
                dacval[t] = self.param.KparDiff.applyK(x, lnu)
            dAffcval = []
            # if self.affineDim > 0:
            #     dAffcval[t, :] = (np.dot(self.affineBasis.T, np.vstack([np.dot(lnu.T, x).reshape([dim2,1]), lnu.sum(axis=0).reshape([self.dim,1])]))).flatten()
            if not self.volumeOnly:
                lv /= normNu.reshape([nu.shape[0], 1])
                lv -= np.multiply(nu, np.multiply(nu, lv).sum(axis=1).reshape([nu.shape[0], 1]))
            lvf = lv[fk[:,0]] + lv[fk[:,1]] + lv[fk[:,2]]
            dnu = np.zeros(x.shape)
            foo = np.cross(xDef2-xDef1, lvf)
            for kk,j in enumerate(fk[:,0]):
                dnu[j, :] += foo[kk,:]
            foo = np.cross(xDef0-xDef2, lvf)
            for kk,j in enumerate(fk[:,1]):
                dnu[j, :] += foo[kk,:]
            foo = np.cross(xDef1-xDef0, lvf)
            for kk,j in enumerate(fk[:,2]):
                dnu[j, :] += foo[kk,:]
            dxcval[t] -= self.fv0ori*dnu


        #print 'testg', (lmb**2).sum() 
        return lmb, dxcval, dacval, dAffcval






    def testConstraintTerm(self, xt, at, Afft):
        eps = 0.00000001
        xtTry = xt + eps*np.random.randn(self.Tsize+1, self.npt, self.dim)
        atTry = at + eps*np.random.randn(self.Tsize, self.npt, self.dim)
        # if self.affineDim > 0:
        #     AfftTry = Afft + eps*np.random.randn(self.Tsize, self.affineDim)
            

        u0 = self.constraintTerm(xt, at, Afft)
        ux = self.constraintTerm(xtTry, at, Afft)
        ua = self.constraintTerm(xt, atTry, Afft)
        [l, dx, da, dA] = self.constraintTermGrad(xt, at, Afft)
        vx = np.multiply(dx, xtTry-xt).sum()/eps
        va = np.multiply(da, atTry-at).sum()/eps
        logging.info('Testing constraints:')
        logging.info('var x: %f %f' %( self.Tsize*(ux[0]-u0[0])/(eps), -vx)) 
        logging.info('var a: %f %f' %( self.Tsize*(ua[0]-u0[0])/(eps), -va)) 
        # if self.affineDim > 0:
        #     uA = self.constraintTerm(xt, at, AfftTry)
        #     vA = np.multiply(dA, AfftTry-Afft).sum()/eps
        #     logging.info('var affine: %f %f' %(self.Tsize*(uA[0]-u0[0])/(eps), -vA ))

    def  objectiveFunDef(self, at, Afft=None, withTrajectory = False, withJacobian = False, fv0 = None, kernel=None,
                         regWeight=None):
        f = super(SurfaceMatchingAtrophy, self).objectiveFunDef(at, Afft=Afft, withTrajectory=True,
                                                                withJacobian=withJacobian, fv0=fv0)
        cstr = self.constraintTerm(f[1], at, Afft)
        obj = f[0]+cstr[0]

        #print f[0], cstr[0]

        if withJacobian:
            return obj, f[1], f[2], cstr[1]
        elif withTrajectory:
            return obj, f[1], cstr[1]
        else:
            return obj

    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = self.param.fun_obj0(self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
            if self.symmetric:
                self.obj0 += self.param.fun_obj0(self.fv0, self.param.KparDist) / (self.param.sigmaError**2)

            (self.obj, self.xt, self.cval) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.obj += self.obj0

            self.fvDef.updateVertices(np.squeeze(self.xt[self.Tsize, ...]))
            self.obj += self.param.fun_obj(self.fvDef, self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
            if self.symmetric:
                self.fvInit.updateVertices(np.squeeze(self.x0))
                self.obj += self.param.fun_obj(self.fvInit, self.fv0, self.param.KparDist) / (self.param.sigmaError**2)

        return self.obj

    # def getVariable(self):
    #     return [self.at, self.Afft]

    def updateTry(self, dir, eps, objRef=None):
        atTry  = self.at - eps * dir.diff
        if self.affineDim > 0:
            AfftTry = self.Afft - eps * dir.aff
        else:
            AfftTry = []

        fv0 = surfaces.Surface(surf=self.fv0)
        if self.symmetric:
            x0Try = self.x0 - eps * dr.initx
            fv0.updateVertices(x0Try)
        else:
            x0Try = None
        foo = self.objectiveFunDef(atTry, AfftTry, fv0 = fv0, withTrajectory=True)
        objTry = 0

        ff = surfaces.Surface(surf=self.fvDef)
        ff.updateVertices(np.squeeze(foo[1][self.Tsize, ...]))
        objTry += self.param.fun_obj(ff, self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
        if self.symmetric:
            ffI = surfaces.Surface(surf=self.fvInit)
            ffI.updateVertices(x0Try)
            objTry += self.param.fun_obj(ffI, self.fv0, self.param.KparDist) / (self.param.sigmaError**2)
        objTry += foo[0]+self.obj0

        if np.isnan(objTry):
            logging.warning('Warning: nan in updateTry')
            return 1e500


        if (objRef == None) | (objTry < objRef):
            self.atTry = atTry
            self.AfftTry = AfftTry
            self.objTry = objTry
            self.x0Try = x0Try
            self.cval = foo[2]


        return objTry


    def covectorEvolution(self, at, Afft, px1):
        M = self.Tsize
        timeStep = 1.0/M
        dim2 = self.dim**2
        A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
        if self.affineDim > 0:
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t]) 
                A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                A[1][t] = AB[dim2:dim2+self.dim]
        xt = evol.landmarkDirectEvolutionEuler(self.x0, at, self.param.KparDiff, affine = A)
        #xt = xJ
        pxt = np.zeros([M+1, self.npt, self.dim])
        pxt[M, :, :] = px1
        
        foo = self.constraintTermGrad(xt, at, Afft)
        lmb = foo[0]
        dxcval = foo[1]
        dacval = foo[2]
        dAffcval = foo[3]
        
        pxt[M, :, :] += timeStep * dxcval[M]
        
        for t in range(M):
            px = np.squeeze(pxt[M-t, :, :])
            z = np.squeeze(xt[M-t-1, :, :])
            a = np.squeeze(at[M-t-1, :, :])
            zpx = np.copy(dxcval[M-t-1])
            zpx += self.param.KparDiff.applyDiffKT(z, px, a, regweight=self.regweight, lddmm=True)
            if self.affineDim > 0:
                pxt[M-t-1, :, :] = np.dot(px, self.affB.getExponential(timeStep*A[0][M-t-1])) + timeStep * zpx
                #zpx += np.dot(px, A[0][M-t-1])
            else:
                pxt[M-t-1, :, :] = px + timeStep * zpx
            

        return pxt, xt, dacval, dAffcval


    def HamiltonianGradient(self, at, Afft, px1, getCovector = False):
        (pxt, xt, dacval, dAffcval) = self.covectorEvolution(at, Afft, px1)

        if self.useKernelDotProduct:
            dat = 2*self.regweight*at - pxt[1:pxt.shape[0],...] - dacval
        else:
            dat = -dacval
            for t in range(self.Tsize):
                dat[t] += self.param.KparDiff.applyK(xt[t], 2*self.regweight*at[t] - pxt[t+1])
        if self.affineDim > 0:
            timeStep = 1.0/self.Tsize
            dAfft = 2*np.multiply(self.affineWeight.reshape([1, self.affineDim]), Afft) 
            #dAfft = 2*np.multiply(self.affineWeight, Afft) - dAffcval
            for t in range(self.Tsize):
                AB = np.dot(self.affineBasis, Afft[t]) 
                A = AB[0:self.dim**2].reshape([self.dim, self.dim])
                #A[1][t] = AB[dim2:dim2+self.dim]
                #dA = np.dot(pxt[t+1].T, xt[t]).reshape([self.dim**2, 1])
                dA = self.affB.gradExponential(timeStep*A, pxt[t+1], xt[t]).reshape([self.dim**2, 1])
                db = pxt[t+1].sum(axis=0).reshape([self.dim,1]) 
                dAff = np.dot(self.affineBasis.T, np.vstack([dA, db]))
                dAfft[t] -=  dAff.reshape(dAfft[t].shape)
            #dAfft = np.divide(dAfft, self.affineWeight.reshape([1, self.affineDim]))
        else:
            dAfft = None
 
        if getCovector == False:
            return dat, dAfft, xt
        else:
            return dat, dAfft, xt, pxt

    # def endPointGradient(self):
    #     px1 = -self.param.fun_objGrad(self.fvDef, self.fv1, self.param.KparDist) / self.param.sigmaError**2
    #     return px1

    def addProd(self, dir1, dir2, beta):
        dir = surfaceMatching.Direction()
        dir.diff = dir1.diff + beta * dir2.diff
        dir.initx = dir1.initx + beta * dir2.initx
        if self.affineDim > 0:
            dir.aff = dir1.aff + beta * dir2.aff
        return dir

    # def copyDir(self, dir0):
    #     dir = surfaceMatching.Direction()
    #     dir.diff = np.copy(dir0.diff)
    #     dir.aff = np.copy(dir0.aff)
    #     return dir

        
    def kernelDotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        for t in range(self.Tsize):
            z = np.squeeze(self.xt[t, :, :])
            gg = np.squeeze(g1.diff[t, :, :])
            u = self.param.KparDiff.applyK(z, gg)
            #if self.affineDim > 0:
                #uu = np.multiply(g1.aff[t], self.affineWeight)
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr.diff[t, :, :])
                res[ll]  +=  np.multiply(ggOld,u).sum()
                if self.affineDim > 0:
                    res[ll] += np.multiply(g1.aff[t], gr.aff[t]).sum() * self.coeffAff
                    #res[ll] += np.multiply(uu, gr.aff[t]).sum()
                ll = ll + 1
        if self.symmetric:
            for ll,gr in enumerate(g2):
                res[ll] += (g1.initx * gr.initx).sum() * self.coeffInitx
      
        return res

    def standardDotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        dim2 = self.dim**2
        for ll,gr in enumerate(g2):
            res[ll]=0
            res[ll] += np.multiply(g1.diff, gr.diff).sum()
            if self.affineDim > 0:
                #uu = np.multiply(g1.aff, self.affineWeight.reshape([1, self.affineDim]))
                #res[ll] += np.multiply(uu, gr.aff).sum() * self.coeffAff
                res[ll] += np.multiply(g1.aff, gr.aff).sum() * self.coeffAff
                #+np.multiply(g1[1][k][:, dim2:dim2+self.dim], gr[1][k][:, dim2:dim2+self.dim]).sum())

        if self.symmetric:
            for ll,gr in enumerate(g2):
                res[ll] += (g1.initx * gr.initx).sum() * self.coeffInitx

        return res



    def getGradient(self, coeff=1.0):
        px1 = -self.endPointGradient()
        #px1.append(np.zeros([self.npoints, self.dim]))
        foo = self.HamiltonianGradient(self.at, self.Afft, px1, getCovector=True)
        grd = surfaceMatching.Direction()
        grd.diff = foo[0] / (coeff*self.Tsize)
        if self.affineDim > 0:
            grd.aff = foo[1] / (self.coeffAff*coeff*self.Tsize)
        if self.symmetric:
            grd.initx = (self.initPointGradient() - foo[3][0,...])/(self.coeffInitx * coeff)
        return grd

    def randomDir(self):
        dirfoo = surfaceMatching.Direction()
        dirfoo.diff = np.random.randn(self.Tsize, self.npt, self.dim)
        dirfoo.initx = np.random.randn(self.npt, self.dim)
        if self.affineDim > 0:
            dirfoo.aff = np.random.randn(self.Tsize, self.affineDim)
        return dirfoo

    # def acceptVarTry(self):
    #     self.obj = self.objTry
    #     self.at = np.copy(self.atTry)
    #     self.Afft = np.copy(self.AfftTry)

    def endOfIteration(self):
        self.iter += 1
        if self.iter >= self.affBurnIn:
            self.coeffAff = self.coeffAff2
        if (self.iter % self.saveRate == 0) :
            (obj1, self.xt, Jt, self.cval) = self.objectiveFunDef(self.at, self.Afft, withJacobian=True)
            logging.info('mean constraint %f max constraint %f' %(np.sqrt((self.cstr**2).sum()/self.cval.size), np.fabs(self.cstr).max()))
            logging.info('saving data')
            self.fvInit.updateVertices(self.x0)

            if self.affine=='euclidean' or self.affine=='translation':
                f = surfaces.Surface(surf=self.fvInit)
                X = self.affB.integrateFlow(self.Afft)
                displ = np.zeros(self.npt)
                dt = 1.0 /self.Tsize ;
                for t in range(self.Tsize+1):
                    U = la.inv(X[0][t])
                    yt = np.dot(self.xt[t,...] - X[1][t, ...], U.T)
                    if t < self.Tsize:
                        at = np.dot(self.at[t,...], U.T)
                        vt = self.param.KparDiff.applyK(yt, at)
                    f.updateVertices(yt)
                    vf = surfaces.vtkFields() ;
                    vf.scalars.append('Jacobian') ;
                    vf.scalars.append(np.exp(Jt[t, :]))
                    vf.scalars.append('displacement')
                    vf.scalars.append(displ)
                    vf.vectors.append('velocity') ;
                    vf.vectors.append(vt)
                    f.saveVTK2(self.outputDir +'/'+self.saveFile+'Corrected'+str(t)+'.vtk', vf)
                    nu = self.fv0ori*f.computeVertexNormals()
                    displ += dt * (vt*nu).sum(axis=1)
                f = surfaces.Surface(surf=self.fv1)
                #logging.info('rotation?: %f' %(np.fabs(np.dot(U, U.T)- np.eye(3)).sum()))
                yt = np.dot(f.vertices - X[1][-1, ...], U.T)
                f.updateVertices(yt)
                f.saveVTK(self.outputDir +'/TargetCorrected.vtk')
                
                
            #self.testConstraintTerm(self.xt, self.at, self.Afft)
            nn = 0 ;
            AV0 = self.fvInit.computeVertexArea()
            nu = self.fv0ori*self.fvInit.computeVertexNormals()
            v = self.v[0,...]
            displ = np.zeros(self.npt)
            dt = 1.0 /self.Tsize ;
            n1 = self.xt.shape[1] ;
            for kk in range(self.Tsize+1):
                self.fvDef.updateVertices(np.squeeze(self.xt[kk, :, :]))
                AV = self.fvDef.computeVertexArea()
                AV = (AV[0]/AV0[0])-1
                vf = surfaces.vtkFields() ;
                vf.scalars.append('Jacobian') ;
                vf.scalars.append(np.exp(Jt[kk, :]))
                vf.scalars.append('Jacobian_T') ;
                vf.scalars.append(AV[:,0])
                vf.scalars.append('Jacobian_N') ;
                vf.scalars.append(np.exp(Jt[kk, :])/(AV[:,0]+1)-1)
                vf.scalars.append('displacement')
                vf.scalars.append(displ)
                if kk < self.Tsize:
                    nu = self.fv0ori*self.fvDef.computeVertexNormals()
                    v = self.v[kk,...]
                    kkm = kk
                else:
                    kkm = kk-1
                if not self.volumeOnly:
                    vf.scalars.append('constraint') ;
                    vf.scalars.append(self.cstr[kkm,:])
                vf.vectors.append('velocity') ;
                vf.vectors.append(self.v[kkm,:])
                vf.vectors.append('normals') ;
                vf.vectors.append(self.nu[kkm,:])
                self.fvDef.saveVTK2(self.outputDir +'/'+self.saveFile+str(kk)+'.vtk', vf)
                displ += dt * (v*nu).sum(axis=1)
        else:
            (obj1, self.xt, Jt, self.cval) = self.objectiveFunDef(self.at, self.Afft, withJacobian=True)
            logging.info('mean constraint %f max constraint %f' %(np.sqrt((self.cstr**2).sum()/self.cval.size), np.fabs(self.cstr).max()))
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.fvInit.updateVertices(self.x0)

    def optimizeMatching(self):
        self.coeffZ = 10.
        self.coeffAff = self.coeffAff2
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = np.sqrt(grd2) / 100
        self.coeffAff = self.coeffAff1
        self.muEps = 1.0
        it = 0
        while (self.muEps > 0.001) & (it<self.maxIter_al)  :
            logging.info('Starting Minimization: gradEps = %f muEps = %f mu = %f' %(self.gradEps, self.muEps,self.mu))
            #self.coeffZ = max(1.0, self.mu)
            cg.cg(self, verb = self.verb, maxIter = self.maxIter_cg, TestGradient = self.testGradient, epsInit=0.1)
            self.coeffAff = self.coeffAff2
            for t in range(self.Tsize+1):
                self.lmb[t, ...] = -self.cval[t, ...]/self.mu
            logging.info('mean lambdas %f' %(np.fabs(self.lmb).sum() / self.lmb.size))
            if self.converged:
                self.gradEps *= .75
                if (((self.cstr**2).sum()/self.cval.size) > self.muEps**2):
                    self.mu *= 0.5
                else:
                    self.muEps = self.muEps /2
            else:
                self.mu *= 0.9
            self.obj = None
            it = it+1
            
            #return self.fvDef

