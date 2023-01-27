import os
import scipy.linalg as spLA
import logging
import time
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
class SurfaceMatchingParam(surfaceMatching.SurfaceMatchingParam):
    def __init__(self, timeStep = .1, KparDiff = None, KparDist = None, KparDiffOut = None, sigmaKernel = 6.5, sigmaKernelOut=6.5, sigmaDist=2.5, sigmaError=1.0, typeKernel='gauss', errorType='varifold'):
        surfaceMatching.SurfaceMatchingParam.__init__(self, timeStep = timeStep, KparDiff = KparDiff, KparDist=KparDist, sigmaKernel =  sigmaKernel, sigmaDist = sigmaDist, sigmaError = sigmaError, typeKernel = typeKernel, errorType=errorType)
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
#        typeConstraint: 'stitched', 'slidingNormal', 'sliding'
#        affine: 'affine', 'euclidean' or 'none'
#        maxIter_cg: max iterations in conjugate gradient
#        maxIter_al: max interation for augmented lagrangian

class SurfaceMatching(surfaceMatching.SurfaceMatching):
    def __init__(self, Template=None, Target=None, fileTempl=None, fileTarg=None, param=None, verb=True, regWeight=1.0, regWeightOut=1.0, affineWeight = 1.0, testGradient=False, mu = 0.1, outputDir='.', saveFile = 'evolution', typeConstraint='stitched', affine='none', rotWeight = None, scaleWeight = None, transWeight = None,  maxIter_cg=1000, maxIter_al=100):
        if Template==None:
            if fileTempl==None:
                logging.error('Please provide a template surface')
                return
            else:
                self.fv0 = []
                for ftmp in fileTempl:
                    self.fv0.append(surfaces.Surface(filename=ftmp))
        else:
            self.fv0 = []
            for ftmp in Template:
                self.fv0.append(surfaces.Surface(surf=ftmp))
        if Target==None:
            if fileTarg==None:
                logging.error('Please provide a target surface')
                return
            else:
                self.fv1 = []
                for ftmp in fileTarg:
                    self.fv1.append(surfaces.Surface(filename=ftmp))
        else:
            self.fv1 = []
            for ftmp in Target:
                self.fv1.append(surfaces.Surface(surf=ftmp))

        self.dim = self.fv0[0].vertices.shape[1]

        self.outputDir = outputDir  
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                logging.error('Cannot save in ' + outputDir)
                return
            else:
                os.mkdir(outputDir)


        self.fvDef = [] 
        self.fvDefB = [] 
        for fv in self.fv0:
            self.fvDef.append(surfaces.Surface(surf=fv))
            self.fvDefB.append(surfaces.Surface(surf=fv))
        self.maxIter_cg = maxIter_cg
        self.maxIter_al = maxIter_al
        self.verb = verb
        self.testGradient = testGradient
        self.regweight = regWeight
        self.regweightOut = regWeightOut
        if param==None:
            self.param = SurfaceMatchingParam()
        else:
            self.param = param            

        self.affine = affine
        affB = AffineBasis(self.dim, affine)
        self.affineDim = affB.affineDim
        self.affineBasis = affB.basis
        self.affineWeight = affineWeight * np.ones([1, self.affineDim])
        #print self.affineDim, affB.rotComp, rotWeight, self.affineWeight
        if (len(affB.rotComp) > 0) & (rotWeight != None):
            self.affineWeight[0,affB.rotComp] = rotWeight
        if (len(affB.simComp) > 0) & (scaleWeight != None):
            self.affineWeight[0,affB.simComp] = scaleWeight
        if (len(affB.transComp) > 0) & (transWeight != None):
            self.affineWeight[0,affB.transComp] = transWeight

             
        self.Tsize = int(round(1.0/self.param.timeStep))
        self.at = []
        self.atTry = []
        self.Afft = []
        self.AfftTry = []
        self.xt = []
        self.nut = []
        self.x0 = []
        self.nu0 = []
        self.nu00 = []
        self.nsurf = len(self.fv0)
        self.npt = np.zeros(self.nsurf)
        self.nf = np.zeros(self.nsurf)
        k=0
        for fv in self.fv0:
            x0 = fv.vertices
            # for q,n in enumerate(x0):
            #     print q, x0[q]
            fk = fv.faces
            xDef1 = x0[fk[:, 0], :]
            xDef2 = x0[fk[:, 1], :]
            xDef3 = x0[fk[:, 2], :]
            nf =  np.cross(xDef2-xDef1, xDef3-xDef1)
            nu0 = np.zeros(x0.shape)
                #print nf[1:10]
            for kk,j in enumerate(fk[:,0]):
                nu0[j, :] += nf[kk,:]
            for kk,j in enumerate(fk[:,1]):
                nu0[j, :] += nf[kk,:]
            for kk,j in enumerate(fk[:,2]):
                nu0[j, :] += nf[kk,:]
                #nu0[fk[:,0], :] += nf
                #nu0[fk[:,1], :] += nf
                #nu0[fk[:,2], :] += nf
            # for q,n in enumerate(nf):
            #     print q, nf[q]
            # for q,n in enumerate(nu0):
            #     print q, nu0[q]
            # for (ll,f) in enumerate(fv.faces):
            #     for v in f:
            #         nu0[v] += fv.surfel[ll]
            self.nu00.append(np.copy(nu0))
            nu0 /= np.sqrt((nu0**2).sum(axis=1)).reshape([x0.shape[0], 1])
            self.npt[k] = x0.shape[0]
            self.nf[k] = fv.faces.shape[0]
            self.x0.append(x0)
            self.nu0.append(nu0)
            self.at.append(np.zeros([self.Tsize, x0.shape[0], x0.shape[1]]))
            self.atTry.append(np.zeros([self.Tsize, x0.shape[0], x0.shape[1]]))
            self.Afft.append(np.zeros([self.Tsize, self.affineDim]))
            self.AfftTry.append(np.zeros([self.Tsize, self.affineDim]))
            #print x0.shape, nu0.shape
            self.xt.append(np.tile(x0, [self.Tsize+1, 1, 1]))
            self.nut.append(np.tile(np.array(nu0), [self.Tsize+1, 1, 1]))
            k=k+1

        self.npoints = self.npt.sum(dtype=int)
        self.nfaces = self.nf.sum(dtype=int)
        self.at.append(np.zeros([self.Tsize, self.npoints, self.dim]))
        self.atTry.append(np.zeros([self.Tsize, self.npoints, self.dim]))
        self.x0.append(np.zeros([self.npoints, self.dim]))
        npt = 0
        for fv0 in self.fv0:
            npt1 = npt + fv0.vertices.shape[0]
            self.x0[self.nsurf][npt:npt1, :] = np.copy(fv0.vertices)
            npt = npt1
        self.xt.append(np.tile(self.x0[self.nsurf], [self.Tsize+1, 1, 1]))

        self.typeConstraint = typeConstraint

        if typeConstraint == 'stitched':
            self.cval = np.zeros([self.Tsize+1, self.npoints, self.dim])
            self.lmb = np.zeros([self.Tsize+1, self.npoints, self.dim])
            self.constraintTerm = self.constraintTermStitched
            self.constraintTermGrad = self.constraintTermGradStitched
            self.useKernelDotProduct = True
            self.dotProduct = self.kernelDotProduct
        elif typeConstraint == 'slidingNormal':
            self.cval = np.zeros([self.Tsize+1, self.npoints])
            self.lmb = np.zeros([self.Tsize+1, self.npoints])
            self.constraintTerm = self.constraintTermSlidingNormal
            self.constraintTermGrad = self.constraintTermGradSlidingNormal
            self.useKernelDotProduct = False
            self.dotProduct = self.standardDotProduct
        elif typeConstraint == 'sliding':
            self.cval = np.zeros([self.Tsize+1, self.nfaces])
            self.lmb = np.zeros([self.Tsize+1, self.nfaces])
            self.constraintTerm = self.constraintTermSliding
            self.constraintTermGrad = self.constraintTermGradSliding
            self.useKernelDotProduct = True
            self.dotProduct = self.kernelDotProduct
            #self.useKernelDotProduct = False
            #self.dotProduct = self.standardDotProduct            
            # if self.param.KparDiff.name == 'none':
            #     self.dotProduct = self.normalizedDotProduct
            #     self.weightDot = 1.
            # else:
            #     #self.dotProduct = self.kernelDotProduct
            #     self.dotProduct = self.standardDotProduct            
        else:
            logging.error('Unrecognized constraint type')
            return
        
        self.mu = mu
        self.obj = None
        self.objTry = None
        self.saveFile = saveFile
        self.gradCoeff = self.fv0[0].vertices.shape[0]
        for k in range(self.nsurf):
            self.fv0[k].saveVTK(self.outputDir+'/Template'+str(k)+'.vtk', normals=self.nu0[k])
            self.fv1[k].saveVTK(self.outputDir+'/Target'+str(k)+'.vtk')


    def constraintTermStitched(self, xt, nut, at, Afft):
        timeStep = 1.0/self.Tsize
        obj=0
        cval = np.zeros(self.cval.shape)
        for t in range(self.Tsize+1):
            zB = np.squeeze(xt[self.nsurf][t,...])
            npt = 0
            for k in range(self.nsurf):
                z = np.squeeze(xt[k][t, ...]) 
                npt1 = npt + self.npt[k]
                cval[t,npt:npt1, ...] = z - zB[npt:npt1, ...]
                npt = npt1

            obj += timeStep * (- np.multiply(self.lmb[t, ...], cval[t,...]).sum() + (cval[t, ...]**2).sum()/(2*self.mu))
        return obj,cval

    def constraintTermSlidingNormal(self, xt, nut, at, Afft):
        timeStep = 1.0/self.Tsize
        obj=0
        cval = np.zeros(self.cval.shape)
        dim2 = self.dim**2
        for t in range(self.Tsize):
            zB = np.squeeze(xt[self.nsurf][t, ...])
            aB = np.squeeze(at[self.nsurf][t, ...])
            npt = 0
            for k in range(self.nsurf):
                a = at[k][t]
                if self.affineDim > 0:
                    AB = np.dot(self.affineBasis, Afft[k][t]) 
                    A = AB[0:dim2].reshape([self.dim, self.dim])
                    b = AB[dim2:dim2+self.dim]
                else:
                    A = np.zeros([self.dim, self.dim])
                    b = np.zeros(self.dim)
                x = np.squeeze(xt[k][t, ...])
                nu = np.squeeze(nut[k][t, ...])
                npt1 = npt + self.npt[k]
                # r = evol_omp.applyK(x, x, a, self.param.KparDiff.sigma, self.param.KparDiff.order,
                #                     x.shape[0], x.shape[0], x.shape[1]) + np.dot(x, A.T) + b
                # r2 = evol_omp.applyK(x, zB, a, self.param.KparDiffOut.sigma, self.param.KparDiffOut.order,
                #                     x.shape[0], zB.shape[0], zB.shape[1]) + np.dot(x, A.T) + b
                r = self.param.KparDiff.applyK(x, a) + np.dot(x, A.T) + b
                r2 = self.param.KparDiffOut.applyK(zB, aB, firstVar=x)
                #print nu.shape, r.shape, r2.shape, cval[t,npt:npt1].shape, npt, npt1
                cval[t,npt:npt1] = np.squeeze(np.multiply(nu, r-r2).sum(axis=1))
                npt = npt1

            obj += timeStep * (- np.multiply(self.lmb[t, ...], cval[t,...]).sum() + (cval[t, ...]**2).sum()/(2*self.mu))
        return obj,cval

    def cstrFun(self, x):
        return (np.multiply(x,x)/2.)
    #u = np.fabs(x)
    #  return u + np.log((1+np.exp(-2*u))/2)

    def derCstrFun(self, x):
        return x
    #u = np.exp(-2*np.fabs(x))
    #  return np.multiply(np.sign(x), np.divide(1-u, 1+u))

    def constraintTermGradStitched(self, xt, nut, at, Afft):
        lmb = np.zeros(self.cval.shape)
        dxcval = []
        dnucval = []
        dacval = []
        dAffcval = []
        for u in xt:
            dxcval.append(np.zeros(u.shape))
        for u in at:
            dacval.append(np.zeros(u.shape))
        for u in Afft:
            dAffcval.append(np.zeros(u.shape))
        for u in nut:
            dnucval.append(np.zeros(u.shape))
        for t in range(self.Tsize+1):
            zB = np.squeeze(xt[self.nsurf][t,...])
            npt = 0
            for k in range(self.nsurf):
                z = np.squeeze(xt[k][t,...]) 
                npt1 = npt + self.npt[k]
                lmb[t, npt:npt1,...] = self.lmb[t, npt:npt1,...] - (z - zB[npt:npt1,...])/self.mu
                dxcval[k][t,...] = lmb[t, npt:npt1,...]
                dxcval[-1][t, npt:npt1,...] = -lmb[t, npt:npt1,...]
                npt = npt1

        return lmb, dxcval, dnucval, dacval, dAffcval

    def constraintTermSliding(self, xt, nut, at, Afft):
        timeStep = 1.0/self.Tsize
        obj=0
        cval = np.zeros(self.cval.shape)
        dim2 = self.dim**2
        for t in range(self.Tsize):
            zB = np.squeeze(xt[self.nsurf][t, ...])
            aB = np.squeeze(at[self.nsurf][t, ...])
            nf = 0
            npt = 0
            r2 = self.param.KparDiffOut.applyK(zB, aB)
            for k in range(self.nsurf):
                nf1 = nf + self.nf[k]
                npt1 = npt + self.npt[k]
                a = at[k][t]
                x = xt[k][t]
                if self.affineDim > 0:
                    AB = np.dot(self.affineBasis, Afft[k][t]) 
                    A = AB[0:dim2].reshape([self.dim, self.dim])
                    b = AB[dim2:dim2+self.dim]
                else:
                    A = np.zeros([self.dim, self.dim])
                    b = np.zeros(self.dim)
                z = zB[npt:npt1, :]
                r = (self.param.KparDiff.applyK(x, a, firstVar=z) + np.dot(z, A.T) + b
                    - r2[npt:npt1, :])
                fk = self.fv0[k].faces
                rf = r[fk[:, 0], :] + r[fk[:, 1], :] + r[fk[:, 2], :]
                xDef0 = z[fk[:, 0], :]
                xDef1 = z[fk[:, 1], :]
                xDef2 = z[fk[:, 2], :]
                nu =  np.cross(xDef1-xDef0, xDef2-xDef0)
                nu /= np.sqrt((nu**2).sum(axis=1)).reshape([nu.shape[0], 1])

                cval[t,nf:nf1] = np.squeeze(np.multiply(nu, rf).sum(axis=1))
                npt = npt1
                nf = nf1

            obj += timeStep * (- np.multiply(self.lmb[t, ...], cval[t,...]).sum() + (cval[t, ...]**2).sum()/(2*self.mu))
            #print 'slidingV2', obj
        return obj,cval

    def constraintTermGradSliding(self, xt, nut, at, Afft):
        lmb = np.zeros(self.cval.shape)
        dxcval = []
        dnucval = []
        dAffcval = []
        dacval = []
        for u in xt:
            dxcval.append(np.zeros(u.shape))
        for u in nut:
            dnucval.append(np.zeros(u.shape))
        for u in at:
            dacval.append(np.zeros(u.shape))
        for u in Afft:
            dAffcval.append(np.zeros(u.shape))
        dim2 = self.dim**2
        for t in range(self.Tsize):
            zB = xt[self.nsurf][t]
            aB = at[self.nsurf][t]
            r2 = self.param.KparDiffOut.applyK(zB, aB)
            npt = 0
            nf = 0
            for k in range(self.nsurf):
                npt1 = npt + self.npt[k]
                nf1 = nf + self.nf[k]
                a = at[k][t]
                x = xt[k][t]
                if self.affineDim > 0:
                    AB = np.dot(self.affineBasis, Afft[k][t]) 
                    A = AB[0:dim2].reshape([self.dim, self.dim])
                    b = AB[dim2:dim2+self.dim]
                else:
                    A = np.zeros([self.dim, self.dim])
                    b = np.zeros(self.dim)
                z = zB[npt:npt1, :]
                fk = self.fv0[k].faces
                nu = np.zeros(x.shape)
                xDef0 = z[fk[:, 0], :]
                xDef1 = z[fk[:, 1], :]
                xDef2 = z[fk[:, 2], :]
                nu =  np.cross(xDef1-xDef0, xDef2-xDef0)
                normNu = np.sqrt((nu**2).sum(axis=1))
                nu /= normNu.reshape([nu.shape[0], 1])

                dv = self.param.KparDiff.applyK(x, a, firstVar=z) + np.dot(z, A.T) + b - r2[npt:npt1, :]
                dvf = dv[fk[:, 0], :] + dv[fk[:, 1], :] + dv[fk[:, 2], :]
                lmb[t, nf:nf1] = self.lmb[t, nf:nf1] - np.multiply(nu, dvf).sum(axis=1)/self.mu
                #lnu = np.multiply(nu, np.mat(lmb[t, npt:npt1]).T)
                lnu0 = np.multiply(nu, lmb[t, nf:nf1].reshape([self.nf[k], 1]))
                lnu = np.zeros([self.npt[k], self.dim])
                for kk,j in enumerate(fk[:,0]):
                    lnu[j, :] += lnu0[kk,:]
                for kk,j in enumerate(fk[:,1]):
                    lnu[j, :] += lnu0[kk,:]
                for kk,j in enumerate(fk[:,2]):
                    lnu[j, :] += lnu0[kk,:]
                #print lnu.shape
                dxcval[k][t] = self.param.KparDiff.applyDiffKT(z, a, lnu, firstVar=x)
                dxcval[self.nsurf][t][npt:npt1, :] += \
                    (self.param.KparDiff.applyDiffKT(x, lnu, a, firstVar=z)
                     - self.param.KparDiffOut.applyDiffKT(zB, lnu, aB, firstVar=z))
                dxcval[self.nsurf][t] -= self.param.KparDiffOut.applyDiffKT(z, aB, lnu, firstVar=zB)
                dxcval[self.nsurf][t][npt:npt1, :] += np.dot(lnu, A)
                if self.useKernelDotProduct:
                    Kxx = self.param.KparDiff.getK(x)
                    dacval[k][t] = spLA.solve(Kxx, self.param.KparDiff.applyK(z, lnu, firstVar=x))
                else:
                    dacval[k][t] = self.param.KparDiff.applyK(z, lnu, firstVar=x)
                if self.affineDim > 0:
                    dAffcval[k][t, :] = (np.dot(self.affineBasis.T, np.vstack([np.dot(lnu.T, z).reshape([dim2,1]), lnu.sum(axis=0).reshape([self.dim,1])]))).flatten()
                if self.useKernelDotProduct:
                    dacval[self.nsurf][t][npt:npt1] = -lnu #self.param.KparDiffOut.applyK(z, lnu, firstVar=zB)
                else:
                    dacval[self.nsurf][t] -= self.param.KparDiffOut.applyK(z, lnu, firstVar=zB)
                lvf = np.multiply(dvf, lmb[t, nf:nf1].reshape([self.nf[k],1]))
                lvf /= normNu.reshape([nu.shape[0], 1])
                lvf -= np.multiply(nu, np.multiply(nu, lvf).sum(axis=1).reshape([nu.shape[0], 1]))
                #lvf = lv[fk[:,0]] + lv[fk[:,1]] + lv[fk[:,2]]
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
                dxcval[self.nsurf][t][npt:npt1, :] -= dnu 
                # for f in self.fv0[k].faces:
                #     lvf = lv[f[0]] + lv[f[1]] + lv[f[2]]
                #     dxcval[kk][t][npt+f[0]] = dxcval[kk][t][npt+f[0]] - np.cross(z[f[2]].T - z[f[1]].T, lvf)
                #     dxcval[kk][t][npt+f[1]] = dxcval[kk][t][npt+f[1]] - np.cross(z[f[0]].T - z[f[2]].T, lvf)
                #     dxcval[kk][t][npt+f[2]] = dxcval[kk][t][npt+f[2]] - np.cross(z[f[1]].T - z[f[0]].T, lvf)
                npt = npt1
                nf = nf1

                #obj += timeStep * (- np.multiply(self.lmb[t, :], cval[t,:]).sum() + np.multiply(cval[t, :], cval[t, :]).sum()/(2*self.mu))
        return lmb, dxcval, dnucval, dacval, dAffcval

    def constraintTermGradSlidingNormal(self, xt, nut, at, Afft):
        lmb = np.zeros(self.cval.shape)
        dxcval = []
        dnucval = []
        dacval = []
        dAffcval = []
        for u in xt:
            dxcval.append(np.zeros(u.shape))
        for u in nut:
            dnucval.append(np.zeros(u.shape))
        for u in at:
            dacval.append(np.zeros(u.shape))
        for u in Afft:
            dAffcval.append(np.zeros(u.shape))
        dim2 = self.dim**2
        for t in range(self.Tsize):
            zB = np.squeeze(xt[self.nsurf][t, :, :])
            aB = np.squeeze(at[self.nsurf][t, :, :])
            npt = 0
            for k in range(self.nsurf):
                a = at[k][t]
                x = np.squeeze(xt[k][t, :, :])
                if self.affineDim > 0:
                    AB = np.dot(self.affineBasis, Afft[k][t]) 
                    A = AB[0:dim2].reshape([self.dim, self.dim])
                    b = AB[dim2:dim2+self.dim]
                else:
                    A = np.zeros([self.dim, self.dim])
                    b = np.zeros(self.dim)
                nu = np.squeeze(nut[k][t, :, :])
                npt1 = npt + self.npt[k]
                r = self.param.KparDiff.applyK(x, a)
                r2 = self.param.KparDiffOut.applyK(zB, aB, firstVar=x)
                lmb[t, npt:npt1] = self.lmb[t, npt:npt1] - np.squeeze(np.multiply(nu, r-r2).sum(axis=1))/self.mu
                lnu = np.multiply(nu, lmb[t, npt:npt1].reshape([self.npt[k], 1]))
                dxcval[k][t] = (self.param.KparDiff.applyDiffKT(x, lnu, a)
                                + self.param.KparDiff.applyDiffKT(x, a, lnu)
                                - self.param.KparDiffOut.applyDiffKT(zB, lnu, aB, firstVar=x))
                dxcval[k][t] += np.dot(lnu, A)
                dxcval[self.nsurf][t] -= self.param.KparDiffOut.applyDiffKT(x, aB, lnu, firstVar=zB)
                dacval[k][t] = self.param.KparDiff.applyK(x, lnu)
                dacval[self.nsurf][t] -= self.param.KparDiffOut.applyK(x, lnu, firstVar=zB) 
                dAffcval[k][t] = np.dot(self.affineBasis.T, np.concatenate(np.dot(lnu.T, x).flatten(), lnu.sum(axis=0)))
                dnucval[k][t] = np.multiply(r-r2, lmb[t, npt:npt1].reshape([self.npt[k], 1]))
                npt = npt1

                #obj += timeStep * (- np.multiply(self.lmb[t, :], cval[t,:]).sum() + np.multiply(cval[t, :], cval[t, :]).sum()/(2*self.mu))
        return lmb, dxcval, dnucval, dacval, dAffcval

    def testConstraintTerm(self, xt, nut, at, Afft):
        xtTry = []
        nutTry = []
        atTry = []
        AfftTry = []
        eps = 0.00000001
        for k in range(self.nsurf):
            xtTry.append(xt[k] + eps*np.random.randn(self.Tsize+1, self.npt[k], self.dim))
        xtTry.append(xt[self.nsurf] + eps*np.random.randn(self.Tsize+1, self.npoints, self.dim))
        for k in range(self.nsurf):
            nutTry.append(nut[k] + eps*np.random.randn(self.Tsize+1, self.npt[k], self.dim))
            #nutTry.append(nut[self.nsurf] + eps*np.random.randn(self.Tsize+1, self.npoints, self.dim))
        for k in range(self.nsurf):
            atTry.append(at[k] + eps*np.random.randn(self.Tsize, self.npt[k], self.dim))
        atTry.append(at[self.nsurf] + eps*np.random.randn(self.Tsize, self.npoints, self.dim))

        if self.affineDim > 0:
            for k in range(self.nsurf):
                AfftTry.append(Afft[k] + eps*np.random.randn(self.Tsize, self.affineDim))
            

        u0 = self.constraintTerm(xt, nut, at, Afft)
        ux = self.constraintTerm(xtTry, nut, at, Afft)
        un = self.constraintTerm(xt, nutTry, at, Afft)
        ua = self.constraintTerm(xt, nut, atTry, Afft)
        [l, dx, dnu, da, dA] = self.constraintTermGrad(xt, nut, at, Afft)
        vx = 0
        for k in range(self.nsurf+1):
            vx += np.multiply(dx[k], xtTry[k]-xt[k]).sum()/eps
        vn = 0
        for k in range(self.nsurf):
            vn += np.multiply(dnu[k], nutTry[k]-nut[k]).sum()/eps
        va = 0
        for k in range(self.nsurf+1):
            va += np.multiply(da[k], atTry[k]-at[k]).sum()/eps
        logging.info('Testing constraints:')
        logging.info('var x: %f %f' %( self.Tsize*(ux[0]-u0[0])/(eps), -vx)) 
        logging.info('var nu: %f %f' %(self.Tsize*(un[0]-u0[0])/(eps), -vn ))
        logging.info('var a: %f %f' %( self.Tsize*(ua[0]-u0[0])/(eps), -va)) 
        if self.affineDim > 0:
            uA = self.constraintTerm(xt, nut, at, AfftTry)
            vA = 0
            for k in range(self.nsurf):
                vA += np.multiply(dA[k], AfftTry[k]-Afft[k]).sum()/eps
            logging.info('var affine: %f %f' %(self.Tsize*(uA[0]-u0[0])/(eps), -vA ))

    def  objectiveFunDef(self, at, Afft, withTrajectory = False, withJacobian = False):
        param = self.param
        timeStep = 1.0/self.Tsize
        xt = []
        nut = []
        dim2 = self.dim**2
        if withJacobian:
            Jt = []
            for k in range(self.nsurf):
                A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
                if self.affineDim > 0:
                    for t in range(self.Tsize):
                        AB = np.dot(self.affineBasis, Afft[k][t]) 
                        A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                        A[1][t] = AB[dim2:dim2+self.dim]
                xJ = evol.landmarkDirectEvolutionEuler(self.x0[k], at[k], param.KparDiff, affine = A, withNormals = self.nu0[k], withJacobian = True)
                xt.append(xJ[0])
                nut.append(xJ[1])
                Jt.append(xJ[2])
            xJ = evol.landmarkDirectEvolutionEuler(self.x0[self.nsurf], at[self.nsurf], param.KparDiffOut, withJacobian=True)
            xt.append(xJ[0])
            Jt.append(xJ[1])
        else:
            for k in range(self.nsurf):
                A = [np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])]
                if self.affineDim > 0:
                    for t in range(self.Tsize):
                        AB = np.dot(self.affineBasis, Afft[k][t]) 
                        A[0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                        A[1][t] = AB[dim2:dim2+self.dim]
                xJ = evol.landmarkDirectEvolutionEuler(self.x0[k], at[k], param.KparDiff, withNormals = self.nu0[k], affine = A)
                xt.append(xJ[0])
                nut.append(xJ[1])
            xt.append(evol.landmarkDirectEvolutionEuler(self.x0[self.nsurf], at[self.nsurf], param.KparDiffOut))
        #print xt[-1, :, :]
        #print obj
        obj=0
        for t in range(self.Tsize):
            zB = np.squeeze(xt[self.nsurf][t, :, :])
            for k in range(self.nsurf):
                z = np.squeeze(xt[k][t, :, :]) 
                a = np.squeeze(at[k][t, :, :])
                ra = param.KparDiff.applyK(z,a)
                obj += self.regweight*timeStep*np.multiply(a, ra).sum()
                if self.affineDim > 0:
                    obj +=  timeStep * np.multiply(self.affineWeight.reshape(Afft[k][t].shape), Afft[k][t]**2).sum()
                #print t,k,obj

            a = np.squeeze(at[self.nsurf][t, :, :])
            ra = param.KparDiffOut.applyK(zB,a)
            obj += self.regweightOut*timeStep*np.multiply(a, ra).sum()

            #print 'obj before constraints:', obj
        cstr = self.constraintTerm(xt, nut, at, Afft)
        obj += cstr[0]

        if withJacobian:
            return obj, xt, Jt, cstr[1]
        elif withTrajectory:
            return obj, xt, cstr[1]
        else:
            return obj

    def objectiveFun(self):
        if self.obj == None:
            self.obj0 = 0
            for fv1 in self.fv1:
                self.obj0 += self.param.fun_obj0(fv1, self.param.KparDist) / (self.param.sigmaError**2)

            (self.obj, self.xt, self.cval) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            selfobj = 2*self.obj0

            npt = 0
            for k in range(self.nsurf):
                npt1 = npt + self.npt[k]
                self.fvDefB[k].updateVertices(np.squeeze(self.xt[-1][self.Tsize, npt:npt1, :]))
                selfobj += self.param.fun_obj(self.fvDefB[k], self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
                self.fvDef[k].updateVertices(np.squeeze(self.xt[k][self.Tsize, :, :]))
                selfobj += self.param.fun_obj(self.fvDef[k], self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
                npt = npt1

                #print 'Deformation based:', self.obj, 'data term:', selfobj, self.regweightOut
            self.obj += selfobj

        return self.obj

    def getVariable(self):
        return [self.at, self.Afft]

    def updateTry(self, dir, eps, objRef=None):
        atTry = []
        for k in range(self.nsurf+1):
            atTry.append(self.at[k] - eps * dir[k].diff)
        AfftTry = []
        if self.affineDim > 0:
            for k in range(self.nsurf):
                AfftTry.append(self.Afft[k] - eps * dir[k].aff)
        foo = self.objectiveFunDef(atTry, AfftTry, withTrajectory=True)
        objTry = 0

        ff = []
        npt = 0
        for k in range(self.nsurf):
            npt1 = npt + self.npt[k]
            ff = surfaces.Surface(surf=self.fvDef[k])
            ff.updateVertices(np.squeeze(foo[1][self.nsurf][self.Tsize, npt:npt1, :]))
            objTry += self.param.fun_obj(ff, self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
            ff.updateVertices(np.squeeze(foo[1][k][self.Tsize, :, :]))
            objTry +=  self.param.fun_obj(ff, self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
            npt = npt1
            #print 'Deformation based:', foo[0], 'data term:', objTry+self.obj0
        objTry += foo[0]+2*self.obj0

        if np.isnan(objTry):
            logging.warning('Warning: nan in updateTry')
            return 1e500


        if (objRef == None) | (objTry < objRef):
            self.atTry = atTry
            self.AfftTry = AfftTry
            self.objTry = objTry
            self.cval = foo[2]


        return objTry


    def covectorEvolution(self, at, Afft, px1, pnu1):
        M = self.Tsize
        timeStep = 1.0/M
        xt = []
        nut = []
        pxt = []
        pnut = []
        A = []
        dim2 = self.dim**2
        for k in range(self.nsurf):
            A.append([np.zeros([self.Tsize, self.dim, self.dim]), np.zeros([self.Tsize, self.dim])])
            if self.affineDim > 0:
                for t in range(self.Tsize):
                    AB = np.dot(self.affineBasis, Afft[k][t]) 
                    A[k][0][t] = AB[0:dim2].reshape([self.dim, self.dim])
                    A[k][1][t] = AB[dim2:dim2+self.dim]
            xJ = evol.landmarkDirectEvolutionEuler(self.x0[k], at[k], self.param.KparDiff, withNormals = self.nu0[k], affine = A[k])
            xt.append(xJ[0])
            nut.append(xJ[1])
            pxt.append(np.zeros([M, self.npt[k], self.dim]))
            pnut.append(np.zeros([M, self.npt[k], self.dim]))
            #print pnu1[k].shape
            pxt[k][M-1, :, :] = px1[k]
            pnut[k][M-1] = pnu1[k]
        
        xt.append(evol.landmarkDirectEvolutionEuler(self.x0[self.nsurf], at[self.nsurf], self.param.KparDiffOut))
        pxt.append(np.zeros([M, self.npoints, self.dim]))
        pxt[self.nsurf][M-1, :, :] = px1[self.nsurf]
        #lmb = np.zeros([self.npoints, self.dim])
        foo = self.constraintTermGrad(xt, nut, at, Afft)
        lmb = foo[0]
        dxcval = foo[1]
        if self.typeConstraint == 'slidingNormal':
            dnucval = foo[2]
        dacval = foo[3]
        dAffcval = foo[4]
        
        for k in range(self.nsurf):
            pxt[k][M-1, :, :] += timeStep * dxcval[k][M]
            if self.typeConstraint == 'slidingNormal':
                pnut[k][M-1, :, :] += timeStep * dnucval[k][M]
        pxt[self.nsurf][M-1, :, :] += timeStep * dxcval[self.nsurf][M]
        
        for t in range(M-1):
            npt = 0
            for k in range(self.nsurf):
                npt1 = npt + self.npt[k]
                px = np.squeeze(pxt[k][M-t-1, :, :])
                z = np.squeeze(xt[k][M-t-1, :, :])
                a = np.squeeze(at[k][M-t-1, :, :])
                zpx = np.copy(dxcval[k][M-t-1])
                # a1 = np.concatenate((px[np.newaxis,...], a[np.newaxis,...], -2*self.regweight*a[np.newaxis,...]))
                # a2 = np.concatenate((a[np.newaxis,...], px[np.newaxis,...], a[np.newaxis,...]))
                #a2 = np.array([a, px, a])
                zpx += self.param.KparDiff.applyDiffKT(z, px, a, regweight=self.regweight, lddmm=True)
                if self.affineDim > 0:
                    zpx += np.dot(px, A[k][0][M-t-1])
                if self.typeConstraint == 'slidingNormal':
                    pnu = np.squeeze(pnut[k][M-t-1, :, :])
                    nu = np.squeeze(nut[k][M-t-1, :, :])
                    zpx -= (self.param.KparDiff.applyDDiffK11(z, nu, a, pnu) +
                            self.param.KparDiff.applyDDiffK12(z, a, nu, pnu))
                    zpnu = -self.param.KparDiff.applyDiffK(z, a, pnu) + dnucval[k][M-t-1]
                    pnut[k][M-t-2, :, :] = np.squeeze(pnut[k][M-t-1, :, :]) + timeStep * zpnu
                    
                pxt[k][M-t-2, :, :] = np.squeeze(pxt[k][M-t-1, :, :]) + timeStep * zpx
                npt = npt1
            px = np.squeeze(pxt[self.nsurf][M-t-1, :, :])
            z = np.squeeze(xt[self.nsurf][M-t-1, :, :])
            a = np.squeeze(at[self.nsurf][M-t-1, :, :])
            zpx = np.copy(dxcval[self.nsurf][M-t-1])
            # a1 = np.concatenate((px[np.newaxis,...], a[np.newaxis,...], -2*self.regweightOut*a[np.newaxis,...]))
            # a2 = np.concatenate((a[np.newaxis,...], px[np.newaxis,...], a[np.newaxis,...]))
            #a1 = [px, a, -2*self.regweightOut*a]
            #a2 = [a, px, a]
            zpx += self.param.KparDiffOut.applyDiffKT(z, px, a, regweight=self.regweightOut, lddmm=True)
            pxt[self.nsurf][M-t-2, :, :] = np.squeeze(pxt[self.nsurf][M-t-1, :, :]) + timeStep * zpx
            

        return pxt, pnut, xt, nut, dacval, dAffcval


    def HamiltonianGradient(self, at, Afft, px1, pnu1, getCovector = False):
        (pxt, pnut, xt, nut, dacval, dAffcval) = self.covectorEvolution(at, Afft, px1, pnu1)

        dat = []
        if self.useKernelDotProduct:
            for k in range(self.nsurf+1):
                dat.append(2*self.regweight*at[k] - pxt[k] - dacval[k])
        else:
            for k in range(self.nsurf+1):
                dat.append(-dacval[k])
            for t in range(self.Tsize):
                npt = 0
                for k in range(self.nsurf):
                    npt1 = npt + self.npt[k]
                    #print k, at[k][t].shape, pxt[k][t].shape, npt, npt1, dat[k][t].shape
                    dat[k][t] += self.param.KparDiff.applyK(xt[k][t], 2*self.regweight*at[k][t] - pxt[k][t])
                    if self.typeConstraint == 'slidingNormal':
                        dat[k][t] -= self.param.KparDiff.applyDiffK2(xt[k][t], nut[k][t], pnut[k][t])
                    npt=npt1
                dat[self.nsurf][t] += self.param.KparDiffOut.applyK(xt[self.nsurf][t], 2*self.regweightOut*at[self.nsurf][t] - pxt[self.nsurf][t])
        dAfft = []
        if self.affineDim > 0:
            for k in range(self.nsurf):
                dAfft.append(2*np.multiply(self.affineWeight, Afft[k]) - dAffcval[k])
                for t in range(self.Tsize):
                    dA = np.dot(pxt[k][t].T, xt[k][t]).reshape([self.dim**2, 1])
                    db = pxt[k][t].sum(axis=0).reshape([self.dim,1]) 
                    dAff = np.dot(self.affineBasis.T, np.vstack([dA, db]))
                    dAfft[k][t] -=  dAff.reshape(dAfft[k][t].shape)
                dAfft[k] = np.divide(dAfft[k], self.affineWeight)
 
        if getCovector == False:
            return dat, dAfft, xt, nut
        else:
            return dat, dAfft, xt, nut, pxt, pnut

    def endPointGradient(self):
        px1 = []
        pxB = np.zeros([self.npoints, self.dim])
        npt = 0 
        for k in range(self.nsurf):
            npt1 = npt + self.npt[k]
            px = -self.param.fun_objGrad(self.fvDef[k], self.fv1[k], self.param.KparDist) / self.param.sigmaError**2
            pxB[npt:npt1, :] = -self.param.fun_objGrad(self.fvDefB[k], self.fv1[k], self.param.KparDist) / self.param.sigmaError**2
            px1.append(px)
            npt = npt1
        px1.append(pxB)
        return px1

    def addProd(self, dir1, dir2, beta):
        dir = []
        for k in range(self.nsurf):
            dir.append(surfaceMatching.Direction())
            dir[k].diff = dir1[k].diff + beta * dir2[k].diff
            if self.affineDim > 0:
                dir[k].aff = dir1[k].aff + beta * dir2[k].aff
        dir.append(surfaceMatching.Direction())
        dir[-1].diff = dir1[-1].diff + beta * dir2[-1].diff
        #     for k in range(self.nsurf):
        #         dir[1].append(dir1[1][k] + beta * dir2[1][k])
        return dir

    def copyDir(self, dir0):
        dir = []
        for d in dir0:
            dir.append(surfaceMatching.Direction())
            dir[-1].diff = np.copy(d.diff)
            dir[-1].aff = np.copy(d.aff)
        return dir

        
    def kernelDotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        for k in range(self.nsurf):
            for t in range(self.Tsize):
                z = np.squeeze(self.xt[k][t, :, :])
                gg = np.squeeze(g1[k].diff[t, :, :])
                u = self.param.KparDiff.applyK(z, gg)
                if self.affineDim > 0:
                    uu = np.multiply(g1[k].aff[t], self.affineWeight)
                ll = 0
                for gr in g2:
                    ggOld = np.squeeze(gr[k].diff[t, :, :])
                    res[ll]  +=  np.multiply(ggOld,u).sum()
                    if self.affineDim > 0:
                        res[ll] += np.multiply(uu, gr[k].aff[t]).sum()
                    ll = ll + 1
      
        for t in range(self.Tsize):
            z = np.squeeze(self.xt[self.nsurf][t, :, :])
            gg = np.squeeze(g1[self.nsurf].diff[t, :, :])
            u = self.param.KparDiffOut.applyK(z, gg)
            ll = 0
            for gr in g2:
                ggOld = np.squeeze(gr[self.nsurf].diff[t, :, :]) 
                res[ll]  +=  np.multiply(ggOld,u).sum() / self.coeffZ
                ll = ll + 1

        return res

    def standardDotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        dim2 = self.dim**2
        for ll,gr in enumerate(g2):
            res[ll]=0
            for k in range(self.nsurf):
                res[ll] += np.multiply(g1[k].diff, gr[k].diff).sum()
                if self.affineDim > 0:
                    uu = np.multiply(g1[k].aff, self.affineWeight)
                    res[ll] += np.multiply(uu, gr[k].aff).sum()
                    #+np.multiply(g1[1][k][:, dim2:dim2+self.dim], gr[1][k][:, dim2:dim2+self.dim]).sum())
            res[ll] += np.multiply(g1[self.nsurf].diff, gr[self.nsurf].diff).sum() / self.coeffZ
        return res



    def getGradient(self, coeff=1.0):
        px1 = self.endPointGradient()
        #px1.append(np.zeros([self.npoints, self.dim]))
        pnu1 = []
        for k in range(self.nsurf):
            pnu1.append(np.zeros(self.nu0[k].shape))
            #for p in px1:
            #print p.sum()
        foo = self.HamiltonianGradient(self.at, self.Afft, px1, pnu1)
        grd = []
        for kk in range(self.nsurf):
            grd.append(surfaceMatching.Direction())
            grd[kk].diff = foo[0][kk] / (coeff*self.Tsize)
            if self.affineDim > 0:
                grd[kk].aff = foo[1][kk] / (coeff*self.Tsize)
        grd.append(surfaceMatching.Direction())
        grd[self.nsurf].diff = foo[0][self.nsurf] * (self.coeffZ/(coeff*self.Tsize))
        return grd

    def randomDir(self):
        dirfoo = []
        for k in range(self.nsurf):
            dirfoo.append(surfaceMatching.Direction())
            dirfoo[k].diff = np.random.randn(self.Tsize, self.npt[k], self.dim)
            if self.affineDim > 0:
                dirfoo[k].aff = np.random.randn(self.Tsize, self.affineDim)
        dirfoo.append(surfaceMatching.Direction())
        dirfoo[-1].diff = np.random.randn(self.Tsize, self.npoints, self.dim)
        return dirfoo

    def acceptVarTry(self):
        self.obj = self.objTry
        self.at = []
        for a in self.atTry:
            self.at.append(np.copy(a))
        self.Afft = []
        for a in self.AfftTry:
            self.Afft.append(np.copy(a))

    def endOfIteration(self):
        (obj1, self.xt, Jt, self.cval) = self.objectiveFunDef(self.at, self.Afft, withJacobian=True)
        logging.info('Time: %f (CPU: %f)  mean constraint %f %f' %(time.time() - self.startTime, time.clock() - self.startTimeClock, np.sqrt((self.cval**2).sum()/self.cval.size), np.fabs(self.cval).sum() / self.cval.size))
        #self.testConstraintTerm(self.xt, self.nut, self.at, self.Afft)
        nn = 0 ;
        for k in range(self.nsurf):
            AV0 = self.fv0[k].computeVertexArea()
            n1 = self.xt[k].shape[1] ;
            for kk in range(self.Tsize+1):
                self.fvDefB[k].updateVertices(np.squeeze(self.xt[-1][kk, nn:nn+n1, :]))
                AV = self.fvDefB[k].computeVertexArea()
                AV = (AV[0]/AV0[0])
                #self.fvDefB[k].saveVTK(self.outputDir +'/'+ self.saveFile+str(k)+'Out'+str(kk)+'.vtk', scalars = Jt[-1][kk, nn:nn+n1], scal_name='Jacobian')
                vf = surfaces.vtkFields() ;
                vf.scalars.append('Jacobian') ;
                vf.scalars.append(np.exp(Jt[-1][kk, nn:nn+n1]))
                vf.scalars.append('Jacobian_T') ;
                vf.scalars.append(AV[:,0])
                vf.scalars.append('Jacobian_N') ;
                vf.scalars.append(np.exp(Jt[-1][kk, nn:nn+n1])/(AV[:,0]))
                #self.fvDefB[k].saveVTK(self.outputDir +'/'+ self.saveFile+str(k)+'Out'+str(kk)+'.vtk', scalars = AV[:,0], scal_name='Jacobian')
                self.fvDefB[k].saveVTK2(self.outputDir +'/'+ self.saveFile+str(k)+'Out'+str(kk)+'.vtk', vf)
                self.fvDef[k].updateVertices(np.squeeze(self.xt[k][kk, :, :]))
                AV = self.fvDef[k].computeVertexArea()
                AV = (AV[0]/AV0[0])
                vf = surfaces.vtkFields() ;
                vf.scalars.append('Jacobian') ;
                vf.scalars.append(np.exp(Jt[k][kk, :]))
                vf.scalars.append('Jacobian_T') ;
                vf.scalars.append(AV[:,0])
                vf.scalars.append('Jacobian_N') ;
                vf.scalars.append(np.exp(Jt[k][kk, :])/(AV[:,0]))
                #self.fvDef[k].saveVTK(self.outputDir +'/'+self.saveFile+str(k)+'In'+str(kk)+'.vtk', scalars = Jt[k][kk, :], scal_name='Jacobian')
                #self.fvDef[k].saveVTK(self.outputDir +'/'+self.saveFile+str(k)+'In'+str(kk)+'.vtk', scalars = AV[:,0], scal_name='Jacobian')
                self.fvDef[k].saveVTK2(self.outputDir +'/'+self.saveFile+str(k)+'In'+str(kk)+'.vtk', vf)
            nn += n1

    def optimizeMatching(self):
        self.coeffZ = 1.
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = np.sqrt(grd2) / 100
        self.muEps = 1.0
        self.restartRate = 100
        it = 0
        self.startTimeClock = time.clock()
        self.startTime = time.time()
        while (self.muEps > 0.05) & (it<self.maxIter_al):
            logging.info('Starting Minimization: gradEps = %f muEps = %f mu = %f' %(self.gradEps, self.muEps,self.mu))
            #self.coeffZ = max(1.0, self.mu)
            cg.cg(self, verb = self.verb, maxIter = self.maxIter_cg, TestGradient = self.testGradient, epsInit=0.1)
            for t in range(self.Tsize+1):
                self.lmb[t, ...] -= self.cval[t, ...]/self.mu
            logging.info('mean lambdas %f' %(np.fabs(self.lmb).sum() / self.lmb.size))
            if self.converged:
                self.gradEps *= .75
                if (((self.cval**2).sum()/self.cval.size) > self.muEps**2):
                    self.mu *= 0.01
                else:
                    self.muEps = self.muEps /2
            else:
                self.mu *= 0.9
            self.obj = None
            it = it+1
            
            #return self.fvDef

