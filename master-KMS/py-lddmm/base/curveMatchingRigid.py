import os
import glob
import matplotlib
matplotlib.use("TKAgg")
import scipy.linalg as la
import scipy.optimize as sopt
from . import curves
from . import conjugateGradient as cg, grid, matchingParam, pointEvolution as evol, loggingUtils, bfgs
from .affineBasis import *
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from .kernelFunctions import Kernel
import logging
from tqdm import *

## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
class CurveMatchingRigidParam(matchingParam.MatchingParam):
    def __init__(self, timeStep = .1, algorithm = 'bfgs', Wolfe=False, KparDiff = None, KparDist = None, sigmaKernel = 6.5, sigmaDist=2.5, sigmaError=1.0,
                 errorType = 'measure', typeKernel='gauss', internalCost=None):
        matchingParam.MatchingParam.__init__(self, timeStep=timeStep, KparDiff = KparDiff, KparDist = KparDist, sigmaKernel = sigmaKernel, sigmaDist=sigmaDist,
                                             sigmaError=sigmaError, errorType = errorType, typeKernel=typeKernel)
          
        self.internalCost = internalCost
                                         
        if errorType == 'current':
            print('Running Current Matching')
            self.fun_obj0 = curves.currentNorm0
            self.fun_obj = curves.currentNormDef
            self.fun_objGrad = curves.currentNormGradient
        elif errorType=='measure':
            print('Running Measure Matching')
            self.fun_obj0 = curves.measureNorm0
            self.fun_obj = curves.measureNormDef
            self.fun_objGrad = curves.measureNormGradient
        elif errorType=='varifold':
            self.fun_obj0 = curves.varifoldNorm0
            self.fun_obj = curves.varifoldNormDef
            self.fun_objGrad = curves.varifoldNormGradient
        elif errorType=='varifoldComponent':
            self.fun_obj0 = curves.varifoldNormComponent0
            self.fun_obj = curves.varifoldNormComponentDef
            self.fun_objGrad = curves.varifoldNormComponentGradient
        else:
            print('Unknown error Type: ', self.errorType)

class Direction:
    def __init__(self):
        self.skew = []
        self.trans = []


## Main class for curve matching
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
class CurveMatchingRigid:
    def __init__(self, Template=None, Target=None, Clamped=None, pltFrame = None,
                 fileTempl=None, fileTarg=None, param=None, maxIter=1000, regWeight = 1.0,
                 verb=True, gradLB = 0.001, saveRate=10, saveTrajectories=False, parpot = None, paramRepell = None,
                 testGradient=False, saveFile = 'evolution', outputDir = '.', pplot=True):
        self.iter = 0
        self.maxIter = maxIter
        self.verb = verb
        self.testGradient = testGradient
        self.regweight = regWeight
        self.minEig = 1e-10
        self.parpot = parpot
        self.rpot = 4.
        self.parRot = 0.
        self.parTrans = 0.
        self.paramRepell = paramRepell
        self.repellCosine = 0.
        self.pltFrame = pltFrame
        if Template is None:
            if fileTempl is None:
                #print 'Please provide a template curve'
                return
            else:
                self.fv0 = curves.Curve(curve=fileTempl)
        else:
            self.fv0 = curves.Curve(curve=Template)
        if Target is None:
            if fileTarg is None:
                print('Please provide a target curve')
                return
            else:
                self.fv1 = curves.Curve(curve=fileTarg)
        else:
            self.fv1 = curves.Curve(curve=Target)



        self.npt = self.fv0.vertices.shape[0]
        self.dim = self.fv0.vertices.shape[1]
        if not(Clamped is None):
            self.fvc = curves.Curve(curve=Clamped)
            self.xc = self.fvc.vertices
        else:
            self.fvc = curves.Curve()
            self.xc = np.zeros([0, self.dim])
        if self.dim == 2:
            xmin = min(self.fv0.vertices[:,0].min(), self.fv1.vertices[:,0].min())
            xmax = max(self.fv0.vertices[:,0].max(), self.fv1.vertices[:,0].max())
            ymin = min(self.fv0.vertices[:,1].min(), self.fv1.vertices[:,1].min())
            ymax = max(self.fv0.vertices[:,1].max(), self.fv1.vertices[:,1].max())
            if Clamped is not None:
                xmin = min(xmin, self.fvc.vertices[:,0].min())
                xmax = max(xmax, self.fvc.vertices[:,0].max())
                ymin = min(ymin, self.fvc.vertices[:,1].min())
                ymax = max(ymax, self.fvc.vertices[:,1].max())
            dx =  xmax-xmin
            dy = ymax-ymin
            self.gsize = 0.005
            self.gskip = int(0.1/self.gsize)
            DXY = 0.25 * max(dx, dy)
            dxy = self.gsize * min(dx,dy)
            #print xmin,xmax, dxy
            [x,y] = np.mgrid[(xmin-DXY):(xmax+DXY):dxy, (ymin-DXY):(ymax+DXY):dxy]
            #print x.shape
            self.gridDef = grid.Grid(gridPoints=[x, y])
            self.gridxy = np.copy(self.gridDef.vertices)


        self.nptc = self.fv0.vertices.shape[0] + self.xc.shape[0]
        if not(self.dim == 2):
            print('This program runs in 2D only')
            return
            
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print('Cannot save in ' + outputDir)
                return
            else:
                os.mkdir(outputDir)
        for f in glob.glob(outputDir+'/*.vtk'):
            os.remove(f)
        self.fvDef = curves.Curve(curve=self.fv0)

        if param==None:
            self.param = CurveMatchingRigidParam()
        else:
            self.param = param
        

        self.x0 = self.fv0.vertices
        ## Attributing connected components to vertices.
        ## *** THIS ASSUMES THAT COMPONENTS ARE DISCONNECTED
        self.ncomponent = self.fv0.component.max() + 1
        self.component = np.zeros(self.x0.shape[0], dtype=int)
        for k in range(self.fv0.faces.shape[0]):
            self.component[self.fv0.faces[k,0]] = self.fv0.component[k]
            self.component[self.fv0.faces[k,1]] = self.fv0.component[k]

        self.Ic = []
        for k in range(self.ncomponent):
            self.Ic.append(np.nonzero(self.component == k)[0])

        self.Tsize = int(round(1.0/self.param.timeStep))
        self.at = np.zeros([self.Tsize, self.ncomponent])
        self.taut = np.zeros([self.Tsize, self.ncomponent, 2])
        self.atTry = np.zeros([self.Tsize, self.ncomponent])
        self.tautTry = np.zeros([self.Tsize, self.ncomponent, 2])
        self.pxt = np.zeros([self.Tsize+1, self.npt, self.dim])
        self.xt = np.tile(self.fv0.vertices, [self.Tsize+1, 1, 1])
        self.obj = None
        self.objTry = None
        self.gradCoeff = self.fv0.vertices.shape[0]
        self.saveFile = saveFile
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        self.fv1.saveVTK(self.outputDir+'/Target.vtk')
        if not (Clamped is None):
            self.fvc.saveVTK(self.outputDir+'/Clamped.vtk')
        self.gradLB = gradLB
        self.saveRate = saveRate 
        self.saveTrajectories = saveTrajectories
        self.pplot = pplot
        if self.pplot:
            fig=plt.figure(1)
            fig.clf()
            ax = fig.gca()
            if len(self.xc) > 0:
                for kf in range(self.fvc.faces.shape[0]):
                    ax.plot(self.fvc.vertices[self.fvc.faces[kf ,:] ,0] ,
                            self.fvc.vertices[self.fvc.faces[kf ,:] ,1] ,color=[0 ,0 ,0])
            for kf in range(self.fv1.faces.shape[0]):
                ax.plot(self.fv1.vertices[self.fv1.faces[kf,:],0], self.fv1.vertices[self.fv1.faces[kf,:],1], color=[0,0,1])
            for kf in range(self.fvDef.faces.shape[0]):
                ax.plot(self.fvDef.vertices[self.fvDef.faces[kf,:],0], self.fvDef.vertices[self.fvDef.faces[kf,:],1], color=[1,0,0], marker='*')
            plt.axis('equal')
            plt.pause(0.1)



    def dataTerm(self, _fvDef):
        obj = self.param.fun_obj(_fvDef, self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
        return obj

    def  objectiveFunDef(self, at, taut, withTrajectory = False, x0 = None):
        if x0 == None:
            x0 = self.fv0.vertices
        param = self.param
        timeStep = 1.0/self.Tsize
        xt  = self.directEvolutionEuler(x0, at, taut)
        obj=0
        for t in range(self.Tsize):
            z = np.squeeze(xt[t, :, :])
            zc = np.concatenate([z,self.xc])
            a = np.squeeze(at[t, :])
            tau = np.squeeze(taut[t, :, :])
            if self.ncomponent == 1:
                a = a[np.newaxis]
                tau = tau[np.newaxis, :]
            Jz = np.zeros(z.shape)
            Jz[:,0] = z[:,1]
            Jz[:,1] = -z[:,0]
            v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
            mu = self.solveK(z,v)[0:self.npt, :]
            obj += self.regweight * timeStep * (mu*v).sum()/2
            # v = np.concatenate([v,np.zeros([self.xc.shape[0], self.dim])])
            # K = param.KparDiff.getK(zc)
            # eK = la.eigh(K)
            # J = np.nonzero(eK[0]>self.minEig)[0]
            # w = np.dot(v.T,eK[1][:,J]).T
            # muv = ((w*w)/eK[0][J, np.newaxis]).sum()
            # obj += self.regweight * timeStep * muv/2
            if not (self.paramRepell is None):
                obj += self.costRepell(z,v) * timeStep

        if withTrajectory:
            return obj, xt
        else:
            return obj

    def costRepell(self, z, v):
        obj = 0
        if self.paramRepell is None:
            return obj
        nv = np.sqrt((v ** 2).sum(axis=1))
        for k in range(self.ncomponent):
            Ik = self.Ic[k]
            for l in range(self.ncomponent):
                if l != k:
                    Il = self.Ic[l]
                    delta = z[Il, np.newaxis, :] - z[np.newaxis, Ik, :]
                    d = np.sqrt((delta ** 2).sum(axis=2))
                    nvd = nv[np.newaxis,Ik]*d
                    dv = np.maximum((v[np.newaxis, Ik, :] * delta).sum(axis=2) - self.repellCosine * nvd, 0)
                    obj += ((dv ** 2) * (d ** (-self.rpot))).sum()
                    # dv = (v[np.newaxis, Ik, :] * delta).sum(axis=2)
                    # obj += (dv * (d ** (-self.rpot))).sum()
            if (len(self.xc) > 0):
                delta = self.xc[:, np.newaxis, :] - z[np.newaxis, Ik, :]
                d = np.sqrt((delta ** 2).sum(axis=2))
                nvd = nv[np.newaxis, Ik] * d
                dv = np.maximum((v[np.newaxis, Ik, :] * delta).sum(axis=2) - self.repellCosine * nvd, 0)
                obj += ((dv ** 2) * (d ** (-self.rpot))).sum()
                # dv = (v[np.newaxis, Ik, :] * delta).sum(axis=2)
                # obj += (dv * (d ** (-self.rpot))).sum()
        obj *= self.paramRepell /2
        return obj

    def gradRepell(self, z, a, tau):
        Jz = np.zeros(z.shape)
        Jz[:, 0] = z[:, 1]
        Jz[:, 1] = -z[:, 0]
        v = a[self.component, np.newaxis] * Jz + tau[self.component, :]
        return self.gradRepellZ(z,v)

    def gradRepellZ(self, z, v):
        grad = np.zeros(z.shape)
        if self.paramRepell is None:
            return grad
        #v = np.concatenate([v,np.zeros([self.xc.shape[0], self.dim])])
        nv = np.sqrt((v ** 2).sum(axis=1))
        for k in range(self.ncomponent):
            Ik = self.Ic[k]
            for l in range(self.ncomponent):
                if l != k:
                    Il = self.Ic[l]
                    delta = z[Il, np.newaxis, :] - z[np.newaxis, Ik, :]
                    d = np.sqrt((delta ** 2).sum(axis=2))
                    deltan = delta / np.maximum(d[:,:,np.newaxis], 1e-8)
                    nvdk = nv[np.newaxis, Ik] * d
                    nvdl = nv[Il, np.newaxis] * d
                    dvk = np.maximum((v[np.newaxis, Ik, :] * delta).sum(axis=2) - self.repellCosine * nvdk, 0)
                    dvl = np.maximum((-v[Il, np.newaxis, :] * delta).sum(axis=2) - self.repellCosine * nvdl, 0)
                    # dvk = (v[np.newaxis, Ik, :] * delta).sum(axis=2)
                    # dvl = (-v[Il, np.newaxis, :] * delta).sum(axis=2)
                    gvk = v[np.newaxis, Ik, :] - self.repellCosine * nv[np.newaxis, Ik, np.newaxis] * deltan
                    gvl = v[Il, np.newaxis,:] + self.repellCosine * nv[Il, np.newaxis, np.newaxis] * deltan
                    # gvk = v[np.newaxis, Ik, :]
                    # gvl = v[Il, np.newaxis,:]
                    d = d[:,:,np.newaxis]
                    grad[Ik,:] += (0.5 * self.rpot) * ((dvk**2+dvl**2)[:,:,np.newaxis]
                                                        * (delta / (d ** (self.rpot + 2)))).sum(axis=0)
                    grad[Ik, :] -= ((dvk[:,:,np.newaxis]*gvk - dvl[:,:,np.newaxis]*gvl)
                                     / (d ** (self.rpot))).sum(axis=0)
                    # grad[Ik,:] += self.rpot * ((dvk+dvl)[:,:,np.newaxis]
                    #                                    * (delta / (d ** (self.rpot + 2)))).sum(axis=0)
                    # grad[Ik, :] -= ((gvk - gvl) / (d ** (self.rpot))).sum(axis=0)
            if len(self.xc)>0:
                delta = self.xc[:, np.newaxis, :] - z[np.newaxis, Ik, :]
                d = np.sqrt((delta ** 2).sum(axis=2))
                deltan = delta / np.maximum(d[:, :, np.newaxis], 1e-8)
                nvdk = nv[np.newaxis, Ik] * d
                dvk = np.maximum((v[np.newaxis, Ik, :] * delta).sum(axis=2) - self.repellCosine * nvdk, 0)
                gvk = v[np.newaxis, Ik, :] - self.repellCosine * nv[np.newaxis, Ik, np.newaxis] * deltan
                # dvk = (v[np.newaxis, Ik, :] * delta).sum(axis=2)
                # gvk = v[np.newaxis, Ik, :]
                dv = np.maximum((v[np.newaxis, Ik, :] * delta).sum(axis=2), 0)
                d = d[:, :, np.newaxis]
                # grad[Ik,:] += self.rpot * (dvk[:,:,np.newaxis] * (delta/(d ** (self.rpot+2)))).sum(axis=0)
                # grad[Ik,:] -= (gvk / (d ** (self.rpot))).sum(axis=0)
                grad[Ik,:] += (0.5*self.rpot) * ((dvk[:,:,np.newaxis]**2) * (delta/(d ** (self.rpot+2)))).sum(axis=0)
                grad[Ik,:] -= ((dvk[:, :, np.newaxis] * gvk) / (d ** (self.rpot))).sum(axis=0)
            grad *= self.paramRepell
        return grad

    def gradRepellV(self, z, v):
        jac = np.zeros(v.shape)
        if self.paramRepell is None:
            return jac

        nv = np.sqrt((v ** 2).sum(axis=1))
        vn = v / np.maximum(nv[:, np.newaxis], 1e-8)
        for k in range(self.ncomponent):
            Ik = self.Ic[k]
            for l in range(self.ncomponent):
                if k != l:
                    Il = self.Ic[l]
                    delta = z[Il, np.newaxis, :] - z[np.newaxis, Ik, :]
                    d = np.sqrt((delta ** 2).sum(axis=2))
                    nvdk = nv[np.newaxis, Ik] * d
                    dvk = np.maximum((v[np.newaxis, Ik, :] * delta).sum(axis=2) - self.repellCosine * nvdk, 0)
                    gvk = delta - self.repellCosine * d[:,:,np.newaxis] * vn[np.newaxis,Ik,:]
                    dvk = (v[np.newaxis, Ik, :] * delta).sum(axis=2)
                    #gvk = delta
                    jac[Ik, :] += ((dvk[:, :, np.newaxis] * gvk) / (d ** self.rpot)[..., np.newaxis]).sum(axis=0)
                    # jac[Ik, :] += (gvk / (d ** self.rpot)[..., np.newaxis]).sum(axis=0)
            if len(self.xc>0):
                delta = self.xc[:, np.newaxis, :] - z[np.newaxis, Ik, :]
                d = np.sqrt((delta ** 2).sum(axis=2))
                nvdk = nv[np.newaxis, Ik] * d
                dvk = np.maximum((v[np.newaxis, Ik, :] * delta).sum(axis=2) - self.repellCosine * nvdk, 0)
                gvk = delta - self.repellCosine * d[:,:,np.newaxis] * vn[np.newaxis, Ik, :]
                jac[Ik, :] += ((dvk[:, :, np.newaxis] * gvk) / (d ** self.rpot)[..., np.newaxis]).sum(axis=0)
                # jac[Ik, :] += (delta / (d ** self.rpot)[..., np.newaxis]).sum(axis=0)

            jac *= self.paramRepell
        return jac

    def testRepellGrad(self, z, v):
        obj0 = self.costRepell(z,v)
        eps = 1e-8
        dv = np.random.normal(0,1,v.shape)
        dz = np.random.normal(0,1,z.shape)
        grd = self.gradRepellV(z,v)
        grd2 = self.gradRepellZ(z,v)
        obj1 = self.costRepell(z, v+eps*dv)
        obj2 = self.costRepell(z+eps*dz, v)
        print('test repell grad v:', (obj1-obj0)/eps, (grd*dv).sum())
        print('test repell grad z:', (obj2-obj0)/eps, (grd2*dz).sum())


    def objectiveFunRigid(self, xt, at, taut, timeStep = None):
        if timeStep is None:
            timeStep = 1.0/self.Tsize
        obj = self.parRot * (at**2).sum()
        for t in range(at.shape[0]):
            for k in range(self.ncomponent):
                J = self.Ic[k]
                c = np.mean(xt[t,J,:], axis=0)
                obj +=  self.parTrans * ((taut[t,k,0] + at[t,k]*c[1])**2 + (taut[t,k,1] - at[t,k]*c[0])**2)
        return timeStep*obj/2

    def gradRigid(self, z, a, tau):
        grad = np.zeros(z.shape)
        for k in range(self.ncomponent):
            J = np.nonzero(self.component == k)[0]
            c = np.mean(z[J, :], axis=0)
            grad[J,0] = self.parTrans*a[k]*(a[k]*c[0]-tau[k,1])/len(J)
            grad[J,1] = self.parTrans*a[k]*(a[k]*c[1]+tau[k,0])/len(J)
        return grad

    def testGradRigid(self,z, a, tau):
        dz = np.random.normal(0,1,z.shape)
        eps = 1e-8
        z2 = z + eps * dz
        obj0 = self.objectiveFunRigid(z[np.newaxis,...], a[np.newaxis,...], tau[np.newaxis,...], timeStep=1.)
        obj = self.objectiveFunRigid(z2[np.newaxis,...], a[np.newaxis,...], tau[np.newaxis,...], timeStep=1.)
        grad = self.gradRigid(z, a, tau)
        print('testGradRigid:', (obj-obj0)/eps, (grad*dz).sum())

    def objectiveFunPotential(self, xt, timeStep=None):
        if self.parpot is None:
            return 0
        obj = 0
        if timeStep is None:
            timeStep = 1.0/self.Tsize
        for k in range(self.ncomponent):
            Ik = np.nonzero(self.component == k)[0]
            xk = xt[:,Ik,:]
            for l in range(k + 1, self.ncomponent):
                Il = np.nonzero(self.component == l)[0]
                xl = xt[:,Il,:]
                for t in range(xt.shape[0]):
                    delta = xk[t, :, np.newaxis, :] - xl[t, np.newaxis, :, :]
                    d = np.sqrt(((delta) ** 2).sum(axis=2))
                    obj += (d ** (-self.rpot)).sum()
            if len(self.xc>0):
                for t in range(xt.shape[0]):
                    d = np.sqrt(((xk[t, :, np.newaxis, :] - self.xc[np.newaxis, :, :]) ** 2).sum(axis=2))
                    obj += (d ** (-self.rpot)).sum()
        return self.parpot * obj * timeStep

    def gradPotential(self, z):
        grad = np.zeros(z.shape)
        if self.parpot is None:
            return grad
        for k in range(self.ncomponent):
            Ik = self.Ic[k]
            for l in range(self.ncomponent):
                if l != k:
                    Il = self.Ic[l]
                    delta = z[Ik, np.newaxis, :] - z[np.newaxis, Il, :]
                    d = np.sqrt(((delta) ** 2).sum(axis=2))[:,:,np.newaxis]
                    grad[Ik,:] += (delta/(d ** (self.rpot+2))).sum(axis=1)
            if len(self.xc)>0:
                delta = z[Ik, np.newaxis, :] - self.xc[np.newaxis, :, :]
                d = np.sqrt(((delta) ** 2).sum(axis=2))[:,:,np.newaxis]
                grad[Ik,:] += (delta/(d ** (self.rpot+2))).sum(axis=1)
        return -self.rpot*self.parpot * grad

    def testGradPotential(self,z):
        dz = np.random.normal(0,1,z.shape)
        eps = 1e-8
        z2 = z + eps * dz
        obj0 = self.objectiveFunPotential(z[np.newaxis,...])
        obj = self.objectiveFunPotential(z2[np.newaxis,...])
        grad = self.gradPotential(z)
        print('testGradPotential:', self.Tsize*(obj-obj0)/eps, (grad*dz).sum())

    def  _objectiveFun(self, at, taut, withTrajectory = False):
        (obj, xt) = self.objectiveFunDef(at, taut, withTrajectory=True)
        self.fvDef.updateVertices(np.squeeze(xt[-1, :, :]))
        obj0 = self.dataTerm(self.fvDef)
        #obj2 = self.objectiveFunPotential(xt)
        obj3 = self.objectiveFunRigid(xt, at, taut)

        if withTrajectory:
            return obj+obj0+obj3, xt
        else:
            return obj+obj0+obj3

    def objectiveFun(self):
        if self.obj is None:
            self.obj0 = self.param.fun_obj0(self.fv1, self.param.KparDist) / (self.param.sigmaError**2)
            (self.obj, self.xt) = self.objectiveFunDef(self.at, self.taut, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.obj += self.obj0 + self.dataTerm(self.fvDef) #+ self.objectiveFunPotential(self.xt)
            self.obj += self.objectiveFunRigid(self.xt, self.at, self.taut)
            #print self.obj0, self.obj

        return self.obj

    def getVariable(self):
        return [self.at, self.taut]

    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        atTry = self.at - eps * dir.skew
        tautTry = self.taut - eps * dir.trans
        foo = self.objectiveFunDef(atTry, tautTry, withTrajectory=True)
        objTry += foo[0]
        #objTry += self.objectiveFunPotential(foo[1])
        objTry += self.objectiveFunRigid(foo[1], atTry, tautTry)

        ff = curves.Curve(curve=self.fvDef)
        ff.updateVertices(np.squeeze(foo[1][-1, :, :]))
        objTry += self.dataTerm(ff)
        if np.isnan(objTry):
            print('Warning: nan in updateTry')
            return 1e500

        if (objRef == None) | (objTry < objRef):
            self.atTry = atTry
            self.objTry = objTry
            self.tautTry = tautTry

        return objTry

    def directEvolutionEuler(self, x0, at, taut, grid=None):
        xt = np.zeros([self.Tsize+1, x0.shape[0], x0.shape[1]])
        xt[0,:,:] = np.copy(x0)
        timeStep = 1.0/self.Tsize
        if grid is not None:
            yt = np.copy(grid)
        for t in range(self.Tsize):
            z = np.squeeze(xt[t, :, :])
            a = np.squeeze(at[t, :])
            tau = np.squeeze(taut[t, :, :])
            Jz = np.zeros(z.shape)
            ca = (np.cos(timeStep*a) - 1)/timeStep
            sa = np.sin(timeStep*a)/timeStep
            if self.ncomponent == 1:
                Jz[:,0] = ca*z[:,0] + sa*z[:,1]
                Jz[:,1] = -sa*z[:,0] + ca*z[:,1]
                vt = Jz+tau
            else:
                Jz[:,0] = ca[self.component]*z[:,0] + sa[self.component]*z[:,1]
                Jz[:,1] = -sa[self.component]*z[:,0] + ca[self.component]*z[:,1]
            #v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
                vt = Jz+tau[self.component,:]
            xt[t + 1, :, :] = z + timeStep * vt
            if grid is not None:
                ax = self.solveK(z, vt)
                yt += timeStep*self.param.KparDiff.applyK(np.concatenate((z, self.xc)), ax, firstVar=yt)
        if grid is None:
            return xt
        else:
            return xt, yt

    def solveK(self,z, v):
        zc = np.concatenate([z,self.xc])
        vc = np.concatenate([v,np.zeros(self.xc.shape)])
        K = self.param.KparDiff.getK(zc)
        try:
            eK = la.eigh(K)
        except Exception:
            raise Exception('Bad value in SolveK')
        J = np.nonzero(eK[0] > self.minEig)[0]
        #D = np.maximum(eK[0], self.minEig)
        w = np.dot(vc.T, eK[1][:, J]).T / eK[0][J, np.newaxis]
        #w = np.dot(vc.T, eK[1]).T / D[:, np.newaxis]
        mu = np.dot(eK[1][:, J], w)
        #mu = np.dot(eK[1], w)

        #mu = mu[0:self.npt,:]
        #print np.fabs(np.dot(K,mu)-v).max()
        #mu = la.solve(K,vc, sym_pos=True)
        return mu

    def center(self,z, kernel = None):
        if kernel is None:
            K = self.param.KparDiff.getK(z)
        else:
            K = kernel.getK(z)
        try:
            eK = la.eigh(K)
        except Exception:
            raise Exception('Bad value in SolveK')
        J = np.nonzero(eK[0] > self.minEig)[0]
        #D = np.maximum(eK[0], self.minEig)
        w = np.dot(z.T, eK[1][:, J]).T / eK[0][J, np.newaxis]
        #w = np.dot(vc.T, eK[1]).T / D[:, np.newaxis]
        mu = np.dot(eK[1][:, J], w)
        s = ((eK[1][:,J].sum(axis=0)**2)/eK[0][J]).sum()
        c = mu.sum(axis=0)/s
        #mu = np.dot(eK[1], w)

        #mu = mu[0:self.npt,:]
        #print np.fabs(np.dot(K,mu)-v).max()
        #mu = la.solve(K,vc, sym_pos=True)
        return c

    def solveMKM(self, z, p):
        zc = np.concatenate([z,self.xc])
        #vc = np.concatenate([v,np.zeros(self.xc.shape)])
        K = self.param.KparDiff.getK(zc)
        try:
            eK = la.eigh(K)
        except Exception:
            raise Exception('Bad Value in solveMKM')
        J = np.nonzero(eK[0] > self.minEig)[0]
        Ki = np.dot(eK[1][:, J], eK[1][:, J].T/eK[0][J,np.newaxis])
        #check = np.dot(K,Ki)
        M = np.zeros([self.nptc*self.dim, self.ncomponent*(1+self.dim)])
        k1 = 0
        for k in range(self.ncomponent):
            I = np.nonzero(self.component==k)[0]
            zk = z[I,:]
            for i in range(zk.shape[0]):
                u = zk[i,:]
                Ju = np.array([u[1],-u[0]])
                mm = np.concatenate([Ju[:,np.newaxis], np.eye(self.dim)], axis=1)
                M[k1+self.dim*i:k1+self.dim*(i+1), k*(1+self.dim):(k+1)*(1+self.dim)] = mm
            k1 += self.dim*zk.shape[0]

        MKM = np.dot(M.T, np.dot(np.kron(Ki,np.eye(self.dim)),M))
        M = M[0:self.npt*self.dim,:]
        for k in range(self.ncomponent):
            I = np.nonzero(self.component == k)[0]
            ck = np.mean(z[I, :], axis=0)
            cc = (ck**2).sum()
            MKM[k*self.dim,k*self.dim] += self.parRot + self.parTrans*cc
            MKM[k * self.dim+1, k * self.dim+1] += self.parTrans
            MKM[k * self.dim + 2, k * self.dim + 2] += self.parTrans
            MKM[k * self.dim, k * self.dim+1] += self.parTrans * ck[1]
            MKM[k * self.dim+1, k * self.dim] += self.parTrans * ck[1]
            MKM[k * self.dim, k * self.dim+2] -= self.parTrans * ck[0]
            MKM[k * self.dim+2, k * self.dim] -= self.parTrans * ck[0]

        try:
            theta = la.solve(MKM, np.dot(M.T,np.ravel(p)))
        except Exception:
            raise Exception('Bad Value in solveMKM')
        return theta

    def solveMKM2(self, z, rho):
        zc = np.concatenate([z,self.xc])
        K = self.param.KparDiff.getK(zc)
        try:
            eK = la.eigh(K)
        except Exception:
            raise Exception('Bad Value in solveMKM2')
        J = np.nonzero(eK[0] > self.minEig)[0]
        Ki = np.dot(eK[1][:, J], eK[1][:, J].T/eK[0][J, np.newaxis])
        M = np.zeros([self.nptc*self.dim, self.ncomponent*(1+self.dim)])
        k1 = 0
        for k in range(self.ncomponent):
            I = np.nonzero(self.component==k)[0]
            zk = z[I,:]
            for i in range(zk.shape[0]):
                    u = zk[i,:]
                    Ju = np.array([u[1],-u[0]])
                    mm = np.concatenate([Ju[:,np.newaxis], np.eye(self.dim)], axis=1)
                    M[k1+self.dim*i:k1+self.dim*(i+1), k*(1+self.dim):(k+1)*(1+self.dim)] = mm
            k1 += self.dim*zk.shape[0]

        MKM = np.dot(M.T, np.dot(np.kron(Ki,np.eye(self.dim)),M))
        for k in range(self.ncomponent):
            I = np.nonzero(self.component == k)[0]
            ck = np.mean(z[I, :], axis=0)
            cc = (ck**2).sum()
            MKM[k*self.dim,k*self.dim] += self.parRot + self.parTrans*cc
            MKM[k * self.dim+1, k * self.dim+1] += self.parTrans
            MKM[k * self.dim + 2, k * self.dim + 2] += self.parTrans
            MKM[k * self.dim, k * self.dim+1] += self.parTrans * ck[1]
            MKM[k * self.dim+1, k * self.dim] += self.parTrans * ck[1]
            MKM[k * self.dim, k * self.dim+2] -= self.parTrans * ck[0]
            MKM[k * self.dim+2, k * self.dim] -= self.parTrans * ck[0]

        try:
            theta = la.solve(MKM, rho)
        except Exception:
            raise Exception('Bad Value in solveMKM2')
        return theta

    def __repellCost(self, theta, z, M):
        cost = 0
        M2 = np.zeros((M.shape[0] / self.dim, self.dim, M.shape[1]))
        for kl in range(self.dim):
            M2[:, kl, :] = M[range(kl, M.shape[0], self.dim), :]
        v = np.dot(M2, theta)
        cost = self.costRepell(z,v)
        # for k in range(self.ncomponent):
        #     Ik = self.Ic[k]
        #     for l in range(self.ncomponent):
        #         if l != k:
        #             Il = self.Ic[l]
        #             delta = z[Il, np.newaxis, :] - z[np.newaxis, Ik, :]
        #             d = np.sqrt((delta**2).sum(axis=2))
        #             dv = (KM2[np.newaxis,Ik,:,:]*delta[:,:,:,np.newaxis]).sum(axis=2)
        #             cost += (np.maximum(np.dot(dv, theta),0)**2/(d**self.rpot)).sum()
        #         if (len(self.xc) > 0):
        #             delta = self.xc[:, np.newaxis, :] - z[np.newaxis, Ik, :]
        #             d = np.sqrt((delta**2).sum(axis=2))
        #             dv = (KM2[np.newaxis, Ik, :,:]*delta[:,:,:,np.newaxis]).sum(axis=2)
        #             cost += (np.maximum(np.dot(dv, theta),0)**2/(d**self.rpot)).sum()
        # cost *= self.paramRepell/2
        # print 'check cost:', cost0, cost
        return cost

    def __repellJac(self, theta, z, M, doTest=False):
        M2 = np.zeros((M.shape[0] / self.dim, self.dim, M.shape[1]))
        for kl in range(self.dim):
            M2[:, kl, :] = M[range(kl, M.shape[0], self.dim), :]
        v = np.dot(M2, theta)
        grad = self.gradRepellV(z,v)
        jac = (M2*grad[...,np.newaxis]).sum(axis=(0,1))
        if doTest:
            self.testRepellJac(theta, z, M)
        return jac

    def testRepellJac(self, theta, z, KM):
        obj0 = self.__repellCost(theta, z, KM)
        eps = 1e-8
        dth = np.random.normal(0,1,theta.shape)
        obj1 = self.__repellCost(theta+eps*dth, z, KM)
        grad = self.__repellJac(theta, z, KM, doTest=False)
        print('test Repell Jac:', (obj1-obj0)/eps, (grad*dth).sum())

    def computeMKM(self, z):
        zc = np.concatenate([z,self.xc])
        K = self.param.KparDiff.getK(zc)
        try:
            eK = la.eigh(K)
        except Exception:
            raise Exception('Bad Value in solveRepell')
        J = np.nonzero(eK[0] > self.minEig)[0]
        Ki = np.dot(eK[1][:, J], eK[1][:, J].T/eK[0][J, np.newaxis])
        M = np.zeros([self.nptc*self.dim, self.ncomponent*(1+self.dim)])
        k1 = 0
        for k in range(self.ncomponent):
            I = self.Ic[k]
            zk = z[I,:]
            for i in range(zk.shape[0]):
                    u = zk[i,:]
                    Ju = np.array([u[1],-u[0]])
                    mm = np.concatenate([Ju[:,np.newaxis], np.eye(self.dim)], axis=1)
                    M[k1+self.dim*i:k1+self.dim*(i+1), k*(1+self.dim):(k+1)*(1+self.dim)] = mm
            k1 += self.dim*zk.shape[0]

        KM = np.dot(np.kron(Ki,np.eye(self.dim)),M)
        MKM = np.dot(M.T, KM)
        for k in range(self.ncomponent):
            I = self.Ic[k]
            ck = np.mean(z[I, :], axis=0)
            cc = (ck**2).sum()
            MKM[k*self.dim,k*self.dim] += self.parRot + self.parTrans*cc
            MKM[k * self.dim+1, k * self.dim+1] += self.parTrans
            MKM[k * self.dim + 2, k * self.dim + 2] += self.parTrans
            MKM[k * self.dim, k * self.dim+1] += self.parTrans * ck[1]
            MKM[k * self.dim+1, k * self.dim] += self.parTrans * ck[1]
            MKM[k * self.dim, k * self.dim+2] -= self.parTrans * ck[0]
            MKM[k * self.dim+2, k * self.dim] -= self.parTrans * ck[0]
        return MKM, KM, M

    def solveRepell(self, z, rho):
        MKM_ = self.computeMKM(z)
        MKM = MKM_[0]
        try:
            theta0 = la.solve(MKM, rho)
        except Exception:
            raise Exception('Bad Value in solveRepell')

        theta = sopt.minimize(method='BFGS',
                              fun = lambda x,mkm,b,z,m: self.regweight*(x*np.dot(mkm,x)).sum()/2
                                                         - (b*x).sum() + self.__repellCost(x,z,m),
                              jac= lambda x,mkm,b,z,m: self.regweight*np.dot(mkm,x) - b + self.__repellJac(x,z,m),
                              args=(MKM, rho, z, MKM_[2]), x0=theta0,options={"maxiter":100})
        #print 'jac:', np.sqrt((theta.jac**2).sum())

        return theta.x

    def computeHamiltonian(self, x0, rho, a0, tau0):
        Jz = np.zeros(x0.shape)
        Jz[:, 0] = x0[:, 1]
        Jz[:, 1] = -x0[:, 0]
        v = a0[self.component, np.newaxis] * Jz + tau0[self.component, :]
        mu0 = self.regweight*self.solveK(x0,v)
        theta = np.zeros(self.ncomponent*(self.dim+1))
        theta[range(0 ,len(theta) ,self.dim + 1)] = a0
        theta[range(1 ,len(theta) ,self.dim + 1)] = tau0[:,0]
        theta[range(2 ,len(theta) ,self.dim + 1)] = tau0[:,1]
        H0 = (rho*theta).sum() - (v*mu0[0:self.npt, :]).sum()/2 - self.costRepell(x0,v) \
                + self.objectiveFunPotential(x0[np.newaxis,...], timeStep=1.)
        return H0


    def gradHamiltonianQ(self, z, rho, a, tau):
        Jmat = np.array([[0, 1], [-1, 0]])
        # Jz[:,0] = z[:,1]
        # Jz[:,1] = -z[:,0]
        v = a[self.component, np.newaxis] * np.dot(z, Jmat.T) + tau[self.component, :]
        # xt[t+1, :, :] = xt[t, :, :] + timeStep * v
        mu = self.solveK(z, v)

        zpot = self.gradPotential(z)
        zrig = self.gradRigid(z, a, tau)
        zrep = self.gradRepellZ(z, v)
        zc = np.concatenate([z, self.xc])
        # a1 = self.regweight * mu[np.newaxis, ...]
        # a2 = mu[np.newaxis, ...]
        zpx = self.param.KparDiff.applyDiffKT(zc, self.regweight*mu, mu)
        zpx = zpx[0:self.npt, :] + zpot - zrig - zrep

        mu0 = np.copy(mu[0:self.npt, :])
        mu = self.regweight * mu0 + self.gradRepellV(z, v)
        # Jz[:, 0] = mu[:, 1]
        # Jz[:, 1] = -mu[:, 0]
        dv = a[self.component, np.newaxis] * np.dot(mu, Jmat)
        zpx -= dv
        drho = np.zeros(self.ncomponent * (self.dim + 1))
        for k in range(self.ncomponent):
            J = self.Ic[k]
            u = np.zeros([len(J), self.dim])
            u[:, 0] = z[J, 1]
            u[:, 1] = -z[J, 0]
            drho[k * (self.dim + 1)] = (zpx[J, :] * u).sum()
            drho[k * (self.dim + 1) + 1:(k + 1) * (self.dim + 1)] = zpx[J, :].sum(axis=0)
        for k in range(self.ncomponent):
            pt = rho[k * (self.dim + 1) + 1:(k + 1) * (self.dim + 1)]
            drho[k * (self.dim + 1)] += -pt[0] * tau[k, 1] + pt[1] * tau[k, 0]
            drho[k * (self.dim + 1) + 1] += -a[k] * pt[1]
            drho[k * (self.dim + 1) + 2] += a[k] * pt[0]
        return drho


    def applyXiT(self, z, mu):
        rho = np.zeros(self.ncomponent*(self.dim + 1))
        for k in range(self.ncomponent):
            J = self.Ic[k]
            u = np.zeros([len(J),self.dim])
            u[:,0] = z[J,1]
            u[:,1] = -z[J,0]
            rho[k * (self.dim + 1)] = (mu[J, :] * u).sum()
            rho[k * (self.dim + 1)+1:(k+1)*(self.dim+1)] = mu[J, :].sum(axis=0)
        return rho

    def gradRigidTheta(self, z, a, tau):
        rho = np.zeros(self.ncomponent*(self.dim + 1))
        for k in range(self.ncomponent):
            I = self.Ic[k]
            ck = np.mean(z[I, :], axis=0)
            cc = (ck**2).sum()
            rho[k*self.dim] += (self.parRot + self.parTrans*cc) * a[k] + self.parTrans * ck[1] * tau[k,0] \
                          - self.parTrans * ck[0] * tau[k, 1]
            rho[k*self.dim+1] += self.parTrans * tau[k,0] + self.parTrans * ck[1] * a[k]
            rho[k*self.dim+2] += self.parTrans * tau[k,1] - self.parTrans * ck[0] * a[k]

    def geodesicEquation(self, T, dt, a0, tau0, pplot = False, nsymplectic = 0):
        fv0 = self.fv0
        x0 = fv0.vertices
        Tsize = int(np.ceil(T/dt))
        xt = np.zeros([Tsize+1, x0.shape[0], x0.shape[1]])
        xt[0,:,:] = np.copy(x0)


        at = np.zeros([Tsize, self.ncomponent])
        at[0,:] = np.copy(a0)
        taut = np.zeros([Tsize, self.ncomponent, self.dim])
        taut[0,:,:] = np.copy(tau0)
        Jz = np.zeros(x0.shape)
        Jz[:, 0] = np.copy(x0[:, 1])
        Jz[:, 1] = -np.copy(x0[:, 0])
        v = a0[self.component, np.newaxis] * Jz + tau0[self.component, :]

        # Ath = np.zeros(ncomponent*(dim+1))
        # Ath[range(0,len(Ath),dim+1)] = self.parRot*a0
        # for j in range(dim):
        #     Ath[range(j+1, len(Ath), dim + 1)] = self.parTrans*tau0[:,j]
        # u0 = self.solveMKM2(x0,Ath)
        # aa0 = u0[range(0,len(Ath),dim+1)]
        # tt0 = np.zeros((ncomponent, dim))
        # for j in range(dim):
        #     tt0[:,j] = u0[range(j+1, len(Ath), dim + 1)]
        # vv = aa0[component, np.newaxis] * Jz + tt0[component, :]

        mu00 = self.solveK(x0,v)
        mu00 = mu00[0:self.npt, :]
        mu0 = mu00 + self.gradRepellV(x0,v)
        rho = self.applyXiT(x0, mu0) + self.gradRigidTheta(x0, a0, tau0)

        theta = np.zeros(self.ncomponent*(self.dim+1))
        theta[range(0 ,len(theta) ,self.dim + 1)] = a0
        theta[range(1 ,len(theta) ,self.dim + 1)] = tau0[:,0]
        theta[range(2 ,len(theta) ,self.dim + 1)] = tau0[:,1]
        H0 = self.computeHamiltonian(x0, rho, a0, tau0)
        #H0 = (rho*theta).sum() - (mu00*v).sum()/2 - self.costRepell(x0,v) + self.objectiveFunPotential(x0[np.newaxis, :,:], timeStep=1.)
             # - self.objectiveFunRigid(x0[np.newaxis, :,:], a0[np.newaxis,:], tau0[np.newaxis,:,:], timeStep=1.)

        fvDef = curves.Curve(curve=fv0)

        timeStep = dt
        lag = int(np.floor(Tsize/100))
        for t in tqdm(range(Tsize)):
            z = np.copy(xt[t, :, :])
            if t>0:
                try:
                    if not (self.paramRepell is None):
                        theta = self.solveRepell(z,rho)
                    else:
                        theta = self.solveMKM2(z,rho)
                except Exception as excp:
                    print('Exception:', excp)
                    print('solved until t=',t*timeStep)
                    return xt[0:t,...], at[0:t,...], taut[0:t,...]
                at[t,:] = theta[range(0,len(theta),self.dim+1)]
                taut[t, :, 0] = theta[range(1, len(theta), self.dim + 1)]
                taut[t, :, 1] = theta[range(2, len(theta), self.dim + 1)]
            ca = np.cos(timeStep*at[t,:])
            sa = np.sin(timeStep*at[t,:])
            Jz[:,0] = ca[self.component]*z[:,0] + sa[self.component]*z[:,1]
            Jz[:,1] = -sa[self.component]*z[:,0] + ca[self.component]*z[:,1]
            #v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
            z2 = Jz + timeStep * taut[t,self.component,:]

            for ks in range(nsymplectic):
                try:
                    if not (self.paramRepell is None):
                        theta = self.solveRepell(z,rho)
                    else:
                        theta = self.solveMKM2(z,rho)
                except Exception as excp:
                    print('Exception:', excp)
                    print('solved until t=',t*timeStep)
                    return xt[0:t,...], at[0:t,...], taut[0:t,...]

                at[t,:] = theta[range(0,len(theta),self.dim+1)]
                taut[t, :, 0] = theta[range(1, len(theta), self.dim + 1)]
                taut[t, :, 1] = theta[range(2, len(theta), self.dim + 1)]
                ca = np.cos(timeStep*at[t, :])
                sa = np.sin(timeStep*at[t, :])
                Jz[:,0] = ca[self.component]*z[:,0] + sa[self.component]*z[:,1]
                Jz[:,1] = -sa[self.component]*z[:,0] + ca[self.component]*z[:,1]
                z2old = z2
                z2 = Jz + timeStep * taut[t,self.component,:]
                err = np.sqrt(((z2-z2old)**2).sum())
                if err < 0.000001:
                    break

            xt[t + 1 ,: ,:] = z2

            z = z2
            a = at[t,:]
            tau = taut[t,:]
            # v = a[self.component, np.newaxis] * Jz + tau[self.component, :]
            # print 'check2:', self.costRepell(z, v)

            try:
                rho -= timeStep * self.gradHamiltonianQ(z, rho, a, tau)
            except Exception as excp:
                print('Exception:', excp)
                print('solved until t=',t*timeStep)
                return xt[0:t,...], at[0:t,...], taut[0:t,...]
            H = self.computeHamiltonian(z, rho, a, tau)

            fvDef.updateVertices(xt[t+1,:,:])
            if pplot and t%lag==0:
                fig=plt.figure(2)
                fig.clf()
                ax = fig.gca()
                if len(self.xc) > 0:
                    for kf in range(self.fvc.faces.shape[0]):
                        ax.plot(self.fvc.vertices[self.fvc.faces[kf, :], 0],
                                self.fvc.vertices[self.fvc.faces[kf, :], 1], color=[0, 0, 0], linewidth=5)
                for kf in range(self.fv1.faces.shape[0]):
                    ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0], self.fv1.vertices[self.fv1.faces[kf, :], 1],
                            color=[0, 0, 1])
                for kf in range(fvDef.faces.shape[0]):
                    ax.plot(fvDef.vertices[fvDef.faces[kf,:],0], fvDef.vertices[fvDef.faces[kf,:],1], color=[1,0,0], linewidth=3)
                for k in range(fvDef.component.max() + 1):
                    I = self.Ic[k]
                    xDef = fvDef.vertices[I, :]
                    ax.plot(np.array([np.mean(xDef[:, 0]), xDef[0, 0]]),
                            np.array([np.mean(xDef[:, 1]), xDef[0, 1]]),
                            color=[0, .5, 0], linewidth=2)
                    ax.plot(np.mean(xt[0:t+2, I, 0], axis=1), np.mean(xt[0:t+2, I, 1], axis=1))
                plt.title('t={0:.4f}, H= {1:.4f}; {2:.4f}'.format(t*timeStep, H, H0))
                plt.axis('equal')
                plt.pause(0.001)
            #H0 = H
        if self.pplot:
            fig=plt.figure(2)
            fig.clf()
            ax = fig.gca()
            for kf in range(fvDef.faces.shape[0]):
                ax.plot(fvDef.vertices[fvDef.faces[kf,:],0], fvDef.vertices[fvDef.faces[kf,:],1], color=[1,0,0], marker='*')
            plt.axis('equal')
            plt.pause(0.1)
        return xt, at, taut



    def geodesicEquation__(self, T, dt, a0, tau0, pplot = False, nsymplectic = 0, plotRatio = 100):
        fv0 = self.fv0
        x0 = fv0.vertices
        Tsize = int(np.ceil(T/dt))
        xt = np.zeros([Tsize+1, x0.shape[0], x0.shape[1]])
        xt[0,:,:] = np.copy(x0)


        at = np.zeros([Tsize, self.ncomponent])
        at[0,:] = np.copy(a0)
        taut = np.zeros([Tsize, self.ncomponent, self.dim])
        taut[0,:,:] = np.copy(tau0)
        Jz = np.zeros(x0.shape)
        Jz[:, 0] = np.copy(x0[:, 1])
        Jz[:, 1] = -np.copy(x0[:, 0])
        v = a0[self.component, np.newaxis] * Jz + tau0[self.component, :]


        mu00 = self.solveK(x0,v)
        mu00 = mu00[0:self.npt, :]
        pt = mu00 + self.gradRepellV(x0,v)
        timeStep = dt
        lag = int(np.floor(Tsize/plotRatio))
        #lag2 = int(np.floor(Tsize/20))
        lag2 = int(np.ceil(plotRatio/20.))*lag
        fvDef = curves.Curve(curve=fv0)
        H0 = 0
        for t in tqdm(range(Tsize)):
            z = np.copy(xt[t, :, :])
            rho = self.applyXiT(z, pt)
            if t>=0:
                try:
                    if not (self.paramRepell is None):
                        theta = self.solveRepell(z,rho)
                    else:
                        theta = self.solveMKM2(z,rho)
                except Exception as excp:
                    print('Exception:', excp)
                    print('solved until t=',t*timeStep)
                    return xt[0:t,...], at[0:t,...], taut[0:t,...]
                at[t,:] = theta[range(0,len(theta),self.dim+1)]
                taut[t, :, 0] = theta[range(1, len(theta), self.dim + 1)]
                taut[t, :, 1] = theta[range(2, len(theta), self.dim + 1)]

            a = at[t, :]
            tau = taut[t, :, :]
            H = self.computeHamiltonian(z, rho, a, tau)
            if t==0:
                H0=H

            ca = np.cos(timeStep*a)
            sa = np.sin(timeStep*a)
            Jz[:,0] = ca[self.component]*z[:,0] + sa[self.component]*z[:,1]
            Jz[:,1] = -sa[self.component]*z[:,0] + ca[self.component]*z[:,1]
            #v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
            xt[t+1,:,:] = Jz + timeStep * tau[self.component,:]

            # Jz[:,0] = z[:,1]
            # Jz[:,1] = -z[:,0]
            # v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
            # #K = self.param.KparDiff.getK(z)
            # mu = self.solveK(z,v)
            # zc = np.concatenate([z, self.xc])
            # a1 = self.regweight*mu[np.newaxis,...]
            # a2 = mu[np.newaxis,...]
            # zpx = self.param.KparDiff.applyDiffKT(zc, a1, a2)[0:self.npt,:] - self.gradRepellZ(z,v) - self.gradRigid(z, a, tau) \
            #         + self.gradPotential(z)
            #
            # #pm = px-mu
            # mu *= self.regweight
            # mu = mu[0:self.npt, :] + self.gradRepellV(z,v)
            # Jz[:,0] = mu[:,1]
            # Jz[:,1] = -mu[:,0]
            # dv = a[self.component, np.newaxis] * Jz
            # #self.testGradPotential(z)
            # #zpx += dv + self.gradPotential(z) - self.gradRepellZ(z,v)
            # zpx += dv
            # ca = np.cos(timeStep*a)
            # sa = np.sin(timeStep*a)
            # pt -= timeStep * zpx
            # pt0 = ca[self.component]*pt[:,0] + sa[self.component]*pt[:,1]
            # pt[:,1] = -sa[self.component]*pt[:,0] + ca[self.component]*pt[:,1]
            # pt[:,0] = pt0
            pt = self.gradHamiltonianQ_(pt, z, a, tau, timeStep, dir = 1)
            fvDef.updateVertices(xt[t+1,:,:])
            if pplot and (t%lag==0 or t==Tsize-1):
                fig=plt.figure(2)
                fig.clf()
                ax = fig.gca()
                if self.pltFrame is not None:
                    ax.plot(self.pltFrame[:, 0],self.pltFrame[:, 1], color=[1, 1, 1])
                if len(self.xc) > 0:
                    for kf in range(self.fvc.faces.shape[0]):
                        ax.plot(self.fvc.vertices[self.fvc.faces[kf, :], 0],
                                self.fvc.vertices[self.fvc.faces[kf, :], 1], color=[1, .5, 0], linewidth=5)
                for kf in range(self.fv1.faces.shape[0]):
                    ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0], self.fv1.vertices[self.fv1.faces[kf, :], 1],
                            color=[.25, .25, .25])
                for kf in range(fvDef.faces.shape[0]):
                    ax.plot(fvDef.vertices[fvDef.faces[kf,:],0], fvDef.vertices[fvDef.faces[kf,:],1], color=[1,0,0], linewidth=3)
                for k in range(fvDef.component.max() + 1):
                    I = self.Ic[k]
                    xDef = fvDef.vertices[I, :]
                    center = np.zeros((t + 2, self.dim))
                    for tt in range(t + 2):
                        center[tt, :] = self.center(xt[tt, I, :])
                    ax.plot(np.array([center[t+1, 0], xDef[0, 0]]),
                            np.array([center[t+1, 1], xDef[0, 1]]),
                            color=[0, .5, 0], linewidth=2)
                    ax.plot(center[0:t + 2, 0], center[0:t + 2, 1])
                plt.title('t={0:.4f}, H= {1:.4f}; {2:.4f}'.format((t+1)*timeStep, H, H0))
                plt.axis('equal')
                if t%lag2==0 or t==Tsize - 1:
                    plt.title('t={0:.4f}'.format((t + 1) * timeStep))
                    plt.savefig(self.outputDir+'/'+ self.saveFile+str(int(t/lag2))+'.png')
                plt.pause(0.001)
        return xt, at, taut


    def computeHamiltonian_(self, p, z, a, tau, timeStep):
        Jz = np.zeros(z.shape)
        Jz[:, 0] = z[:, 1]
        Jz[:, 1] = -z[:, 0]
        v = a[self.component, np.newaxis] * Jz + tau[self.component, :]
        mu0 = self.regweight*self.solveK(z,v)
        theta = np.zeros(self.ncomponent*(self.dim+1))
        theta[range(0 ,len(theta) ,self.dim + 1)] = a
        theta[range(1 ,len(theta) ,self.dim + 1)] = tau[:,0]
        theta[range(2 ,len(theta) ,self.dim + 1)] = tau[:,1]
        ca = np.cos(timeStep * a)
        sa = np.sin(timeStep * a)
        Jz[:, 0] = ca[self.component] * z[:, 0] + sa[self.component] * z[:, 1]
        Jz[:, 1] = -sa[self.component] * z[:, 0] + ca[self.component] * z[:, 1]
        Jz += tau[self.component, :]*timeStep
        H = (p*Jz).sum() - timeStep*(mu0[0:self.npt, :]*v).sum()/2 - timeStep*self.costRepell(z,v) \
            + timeStep* self.objectiveFunPotential(z)
        return H

    # def __minifun__(self, z, v):
    #     mu = self.solveK(z,v)
    #     return (mu*v).sum()/2
    #
    # def __minigrad__(self, z, v):
    #     mu = self.solveK(z,v)
    #     a1 = np.copy(mu[np.newaxis, ...])
    #     a2 = np.copy(mu[np.newaxis, ...])
    #     zpx = -self.param.KparDiff.applyDiffKT(z, a1, a2)
    #     return zpx
    #
    # def __minitest__(self, z, a, tau):
    #     Jz = np.zeros(z.shape)
    #     Jz[:, 0] = z[:, 1]
    #     Jz[:, 1] = -z[:, 0]
    #     v = a[self.component, np.newaxis] * Jz + tau[self.component, :]
    #     obj0 = self.__minifun__(z,v)
    #     eps = 1e-10
    #     dz = np.random.normal(0,1,z.shape)
    #     obj1 = self.__minifun__(z+eps*dz,v)
    #     grad = self.__minigrad__(z,v)
    #     print 'Minitest:', (obj1-obj0)/eps, (grad*dz).sum()

    def gradHamiltonianQ_(self, p, z, a, tau, timeStep, dir = -1):
        zc = np.concatenate([z, self.xc])
        Jz = np.zeros(z.shape)
        Jz[:, 0] = z[:, 1]
        Jz[:, 1] = -z[:, 0]
        if type(a) is float:
            a = a[np.newaxis]
            tau = tau[np.newaxis,:]
        v = a[self.component, np.newaxis] * Jz + tau[self.component, :]
        # K = self.param.KparDiff.getK(z)
        mu = self.solveK(z, v)
        # a1 = self.regweight * mu[np.newaxis, ...]
        # a2 = mu[np.newaxis, ...]
        zpx = self.param.KparDiff.applyDiffKT(zc, self.regweight*mu, mu)[0:self.npt, :] - self.gradRepellZ(z, v) \
            + self.gradPotential(z)

        # pm = px-mu
        mu = self.regweight * mu[0:self.npt, :]
        mu += self.gradRepellV(z, v)
        Jz[:, 0] = mu[:, 1]
        Jz[:, 1] = -mu[:, 0]
        dv = a[self.component, np.newaxis] * Jz
        # self.testGradPotential(z)
        # zpx += dv + self.gradPotential(z) - self.gradRepellZ(z,v)
        zpx += dv
        ca = np.cos(timeStep * a)
        sa = np.sin(timeStep * a)
        if dir < 0:
            Jz[:, 0] = ca[self.component] * p[:, 0] - sa[self.component] * p[:, 1]
            Jz[:, 1] = sa[self.component] * p[:, 0] + ca[self.component] * p[:, 1]
            return Jz + timeStep * zpx
        else:
            p -= timeStep * zpx
            Jz[:, 0] = ca[self.component] * p[:, 0] + sa[self.component] * p[:, 1]
            Jz[:, 1] = -sa[self.component] * p[:, 0] + ca[self.component] * p[:, 1]
            return Jz

    def gradHamiltonianTheta_(self, p, z, a, tau, timeStep):
        # if self.ncomponent==1:
        #     a = a[np.newaxis]
        #     tau = tau[np.newaxis,:]
        da = np.zeros(a.shape)
        dtau = np.zeros(tau.shape)
        Jz = np.zeros(z.shape)
        Jz[:, 0] = z[:, 1]
        Jz[:, 1] = -z[:, 0]
        v = a[self.component, np.newaxis] * Jz + tau[self.component, :]
        #self.testRepellGrad(z,v)
        mu = self.regweight * self.solveK(z, v)[0:self.npt, :]
        if not (self.paramRepell is None):
            mu += self.gradRepellV(z, v)
            # self.testRepellGrad(z,v)
        p1 = mu * Jz
        ca = np.cos(timeStep * a)
        sa = np.sin(timeStep * a)
        Jz[:, 0] = -sa[self.component] * z[:, 0] + ca[self.component] * z[:, 1]
        Jz[:, 1] = -ca[self.component] * z[:, 0] + -sa[self.component] * z[:, 1]
        p1 -= p * Jz
        for j in range(self.ncomponent):
            I = self.Ic[j]
            da[j] = p1[I, :].sum()
            dtau[j, :] = (mu - p)[I, :].sum(axis=0)
        return da, dtau

    def testGradHamiltonian_(self, p, z, a, tau, timeStep):
        obj0 = self.computeHamiltonian_(p, z, a, tau, timeStep)
        eps = 1e-10
        dz = np.random.normal(0,1,z.shape)
        da = np.random.normal(0,1,a.shape)
        dtau = np.random.normal(0,1,tau.shape)
        obj1 = self.computeHamiltonian_(p, z+eps*dz, a, tau, timeStep)
        obj2 = self.computeHamiltonian_(p, z, a+eps*da, tau, timeStep)
        obj3 = self.computeHamiltonian_(p, z, a, tau+eps*dtau, timeStep)
        gradQ = self.gradHamiltonianQ_(p, z, a, tau, timeStep)
        gradTh = self.gradHamiltonianTheta_(p, z, a, tau, timeStep)
        print('Test Hamiltonian, Q:', (obj1-obj0)/eps, (gradQ*dz).sum())
        print('Test Hamiltonian, a:', (obj2-obj0)/eps, -timeStep*(gradTh[0]*da).sum())
        print('Test Hamiltonian, tau:', (obj3-obj0)/eps, -timeStep*(gradTh[1]*dtau).sum())
        #self.__minitest__(z,a,tau)

    def hamiltonianCovector(self, px1, affine = None):
        x0 = self.x0
        at = self.at
        taut = self.taut
        KparDiff = self.param.KparDiff
        N = x0.shape[0]
        dim = x0.shape[1]
        M = at.shape[0]
        timeStep = 1.0/M
        xt = self.directEvolutionEuler(x0, at, taut)
        px1 -= self.gradPotential(xt[-1,:,:])*timeStep

        pxt = np.zeros([M+1, N, dim])
        pxt[M, :, :] = px1
        foo = curves.Curve(curve=self.fv0)
        for t in range(M):
            px = pxt[M-t, :, :]
            z = xt[M-t-1, :, :]
            #zc = np.concatenate([z, self.xc])
            a = at[M-t-1, :]
            tau = taut[M-t-1, :, :]
            #self.testGradHamiltonian_(px, z, a, tau, timeStep)
            # Jz = np.zeros(z.shape)
            # Jz[:,0] = z[:,1]
            # Jz[:,1] = -z[:,0]
            # v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
            # #K = self.param.KparDiff.getK(z)
            # mu = self.solveK(z,v)
            # foo.updateVertices(z)
            # a1 = self.regweight*mu[np.newaxis,...]
            # a2 = mu[np.newaxis,...]
            # zpx = self.param.KparDiff.applyDiffKT(zc, a1, a2)[0:self.npt, :] - self.gradRepellZ(z,v)
            #
            # #pm = px-mu
            # mu = self.regweight * mu[0:self.npt, :]
            # mu += self.gradRepellV(z,v)
            # Jz[:,0] = mu[:,1]
            # Jz[:,1] = -mu[:,0]
            # dv = a[self.component, np.newaxis] * Jz
            # #self.testGradPotential(z)
            # #zpx += dv + self.gradPotential(z) - self.gradRepellZ(z,v)
            # zpx += dv
            # ca = np.cos(timeStep*a)
            # sa = np.sin(timeStep*a)
            # Jz[:,0] = ca[self.component]*px[:,0] - sa[self.component]*px[:,1]
            # Jz[:,1] = sa[self.component]*px[:,0] + ca[self.component]*px[:,1]
            #pxt[M-t-1, :, :] = Jz + timeStep * zpx
            pxt[M-t-1, :, :] = self.gradHamiltonianQ_(px, z, a, tau, timeStep)
        return pxt, xt


    def hamiltonianGradient(self, px1):
        foo = curves.Curve(curve=self.fv0)
        timeStep = 1.0/self.Tsize
        (pxt, xt) = self.hamiltonianCovector(px1)
        at = self.at        
        dat = np.zeros(at.shape)
        taut = self.taut
        dtaut = np.zeros(taut.shape)
        for k in range(at.shape[0]):
            z = np.squeeze(xt[k,...])
            foo.updateVertices(z)
            a = at[k, :]
            tau = taut[k, :, :]
            px = pxt[k+1, :, :]
            grd = self.gradHamiltonianTheta_(px, z, a, tau, timeStep)
            # Jz = np.zeros(z.shape)
            # Jz[:,0] = z[:,1]
            # Jz[:,1] = -z[:,0]
            # v = a[self.component, np.newaxis] * Jz + tau[self.component,:]
            # mu = self.regweight * self.solveK(z,v)[0:self.npt,:]
            # if not (self.paramRepell is None):
            #     mu += self.gradRepellV(z, v)
            #     #self.testRepellGrad(z,v)
            # p1 = mu * Jz
            # ca = np.cos(timeStep*a)
            # sa = np.sin(timeStep*a)
            # Jz[:,0] = -sa[self.component]*z[:,0] + ca[self.component]*z[:,1]
            # Jz[:,1] = -ca[self.component]*z[:,0] + -sa[self.component]*z[:,1]
            # p1 -= px*Jz
            # for j in range(self.ncomponent):
            #     I = self.Ic[j]
            #     dat[k, j] = p1[I,:].sum()
            #     dtaut[k,j,:] = (mu-px)[I,:].sum(axis=0)
            dat[k,:] = grd[0]
            dtaut[k,:] = grd[1]

        return dat, dtaut, xt, pxt


    def endPointGradient(self):
        px = self.param.fun_objGrad(self.fvDef, self.fv1, self.param.KparDist)
        return px / self.param.sigmaError**2


    def getGradient(self, coeff=1.0):
        px1 = -self.endPointGradient()
        foo = self.hamiltonianGradient(px1)
        grd = Direction()
        grd.skew = foo[0]/(coeff*self.Tsize)
        grd.trans = foo[1]/(coeff*self.Tsize)
        self.pxt = foo[3]
        return grd



    def addProd(self, dir1, dir2, beta):
        dir = Direction()
        dir.skew = dir1.skew + beta * dir2.skew
        dir.trans = dir1.trans + beta * dir2.trans
        return dir

    def prod(self, dir1, beta):
        dir = Direction()
        dir.skew = beta * dir1.skew
        dir.trans = beta * dir1.trans
        return dir

    def copyDir(self, dir0):
        dir = Direction()
        dir.skew = np.copy(dir0.skew)
        dir.trans = np.copy(dir0.trans)

        return dir


    def randomDir(self):
        dirfoo = Direction()
        dirfoo.skew = np.random.randn(self.Tsize, self.ncomponent)
        dirfoo.trans = np.random.randn(self.Tsize, self.ncomponent, self.dim)
        return dirfoo

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        #dim2 = self.dim**2
        for t in range(self.Tsize):
            gs = np.squeeze(g1.skew[t, :])
            gt = np.squeeze(g1.trans[t, :, :])
            for ll,gr in enumerate(g2):
                ggs = np.squeeze(gr.skew[t, :])
                ggt = np.squeeze(gr.trans[t, :, :])
                res[ll] += (gs*ggs).sum() + (gt*ggt).sum()

        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.at = np.copy(self.atTry)
        self.taut = np.copy(self.tautTry)

    def endOfIteration(self):
        (obj1, self.xt) = self.objectiveFunDef(self.at, self.taut, withTrajectory=True)
        self.iter += 1

        if self.saveRate > 0 and self.iter%self.saveRate==0:
            for kk in range(self.Tsize+1):
                self.fvDef.updateVertices(np.squeeze(self.xt[kk, :, :]))
                self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk')
            yy = self.directEvolutionEuler(self.x0, self.at, self.taut, grid=self.gridxy)
            self.gridDef.vertices = np.copy(yy[1][:, :])
            if self.pplot:
                plt.axis('equal')
                plt.savefig(self.outputDir +'/'+ self.saveFile+'.png')
                plt.pause(0.1)
                fig=plt.figure(2)
                fig.clf()
                ax = fig.gca()
                nr = self.gridDef.nrow
                nc = self.gridDef.ncol
                for k in range(0, nr, self.gskip):
                    I = np.array(range(0,nc), dtype=int)
                    xg = self.gridDef.vertices[k*nc + I,0]
                    yg = self.gridDef.vertices[k*nc + I,1]
                    ax.plot(xg, yg, color=[0,0,0])
                for k in range(0, nc, self.gskip):
                    I = np.array(range(0,nr), dtype=int)
                    xg = self.gridDef.vertices[I*nc + k,0]
                    yg = self.gridDef.vertices[I*nc + k,1]
                    ax.plot(xg, yg, color=[0,0,0])
                for kf in range(self.fvDef.faces.shape[0]):
                    ax.plot(self.fvDef.vertices[self.fvDef.faces[kf, :], 0],
                            self.fvDef.vertices[self.fvDef.faces[kf, :], 1],
                            color=[1, 0, 0], linewidth=1)
                plt.axis('equal')
                plt.savefig(self.outputDir +'/'+ self.saveFile+'Grid.png')
                plt.pause(0.1)
            #self.geodesicEquation__(1., 1./self.Tsize, self.at[0, :], self.taut[0, :, :], pplot=True, plotRatio=10)
            #self.__geodesicEquation__(self.Tsize, self.pxt[0, :,:])
        else:
            self.fvDef.updateVertices(np.squeeze(self.xt[self.Tsize, :, :]))

        if self.pplot:
            fig=plt.figure(1)
            fig.clf()
            ax = fig.gca()
            t = self.Tsize
            if self.pltFrame is not None:
                ax.plot(self.pltFrame[:, 0], self.pltFrame[:, 1], color=[1, 1, 1])
            if len(self.xc) > 0:
                for kf in range(self.fvc.faces.shape[0]):
                    ax.plot(self.fvc.vertices[self.fvc.faces[kf, :], 0],
                            self.fvc.vertices[self.fvc.faces[kf, :], 1], color= [1, .5, 0], linewidth=5)
            for kf in range(self.fv0.faces.shape[0]):
                ax.plot(self.fv0.vertices[self.fv0.faces[kf, :], 0],
                        self.fv0.vertices[self.fv0.faces[kf, :], 1], color=[.25, .25, .25])
            for kf in range(self.fv1.faces.shape[0]):
                ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0],
                        self.fv1.vertices[self.fv1.faces[kf, :], 1], color=[.25, .25, .25])
            for kf in range(self.fvDef.faces.shape[0]):
                ax.plot(self.fvDef.vertices[self.fvDef.faces[kf, :], 0],
                        self.fvDef.vertices[self.fvDef.faces[kf, :], 1],
                        color=[1, 0, 0], linewidth=3)
            for k in range(self.ncomponent):
                I = self.Ic[k]
                xDef = self.fvDef.vertices[I, :]
                center = np.zeros((t+1,self.dim))
                for tt in range(t+1):
                    center[tt,:] = self.center(self.xt[tt,I,:])
                ax.plot(np.array([center[t,0], xDef[0, 0]]),
                        np.array([center[t,1], xDef[0, 1]]),
                        color=[0, .5, 0], linewidth=2)
                ax.plot(center[0:t+1,0], center[0:t + 1, 1])

                # ax.plot(np.array([np.mean(xDef[:, 0]), xDef[0, 0]]),
                #         np.array([np.mean(xDef[:, 1]), xDef[0, 1]]),
                #         color=[0, .5, 0], linewidth=2)
                # ax.plot(np.mean(self.xt[0:t + 1, I, 0], axis=1), np.mean(self.xt[0:t + 1, I, 1], axis=1))
            #plt.axis('equal')
            #plt.title('t={0:.3f}'.format(float(t) / self.Tsize))
            # if len(self.xc) > 0:
            #     for kf in range(self.fvc.faces.shape[0]):
            #         ax.plot(self.fvc.vertices[self.fvc.faces[kf, :], 0],
            #                 self.fvc.vertices[self.fvc.faces[kf, :], 1], color=[1, 0, 0])
            # for kf in range(self.fv1.faces.shape[0]):
            #     ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0],
            #             self.fv1.vertices[self.fv1.faces[kf, :], 1], color=[0, 0, 1])
            # for kf in range(self.fvDef.faces.shape[0]):
            #     ax.plot(self.fvDef.vertices[self.fvDef.faces[kf,:],0], self.fvDef.vertices[self.fvDef.faces[kf,:],1], color=[1,0,0], marker='*')


                

    def endOptim(self):
        fig = plt.figure(10)
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Euclidean LDDMM')
        writer = FFMpegWriter(fps=5, metadata=metadata)
        with writer.saving(fig, self.outputDir + '/' + self.saveFile + ".mp4", 100):
            for t in range(self.Tsize + 1):
                self.fvDef.updateVertices(self.xt[t, :, :])
                fig.clf()
                ax = fig.gca()
                if self.pltFrame is not None:
                    ax.plot(self.pltFrame[:, 0], self.pltFrame[:, 1], color=[1, 1, 1])
                if len(self.xc) > 0:
                    for kf in range(self.fvc.faces.shape[0]):
                        ax.plot(self.fvc.vertices[self.fvc.faces[kf, :], 0],
                                self.fvc.vertices[self.fvc.faces[kf, :], 1], color=[1, .5, 0], linewidth=5)
                for kf in range(self.fv0.faces.shape[0]):
                    ax.plot(self.fv0.vertices[self.fv0.faces[kf, :], 0],
                            self.fv0.vertices[self.fv0.faces[kf, :], 1], color=[.25, .25, .25])
                for kf in range(self.fv1.faces.shape[0]):
                    ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0],
                            self.fv1.vertices[self.fv1.faces[kf, :], 1], color=[.25, .25, .25])
                for kf in range(self.fvDef.faces.shape[0]):
                    ax.plot(self.fvDef.vertices[self.fvDef.faces[kf, :], 0],
                            self.fvDef.vertices[self.fvDef.faces[kf, :], 1],
                            color=[1, 0, 0], linewidth=3)
                for k in range(self.ncomponent):
                    I = self.Ic[k]
                    xDef = self.fvDef.vertices[I, :]
                    ax.plot(np.array([np.mean(xDef[:, 0]), xDef[0, 0]]),
                            np.array([np.mean(xDef[:, 1]), xDef[0, 1]]),
                            color=[0, .5, 0], linewidth=2)
                    ax.plot(np.mean(self.xt[0:t + 1, I, 0], axis=1), np.mean(self.xt[0:t + 1, I, 1], axis=1))
                plt.axis('equal')
                plt.title('t={0:.3f}'.format(float(t) / self.Tsize))
                writer.grab_frame()
                plt.savefig(self.outputDir +'/'+ self.saveFile+str(t)+'.png')
                plt.pause(0.001)
        if self.saveRate==0 or self.iter%self.saveRate > 0:
            for kk in range(self.Tsize+1):
                self.fvDef.updateVertices(np.squeeze(self.xt[kk, :, :]))
                self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk')
        self.defCost = self.obj - self.obj0 - self.dataTerm(self.fvDef)   


    def optimizeMatching(self, a0=None, tau0= None):
        if a0 is not None:
            self.at = a0
        if tau0 is not None:
            self.taut = tau0

        if a0 is not None or tau0 is not None:
            self.xt = self.directEvolutionEuler(self.x0, self.at, self.taut)
            self.fvDef.updateVertices((self.xt[-1,:,:]))
            if self.pplot:
                fig=plt.figure(5)
                fig.clf()
                t = self.Tsize
                fig.clf()
                ax = fig.gca()
                if len(self.xc) > 0:
                    for kf in range(self.fvc.faces.shape[0]):
                        ax.plot(self.fvc.vertices[self.fvc.faces[kf, :], 0],
                                self.fvc.vertices[self.fvc.faces[kf, :], 1], color=[1, .5, 0], linewidth=5)
                for kf in range(self.fv0.faces.shape[0]):
                    ax.plot(self.fv0.vertices[self.fv0.faces[kf, :], 0],
                            self.fv0.vertices[self.fv0.faces[kf, :], 1], color=[0, 0, 1])
                for kf in range(self.fv1.faces.shape[0]):
                    ax.plot(self.fv1.vertices[self.fv1.faces[kf, :], 0],
                            self.fv1.vertices[self.fv1.faces[kf, :], 1], color=[0, 0, 1])
                for kf in range(self.fvDef.faces.shape[0]):
                    ax.plot(self.fvDef.vertices[self.fvDef.faces[kf, :], 0],
                            self.fvDef.vertices[self.fvDef.faces[kf, :], 1],
                            color=[1, 0, 0], linewidth=3)
                for k in range(self.ncomponent):
                    I = self.Ic[k]
                    xDef = self.fvDef.vertices[I, :]
                    center = np.zeros((t+1,self.dim))
                    for tt in range(t+1):
                        center[tt,:] = self.center(self.xt[tt,I,:])
                    ax.plot(np.array([center[t,0], xDef[0, 0]]),
                            np.array([center[t,1], xDef[0, 1]]),
                            color=[0, .5, 0], linewidth=2)
                    ax.plot(center[0:t+1,0], center[0:t + 1, 1])
                plt.axis('equal')

        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(self.gradLB, np.sqrt(grd2) / 10000000)
        print('Gradient bound:', self.gradEps)
        kk = 0
        while os.path.isfile(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk'):
            os.remove(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk')
            kk += 1

        if self.param.algorithm == 'bfgs':
            bfgs.bfgs(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient)
        else:
            cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient)
        #return self.at, self.xt


    def move(self, x, theta=0, b=(0,0)):
        y = np.zeros(x.shape)
        y[:,0] = np.cos(theta) * x[:,0] + np.sin(theta)*x[:,1] + b[0]
        y[:,1] = -np.sin(theta) * x[:,0] + np.cos(theta)*x[:,1] + b[1]
        return y



    def __circle(self,N,r):
        t = np.arange(0., 2 * np.pi, 2*np.pi/N)
        x = np.zeros((len(t), 2))
        x[:, 0] = r * np.cos(t)
        x[:, 1] = r * np.sin(t)
        return x

    def __c(self,N,r, d):
        t1 = 7 * np.pi / 4
        t0 = np.pi / 4

        t = np.arange(t0, t1, (t1-t0)/(10*N))
        x1 = np.zeros((len(t), 2))
        rx = r + d * (1 - np.exp(-100 * (t - t0) * (t1 - t)))
        x1[:, 0] = rx * np.cos(t)
        x1[:, 1] = rx * np.sin(t)
        t = np.arange(t1, t0, -(t1-t0)/(10*N))
        x2 = np.zeros((len(t), 2))
        rx = r - d * (1 - np.exp(-100 * (t - t0) * (t1 - t)))
        x2[:, 0] = rx * np.cos(t)
        x2[:, 1] = rx * np.sin(t)
        x = curves.remesh(np.concatenate((x1, x2)), N)
        return x

    def __bunny(self, N0, r, a1, th1, a2, th2):
        N = 500
        t = np.arange(0., 2 * np.pi, 2*np.pi/N)
        rho = r*np.ones(t.shape)
        j1 = int(np.floor(N*(th1 - 0.2)/(2*np.pi)))
        j2 = int(np.ceil(N*(th1 + 0.2)/(2*np.pi)))
        rho[j1:j2] += a1 * np.sin(np.pi*(t[j1:j2]-t[j1])/(t[j2]-t[j1]))
        j1 = int(np.floor(N*(th2 - 0.2)/(2*np.pi)))
        j2 = int(np.ceil(N*(th2 + 0.2)/(2*np.pi)))
        rho[j1:j2] += a2 * np.sin(np.pi*(t[j1:j2]-t[j1])/(t[j2]-t[j1]))
        x = np.zeros((len(t), 2))
        x[:, 0] = rho * np.cos(t)
        x[:, 1] = rho * np.sin(t)
        x = curves.remesh(x, N0)
        return x

    def __ellipse(self, N0,a,b,theta):
        N = 500
        t = np.arange(0., 2 * np.pi, 2*np.pi/N)
        x = np.zeros((len(t), 2))
        x[:, 0] = a * np.cos(theta) * np.cos(t) + b * np.sin(theta) * np.sin(t)
        x[:, 1] = -a * np.sin(theta) * np.cos(t) + b * np.cos(theta) * np.sin(t)
        x = curves.remesh(x, N0)
        return x

    def __square(self,N,r):
        t = np.arange(0., 1., 4./N)[:,np.newaxis]
        x = np.concatenate([t*[1,0], [1,0] + t*[0,-1], [1,-1] + t*[-1,0], [0,-1]+t*[0,1]])
        x -= [.5,-.5]
        x *= 2*r
        return x

    def __rectangle(self,N, r1, r2):
        t = np.arange(0., 1., 4./N)[:,np.newaxis]
        x = np.concatenate([t*[1,0], [1,0] + t*[0,-1], [1,-1] + t*[-1,0], [0,-1]+t*[0,1]])
        x -= [.5,-.5]
        x[:,0] *= 2*r1
        x[:,1] *= 2*r2
        return x

    def __hline(self,N,r):
        t = np.arange(0., 1., 1./N)[:,np.newaxis]
        x = t*[2,0] - 1
        return r*x

    def __vline(self,N,r):
        t = np.arange(0., 1., 1./N)[:,np.newaxis]
        x = t*[0,2] - [0,1]
        return r*x

    def __line(self,N,r, theta):
        t = np.arange(0., 1., 1./N)[:,np.newaxis]
        x = 2*t*[np.cos(theta),np.sin(theta)] - 1
        return r*x

    def shootingScenario(self, scenario = 1, T=5., dt=0.001):
        dirOut = '/Users/younes'
        if os.path.isfile(dirOut + '/Development/Results/curveShootingRigid/info.tex'):
            os.remove(dirOut + '/Development/Results/curveShootingRigid/info.tex')
        loggingUtils.setup_default_logging(dirOut + '/Development/Results/curveShootingRigid', fileName='info.txt',
                                           stdOutput=True)
        sigmaDist = 2.0
        sigmaError = 0.01
        fileName = 'Shooting'
        xframe = None
        fvc = None

        if scenario == 1:
            sigma = .2
            K1 = Kernel(name='laplacian' ,sigma=sigma)
            sm = CurveMatchingRigidParam(timeStep=dt / T ,KparDiff=K1 ,sigmaDist=sigmaDist ,sigmaError=sigmaError ,
                                         errorType='varifold')
            x = self.__circle(50, 0.25)
            x1 = self.__circle(25, 0.1)
            # x = self.__square(50, 0.25)
            fv0 = curves.Curve(curve=[curves.Curve(pointSet=x + np.array([-3, 0])),
                                      curves.Curve(pointSet=x + np.array([-1.1, 1.])),
                                      curves.Curve(pointSet=x + np.array([-1.3, -1.])),
                                      curves.Curve(pointSet=x + np.array([-1.2, 0]))])
            xframe = self.__rectangle(200, 4, 3)
            fvc = curves.Curve(pointSet=xframe)
            #fvc = curves.Curve(curve=[curves.Curve(pointSet=x),curves.Curve(pointSet=x1 + np.array([0,1])), curves.Curve(pointSet=x1 + np.array([0,-1]))])
            mx = self.center(x + np.array([-3, 0]), kernel=K1)
            a0 = np.array([15, 0, 0, 0])
            tau0 = np.array([[20-15*mx[1], 15*mx[0]], [0, 0], [0,0], [0, 0]])
            fileName = 'balls'
        elif scenario == 2:
            sigma = .5
            K1 = Kernel(name='laplacian' ,sigma=sigma, order=1)
            sm = CurveMatchingRigidParam(timeStep=dt / T ,KparDiff=K1 ,sigmaDist=sigmaDist ,sigmaError=sigmaError ,
                                         errorType='varifold')
            x = self.__circle(25, 0.1)
            #x = self.__bunny(50, 0.1, 0.15, 0.5*np.pi, 0.15, 0.75*np.pi)
            fv0 = curves.Curve(pointSet=x + np.array([-1 , .25]))
            xframe = self.__rectangle(200, 1.0, 0.5)
            #x1 = self.__circle(25, 0.1)
            x1 = self.__vline(50, 0.25)
            #fvc = curves.Curve(curve=[curves.Curve(pointSet=x),curves.Curve(pointSet=x1 + np.array([0,-.5]))])
            fvc = curves.Curve(pointSet=x1 + np.array([0, -.45]))
            #fvc = curves.Curve(curve=[curves.Curve(pointSet=x1 + np.array([0, -.45])), curves.Curve(pointSet=x)])
            #fvc = curves.Curve(pointSet=x1 + np.array([0,-1.5]))
            a0 = np.array([0])
            tau0 = np.array([[5.,0]])
            fileName = 'oneBallAttractor'
        elif scenario == 3:
            sigma = .2
            K1 = Kernel(name='laplacian', sigma=sigma)
            sm = CurveMatchingRigidParam(timeStep=dt / T, KparDiff=K1, sigmaDist=sigmaDist, sigmaError=sigmaError,
                                         errorType='varifold')
            x = self.__circle(25, 0.1)
            mx = self.center(x, kernel=K1)
            #x = self.__bunny(50, 0.05, 0.1, 0.5*np.pi, 0.1, 0.75*np.pi)
            x1 = self.__circle(25, 0.1)
            # x = self.__square(50, 0.25)
            offset = np.array([-1, 0.5318])
            fv0 = [curves.Curve(pointSet=x + offset),
                   #                   curves.Curve(pointSet=x + np.array([.9, 1.])),
                   #                   curves.Curve(pointSet=x + np.array([.7, -1.])),
                   curves.Curve(pointSet=x - offset)]
            xframe = self.__rectangle(200, 2, 1.5)
            x = self.__square(200, 4)
            fvc = curves.Curve(pointSet=x)
            fvc = None
            fileName = "twoBunniesDanse"
            # fvc = curves.Curve(curve=[curves.Curve(pointSet=x),curves.Curve(pointSet=x1 + np.array([0,1])), curves.Curve(pointSet=x1 + np.array([0,-1]))])
            # a0 = np.array([0, 0, 0, 0])
            # tau0 = np.array([[10, 0], [0, 0], [0,0], [0, 0]])
            mx1 = mx + offset
            mx2 = mx - offset

            a0 = -10*np.array([1, 1])
            #a0 = np.array([0,0])
            tau0 = np.array([[10-a0[0]*mx1[1], a0[0]*mx1[0]], [-10-a0[1]*mx2[1], a0[1]*mx2[0]]])
        elif scenario == 30:
            sigma = 1.
            K1 = Kernel(name='laplacian', sigma=sigma)
            sm = CurveMatchingRigidParam(timeStep=dt / T, KparDiff=K1, sigmaDist=sigmaDist, sigmaError=sigmaError,
                                         errorType='varifold')
            #x = self.__bunny(50, 0.25, 0.5, 0.5*np.pi, 0.5, 0.75*np.pi)
            x = self.__c(50, 0.5, 0.2)
            # x = self.__square(50, 0.25)
            #x = curves.remesh(x, 50)
            fv0 = curves.Curve(pointSet=x + np.array([-1, 0]))
            #fv0.resample(0.1)
            mx = self.center(fv0.vertices, kernel=K1)
            #fv0 = curves.Curve(pointSet= fv0.vertices)
                   #                   curves.Curve(pointSet=x + np.array([.9, 1.])),
                   #                   curves.Curve(pointSet=x + np.array([.7, -1.])),
            xframe = self.__square(200, 6)
            fvc = curves.Curve(pointSet=xframe)
            fvc = None
            fileName = "oneC"
            # fvc = curves.Curve(curve=[curves.Curve(pointSet=x),curves.Curve(pointSet=x1 + np.array([0,1])), curves.Curve(pointSet=x1 + np.array([0,-1]))])
            # a0 = np.array([0, 0, 0, 0])
            # tau0 = np.array([[10, 0], [0, 0], [0,0], [0, 0]])
            a0 = np.array([10])
            tau0 = np.array([[10 - 10*mx[1], 0 + 10*mx[0]]])
        elif scenario == 4:
            sigma = .75
            K1 = Kernel(name='laplacian' ,sigma=sigma)
            sm = CurveMatchingRigidParam(timeStep=dt / T ,KparDiff=K1 ,sigmaDist=sigmaDist ,sigmaError=sigmaError ,
                                         errorType='varifold')
            x = self.__ellipse(50, 0.33, 0.33, 0)
            #fv0 = [curves.Curve(pointSet=x + np.array([-1, 0])), curves.Curve(pointSet=x)]
            fv0 = curves.Curve(pointSet=x + np.array([-1, 0]))
            x = self.__square(200, 4)
            fvc = curves.Curve(pointSet=x)
            #fvc = curves.Curve(curve=[curves.Curve(pointSet=x),curves.Curve(pointSet=x1 + np.array([0,1])), curves.Curve(pointSet=x1 + np.array([0,-1]))])
            a0 = np.array([10])
            tau0 = np.array([[10, -10]])
            #a0 = np.array([10,0])
            #tau0 = np.array([[10, -10],[0,0]])
        elif scenario == 5:
            sigma = .1
            K1 = Kernel(name='laplacian' ,sigma=sigma)
            sm = CurveMatchingRigidParam(timeStep=dt / T ,KparDiff=K1 ,sigmaDist=sigmaDist ,sigmaError=sigmaError ,
                                         errorType='varifold')
            x1 = self.__line(25, 0.1, 0) + [0,-1]
            x2 = self.__line(25, 0.1, np.pi/3) + [0.,-.5]
            x3 = self.__line(25, 0.1, np.pi/6) + [0.4,-0.6]
            x4 = self.__line(25, 0.1, np.pi/2) + [0,0]
            x5 = self.__line(25, 0.1, np.pi/4) + [-0.2,0.9]
            fv0 = [curves.Curve(pointSet=x1), curves.Curve(pointSet=x2), curves.Curve(pointSet=x3),
                   curves.Curve(pointSet=x4), curves.Curve(pointSet=x5)]
            x = self.__square(200, 4)
            fvc = curves.Curve(pointSet=x)
            #fvc = curves.Curve(curve=[curves.Curve(pointSet=x),curves.Curve(pointSet=x1 + np.array([0,1])), curves.Curve(pointSet=x1 + np.array([0,-1]))])
            a0 = np.array([10,0,-5,2,1])
            tau0 = np.array([[10, -10],[3,0],[-2,5],[4,-2], [0,-1]])
        else:
            a0 = None
            tau0 = None
            fv0 = None
            fvc = None
            sm = CurveMatchingRigidParam()

        f = CurveMatchingRigid(Template=fv0 ,Target=fv0 ,Clamped=fvc , pltFrame = xframe,
                               outputDir=dirOut + '/Development/Results/curveRigid' ,param=sm ,
                               testGradient=True ,gradLB=1e-5 ,saveTrajectories=True ,
                               regWeight=1. ,maxIter=10000)
        return f, a0, tau0, fileName


    def runLandmarks(self, dt=0.001, T=.5):
        fig = plt.figure(5)
        x0 = np.array([[0,.28], [1,0]])
        a0 = np.array([[2.75,0], [-2.75,0]])
        sigma = .2
        K1 = Kernel(name='gauss', sigma=sigma, order=1)
        L = evol.landmarkEPDiff(int(np.ceil(T/dt)), x0, a0, K1)
        xt = L[0]
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Euclidean LDDMM')
        writer = FFMpegWriter(fps=5, metadata=metadata)
        dirMov = '/Users/younes/OneDrive - Johns Hopkins University/TALKS/MECHANICAL/Videos/'
        with writer.saving(fig, dirMov + "LandmarksDanse.mp4", 100):
            for t in range(0,xt.shape[0], np.maximum(1,xt.shape[0]/100)):
                fig.clf()
                ax = fig.gca()
                for k in range(xt.shape[1]):
                    ax.plot(xt[0:t+1,k,0], xt[0:t+1,k,1], markersize=10)
                    ax.plot(xt[t, k, 0], xt[t, k, 1], markersize=10, marker='o')
                plt.axis('equal')
                plt.title('t={0:.3f}'.format(t*dt))
                writer.grab_frame()
                plt.pause(0.001)

        logging.shutdown()
        plt.ioff()
        plt.show()



    def runShoot(self, dt=0.001, T=.5):
        plt.ion()
        S = self.shootingScenario(30,dt=dt, T=T)
        f = S[0]
        a0 = S[1]
        tau0 = S[2]
        fileName = S[3] + 'LDDMM'
        f.parpot = None
        f.paramRepell = None
        geod = f.geodesicEquation__(T, dt, a0, tau0, pplot=True, nsymplectic=0)
        # f.__geodesicEquation__(f.Tsize, f.fv0, f.pxt[0,:,:])
        fig = plt.figure(2)
        fvDef = curves.Curve(curve=f.fv0)
        xt = geod[0]
        FFMpegWriter = manimation.writers['ffmpeg']
        metadata = dict(title='Euclidean LDDMM')
        writer = FFMpegWriter(fps=5, metadata=metadata)
        dirMov = '/Users/younes/OneDrive - Johns Hopkins University/TALKS/MECHANICAL/Videos/'
        center = np.zeros((xt.shape[0], f.ncomponent, f.dim))
        for k in range(f.ncomponent):
            I = f.Ic[k]
            for tt in range(xt.shape[0]):
                center[tt, k, :] = f.center(xt[tt, I, :])
        with writer.saving(fig, dirMov+ fileName + ".mp4", 100):
            for t in range(0,xt.shape[0], np.maximum(1,xt.shape[0]/100)):
                fig.clf()
                ax = fig.gca()
                fvDef.updateVertices(xt[t,:,:])
                if len(f.xc)>0:
                    for kf in range(f.fvc.faces.shape[0]):
                        ax.plot(f.fvc.vertices[f.fvc.faces[kf, :], 0], f.fvc.vertices[f.fvc.faces[kf, :], 1], color=[0, 0, 0], linewidth=5)
                for kf in range(fvDef.faces.shape[0]):
                    ax.plot(fvDef.vertices[fvDef.faces[kf, :], 0], fvDef.vertices[fvDef.faces[kf, :], 1], color=[1, 0, 0], linewidth=3)
                for k in range(f.ncomponent):
                    I = f.Ic[k]
                    xDef = fvDef.vertices[I, :]
                    ax.plot(np.array([center[t, k, 0], xDef[0, 0]]),
                            np.array([center[t, k, 1], xDef[0, 1]]),
                            color=[0, .5, 0], linewidth=2)
                    ax.plot(center[0:t, k, 0], center[0:t, k, 1])
                    # ax.plot(np.array([np.mean(xDef[:,0]),xDef[0, 0]]),
                    #         np.array([np.mean(xDef[:,1]), xDef[0, 1]]),
                    #         color = [0,.5, 0], linewidth=2)
                    # ax.plot(np.mean(xt[0:t+1,I,0], axis=1), np.mean(xt[0:t+1,I,1], axis=1))
                plt.axis('equal')
                plt.title('t={0:.3f}'.format(t*dt))
                writer.grab_frame()
                plt.pause(0.001)

        logging.shutdown()
        plt.ioff()
        plt.show()
        return f

    def runMatch(self):
        plt.ion()
        N = 50
        r = 0.25
        #x = self.move(self.__circle(N,r ), b=(-1.5,1))
        x = self.__circle(N,r )
        #y = self.move(self.__c(N,2*r, r/4 ), b = (-1.2,0.8))
        fv0 = curves.Curve(pointSet=x + np.array([-0.5, 0.5]))
        fv1 = curves.Curve(pointSet=x + np.array([0.5, 0.5]))
        #fvc = curves.Curve(pointSet=0.25*x+np.array([1,0.5]))
        # fv0 = curves.Curve(curve=[curves.Curve(pointSet=x),
        #                           curves.Curve(pointSet=y)])
        # fv1 = curves.Curve(curve=[curves.Curve(pointSet=self.move(x, theta = 3*np.pi/4, b=(-0.5,1))),
        #                           curves.Curve(pointSet=self.move(y, theta = 3*np.pi/4, b=(1.5,1)))])
        # #fv1 = curves.Curve(pointSet=x+np.array([1.5, 0]))
        #fv0.component = np.zeros(fv0.component.shape, dtype=int)
        #fv1.component = np.zeros(fv1.component.shape, dtype=int)
        x = self.__vline(100, .5)
        fvc = curves.Curve(pointSet=x + [0, 0])
        xframe = self.__rectangle(100, 1.5, 1.0) + [0, 0.5]
        #fvc = curves.Curve(curve=[curves.Curve(pointSet=x - [.5,.75]),curves.Curve(pointSet=x + [.5,.75])])
        #fvc = None

        sigma = .1
        K1 = Kernel(name='laplacian', sigma=sigma)
        sigmaDist = 2.
        sigmaError = .1
        prec = 0.05
        dirOut = '/Users/younes'
        fileOut = '/Development/Results/curveMatchingRigidObstacle'

        if os.path.isfile(dirOut + '/Development/Results/curveMatchingRigid/info.tex'):
            os.remove(dirOut + '/Development/Results/curveMatchingRigid/info.tex')
        loggingUtils.setup_default_logging(dirOut + fileOut, fileName='info.txt',
                                           stdOutput=True)

        sm = CurveMatchingRigidParam(timeStep=prec, algorithm='bfgs', KparDiff=K1, sigmaDist=sigmaDist, sigmaError=sigmaError, errorType='varifoldComponent')
        f = CurveMatchingRigid(Template=fv0, Target=fv1, Clamped=fvc, pltFrame=xframe, outputDir=dirOut + fileOut, param=sm,
                          testGradient=True, gradLB=1e-5, saveTrajectories=True, regWeight=1., maxIter=10000,paramRepell= None)

        c0 = f.center(fv0.vertices)
        c1 = f.center(fv1.vertices)
        tau0 = c1 - c0
        fvTmp = curves.Curve(curve=fv0)
        fvTmp1 = curves.Curve(curve=fv1)
        x0 = fv0.vertices - c0
        x1 = fv1.vertices - c1
        fvTmp1.updateVertices(x1)
        d0 = 0
        a0 = 0
        for k in range(100):
            a = 2*np.pi*k/100
            c = np.cos(a)
            s = np.sin(a)
            x = np.zeros(fv0.vertices.shape)
            x[:,0] = c*x0[:,0] + s*x0[:,1]
            x[:,1] = -s*x0[:,0] + c*x0[:,1]
            fvTmp.updateVertices(x)
            d = curves.varifoldNorm(fvTmp, fvTmp1, f.param.KparDist)
            if k==0 or d<d0:
                a0 = a
                d0 = d

        tau = np.zeros((f.Tsize, f.ncomponent, 2))
        cos0 = (np.cos(prec*a0) - 1)/prec
        sin0 = np.sin(prec*a0)/prec
        ac0 = np.array([cos0 *c0[0] + sin0*c0[1],
                        -sin0 * c0[0] + cos0 * c0[1]])
        ac1 = np.array([cos0 *c1[0] + sin0*c1[1],
                        -sin0 * c1[0] + cos0 * c1[1]])
        for k in range(tau.shape[0]):
            t = (k)*prec
            tau[k,:,:] = tau0 - t*ac1 - (1-t)*ac0

        a0 = np.tile(a0, (f.Tsize, f.ncomponent))

        #f.optimizeMatching(tau0=tau, a0 = a0)
        f.optimizeMatching()
        # f.optimizeMatching(a0=np.array([0, 0]), tau0=0 * np.array([[1, -1], [0, 0]]))
        # f.param.sigmaError = 0.02
        # f.obj = None
        # f.optimizeMatching()
        # # f.optimizeMatching(a0=np.array([0, 0]), tau0=0 * np.array([[1, -1], [0, 0]]))
        # f.param.sigmaError = 0.004
        # f.maxIter = 1000
        # f.obj = None
        # f.optimizeMatching()

        #f.geodesicEquation(f.Tsize, f.at[0,:], f.taut[0,:,:])
        #f.__geodesicEquation__(f.Tsize, f.fv0, f.pxt[0,:,:])

        logging.shutdown()
        plt.ioff()
        plt.show()
        return f

#if __name__ == "__main__":
#    CurveMatchingRigid().runMatch()
