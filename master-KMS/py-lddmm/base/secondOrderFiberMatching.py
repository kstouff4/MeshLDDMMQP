from . import surfaces
from .pointSets import *
from . import conjugateGradient as cg, kernelFunctions as kfun, pointEvolution as evol
from .affineBasis import *

## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
class SurfaceMatchingParam:
    def __init__(self, timeStep = .1, KparDiff = None, KparDist =
                 None, sigmaKernel = 6.5, sigmaDist=2.5,
                 sigmaError=1.0, errorType = 'measure',  typeKernel='gauss'):
        self.timeStep = timeStep
        self.sigmaKernel = sigmaKernel
        self.sigmaDist = sigmaDist
        self.sigmaError = sigmaError
        self.typeKernel = typeKernel
        self.errorType = errorType
        if errorType == 'current':
            self.fun_obj0 = surfaces.currentNorm0
            self.fun_obj = surfaces.currentNormDef
            self.fun_objGrad = surfaces.currentNormGradient
        elif errorType=='measure':
            self.fun_obj0 = surfaces.measureNorm0
            self.fun_obj = surfaces.measureNormDef
            self.fun_objGrad = surfaces.measureNormGradient
        else:
            print('Unknown error Type: ', self.errorType)
        if KparDiff == None:
            self.KparDiff = kfun.Kernel(name = self.typeKernel, sigma = self.sigmaKernel)
        else:
            self.KparDiff = KparDiff
        if KparDist == None:
            self.KparDist = kfun.Kernel(name = 'gauss', sigma = self.sigmaDist)
        else:
            self.KparDist = KparDist

class Direction:
    def __init__(self):
        self.diff = []


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
class SurfaceMatching:

    def __init__(self, Template=None, Target=None, Fiber=None, fileTempl=None, fileTarg=None, fileFiber=None, param=None, initialMomentum=None,
                 maxIter=1000, regWeight = 1.0, verb=True,
                 subsampleTargetSize=-1, testGradient=True, saveFile = 'evolution', outputDir = '.'):
        if Template==None:
            if fileTempl==None:
                print('Please provide a template surface')
                return
            else:
                self.fv0 = surfaces.Surface(surf=fileTempl)
        else:
            self.fv0 = surfaces.Surface(surf=Template)
        if Target==None:
            if fileTarg==None:
                print('Please provide a list of target surfaces')
                return
            else:
                self.fv1 = []
                for f in fileTarg:
                    self.fv1.append(surfaces.Surface(surf=f))
        else:
            self.fv1 = []
            for s in Target:
                self.fv1.append(surfaces.Surface(surf=s))

        if Fiber==None:
            if fileFiber is None:
                print('Please provide fiber structure')
                return
            else:
                (self.y0, self.v0) = read3DVectorField(fileFiber)
        else:
            self.y0 = np.copy(Fiber[0])
            self.v0 = np.copy(Fiber[1])

            #print np.fabs(self.fv1.surfel-self.fv0.surfel).max()

        self.nTarg = len(self.fv1)
        self.saveRate = 10
        self.iter = 0
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print('Cannot save in ' + outputDir)
                return
            else:
                os.mkdir(outputDir)
        self.dim = self.fv0.vertices.shape[1]
        self.maxIter = maxIter
        self.verb = verb
        self.testGradient = testGradient
        self.regweight = regWeight

        if param==None:
            self.param = SurfaceMatchingParam()
        else:
            self.param = param

        self.fv0Fine = surfaces.Surface(surf=self.fv0)
        if (subsampleTargetSize > 0):
            self.fv0.Simplify(subsampleTargetSize)
            v0 = self.fv0.surfVolume()
            for s in self.fv1:
                v1 = s.surfVolume()
                if (v0*v1 < 0):
                    s.flipFaces()
            print('simplified template', self.fv0.vertices.shape[0])
        self.x0 = self.fv0.vertices
        self.fvDef = []
        for k in range(self.nTarg):
            self.fvDef.append(surfaces.Surface(surf=self.fv0))
        self.npt = self.y0.shape[0]
        self.Tsize1 = int(round(1.0/self.param.timeStep))
        self.Tsize = self.nTarg*self.Tsize1
        self.rhot = np.zeros([self.Tsize, self.y0.shape[0]])
        self.rhotTry = np.zeros([self.Tsize, self.y0.shape[0]])
        if initialMomentum==None:
            self.xt = np.tile(self.x0, [self.Tsize+1, 1, 1])
            self.a0 = np.zeros([self.y0.shape[0], self.x0.shape[1]])
            self.at = np.tile(self.a0, [self.Tsize+1, 1, 1])
            self.yt = np.tile(self.y0, [self.Tsize+1, 1, 1])
            self.vt = np.tile(self.v0, [self.Tsize+1, 1, 1])
        else:
            self.a0 = initialMomentum
            (self.xt, self.at, self.yt, self.vt)  = evol.secondOrderFiberEvolution(self.x0, self.a0, self.y0, self.v0, self.rhot, self.param.KparDiff)

        self.obj = None
        self.objTry = None
        self.gradCoeff = self.fv0.vertices.shape[0]
        self.saveFile = saveFile
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        for k,s in enumerate(self.fv1):
            s.saveVTK(self.outputDir+'/Target'+str(k)+'.vtk')

    def setOutputDir(self, outputDir):
        self.outputDir = outputDir
        if not os.access(outputDir, os.W_OK):
            if os.access(outputDir, os.F_OK):
                print('Cannot save in ' + outputDir)
                return
            else:
                os.makedirs(outputDir)


    def dataTerm(self, _fvDef):
        obj = 0
        for k,s in enumerate(_fvDef):
            obj += self.param.fun_obj(s, self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
        return obj

    def  objectiveFunDef(self, rhot, withTrajectory = False, withJacobian=False, Init = None):
        if Init == None:
            x0 = self.x0
            y0 = self.y0
            v0 = self.v0
            a0 = self.a0
        else:
            x0 = Init[0]
            y0 = Init[2]
            v0 = Init[3]
            a0 = Init[1]
            
        param = self.param
        timeStep = 1.0/self.Tsize
        dim2 = self.dim**2
        #print a0.shape
        if withJacobian:
            (xt, at, yt, vt, Jt)  = evol.secondOrderFiberEvolution(x0, a0, y0, v0, rhot, param.KparDiff, withJacobian=True)
        else:
            (xt, at, yt, vt)  = evol.secondOrderFiberEvolution(x0, a0, y0, v0, rhot, param.KparDiff)
        #print xt[-1, :, :]
        #print obj
        obj=0
        for t in range(self.Tsize):
            v = np.squeeze(vt[t, :, :])
            rho = np.squeeze(rhot[t, :])
            
            obj = obj + timeStep*((rho**2) * (v**2).sum(axis=1)).sum()/2
            #print xt.sum(), at.sum(), obj
        if withJacobian:
            return obj, xt, at, yt, vt, Jt
        elif withTrajectory:
            return obj, xt, at, yt, vt
        else:
            return obj

    def  _objectiveFun(self, rhot, withTrajectory = False):
        (obj, xt, at, yt, vt) = self.objectiveFunDef(rhot, withTrajectory=True)
        for k in range(self.nTarg):
            self.fvDef[k].updateVertices(np.squeeze(xt[(k+1)*self.Tsize1, :, :]))
        obj0 = self.dataTerm(self.fvDef)

        if withTrajectory:
            return obj+obj0, xt, at, yt, vt
        else:
            return obj+obj0

    def objectiveFun(self):
        if self.obj == None:
            (self.obj, self.xt, self.at, self.yt, self.vt) = self.objectiveFunDef(self.rhot, withTrajectory=True)
            self.obj0 = 0
            for k in range(self.nTarg):
                self.obj0 += self.param.fun_obj0(self.fv1[k], self.param.KparDist) / (self.param.sigmaError**2)
                foo = surfaces.Surface(surf=self.fvDef[k])
                self.fvDef[k].updateVertices(np.squeeze(self.xt[(k+1)*self.Tsize1, :, :]))
                foo.computeCentersAreas()
            self.obj += self.obj0 + self.dataTerm(self.fvDef)
            #print self.obj0,  self.dataTerm(self.fvDef)
        return self.obj

    def getVariable(self):
        return self.rhot

    def updateTry(self, dir, eps, objRef=None):
        objTry = self.obj0
        rhotTry = self.rhot - eps * dir.diff
        foo = self.objectiveFunDef(rhotTry, withTrajectory=True)
        objTry += foo[0]

        ff = [] 
        for k in range(self.nTarg):
            ff.append(surfaces.Surface(surf=self.fvDef[k]))
            ff[k].updateVertices(np.squeeze(foo[1][(k+1)*self.Tsize1, :, :]))
        objTry += self.dataTerm(ff)
        if np.isnan(objTry):
            print('Warning: nan in updateTry')
            return 1e500

        if (objRef == None) | (objTry < objRef):
            self.rhotTry = rhotTry
            self.objTry = objTry
            #print 'objTry=',objTry, dir.diff.sum()

        return objTry



    def endPointGradient(self):
        px = []
        for k in range(self.nTarg):
            px.append(-self.param.fun_objGrad(self.fvDef[k], self.fv1[k], self.param.KparDist)/ self.param.sigmaError**2)
        return px 


    def getGradient(self, coeff=1.0):
        px1 = self.endPointGradient()
        pa1 = []
        py1 = []
        pv1 = []
        for k in range(self.nTarg):
            pa1.append(np.zeros(self.a0.shape))
            py1.append(np.zeros(self.y0.shape))
            pv1.append(np.zeros(self.v0.shape))
        foo = evol.secondOrderFiberGradient(self.x0, self.a0, self.y0, self.v0, self.rhot, px1, pa1, py1, pv1,
                                            self.param.KparDiff, times = (1+np.array(range(self.nTarg)))*self.Tsize1)
        grd = Direction()
        grd.diff = foo[0]/(coeff*self.rhot.shape[0])
        return grd



    def addProd(self, dir1, dir2, beta):
        dir = Direction()
        dir.diff = dir1.diff + beta * dir2.diff
        return dir

    def copyDir(self, dir0):
        dir = Direction()
        dir.diff = np.copy(dir0.diff)

        return dir


    def randomDir(self):
        dirfoo = Direction()
        dirfoo.diff = np.random.randn(self.Tsize, self.npt)
        return dirfoo

    def dotProduct(self, g1, g2):
        res = np.zeros(len(g2))
        gg = g1.diff
        vn = (self.vt[0:-1,...]**2).sum(axis=2)
        ll = 0
        for gr in g2:
            ggOld = gr.diff
            res[ll]  = (ggOld*vn*gg).sum()
            #res[ll]  = (ggOld*gg).sum()
            ll = ll+1

        return res

    def acceptVarTry(self):
        self.obj = self.objTry
        self.rhot = np.copy(self.rhotTry)
        #print self.at

    def endOfIteration(self):
        self.iter += 1
        if (self.iter % self.saveRate == 0):
            (obj1, self.xt, self.at, self.yt, self.vt) = self.objectiveFunDef(self.rhot, withTrajectory=True)
            for k in range(self.nTarg):
                self.fvDef[k].updateVertices(np.squeeze(self.xt[(k+1)*self.Tsize1, :, :]))
            (xt, at, yt, vt, zt, Jt)  = evol.secondOrderFiberEvolution(self.x0, self.a0, self.y0, self.v0,  self.rhot, self.param.KparDiff,
                                                           withPointSet = self.fv0Fine.vertices, withJacobian=True)
            fvDef = surfaces.Surface(surf=self.fv0Fine)
            for kk in range(self.Tsize+1):
                fvDef.updateVertices(np.squeeze(zt[kk, :, :]))
                fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = Jt[kk, :], scal_name='Jacobian')
                vn = np.squeeze(vt[kk,...])
                vn = vn/np.sqrt((vn**2).sum(axis=1))[:, np.newaxis]
                savePoints(self.outputDir +'/fiber_'+ self.saveFile+str(kk)+'.vtk', yt[kk, ...], vector=vn)
                #self.fvDef.saveVTK(self.outputDir +'/'+ self.saveFile+str(kk)+'.vtk', scalars = self.idx, scal_name='Labels')
        else:
            (obj1, self.xt, self.at, self.yt, self.vt) = self.objectiveFunDef(self.rhot, withTrajectory=True)
            for k in range(self.nTarg):
                self.fvDef[k].updateVertices(np.squeeze(self.xt[(k+1)*self.Tsize1, :, :]))


    def optimizeMatching(self):
        #print 'dataterm', self.dataTerm(self.fvDef)
        #print 'obj fun', self.objectiveFun(), self.obj0
        grd = self.getGradient(self.gradCoeff)
        [grd2] = self.dotProduct(grd, [grd])

        self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
        print('Gradient lower bound:', self.gradEps)
        #print 'x0:', self.x0
        #print 'y0:', self.y0
        
        cg.cg(self, verb = self.verb, maxIter = self.maxIter,TestGradient=self.testGradient, epsInit=0.1)
        #return self.at, self.xt

