import numpy as np
import numpy.linalg as la
import logging
from . import pointEvolution as evol
from .surfaces import Surface, extract_components_
from .curves import Curve
from .curve_distances import measureNorm0, measureNormDef, measureNormGradient
from .curve_distances import currentNorm0, currentNormDef, currentNormGradient
from .curve_distances import varifoldNorm0, varifoldNormDef, varifoldNormGradient
from . import pointSets
from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from .surfaceMatching import SurfaceMatchingParam, SurfaceMatching
from .surfaceSection import SurfaceSection, Surf2SecDist, Surf2SecGrad, Hyperplane, readFromTXT




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
class SurfaceToSectionsMatching(SurfaceMatching):
    def __init__(self, Template=None, Target=None, param=None, maxIter=1000, passenger = None, componentMap=None,
                 regWeight = 1.0, affineWeight = 1.0, internalWeight=1.0, mode="normal",
                 subsampleTargetSize=-1, affineOnly = False,
                 rotWeight = None, scaleWeight = None, transWeight = None, symmetric = False,
                 saveFile = 'evolution', select_planes = None, forceClosed = False,
                 saveTrajectories = False, affine = 'none', outputDir = '.',pplot=True):
        self.forceClosed = forceClosed
        self.colors = ('b', 'm', 'g', 'r', 'y', 'k')
        super().__init__(Template=Template, Target=Target, param=param, maxIter=maxIter, passenger = passenger,
                 regWeight = regWeight, affineWeight = affineWeight, internalWeight=internalWeight, mode=mode,
                 subsampleTargetSize=subsampleTargetSize, affineOnly = affineOnly,
                 rotWeight = rotWeight, scaleWeight = scaleWeight, transWeight = transWeight, symmetric = symmetric,
                 saveFile = saveFile,
                 saveTrajectories = saveTrajectories, affine = affine, outputDir = outputDir, pplot=pplot)

        self.hyperplanes = None

        self.set_fun(self.param.errorType)
        self.set_template_and_target(Template, Target, subsampleTargetSize, misc = [select_planes, componentMap])
        self.match_landmarks = False
        self.def_lmk = None
        print(f'Template has {self.fv0.vertices.shape[0]} vertices')
        print(f'There are {self.hyperplanes.shape[0]} planes')
        print(f'There are {len(self.fv1)} target curves')


        #print(self.componentMap)
        self.saveRate = 25
        # self.forceLineSearch = False



    def set_template_and_target(self, Template, Target, subsampleTargetSize=-1, misc = None):
        if Template is None:
            logging.error('Please provide a template surface')
            return
        else:
            self.fv0 = Surface(surf=Template)

        if type(Target) in (list, tuple):
            if type(Target[0]) == str:
                self.fv1, self.hyperplanes = readFromTXT(Target, forceClosed=self.forceClosed)
            else:
                self.fv1 = Target[0]
                self.hyperplanes = Target[1]
        elif type(Target) == str:
            self.fv1, self.hyperplanes = readFromTXT(Target, forceClosed=self.forceClosed)
        else:
            logging.error('Target must be a list or tuple of SurfaceSection')
            return

        # if fv1 is not None:
        #     c = []
        #     found_h = np.zeros(len(fv1), dtype=bool)
        #     for k in range(self.hyperplanes.shape[0]):
        #         c.append([])
        #         for i in range(len(fv1)):
        #             if not found_h[i]:
        #                 u = min(np.fabs(self.hyperplanes[k, :3] - fv1[i].hyperplane.u).sum()
        #                         + np.fabs(self.hyperplanes[k, 3] - fv1[i].hyperplane.offset),
        #                         np.fabs(self.hyperplanes[k, :3] + fv1[i].hyperplane.u).sum()
        #                         + np.fabs(self.hyperplanes[k, 3] + fv1[i].hyperplane.offset))
        #                 if u<1e-2:
        #                     c[k].append(fv1[i].curve)
        #                     found_h[i] = True
        #
        #     self.fv1 = []
        #     for i in range(len(c)):
        #         cv = Curve(curve=c[i])
        #         h = Hyperplane(u=self.hyperplanes[i,:3], offset=self.hyperplanes[3])
        #         self.fv1.append(SurfaceSection(curve=cv, hyperplane=h, hypLabel=i))

        if misc is None:
            select_planes = None
            componentMap = None
        else:
            select_planes = misc[0]
            componentMap = misc[1]

        if select_planes is not None:
            self.selectPlanes((select_planes))
        self.fvInit = Surface(surf=self.fv0)
        self.fix_orientation()
        self.fv0.saveVTK(self.outputDir+'/Template.vtk')
        outCurve = ()
        normals = np.zeros((0, 3))
        npt = 0
        for k in range(self.hyperplanes.shape[0]):
            if componentMap is not None:
                target_label = np.where(componentMap[:, k])[0]
                surf_, J = self.fv0.extract_components(target_label)
            else:
                surf_ = self.fv0
            h = Hyperplane(u=self.hyperplanes[k, :3], offset=self.hyperplanes[k, 3])
            f2 = SurfaceSection(hyperplane=h, surf=surf_)
            npt += f2.curve.vertices.shape[0]
            outCurve += (f2.curve,)
            normals = np.concatenate((normals, f2.normals), axis=0)
        labels = np.zeros(npt, dtype=int)
        npt = 0
        for k, f in enumerate(outCurve):
            npt2 = npt + f.vertices.shape[0]
            labels[npt:npt2] = k + 1
            npt = npt2
        c = Curve(outCurve)
        c.saveVTK(self.outputDir + f'/InitialCurves.vtk', cell_normals=normals, scalars=labels, scal_name='labels')

        if self.fv1:
            outCurve = ()
            normals = np.zeros((0,3))
            npt = 0
            for k,f in enumerate(self.fv1):
                npt += f.curve.vertices.shape[0]
                outCurve += (f.curve,)
                normals = np.concatenate((normals, f.normals), axis=0)
            labels = np.zeros(npt, dtype=int)
            npt = 0
            for k,f in enumerate(self.fv1):
                npt2 = npt + f.curve.vertices.shape[0]
                labels[npt:npt2] = k+1
                npt = npt2
            c = Curve(outCurve)
            c.saveVTK(self.outputDir+f'/TargetCurves.vtk', cell_normals=normals, scalars=labels, scal_name='labels')
        self.dim = self.fv0.vertices.shape[1]

        self.ncomp_template = self.fv0.component.max()+1
        self.ncomp_target = len(self.fv1)
        if componentMap is None or componentMap.shape[0] != self.ncomp_template \
            or componentMap.shape[1] != self.hyperplanes.shape[0]:
            self.componentMap = np.ones((self.ncomp_template, self.hyperplanes.shape[0]), dtype=bool)
        else:
            self.componentMap = componentMap
        self.fv0.getEdges()
        self.component_structure = []
        for k,f in enumerate(self.fv1):
            target_label = np.where(self.componentMap[:, f.hypLabel])[0]
            self.component_structure.append(extract_components_(target_label, self.fv0.vertices.shape[0], self.fv0.faces,
                                                                self.fv0.component,
                                                                edge_info = (self.fv0.edges, self.fv0.faceEdges, self.fv0.edgeFaces)))


    def selectPlanes(self, hs):
        fv1 = ()
        for k,f in enumerate(self.fv1):
            if f.hypLabel in hs:
                fv1 += (f,)
        self.fv1 = fv1

    # def readTargetFromTXT(self, filename):
    #     self.fv1 = ()
    #     with open(filename, 'r') as f:
    #         s = f.readline()
    #         nc = int(s)
    #         for i in range(nc):
    #             s = f.readline()
    #             npt = int(s)
    #             pts = np.zeros((npt,3))
    #             for j in range(npt):
    #                 s = f.readline()
    #                 pts[j,:] = s.split()
    #             c = Curve(pts)
    #             self.fv1 += (SurfaceSection(curve=c),)
    #     uo = np.zeros((nc, 4))
    #     hyp = np.zeros(nc, dtype=int)
    #     self.area = np.zeros(nc)
    #     self.outer = np.zeros(nc, dtype=bool)
    #     nh = 0
    #     tol = 1e-5
    #     hk = np.zeros(4)
    #     for k,f in enumerate(self.fv1):
    #         self.area[k] = f.curve.enclosedArea()
    #         found = False
    #         hk[:3] = f.hyperplane.u
    #         hk[3] = f.hyperplane.offset
    #         if k > 0:
    #             dst = np.sqrt(((hk - uo[:nh, :])**2).sum(axis=1))
    #             dst2 = np.sqrt(((hk + uo[:nh, :])**2).sum(axis=1))
    #             if dst.min() < tol:
    #                 i = np.argmin(dst)
    #                 hyp[k] = i
    #                 found = True
    #             elif dst2.min() < tol:
    #                 i = np.argmin(dst)
    #                 hyp[k] = i
    #                 found = True
    #         if not found:
    #             uo[nh, :] = hk
    #             hyp[k] = nh
    #             nh += 1
    #     self.hyperplanes = uo[:nh, :]
    #     self.inHyperplane = hyp
    #     for k,f in enumerate(self.fv1):
    #         f.hyperplane.u = uo[hyp[k], :3]
    #         f.hyperplane.offset = uo[hyp[k], 3]
    #
    #     for k in range(nh):
    #         J = np.nonzero(self.inHyperplane == k)[0]
    #         j = np.argmax(np.fabs(self.area[J]))
    #         self.outer[J[j]] = True
    #
    #     eps = 1e-4
    #     for k,f in enumerate(self.fv1):
    #         c = Curve(curve=f.curve)
    #         n = np.zeros(c.vertices.shape)
    #         for i in range(c.faces.shape[0]):
    #             n[c.faces[i, 0], :] += f.normals[i]/2
    #             n[c.faces[i, 1], :] += f.normals[i] / 2
    #         c.updateVertices(c.vertices + eps*n)
    #         a = np.fabs(c.enclosedArea())
    #         if (a > self.area[k] and not self.outer[k]) or \
    #                 (a < self.area[k] and self.outer[k]):
    #             f.curve.flipFaces()
    #             f.normals *= -1
    #

    def fix_orientation(self, fv1=None):
        self.fv0ori = 1
        self.fv1ori = 1



    def initial_plot(self):
        fig = plt.figure(3)
        ax = Axes3D(fig)#, auto_add_to_figure=False)
        fig.add_axes(ax)
        lim1 = self.addSurfaceToPlot(self.fv0, ax, ec='k', fc='r', al=0.2)
        for k,f in enumerate(self.fv1):
            lim0 = self.addCurveToPlot(f, ax, ec=self.colors[k%len(self.colors)], fc='b', lw=5)
            for i in range(3):
                lim1[i][0] = min(lim0[i][0], lim1[i][0])
                lim1[i][1] = max(lim0[i][1], lim1[i][1])
        ax.set_xlim(lim1[0][0], lim1[0][1])
        ax.set_ylim(lim1[1][0], lim1[1][1])
        ax.set_zlim(lim1[2][0], lim1[2][1])
        fig.canvas.flush_events()

    def set_fun(self, errorType, vfun=None):
        self.param.errorType = errorType
        if errorType == 'current':
            #print('Running Current Matching')
            self.fun_obj0 = partial(currentNorm0, KparDist=self.param.KparDist, weight=1.)
            self.fun_obj = partial(currentNormDef, KparDist=self.param.KparDist, weight=1.)
            self.fun_objGrad = partial(currentNormGradient, KparDist=self.param.KparDist, weight=1.)
        elif errorType=='measure':
            #print('Running Measure Matching')
            self.fun_obj0 = partial(measureNorm0, KparDist=self.param.KparDist, cpu=True)
            self.fun_obj = partial(measureNormDef,KparDist=self.param.KparDist, cpu=True)
            self.fun_objGrad = partial(measureNormGradient,KparDist=self.param.KparDist, cpu=True)
        elif errorType=='varifold':
            self.fun_obj0 = partial(varifoldNorm0, KparDist=self.param.KparDist, weight=1.)
            self.fun_obj = partial(varifoldNormDef, KparDist=self.param.KparDist, weight=1.)
            self.fun_objGrad = partial(varifoldNormGradient, KparDist=self.param.KparDist, weight=1.)
        else:
            print('Unknown error Type: ', self.param.errorType)


    def dataTerm(self, _fvDef, fv1 = None, _fvInit = None, _lmk_def = None, lmk1 = None):
        #print('starting dt')
        if fv1 is None:
            fv1 = self.fv1
        obj = 0
        for k,f in enumerate(fv1):
            #logging.info(f'     Target {k}')
                                #target_label=np.where(self.componentMap[:,k])[0]) # curveDist0=self.fun_obj0) plot=101+k)
            obj += Surf2SecDist(_fvDef, f, self.fun_obj, target_comp_info=self.component_structure[k])
        obj /= self.param.sigmaError**2
        #print('ending dt')
        return obj

    def objectiveFun(self):
        if self.obj is None:
            self.obj0 = 0
            for f in self.fv1:
                self.obj0 += self.fun_obj0(f.curve) / (self.param.sigmaError**2)
            (self.obj, self.xt) = self.objectiveFunDef(self.at, self.Afft, withTrajectory=True)
            self.fvDef.updateVertices(np.squeeze(self.xt[-1, :, :]))
            self.obj += self.obj0 + self.dataTerm(self.fvDef)
        return self.obj


    def endPointGradient(self, endPoint=None):
        #print('starting epg')
        if endPoint is None:
            endPoint = self.fvDef
        px = np.zeros(endPoint.vertices.shape)
        for k,f in enumerate(self.fv1):
            #print(k, endPoint.vertices.shape)
            px += Surf2SecGrad(endPoint, f, self.fun_objGrad, target_comp_info=self.component_structure[k])
                               #target_label=np.where(self.componentMap[:,k])[0])
        #print('ending epg')
        return px / self.param.sigmaError**2

    def saveCorrectedTarget(self, X0, X1):
        U = la.inv(X0[-1])
        outCurve = ()
        for k, f0 in enumerate(self.fv1):
            fc = Curve(curve=f0.curve)
            yyt = np.dot(fc.vertices - X1[-1,...], U.T)
            fc.updateVertices(yyt)
            outCurve += (fc,)
        fc = Curve(curve=outCurve)
        fc.saveVTK(self.outputDir + f'/TargetCurveCorrected.vtk')


    def addCurveToPlot(self, fv1, ax, ec = 'b', fc = 'r', al=.5, lw=1):
        return fv1.curve.addToPlot(ax, ec = ec, fc = fc, al=al, lw=lw)


    def plotAtIteration(self):
        fig = plt.figure(4)
        fig.clf()
        ax = Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)
        lim1 = self.addSurfaceToPlot(self.fvDef, ax, ec='k', fc='r', al=0.2)
        for k, f in enumerate(self.fv1):
            lim0 = self.addCurveToPlot(f, ax, ec=self.colors[k % len(self.colors)], fc='b', lw=5)
            for i in range(3):
                lim1[i][0] = min(lim0[i][0], lim1[i][0])
                lim1[i][1] = max(lim0[i][1], lim1[i][1])
        ax.set_xlim(lim1[0][0], lim1[0][1])
        ax.set_ylim(lim1[1][0], lim1[1][1])
        ax.set_zlim(lim1[2][0], lim1[2][1])
        fig.canvas.flush_events()

    def saveHdf5(self, fileName):
        pass


    def endOfIteration(self, forceSave=False):
        super().endOfIteration(forceSave = forceSave)
        if forceSave or self.iter % self.saveRate == 0:
            outCurve = ()
            normals = np.zeros((0,3))
            npt = 0
            for k in range(self.hyperplanes.shape[0]):
                target_label = np.where(self.componentMap[:, k])[0]
                h = Hyperplane(u=self.hyperplanes[k,:3], offset=self.hyperplanes[k,3])
                surf_, J = self.fvDef.extract_components(target_label)
                f2 = SurfaceSection(hyperplane=h, surf=surf_)
                npt += f2.curve.vertices.shape[0]
                outCurve += (f2.curve,)
                normals = np.concatenate((normals, f2.normals), axis=0)
            labels = np.zeros(npt, dtype=int)
            npt = 0
            for k,f in enumerate(outCurve):
                npt2 = npt + f.vertices.shape[0]
                labels[npt:npt2] = k+1
                npt = npt2
            c = Curve(outCurve)
            c.saveVTK(self.outputDir+f'/DeformedCurves.vtk', cell_normals=normals, scalars=labels, scal_name='labels')




    # def optimizeMatching(self):
    #     #print 'dataterm', self.dataTerm(self.fvDef)
    #     #print 'obj fun', self.objectiveFun(), self.obj0
    #     self.coeffAff = self.coeffAff2
    #     grd = self.getGradient(self.gradCoeff)
    #     [grd2] = self.dotProduct(grd, [grd])
    #
    #     self.gradEps = max(0.001, np.sqrt(grd2) / 10000)
    #     self.epsMax = 5.
    #     logging.info('Gradient lower bound: %f' %(self.gradEps))
    #     self.coeffAff = self.coeffAff1
    #     if self.param.algorithm == 'cg':
    #         cg.cg(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=0.1)
    #     elif self.param.algorithm == 'bfgs':
    #         bfgs.bfgs(self, verb = self.verb, maxIter = self.maxIter, TestGradient=self.testGradient, epsInit=1.,
    #                   Wolfe=self.param.wolfe, memory=50)
    #     #return self.at, self.xt

