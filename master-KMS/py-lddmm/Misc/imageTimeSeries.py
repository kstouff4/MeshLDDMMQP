from . import conjugateGradient, loggingUtils
import numpy
import optparse
import os
import shutil
import logging
import multiprocessing
from . import fftHelper
from . import imageTimeSeriesConfig
#import tvtk
#import gradientDescent
import numexpr as ne

class ImageTimeSeries(object):

    def __init__(self, output_dir, config_name):
        # can override these in configuration scripts
        self.num_points = None
        self.domain_max = None
        self.dx = None
        self.output_dir = output_dir
        imageTimeSeriesConfig.configure(self, config_name)

        # general configuration
        self.mu = numpy.zeros((self.rg.num_nodes, 3, self.num_times))
        self.v = numpy.zeros((self.rg.num_nodes, 3, self.num_times))
        self.objTry = 0.
        self.mu_state = self.mu.copy()
        self.optimize_iteration = 0

        # initialize the V kernel multiplier
        i = complex(0,1)
        r_sqr_xsi = (numpy.power(i*self.rg.xsi_1,2) + \
                            numpy.power(i*self.rg.xsi_2,2) + \
                            numpy.power(i*self.rg.xsi_3,2))
        self.KernelV = 1.0 / numpy.power(self.gamma - self.alpha * (r_sqr_xsi), \
                            self.Lpower)

        self.pool = multiprocessing.Pool(self.pool_size)

        test_mu = numpy.zeros_like(self.mu[...,0])
        test_mu[1850,0] = 1.0
        test_v = fftHelper.applyKernel(test_mu, self.rg.dims, \
                            self.rg.num_nodes, self.KernelV, \
                            self.rg.element_volumes[0])
        self.rg.create_vtk_sg()
        self.rg.add_vtk_point_data(test_v.real, "test_v")
        self.rg.vtk_write(0, "kernel_test", self.output_dir)

        self.cache_v = False
        self.updateEvolutions()
        self.v_state = self.v.copy()

    def get_sim_data(self):
        return [self.rg, self.num_points, self.num_times]

    def kMu(self, async=False):
        rg, N, T = self.get_sim_data()
        if async:
            res = []
            for t in range(T):
                res.append(self.pool.apply_async(fftHelper.applyKernel, \
                        args=(self.mu[:,:,t].copy(), rg.dims, rg.num_nodes,\
                        self.KernelV, rg.element_volumes[0])))
            for t in range(T):
                self.v[:,:,t] = res[t].get(timeout=self.pool_timeout)
        else:
            for t in range(T):
              self.v[:,:,t] = fftHelper.applyKernel(self.mu[:,:,t], rg.dims, \
                              rg.num_nodes, self.KernelV, \
                              rg.element_volumes[0])
        self.v = self.v.real

    def updateEvolutions(self):
        rg, N, T = self.get_sim_data()
        self.I_interp[:,0] = self.I[:,0].copy()
        for t in range(T-1):
            vt = self.v[:,:,t]
            dt = self.dt
            #w = -1.0 * vt * dt
            w = ne.evaluate("-1.0 *  vt * dt")
            w = w.reshape((rg.dims[0], rg.dims[1], rg.dims[2], 3))
            self.I_interp[:,t+1] = rg.grid_interpolate3d(self.I_interp[:,t], \
                                w, useFortran=True)
        if self.verbose_file_output:
            self.writeData("evolutions%d" % (self.optimize_iteration))

    def writeData(self, name):
        rg, N, T = self.get_sim_data()
        for t in range(T):
            rg.create_vtk_sg()
            rg.add_vtk_point_data(self.v[:,:,t], "v")
            rg.add_vtk_point_data(self.I[:,t], "I")
            rg.add_vtk_point_data(self.I_interp[:,t], "I_interp")
            rg.add_vtk_point_data(self.I[:,t]-self.I_interp[:,t], "diff")
            rg.add_vtk_point_data(self.p[:,t], "p")
            rg.add_vtk_point_data(self.mu[:,:,t], "mu")
            rg.vtk_write(t, name, output_dir=self.output_dir)
            self.sc.data = self.I_interp[:,t]
            self.sc.saveAnalyze("%s/%s_I_%d" % (self.output_dir, name, \
                                 t), rg.num_points)

    def cacheDir(self, mu_dir):
        self.cache_v = True
        self.cache_v_dir = numpy.zeros_like(self.v)
        rg, N, T = self.get_sim_data()
        res = []
        for t in range(T):
            res.append(self.pool.apply_async(fftHelper.applyKernel, \
                            args=(mu_dir[:,:,t].copy(), rg.dims, rg.num_nodes,\
                            self.KernelV, rg.element_volumes[0])))
        for t in range(T):
            self.cache_v_dir[:,:,t] = res[t].get(timeout=self.pool_timeout)

    # **********************************************************************
    # Implementation of Callback functions for non-linear conjugate gradient
    # **********************************************************************
    def getVariable(self):
        return self

    def objectiveFun(self):
        rg, N, T = self.get_sim_data()
        obj = 0.
        term2 = 0.
        for t in range(T):
            if t<T-1:
                kn = 0.
                kn += numpy.dot(self.mu[:,0,t], self.v[:,0,t])
                kn += numpy.dot(self.mu[:,1,t], self.v[:,1,t])
                kn += numpy.dot(self.mu[:,2,t], self.v[:,2,t])
                obj += self.dt * kn
            if t in range(0, self.num_times, self.num_times_disc):
                term2 += numpy.power(self.I_interp[:,t] - self.I[:,t],2).sum()
        term2 *= rg.element_volumes[0]
        total_fun = obj + 1./numpy.power(self.sigma,2) * term2
        logging.info("term1: %e, term2: %e, tot: %e" % (obj, term2, total_fun))
        return total_fun

    def updateTry(self, direction, eps, objRef=None):
        rg, N, T = self.get_sim_data()
        self.last_dir = eps * direction
        mu_old = self.mu.copy()
        v_old = self.v.copy()
        Ii_old = self.I_interp.copy()
        self.mu = self.mu_state - direction * eps
        if not self.cache_v:
            self.kMu(async=True)
        else:
            self.v = self.v_state - self.cache_v_dir * eps
        self.updateEvolutions()
        objTry = self.objectiveFun()
        if (objRef != None) and (objTry > objRef):
            self.mu = mu_old
            self.v = v_old
            self.I_interp = Ii_old
        return objTry

    def acceptVarTry(self):
        rg, N, T = self.get_sim_data()
        for t in range(T):
            self.mu_state[:,:,t] = self.mu[:,:,t].copy()
            self.v_state[...,t] = self.v[...,t].copy()
        self.cache_v = False

    def getGradient(self, coeff=1.0):
        rg, N, T = self.get_sim_data()
        dt = self.dt
        self.p[...] = 0.
        vol = rg.element_volumes[0]
        gIi = numpy.empty((rg.num_nodes,3,T))
        pr = numpy.empty((rg.num_nodes,1,T))
        pr[:,0,T-1] = 2./numpy.power(self.sigma, 2) * vol * \
                                    (-self.I[:,T-1]+self.I_interp[:,T-1])
        for t in range(T-1,-1,-1):
            p1 = pr[:,0,t]
            if (t in range(0, self.num_times, self.num_times_disc)):
                if t!=T-1:
                    s = 2./numpy.power(self.sigma,2)
                    It = self.I[:,t]
                    Iit = self.I_interp[:,t]
                    #p1 = (p1 - s * vol * (It - Iit))
                    p1 = ne.evaluate("(p1 - s * vol * (It - Iit))")
            v = self.v[:,:,t]
            v.shape = ((rg.dims[0], rg.dims[1], rg.dims[2], 3))
            (p_new, gI_interp) = rg.grid_interpolate3d_dual_and_grad( \
                            p1.reshape(rg.dims), self.I_interp[:,t], v, dt, \
                            True)
            if t>0:
                pr[:,0,t-1] = p_new[...]
            gIi[...,t] = gI_interp[...]
        mu = self.mu
        #grad = 2*mu - pr * gIi
        grad = ne.evaluate("2*mu - pr * gIi")
        self.p = pr[:,0,:]
        #retGrad = coeff * grad
        retGrad = ne.evaluate("coeff * grad")
        #import pdb
        #pdb.set_trace()
        if self.verbose_file_output:
            for t in range(T):
                rg.create_vtk_sg()
                rg.add_vtk_point_data(gIi[...,t], "gIi")
                rg.vtk_write(t, "gradE", self.output_dir)
        return retGrad

    def dotProduct(self, g1, g2):
        rg, N, T = self.get_sim_data()
        prod = numpy.zeros(len(g2))
        vol = rg.element_volumes[0]
        for ll in range(len(g2)):
            gr = g2[ll]
            res = []
            for t in range(T):
                res.append(self.pool.apply_async(fftHelper.applyKernel, \
                        args=(gr[:,:,t].copy(), rg.dims, rg.num_nodes,\
                        self.KernelV, rg.element_volumes[0])))
            for t in range(T):
                kgr = res[t].get(timeout=self.pool_timeout)
                for d in range(self.dim):
                    prod[ll] += self.dt * numpy.dot(g1[:,d,t], kgr[:,d])
        return prod

    def endOfIteration(self):
        self.optimize_iteration += 1
        if (self.optimize_iteration % self.write_iter == 0):
            self.writeData("iter%d" % (self.optimize_iteration))

    def endOptim(self):
        self.writeData("final")
    # ***********************************************************************
    # end of non-linear cg callbacks
    # ***********************************************************************

    def computeMatching(self):
        conjugateGradient.cg(self, True, maxIter = 500, TestGradient=False, \
                             epsInit=self.cg_init_eps)
        #gradientDescent.descend(self, True, maxIter=1000, TestGradient=False,\
        #                    epsInit=self.cg_init_eps)
        return self

    def computeMaps(self, landmarks=[]):
        rg, N, T = self.get_sim_data()
        h = numpy.zeros_like(self.v)
        b = numpy.zeros_like(self.v)
        h[...,0] = rg.nodes
        b[...,0] = rg.nodes
        Jh = numpy.zeros((rg.num_nodes, T))
        Jh[:,0] = numpy.ones(rg.num_nodes)
        Jb = numpy.zeros((rg.num_nodes, T))
        Jb[:,0] = numpy.ones(rg.num_nodes)
        land_t = numpy.zeros((len(landmarks),3,T))
        if len(landmarks) > 0:
            land_t[...,0] = numpy.array(landmarks)
            lmk_coords = [rg.barycentric_coordinates(lmk) for lmk in landmarks]
        for t in range(T-1):
            vt = self.v[:,:,t]
            vt_flip = self.v[:,:,T-1-t]
            dt = self.dt
            #w = -1.0 *  vt * dt
            w = ne.evaluate("-1.0 *  vt * dt")
            w = w.reshape((rg.dims[0], rg.dims[1], rg.dims[2], 3))
            for d in range(self.dim):
                h[:,d,t+1] = rg.interpolate3d(h[:,d,t],w, True)
            #w = 1.0 *  vt_flip * dt
            w = ne.evaluate("1.0 *  vt_flip * dt")
            w = w.reshape((rg.dims[0], rg.dims[1], rg.dims[2], 3))
            for d in range(self.dim):
                b[:,d,t+1] = rg.interpolate3d(b[:,d,t], w, True)
            dh0 = rg.gradient(h[:,0,t+1]).real
            dh1 = rg.gradient(h[:,1,t+1]).real
            dh2 = rg.gradient(h[:,2,t+1]).real
            db0 = rg.gradient(b[:,0,t+1]).real
            db1 = rg.gradient(b[:,1,t+1]).real
            db2 = rg.gradient(b[:,2,t+1]).real
            Jh[:,t+1] = dh0[:,0]*(dh1[:,1]*dh2[:,2]-dh2[:,1]*dh1[:,2]) - \
                dh0[:,1]*(dh1[:,0]*dh2[:,2]-dh2[:,0]*dh1[:,2]) + \
                dh0[:,2]*(dh1[:,0]*dh2[:,1]-dh2[:,0]*dh1[:,1])
            Jb[:,t+1] = db0[:,0]*(db1[:,1]*db2[:,2]-db2[:,1]*db1[:,2]) - \
                db0[:,1]*(db1[:,0]*db2[:,2]-db2[:,0]*db1[:,2]) + \
                db0[:,2]*(db1[:,0]*db2[:,1]-db2[:,0]*db1[:,1])
            # gradient above not computed at boundary, set J to 1 there
            Jh[rg.edge_nodes, t+1] = 1.
            Jb[rg.edge_nodes, t+1] = 1.
            # evolve the landmark set
            for lmk_j, lmk in enumerate(landmarks):
                inodes = rg.elements[lmk_coords[lmk_j][0],:]
                ax = lmk_coords[lmk_j][1][0]
                ay = lmk_coords[lmk_j][1][1]
                az = lmk_coords[lmk_j][1][2]
                for d in range(self.dim):
                    ib = b[:,d,t+1]
                    land_t[lmk_j,d,t+1] = ib[inodes[0]]*(1-ax)*(1-ay)*(1-az) + \
                                         ib[inodes[1]]*(ax)*(1-ay)*(1-az) + \
                                         ib[inodes[3]]*(1-ax)*(ay)*(1-az) + \
                                         ib[inodes[2]]*(ax)*(ay)*(1-az) + \
                                         ib[inodes[4]]*(1-ax)*(1-ay)*(az) + \
                                         ib[inodes[5]]*(ax)*(1-ay)*(az) + \
                                         ib[inodes[7]]*(1-ax)*(ay)*(az) + \
                                         ib[inodes[6]]*(ax)*(ay)*(az)
        for t in range(T):
            rg.create_vtk_sg()
            rg.add_vtk_point_data(h[...,t], "h")
            rg.add_vtk_point_data(h[...,t]-rg.nodes, "hd")
            rg.add_vtk_point_data(b[...,t], "b")
            rg.add_vtk_point_data(b[...,t]-rg.nodes, "bd")
            rg.add_vtk_point_data(Jh[:,t], "Jh")
            rg.add_vtk_point_data(Jb[:,t], "Jb")
            rg.add_vtk_point_data(self.v[:,:,t], "v")
            rg.add_vtk_point_data(self.I[:,t], "I")
            rg.add_vtk_point_data(self.I_interp[:,t], "I_interp")
            rg.add_vtk_point_data(self.I[:,t]-self.I_interp[:,t], "diff")
            rg.add_vtk_point_data(self.p[:,t], "p")
            rg.add_vtk_point_data(self.mu[:,:,t], "mu")
            rg.vtk_write(t, "maps", self.output_dir)

    def loadData(self, fbase):
        rg, N, T = self.get_sim_data()
        res = []
        for t in range(T):
            res.append(self.pool.apply_async(loadData_for_async, \
                                args=(fbase, t)))
        for t in range(T):
            (v,I,I_interp,p,mu,mu_state) = \
                                res[t].get(timeout=self.pool_timeout)
            self.v[...,t] = v
            self.I[...,t] = I
            self.I_interp[...,t] = I_interp
            self.p[...,t] = p
            self.mu[...,t] = mu
            self.mu_state[...,t] = mu_state
            logging.info("set data for time %d." % (t))

    def reset(self):
        fbase = "/cis/home/clr/compute/time_series/lung_data_1/iter250_mesh256_"
        self.load_data(fbase)

