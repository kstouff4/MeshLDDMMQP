#! /usr/bin/python

import numpy
import logging

import scipy.integrate


#class Derivative_Evaluator(object):
#
#    def __init__(self, rg, v):
#        self.v = v
#        self.rg = rg
#
#    def evaluate(self, f_in, t0):
#        rg = self.rg
#        v = self.v
#        f = f_in.reshape((rg.num_nodes, 3))
#        import pdb
#        pdb.set_trace()
#        dvv = numpy.empty((rg.num_nodes,3))
#        for k in range(3):
#            dvv[:,k] = -1 * numpy.multiply( \
#                            rg.gradient(f[:,k]).real, v[:,:,t0]).sum(axis=1)
#        return dvv.reshape(rg.num_nodes * 3)
#
#def integrate(rg, N, T, v, Nlagrange, dt):
#    de = Derivative_Evaluator(rg, v)
#    y = rg.nodes.reshape(rg.num_nodes * 3)
#    f = scipy.integrate.odeint(de.evaluate, y, range(T))
#    fout = numpy.empty((rg.num_nodes, 3, T))
#    for t in range(T):
#        fout[...,t] = f[t,...].reshape((rg.num_nodes,3))
#    return fout

def integrate(rg, N, T, v, Nlagrange, dt):
    f = numpy.empty((rg.num_nodes, 3, T))
    f[:,:,0] = rg.nodes.copy()
    [w, e, s, n, d, u] = rg.grid_neighbors()
    for t in range(1,T):
        dvv = numpy.empty((rg.num_nodes,3))
        for k in range(3):
            dvv[:,k] = -1 * numpy.multiply( \
                            rg.gradient(f[:,k,t-1]).real, v[:,:,t-1]).sum(axis=1)
        f[:,:,t] = f[:,:,t-1] + dt * dvv
    return f


def integrate_upwind(rg, N, T, v, Nlagrange, dt):
    f = numpy.zeros((rg.num_nodes, 3, T))
    f[:,:,0] = rg.nodes.copy()
    [w, e, s, n, d, u] = rg.grid_neighbors()
    maxc = -1
    rhs = numpy.empty(len(rg.nodes))
    c = numpy.empty(len(rg.nodes))
    for t in range(1,T):
        for outer_dim in range(rg.dim):
            rhs = 0.
            c = 0.
            for dim in range(rg.dim):
                c +=  numpy.abs(v[:,dim,t-1] * dt / rg.dx[dim])
                ap = numpy.maximum(v[:,dim,t-1], 0)
                am = numpy.minimum(v[:,dim,t-1], 0)
                if dim==0:
                    fm = f[:,outer_dim,t-1] - f[w,outer_dim,t-1]
                    fp = f[e,outer_dim,t-1] - f[:,outer_dim,t-1]
                elif dim==1:
                    fm = f[:,outer_dim,t-1] - f[s,outer_dim,t-1]
                    fp = f[n,outer_dim,t-1] - f[:,outer_dim,t-1]
                elif dim==2:
                    fm = f[:,outer_dim,t-1] - f[d,outer_dim,t-1]
                    fp = f[u,outer_dim,t-1] - f[:,outer_dim,t-1]
                rhs += (ap*fm + am*fp)
                # ***** upwinded euler method
            tempmax = numpy.max(c)
            if tempmax > maxc:
                maxc = tempmax
            if numpy.any(c > .9):
                logging.info("Warning: CFL violated for dim = %d." % outer_dim)
                logging.info("max c: %f" % numpy.max(c))
            f[:,outer_dim,t] += f[:,outer_dim,t-1] - (dt/rg.dx[outer_dim]) * rhs
    #logging.info("max c: %f" % maxc)
    return f

def integrate2(rg, N, T, v, Nlagrange, dt):

    F_ind_in = numpy.zeros((rg.num_nodes, 3, T))
    for t in range(T):
        F_ind_in[:,:,t] = rg.nodes.copy()

    v2 = numpy.zeros((rg.num_nodes, 3, 2*T - 1))
    v2[:,:,range(0,2*T-1,2)] = v
    v2[:,:,range(1,2*T-2,2)] = .5 * (v[:,:,range(0,T-1,1)] + v[:,:,range(1,T,1)])

    F_ind2 = numpy.zeros((rg.num_nodes, 3, 2*T-1))
    F_ind2[:,:,range(0,2*T-1,2)] = F_ind_in
    F_ind2[:,:,range(1,2*T-2,2)] = .5 * (F_ind_in[:,:,range(0,T-1,1)] + \
                        F_ind_in[:,:,range(1,T,1)])

    T2 = 2*T-1
    for t in range(2,T2):
        v_use = (v2[:,:,t] + v2[:,:,t-1])/2.0

        alpha = numpy.zeros((rg.num_nodes,3))

        for k in range(Nlagrange):
            xa = rg.nodes - alpha
            for d in range(3):
                alpha[:,d] = rg.grid_interpolate(v_use[:,d], xa).real * dt
        xa = rg.nodes - alpha
        for d in range(3):
            F_ind2[:,d,t] = rg.grid_interpolate(F_ind2[:,d,t-2], xa).real

    F_ind = F_ind2[:,:,range(0,2*T-1,2)]
    return F_ind

def integrate3(rg, N, T, v, Nlagrange, F_ind_in, dt):

    dx = rg.dx

    v2 = numpy.zeros((N**2, 3, 2*T - 1))
    v2[:,:,range(0,2*T-1,2)] = v
    v2[:,:,range(1,2*T-2,2)] = .5 * (v[:,:,range(0,T-1,1)] + v[:,:,range(1,T,1)])

    F_ind2 = numpy.zeros((N**2, 2*T-1))
    F_ind2[:,range(0,2*T-1,2)] = F_ind_in
    F_ind2[:,range(1,2*T-2,2)] = .5 * (F_ind_in[:,range(0,T-1,1)] + F_ind_in[:,range(1,T,1)])

    T2 = 2*T-1

    F = numpy.zeros((N,N,T2))

    (indexx, indexy) = numpy.meshgrid(range(N), range(N))
    indexx = indexx.astype(int)
    indexy = indexy.astype(int)

    for t in range(T2):
        F[:,:,t] = numpy.reshape(F_ind2[:,t], (N, N))

    for t in range(2,T2):
        vx_use = (v2[:,0,t] + v2[:,0,t-1])/2.0
        vy_use = (v2[:,1,t] + v2[:,1,t-1])/2.0

        alphax = numpy.zeros((N,N))
        alphay = numpy.zeros((N,N))

        for k in range(Nlagrange):
          stepsx = alphax / dx / 2.0
          stepsy = alphay / dx / 2.0

          px = numpy.floor(stepsx)
          py = numpy.floor(stepsy)

          ax = stepsx - px
          ay = stepsy - py

          pxindex = (indexx + px).astype(int)
          pyindex = (indexy + py).astype(int)
          pxindex_x = (pxindex + 1).astype(int)
          pyindex_y = (pyindex + 1).astype(int)

          pxindex[(pxindex<0)] = 0
          pyindex[(pyindex<0)] = 0
          pxindex_x[(pxindex_x<0)] = 0
          pyindex_y[(pyindex_y<0)] = 0

          pxindex[(pxindex>N-1)] = N-1
          pyindex[(pyindex>N-1)] = N-1
          pxindex_x[(pxindex_x>N-1)] = N-1
          pyindex_y[(pyindex_y>N-1)] = N-1

          pindex = (pxindex + (N)*(pyindex)).astype(int)
          pindex_x = (pxindex_x + (N)*(pyindex)).astype(int)
          pindex_y = (pxindex + (N)*(pyindex_y)).astype(int)
          pindex_xy = (pxindex_x + (N)*(pyindex_y)).astype(int)

          alphax = dt*(vx_use[pindex] * (1-ax) * (1-ay) + vx_use[pindex_x] * (ax) * (1-ay) + vx_use[pindex_y] * (1-ax)*(ay) + vx_use[pindex_xy]*(ax)*(ay))
          alphay = dt*(vy_use[pindex] * (1-ax) * (1-ay) + vy_use[pindex_x] * (ax) * (1-ay) + vy_use[pindex_y] * (1-ax)*(ay) + vy_use[pindex_xy]*(ax)*(ay))

        #update function
        stepsx = alphax / dx
        stepsy = alphay / dx

        px = numpy.floor(stepsx)
        py = numpy.floor(stepsy)

        ax = stepsx - px
        ay = stepsy - py

        pxindex = (indexx - px).astype(int)
        pyindex = (indexy - py).astype(int)
        pxindex_x = (pxindex - 1).astype(int)
        pyindex_y = (pyindex - 1).astype(int)

        pxindex[(pxindex<0)] = 0
        pyindex[(pyindex<0)] = 0
        pxindex_x[(pxindex_x<0)] = 0
        pyindex_y[(pyindex_y<0)] = 0

        pxindex[(pxindex>N-1)] = N-1
        pyindex[(pyindex>N-1)] = N-1
        pxindex_x[(pxindex_x>N-1)] = N-1
        pyindex_y[(pyindex_y>N-1)] = N-1

        pindex = (pxindex + (N)*(pyindex)).astype(int)
        pindex_x = (pxindex_x + (N)*(pyindex)).astype(int)
        pindex_y = (pxindex + (N)*(pyindex_y)).astype(int)
        pindex_xy = (pxindex_x + (N)*(pyindex_y)).astype(int)

        F[:,:,t] = F_ind2[pindex,(t-2)]*(1 - ax)*(1 - ay) + F_ind2[pindex_x, (t-2)]*(ax)*(1-ay) + F_ind2[pindex_y, (t-2)]*(1-ax)*(ay) + F_ind2[pindex_xy, (t-2)]*(ax)*(ay)
        for t in range(T2):
          F_ind2[:,t] = numpy.reshape(F[:,:,t], (N**2))

    F_ind = F_ind2[:,range(0,2*T-1,2)]
    return F_ind
