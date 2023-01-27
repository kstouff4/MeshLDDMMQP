import numpy as np
import scipy as sp
import scipy.linalg as la
from base.curves import Curve, remesh
from base import kernelFunctions as kff, pointEvolution as evol
import matplotlib.pyplot as plt

Npts = 200


def diffusion(c0, t1, K, dt = 0.1, iso_par = 1., area_par=1., peri_par = 1.0, stdev=1.,
              K2=None, r0=None, l0=None, a0=None, force=[1,0], force_par = 0, grav_par = 1.0, lmk0=[], lmk_par=0.):
    force = np.array(force)
    t = 0
    powlmk = -6.
    if a0 is None:
        a0 = c0.enclosedArea()
    if l0 is None:
        l0 = c0.length()
    if r0 is None:
        r0 = l0 ** 2 / (a0 * 4 * np.pi)
    c = Curve(curve=c0)
    nlmk = 0
    if lmk0 is None:
        lmk = None
    else:
        lmk = np.copy(lmk0)
        centers = np.zeros((len(lmk0),2))
        for k,cc in enumerate(lmk):
            centers[k,:] = np.mean(cc.vertices, axis=0)
            nlmk += cc.vertices.shape[0]
    m0 = np.mean(c.vertices, axis=0)[np.newaxis,:]
    fig = plt.figure(1)
    fig.clf()
    sdt = np.sqrt(dt)
    fig.clf()
    ax = fig.gca()
    ax.plot(c.vertices[:, 0], c.vertices[:, 1], color=[0, 0, 1])
    if not (lmk is None):
        for cl in lmk:
            ax.plot(cl.vertices[:, 0], cl.vertices[:, 1], color=[0, 1, 0], marker='o')
        for k in range(len(lmk)):
            ax.plot(centers[k,0], centers[k,1], color=[1,0,0], marker = 'o')
    plt.axis('equal')
    #axs = ax.axis()
    plt.pause(0.1)
    if K2 is None:
        K2 = K
    for t in sp.arange(0,t1,dt):
        a = c.enclosedArea()
        l = c.length()
        x = np.roll(c.vertices, 5, axis=0)
        nv0 = x.shape[0]
        xInlmk = np.zeros((0, 2))
        xInlmkC = np.zeros((0, 2))
        if not(lmk is None):
            for cl in lmk:
                xInlmk = np.concatenate((xInlmk, cl.vertices), axis=0)
            xInlmkC = np.concatenate((xInlmk, centers), axis=0)
        #xlmk = np.concatenate((x, xInlmk), axis=0)
        xlmkC = np.concatenate((x, xInlmkC), axis=0)
        dx = (np.roll(x,-1, axis=0) - np.roll(x,1, axis=0))/2
        nu = np.zeros(dx.shape)
        nu[:,0] = -dx[:,1]
        nu[:,1] = dx[:,0]
        #ka = c.computeCurvature()[2]
        dx = np.roll(x,-1, axis=0) - x
        ndx = np.sqrt((dx**2).sum(axis=1))[:, np.newaxis]
        dx = dx/ndx
        ka = dx - np.roll(dx,1, axis=0)
        vC = np.zeros((len(lmk),2))
        vlmk = np.zeros((nlmk,2))
        vlmk2 = np.zeros((nlmk,2))
        #vx = -lbd*(-(l**2/a**2)*nu + 2*l*ka/a)*(l**2/(a*4*np.pi)-1.5) - mu*(l-l0)*ka - xi * (a-a0)*nu
        vx = (-iso_par*(-(l**2/a**2)*nu + 2*l*ka/a)*(l**2/(a*4*np.pi)-r0) - peri_par*(l-l0)*ka
              - area_par * (a-a0)*nu)
        vf = force_par * (np.maximum((force[:,np.newaxis,:]*nu[np.newaxis,:,:]).sum(axis=2),0)[:,:,np.newaxis]*nu[np.newaxis, :, :]).sum(axis=0)
        vx += vf
        nvf = np.sqrt((vf**2).sum(axis=1))
        if not(lmk is None):
            nrm0 = np.maximum(1e-8, np.sqrt(((x[np.newaxis, :, :] - xInlmk[:, np.newaxis, :]) ** 2).sum(axis=2))) ** (
            powlmk - 2.)
            nrm1 = np.maximum(1e-8, np.sqrt(((xInlmk[np.newaxis, :, :] - xInlmk[:, np.newaxis, :]) ** 2).sum(axis=2))) ** (
            powlmk - 2.)
            vx += lmk_par * powlmk * ((x[np.newaxis, :, :] - xInlmk[:, np.newaxis, :])*nrm0[...,np.newaxis]).sum(axis=0)
            vlmk = lmk_par * powlmk * ((xInlmk[:, np.newaxis, :] - x[np.newaxis, :, :])*nrm0[...,np.newaxis]).sum(axis=1)
            vlmk += lmk_par * powlmk * ((xInlmk[np.newaxis, :, :] - xInlmk[:, np.newaxis, :])*nrm1[...,np.newaxis]).sum(axis=0)
            k0=0
            for k,cl in enumerate(lmk):
                vlmk2[k0:k0+cl.vertices.shape[0], :] = -grav_par * (centers[k,:] - cl.vertices)
                k0 += cl.vertices.shape[0]
                #vC[k,:] =  grav_par * (centers[k,:] - np.mean(cl.vertices, axis=0))
            vlmk += vlmk2
            #vx = np.concatenate((vx,vlmk), axis=0)
        Kx = K2.getK(x)
        Sx = np.real(la.sqrtm(Kx))
        v = -dt*vx + sdt*stdev*la.solve(Sx,np.random.normal(size=x.shape))
        v = np.concatenate((v, -dt*vlmk, dt*vC), axis=0)
        normv = np.sqrt((v*(K.applyK(xlmkC,v))).sum())
        Ns = min(1000, 1+int(np.ceil(20*normv)))
        xt, at = evol.landmarkEPDiff(Ns, xlmkC, v, K)
        x = remesh(xt[-1,0:x.shape[0],:],N=int(Npts))
        #Kv = v #np.dot(Kx,v)
        #x = x+Kv
        m = np.mean(x, axis=0)[np.newaxis, :]
        x = x - (m-m0)
        xend = xt[-1,:,:]
        xend = xend - (m-m0)
        c = Curve(pointSet= x)
        x1 = np.append(x, [x[0,:]], axis=0)
        if np.fabs(100.*t/t1 - np.floor(100.*t/t1)) < 1e-4:
            fig.clf()
            ax = fig.gca()
            for k in range(x1.shape[0]):
                ax.plot(x1[k,0], x1[k,1], color=[0, 0, 1], marker='*')
            ax.plot(x1[0,0], x1[0,1], color=[0, 1, 1], marker='o')
            ax.plot(x1[:, 0], x1[:, 1], color=[1, 0, 1], marker='*')
            ax.plot(x1[:, 0], x1[:, 1], color=[1, 0, 0])
            #plt.quiver(x1[:, 0], x1[:, 1], -vf[0:nv0, 0], -vf[0:nv0, 1], scale=100)
            #plt.quiver(x1[:, 0], x1[:, 1], -nu[0:nv0, 0], -nu[0:nv0, 1])
            plt.axis('equal')
            #ax.axis(axs)
            plt.pause(0.1)
        j0 = 0
        if not(lmk is None):
            for cl in lmk:
                cl.updateVertices(xend[j0+nv0:j0+nv0+cl.vertices.shape[0],:])
                for k in range(cl.vertices.shape[0]):
                    ax.plot(cl.vertices[k,0], cl.vertices[k,1], color=[0, 1, 0], marker='o')
                plt.quiver(cl.vertices[:,0], cl.vertices[:,1], -vlmk[j0:j0+cl.vertices.shape[0], 0], -vlmk[j0:j0+cl.vertices.shape[0], 1])
                j0 += cl.vertices.shape[0]
            for k in range(len(lmk)):
                centers[k,:] = xend[j0+nv0, :]
                ax.plot(centers[k, 0], centers[k, 1], color=[1, 0, 0], marker='o')
                j0 += 1
        #plt.quiver(x[:,0], x[:,1], ka[:,0], ka[:,1])
        #print (c.enclosedArea() - a)/dt, (nu*Kv).sum()/dt
        #print lbd*(c.length()**2/c.enclosedArea() - l**2/a)/dt, (v*Kv).sum()/dt**2
        print(t,Ns, l-l0, a-a0, l**2/(a*4*np.pi) -r0, x1.shape[0])
        #t += dt
    return c


def main_run():
    plt.ion()
    M = 10
    t = np.arange(0., 2 * np.pi, 0.05)
    a = 2.5
    b = 1.5
    c = 0.
    x0 = 10
    y0 = 2
    x = x0 + a * np.cos(t) * (1 - c * np.cos(t))
    y = y0 + b * np.sin(t) * (1 - c * np.cos(t))
    X = np.zeros([t.shape[0], 2])
    X[:, 0] = x
    X[:, 1] = y
    X = remesh(X, N=Npts)
    N = X.shape[0]
    f = np.zeros([N, 2], dtype=int)
    f[0:N - 1, 0] = range(0, N - 1)
    f[0:N - 1, 1] = range(1, N)
    f[N - 1, :] = [N - 1, 0]
    ftemp = Curve(FV=(f, X))

    lmk = []
    x = x0 + a/2 + 0.5 * np.cos(t)
    y = y0 + 0.5 * np.sin(t)
    X = np.zeros([t.shape[0], 2])
    X[:, 0] = x
    X[:, 1] = y
    X = remesh(X, N=10)
    lmk.append(Curve(pointSet=X))
    x = x0 - a/2 + 0.5 * np.cos(t)
    y = y0 + 0.5 * np.sin(t)
    X = np.zeros([t.shape[0], 2])
    X[:, 0] = x
    X[:, 1] = y
    X = remesh(X, N=10)
    lmk.append(Curve(pointSet=X))
    x = x0 + 0.5 * np.cos(t)
    y = y0 - b/2 + 0.5 * np.sin(t)
    X = np.zeros([t.shape[0], 2])
    X[:, 0] = x
    X[:, 1] = y
    X = remesh(X, N=10)
    lmk.append(Curve(pointSet=X))

    #frc = [[1,0],[np.cos(2*np.pi/3),np.sin(2*np.pi/3)],[np.cos(4*np.pi/3),np.sin(4*np.pi/3)]]
    frc = [[1,0],[-1,0],[0,5], [0,-5]]

    K = kff.Kernel(name='laplacian', sigma = 0.5, order=3)
    K2 = kff.Kernel(name='laplacian', sigma = 0.5, order=2)
    c = []
    for s in [0.00001, 0.25, 0.5, 1.0, 1.5, 2.0]:
        c.append(diffusion(ftemp,.1,K, dt=0.0001,iso_par=0., peri_par = 10., area_par = 100., stdev=s, r0 = 1.5,
              lmk0=[], lmk_par=.001, K2=K2, force=[[1,0]], force_par=1000., grav_par=50.))
    plt.ioff()
    plt.show()
    return c

if __name__=="__main__":
    main_run()
