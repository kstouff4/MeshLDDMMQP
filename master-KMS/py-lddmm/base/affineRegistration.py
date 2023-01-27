import numpy as np
from numba import jit
import numpy.linalg as linalg
from scipy.linalg import expm
from scipy.optimize import minimize
import logging

def randomRotation(dim):
    A = np.random.normal(0,1, (dim, dim))
    A = A - A.T
    return expm(A)

def rigidRegistrationLmk(x, y):
    N = x.shape[0]
    mx = x.sum(axis=0)/N
    my = y.sum(axis=0)/N
    rx = x-mx
    ry = y-my

    M = np.dot(ry.T, rx)
    sM = linalg.inv(np.real(sqrtm(np.dot(M.T, M))))
    R = M*sM
    T = my - np.dot(mx, R.T)
    return R,T

@jit(nopython=True)
def rotpart(A, rotation=True):
    U, S, Vh = linalg.svd(A)
    if rotation and linalg.det(A)<0:
        j = np.argmin(S)
        Vh[j,:] *= -1

    R = np.dot(U, Vh)
    return R

@jit(nopython=True)
def sqrtm(A):
    U_, S_, Vh_ = linalg.svd(A)
    # dg = np.zeros(S_.shape)
    # for i in range(S_.shape[0]):
    #     dg[i,i] = 1 / np.sqrt(S_[i])
    dg = np.diag(1/np.sqrt(S_))
    sU = np.dot(Vh_.T, np.dot(dg, U_.T))
    return sU

# sU = linalg.inv(np.real(sqrtm(np.dot(U.T, U))))

def saveRigid(filename, R, T):
    with open(filename,'w') as fn:
        for k,r in enumerate(R):
            str = '{0: f} {1: f} {2: f} {3: f} \n'.format(r[0], r[1], r[2], T[0,k])
            fn.write(str)

            
@jit(nopython=True, parallel=True)
def _flipMidPoint(Y,X):
    M = Y.shape[0]
    dim = Y.shape[1]
    mx = X.sum(axis=0)/X.shape[0]
    my = Y.sum(axis=0)/M

    mid = (mx+my).reshape((1, dim))/2
    u = (mx - my).reshape((1, dim))
    nu = np.sqrt((u**2).sum())
    if (nu < 1e-10):
        u = np.zeros((1,dim))
        u[0,0] = 1
    else:
        u = u/nu
    S = np.eye(3) - 2* np.dot(u.T, u)
    T = 2*(mid * u).sum() * u
    Z = np.dot(Y, S.T) + T

    return Z, S, T

@jit(nopython=True)
def objective_and_gradient_varifold(u, x, ys, wxy, sigma, test=True,us=None):
    grad = np.zeros(u.shape)
    dimn = x.shape[1]
    if dimn == 3:
        ur = u[:3]
        T = u[3:]
        t = np.sqrt((ur ** 2).sum())
        if t > 1e-10:
            st = np.sin(t)
            ct = np.cos(t)
            a1 = st / t
            a2 = (1 - ct) / t
            unorm = ur / t
            ucx = np.cross(np.expand_dims(unorm, 0), x)
            udx = (np.expand_dims(unorm, 0) * x).sum(axis=1)
            Rx = ct * x + st * ucx + (1-ct) * np.expand_dims(udx, 1) * np.expand_dims(unorm, 0)
            # Katie Add
            if (us is not None):
                Rx = Rx@us # assume us is in form of scale for each dimension
            xx = (Rx + T) / sigma
            xy = np.expand_dims(xx, 1) - np.expand_dims(ys, 0)
            dxy = (xy ** 2).sum(axis=2)
            Kxy = np.exp(-dxy / 2) * wxy
            obj = -Kxy.sum()
            dKxy = - np.sum(np.expand_dims(Kxy, 2) * xy, axis = 1)
            # da1 = (t * ct - st) / (t ** 2)
            # da2 = (t * st - 2 * (1 - ct)) / (t ** 2)
            da1 = ct - a1
            da2 = st - 2 * a2
            dKxyx = (dKxy * x).sum()
            dKxyu = dKxy[:,0] * unorm[0] + dKxy[:,1] * unorm[1] + dKxy[:,2] * unorm[2]
                #(dKxy * np.expand_dims(ur, (0,1))).sum(axis=2)
            dRx = (da1 * (dKxy * ucx).sum() - st * dKxyx + da2 * (dKxyu * udx).sum()) * unorm \
                  + a1 * np.sum(np.cross(x, dKxy), axis=0) \
                  + a2 * ((np.expand_dims(dKxyu, 1) * x + dKxy * np.expand_dims(udx, 1)).sum(axis=0))
            dRx = -dRx / sigma
            dT = - np.sum(dKxy, axis=0) / sigma
        else:
            xx = (x + T) / sigma
            xy = np.expand_dims(xx, 1) - np.expand_dims(ys, 0)
            dxy = (xy ** 2).sum(axis=2)
            Kxy = np.exp(-dxy / 2) * wxy
            obj = -Kxy.sum()
            dKxy = - np.sum(np.expand_dims(Kxy, 2) * xy, axis = 1)
            dRx = -np.sum(np.sum(np.cross(np.expand_dims(x, 1), dKxy), axis=0), axis=0)/sigma
            dT = - np.sum(dKxy, axis=0) / sigma
        grad[:3] = dRx
        grad[3:] = dT
        #print(f'obj={obj:.4f}; grad = {np.fabs(grad).max():.4f}')
    else:
        ct = np.cos(u[0])
        st = np.sin(u[0])
        R = np.array([[ct, -st], [st, ct]])
        Rx = x @ R.T
        # Katie added
        if (us is not None):
            Rx = Rx @ us
        T = u[1:]
        xx = (Rx + T) / sigma
        xy = np.expand_dims(xx, 1) - np.expand_dims(ys, 0)
        dxy = (xy ** 2).sum(axis=2)
        Kxy = np.exp(-dxy / 2) * wxy
        obj = -Kxy.sum()
        dKxy = - np.expand_dims(Kxy, 2) * xy
        dR = np.array([[-st, -ct], [ct, -st]])
        dRx = (dKxy * np.expand_dims((x @ dR.T), 1)).sum()
        dT = np.sum(np.sum(dKxy, axis=0), axis=0)
        grad[:1] = -dRx / sigma
        grad[1:] = -dT / sigma
        #print(f'theta = {u[0]:.4f}, obj={obj:.4f}; grad = {np.fabs(grad).max():.4f}')

    if test:
        eps = 1e-8
        du = np.random.normal(0,1, u.shape)
        # t = np.sqrt((u[:3] ** 2).sum())
        # du[:3] = (du[:3] * u[:3]).sum() * u[:3] / t**2
        obj1, foo = objective_and_gradient_varifold(u+eps*du, x, ys, wxy, sigma, test=False)
        obj2, foo = objective_and_gradient_varifold(u - eps * du, x, ys, wxy, sigma, test=False)
        #print(f'test gradient {(obj1 - obj2)/(2*eps):.6f}, {(du*grad).sum():.6f}')
    return obj, grad


def rigidRegistration_varifold(surfaces, weights=None, sigma = 1., ninit=1):
    x = surfaces[0]
    y = surfaces[1]
    Nsurf = x.shape[0]
    Msurf = y.shape[0]
    dimn = x.shape[1]
    if weights is None:
        wxy = np.ones((Nsurf, Msurf))
    else:
        wxy = weights

    ys = y/sigma

    def getRotation(u):
        if dimn == 3:
            ur = u[:3]
            t = np.sqrt((ur ** 2).sum())
            if t <1e-10:
                R = np.eye(3)
            else:
                A = np.array([[0, -ur[2], ur[1]], [ur[2], 0, -ur[0]], [-ur[1], ur[0], 0]])
                R = np.eye(3) + ((1 - np.cos(t)) / (t ** 2)) * (np.dot(A, A)) + (np.sin(t) / t) * A
            T = u[3:]
        else:
            ct = np.cos(u[0])
            st = np.sin(u[0])
            R = np.array([[ct, -st], [st, ct]])
            T = u[1:]
        return R, T

    if dimn == 2:
        bestx = np.zeros(3)
    else:
        bestx = np.zeros(6)
    bestobj, foo = objective_and_gradient_varifold(bestx, x, ys, wxy, sigma, test=False)

    mx = np.mean(x,axis=0)
    my = np.mean(y,axis=0)
    x00 = []
    if dimn == 2:
        for k in range(ninit):
            x0 = np.zeros(3)
            x0[0] = 2*k*np.pi/ninit
            R, T = getRotation(x0)
            x0[1:] = my - R@mx
            x00.append(x0)
            #x0[0] = np.random.uniform(0, 2*np.pi, 1)
    else:
        if ninit == 1:
            x0 = np.zeros(6)
            R, T = getRotation(x0)
            x0[3:] = my - R @ mx
            x00.append(x0)
        else:
            for i1 in range(2):
                for i2 in range(2):
                    for i3 in range(2):
                        t = i1*np.pi
                        phi = i2*np.pi
                        psi = i3*np.pi
                        x0 = np.zeros(6)
                        x0[0] = t*np.cos(phi)*np.cos(psi)
                        x0[1] = t * np.cos(phi) * np.sin(psi)
                        x0[2] = t * np.sin(phi)
                        R, T = getRotation(x0)
                        x0[3:] = my - R@mx
                        x00.append(x0)
    for k, x0 in enumerate(x00):
        logging.info(f'Rigid registration Initialization {k + 1}')
        res = minimize(objective_and_gradient_varifold, x0, args= (x, ys, wxy, sigma, False),
                       method='BFGS', jac=True, options={'maxiter':10000})
        if res.fun < bestobj:
            logging.info(f'found better solution {bestobj:0.4f} {res.fun:0.4f}')
            bestx = np.copy(res.x)
            bestobj = res.fun
            #print(f'Current optimal: theta = {bestx[0]:.4f}, obj={bestobj:.4f}')
    #print(f'Final Optimal: theta = {bestx[0]:.4f}, obj={bestobj:.4f}')
    R, T = getRotation(bestx)
    return R, T


@jit(forceobj=True, parallel=True, debug=True)
def rigidRegistration__(surfaces = None, temperature = 1.0, rotWeight = 1.0, rotationOnly=False,
                      translationOnly=False, flipMidPoint=False, init = None,
                      annealing = True, verb=False, landmarks = None, normals = None, image=None):
#  [R, T] = rigidRegistrationSurface(X0, Y0, t)
# compute rigid registration using soft-assign maps
# computes R (orthogonal matrix) and T (translation so that Y0 ~ R*X0+T
# X0 and Y0 are point sets (size npoints x dimension)
# OPT.t: scale parameter for soft assign(default : 1)
# OPT.rotationOnly: 1 to estimate rotations only, 0 for rotations and
# symmetries (0: default)

#  Warning: won't work with large data sets (npoints should be less than a
#  few thousands).

    if (surfaces is None):
        surf = False
        norm = False
        norm0 = np.zeros((0,0))
        norm1 = np.zeros((0,0))
        N=0
        M=0
        Nsurf = 0
        Msurf = 0
        if landmarks is None:
            print('Provide either surface points or landmarks or both')
            return
        else:
            lmk = True
            Nlmk = landmarks[0].shape[0]
            X0 = landmarks[0]
            Y0 = landmarks[1]
    else:
        surf = True
        X0 = surfaces[0]
        Y0 = surfaces[1]
        Nsurf = X0.shape[0]
        Msurf = Y0.shape[0]
        if image is None:
            wxx = np.ones((Nsurf, Nsurf))
            wxy = np.ones((Nsurf, Msurf))
        else:
            wxx = image[0].T @ image[0]
            wxy = image[0].T @ image[1]

        if landmarks is None:
            lmk = False
            N = Nsurf
            M = Msurf
            Nlmk = 0
        else:
            lmk = True
            Nlmk = landmarks[0].shape[0]
            N = Nsurf + Nlmk
            M = Msurf + Nlmk
            X0 = np.concatenate((X0, landmarks[0]))
            Y0 = np.concatenate((Y0, landmarks[1]))
        if normals is None:
            norm = False
            norm0 = np.zeros(X0.shape)
            norm1 = np.zeros(Y0.shape)
        else:
            norm = True
            norm0 = normals[0]
            norm1 = normals[1]

    #norm = False
    if lmk:
        if landmarks is not None and len(landmarks)==3:
            lmkWeight = landmarks[2]
        else:
            lmkWeight = 1.0
    else:
        lmkWeight=0

    if norm:
        if normals is not None and len(normals) == 3:
            normWeight = normals[2]
        else:
            normWeight = 1
    else:
        normWeight=0

    rotWeight *= X0.shape[0]
            
    t1 = temperature
    dimn = X0.shape[1]

    if flipMidPoint:
        [Y1, S1, T1] = _flipMidPoint(Y0, X0)
        if norm:
            norm0 *= -1
    else:
        S1 = np.eye(dimn)
        T1 = np.zeros((1, dimn))
        #print S1, T1
        Y1 = Y0

    if init is None:
        R = np.eye(dimn)
        T = np.zeros(((1, dimn)))
    else:
        R = init[0]
        T = init[1]


    if surf:
        # Alternate minimization for surfaces with or without landmarks
        if init is None:
            R = np.eye(dimn)
            T = np.zeros(((1, dimn)))
        else:
            R = init[0]
            T = init[1]

        RX = np.dot(X0,  R.T) + T
        RX2 = (RX**2).sum(axis=1)
        Y12 = (Y1**2).sum(axis=1)
        # d = np.zeros((RX.shape[0], Y1.shape[0]))
        # for i in range(RX.shape[0]):
        #     for j in range(Y1.shape[0]):
        #         d[i,j] = ((RX[i,:] - Y1[j,:])**2).sum(axis=1)
        d = RX2.reshape((RX2.shape[0], 1)) - 2 * np.dot(RX, Y1.T) + Y12.reshape((1, Y12.shape[0]))
        if norm:
            Rn = np.dot(norm0, R.T)
            for i in range(Nsurf):
                for j in range(Msurf):
                    d[i,j] += normWeight * ((Rn[i, :] - norm1[j,:])**2).sum()
            # u1 = np.reshape((Rn**2).sum(axis=1), [Nsurf,1])
            # u2 = np.reshape((norm1**2).sum(axis=1), [1,Msurf])
            # d[0:Nsurf, 0:Msurf] += normWeight*(np.reshape((Rn**2).sum(axis=1), [Nsurf,1]) - 2 * np.dot(Rn, norm1.T) +
            #                                    np.reshape((norm1**2).sum(axis=1), [1,Msurf]))
        dSurf = d[0:Nsurf, 0:Msurf]
        Rold = np.copy(R)
        Told = np.copy(T)
        if annealing:
            t0 = 10*t1
        else:
            t0 = t1
        t = t0
        c = .89
        w1 = np.zeros((N,M))
        w2 = np.zeros((N,M))
        if lmk:
            w1[Nsurf:N, Msurf:M] = lmkWeight*np.eye(Nlmk)
            w2[Nsurf:N, Msurf:M] = lmkWeight*np.eye(Nlmk)

        for k  in range(10000):
            # new weights
            if annealing and (k < 21):
                t = t*c 
            dmin = dSurf.min()
            wSurf = np.minimum((dSurf-dmin)/t, 500.)
            wSurf = np.exp(-wSurf)
            #wSurf = w[0:Nsurf, 0:Msurf] 
    
            #    w = sinkhorn(w, 100) ;
            Z  = wSurf.sum(axis=1)
            w1Surf = wSurf / Z.reshape((Z.shape[0], 1))
            Z  = wSurf.sum(axis = 0)
            w2Surf = wSurf / Z.reshape((1, Z.shape[0]))
            w1[0:Nsurf, 0:Msurf] = w1Surf
            w2[0:Nsurf, 0:Msurf] = w2Surf
            w = w1 + w2
            # ener = rotWeight*(dimn-np.trace(R)) + (w*d + t*(w1Surf*np.log(w1Surf) + w2Surf * np.log(w2Surf))).sum()
            #if verb:
            #    print 'ener = ', ener


            # new transformation
            wX = np.dot(w.T, X0).sum(axis=0)
            wY = np.dot(w, Y1).sum(axis=0)

            Z = w.sum()
            #print Z, dSurf.min(), dSurf.max()
            mx = wX.reshape((1, dimn))/Z
            my = wY.reshape((1, dimn))/Z
            Y = Y1 - my
            X = X0 - mx
    
            if not translationOnly: 
                U = np.dot( np.dot(w, Y).T, X) + rotWeight * np.eye(dimn)
                if norm:
                    U += normWeight * np.dot(np.dot(w1Surf+w2Surf, norm0).T, norm1)
                R = rotpart(U, rotation=rotationOnly)
                # if rotationOnly:
                #     R = rotpart(U)
                # else:
                #     sU = linalg.inv(np.real(linalg.sqrtm(np.dot(U.T, U))))
                #     R = np.dot(U, sU)

            T = my - np.dot(mx, R.T)
            #print R, T
            RX = np.dot(X0, R.T) + T
        
            d = (RX**2).sum(axis=1).reshape((RX.shape[0], 1)) - 2 * np.dot(RX, Y1.T) + (Y1**2).sum(axis=1).reshape((1, Y.shape[0]))
            if norm:
                Rn = np.dot(norm0 ,R.T)
                d[0:Nsurf, 0:Msurf] += normWeight * ((Rn ** 2).sum(axis=1).reshape((Nsurf ,1)) - 2 * np.dot(Rn ,norm1.T) + (
                            norm1 ** 2).sum(axis=1).reshape((1 ,Msurf)))
            dSurf = d[0:Nsurf, 0:Msurf]
            ener = rotWeight*(dimn-np.trace(R)) + (w*d).sum() + t*((w1Surf*np.log(w1Surf)) + (w2Surf*np.log(w2Surf))).sum()
            #ener = rotWeight*(3-np.trace(R)) + (np.multiply(w, d) + t*(np.multiply(w1, np.log(w1)) + np.multiply(w2, np.log(w2)))).sum()

            if verb:
                #str_ = 'ener ' + str(ener)
                print(ener, np.fabs((R-Rold)).sum(), np.fabs((T-Told)).sum())
                print(R)
                # print('ener = {0:.3f} var = {0:.3f} {0:.3f}'.format(ener, np.fabs((R-Rold)).sum(), np.fabs((T-Told)).sum()))

            if (k > 21) and (np.fabs((R-Rold)).sum() < 1e-3) and (np.fabs((T-Told)).sum() < 1e-2):
                break
            else:
                Told = np.copy(T)
                Rold = np.copy(R)
        if flipMidPoint:
            R = np.dot(R, S1)
            T = np.dot(T - T1, S1.T)

        return R, T
    elif lmk:
        # landmarks only
        mx = X0.sum(axis=0)/Nlmk
        my = Y1.sum(axis=0)/Nlmk
        Y = Y1 - my.reshape((1, dimn))
        X = X0 - mx.reshape((1, dimn))
    
        if not translationOnly: 
            U = np.dot(Y.T, X) #+ rotWeight * np.eye(dimn)
            #print(U)
            if rotationOnly:
                R = rotpart(U)
            else:
                UU = np.dot(U.T, U)
                sU = sqrtm(UU)
                R = np.dot(U, sU)
        else:
            R = np.eye(dimn)

        T_ = np.dot(mx.reshape((1, dimn)), R.T)
        T = my.reshape((1, dimn)) - T_

        if flipMidPoint:
            R = np.dot(R, S1)
            T = np.dot(T - T1, S1.T)

        return R,T
    return R, T



def rigidRegistration(surfaces=None, temperature=1.0, rotWeight=1.0, rotationOnly=False,
                      translationOnly=False, flipMidPoint=False,
                      annealing=True, verb=False, landmarks=None, normals=None):
    R, T = rigidRegistration__(surfaces=surfaces, temperature=temperature, rotWeight=rotWeight, rotationOnly=rotationOnly,
                          translationOnly=translationOnly, flipMidPoint=flipMidPoint,
                          annealing=annealing, verb=verb, landmarks=landmarks, normals=normals)

    # R = linalg.inv(R)
    # T = - np.dot(T, R.T)
    # T = T.reshape((1, T.shape[0]))
    #
    # #print R,T
    # if flipMidPoint:
    #     T += np.dot(T1, R.T)
    #     R = np.dot(R, S1)

        #print R, T
    return R,T


def rigidRegistration_multi(surfaces, temperature=1.0, rotWeight=1.0, rotationOnly=False, translationOnly=False,
                            annealing=True, verb=False):
    #  [R, T] = rigidRegistrationSurface(X0, Y0, t)
    # compute rigid registration using soft-assign maps
    # computes R (orthogonal matrix) and T (translation so that Y0 ~ R*X0+T
    # X0 and Y0 are point sets (size npoints x dimension)
    # OPT.t: scale parameter for soft assign(default : 1)
    # OPT.rotationOnly: 1 to estimate rotations only, 0 for rotations and
    # symmetries (0: default)

    #  Warning: won't work with large data sets (npoints should be less than a
    #  few thousands).

    X0 = surfaces[1]
    Y1 = surfaces[0]
    Nsurf = np.zeros(len(X0), dtype=int)
    Msurf = np.zeros(len(Y1), dtype=int)
    for k,s in enumerate(X0):
        Nsurf[k] = s.shape[0]
    for k,s in enumerate(Y1):
        Msurf[k] = s.shape[0]


    rotWeight *= Nsurf.sum()

    t1 = temperature

    dimn = X0[0].shape[1]
    ns = len(X0)

    # Alternate minimization for surfaces with or without landmarks
    R = np.eye(dimn)
    T = np.zeros([1, dimn])
    RX = []
    dSurf = []
    for k in range(ns):
        RX.append(np.dot(X0[k], R.T) + T)
        dSurf.append((RX[k] ** 2).sum(axis=1).reshape([Nsurf[k], 1]) - 2 * np.dot(RX[k], Y1[k].T)
                 + (Y1[k] ** 2).sum(axis=1).reshape([1, Msurf[k]]))
    Rold = np.copy(R)
    Told = np.copy(T)
    t0 = 10 * t1
    t = t0
    c = .89
    w1 = []
    w2 = []
    w = []
    for k in range(ns):
        w1.append(np.zeros([Nsurf[k], Msurf[k]]))
        w2.append(np.zeros([Nsurf[k], Msurf[k]]))
        w.append(np.zeros([Nsurf[k], Msurf[k]]))

    for k0 in range(10000):
        # new weights
        if annealing and (k0 < 21):
            t = t * c
        for k in range(ns):
            dmin = dSurf[k].min()
            wSurf = np.minimum((dSurf[k] - dmin) / t, 500.)
            wSurf = np.exp(-wSurf)
            # wSurf = w[0:Nsurf, 0:Msurf]

            #    w = sinkhorn(w, 100) ;
            Z = wSurf.sum(axis=1)
            w1[k] = wSurf / Z[:, np.newaxis]
            Z = wSurf.sum(axis=0)
            w2[k] = wSurf / Z[np.newaxis, :]
            w[k] = w1[k] + w2[k]
            # ener = rotWeight*(3-np.trace(R)) + (np.multiply(w, d) + t*(np.multiply(w1Surf, np.log(w1Surf)) + np.multiply(w2Surf, np.log(w2Surf)))).sum()
            # if verb:
            #    print 'ener = ', ener

            # new transformation
        mx = np.zeros(dimn)
        my = np.zeros(dimn)
        Z = 0
        for k in range(ns):
            wX = np.dot(w[k].T, X0[k]).sum(axis=0)
            wY = np.dot(w[k], Y1[k]).sum(axis=0)

            Z += w[k].sum()
            # print Z, dSurf.min(), dSurf.max()
            mx += wX
            my += wY
        mx /= Z
        my /= Z

        Y = []
        X = []
        for k in range(ns):
            Y.append(Y1[k] - my)
            X.append(X0[k] - mx)

        if not translationOnly:
            U = np.zeros((dimn, dimn))
            for k in range(ns):
                U += np.dot(np.dot(w[k], Y[k]).T, X[k])
            U += rotWeight * np.eye(dimn)
            if rotationOnly:
                R = rotpart(U)
            else:
                sU = linalg.inv(np.real(sqrtm(np.dot(U.T, U))))
                R = np.dot(U, sU)

        T = my - np.dot(mx, R.T)
        # print R, T
        RX = []
        dSurf = []
        for k in range(ns):
            RX.append(np.dot(X0[k], R.T) + T)
            dSurf.append((RX[k] ** 2).sum(axis=1).reshape([Nsurf[k], 1]) - 2 * np.dot(RX[k], Y1[k].T)
                     + (Y1[k] ** 2).sum(axis=1).reshape([1, Msurf[k]]))

        ener = rotWeight * (dimn - np.trace(R))
        for k in range(ns):
            ener += (w[k] * dSurf[k]).sum() + t * (
                    (w1[k] * np.log(w1[k])) + (w2[k] * np.log(w2[k]))).sum()
        # ener = rotWeight*(3-np.trace(R)) + (np.multiply(w, d) + t*(np.multiply(w1, np.log(w1)) + np.multiply(w2, np.log(w2)))).sum()

        if verb:
            print('ener = ', ener, 'var = ', np.fabs((R - Rold)).sum(), np.fabs((T - Told)).sum())

        if (k0 > 21) and (np.fabs((R - Rold)).sum() < 1e-3) and (np.fabs((T - Told)).sum() < 1e-2):
            break
        else:
            Told = np.copy(T)
            Rold = np.copy(R)

    R = linalg.inv(R)
    T = - np.dot(T, R.T)
    T = T.reshape([1, dimn])

    # print R,T
    return R, T

