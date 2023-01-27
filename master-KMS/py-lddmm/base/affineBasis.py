import numpy as np
from scipy.sparse import coo_matrix

class AffineBasis:
    def __init__(self, dim=3, affine='affine'):
        u = 1.0/np.sqrt(2.0)
        dimSym = (dim * (dim+1))// 2
        dimSkew = (dim * (dim-1))// 2
        translation = False
        sym = False
        skew_sym = False
        scale = False
        diagonal = False
        self.dim = dim
        if dim > 25:
            self.sparse = True
        else:
            self.sparse = False
        self.rotComp = []
        self.simComp = []
        self.transComp = []
        self.diagComp = []
        self.symComp = []
        if affine == 'affine':
            translation = True
            skew_sym = True
            sym = True
            scale = True
            diagonal = True
            #self.affineDim = 2* (dimSym + dim)
        elif affine == 'similitude':
            translation = True
            skew_sym = True
            scale = True
        elif affine == 'euclidean':
            translation = True
            skew_sym = True
        elif affine == 'translation':
            translation = True
        elif affine == 'diagonal':
            translation = True
            scale = True
            diagonal = True
        else:
            self.basis = []

        self.affineDim = dim * translation + dim * diagonal + scale + skew_sym * dimSkew + sym * dimSkew

        if self.sparse:
            maxDim = 2 * (dimSkew + dim)* self.affineDim
            I = np.zeros(maxDim, dtype=int)
            J = np.zeros(maxDim, dtype=int)
            V = np.zeros(maxDim)

            kk = 0
            k = 0
            if translation:
                k0 = k
                for i in range(dim):
                    I[kk] = i + dim**2
                    J[kk] = k
                    V[kk] = 1
                    kk += 1
                    k += 1
                self.transComp = range(k0, k)
            if skew_sym:
                k0 = k
                for i in range(dim):
                    for j in range(i+1, dim):
                        I[kk] = i*dim+j
                        J[kk] = k
                        V[kk] = u
                        kk += 1
                        I[kk] = j*dim+i
                        J[kk] = k
                        V[kk] = -u
                        kk += 1
                        k+=1
                self.rotComp = range(k0,k)
            if scale:
                k0=k
                for i in range(dim):
                    I[kk] = i*dim+i
                    J[kk] = k
                    V[kk] = 1./np.sqrt(dim)
                    kk += 1
                k += 1
                self.simComp = range(k0, k)
            if diagonal:
                k0 = k
                for i in range(dim-1):
                    uu = np.sqrt((1 - 1.0/(i+2)))/(i+1.)
                    I[kk] = (i+1)*dim + (i+1)
                    J[kk] = k
                    V[kk] = np.sqrt(1 - 1.0/(i+2))
                    kk += 1
                    for j in range(i+1):
                        I[kk] = j*dim+j
                        J[kk] = k
                        V[kk] = -uu
                        kk += 1
                    k += 1
                self.diagComp = range(k0, k)
            if sym:
                k0 = k
                for i in range(dim):
                    for j in range(i+1, dim):
                        I[kk] = i*dim+j
                        J[kk] = k
                        V[kk] = u
                        kk += 1
                        I[kk] = j*dim+i
                        J[kk] = k
                        V[kk] = u
                        kk += 1
                        k+=1
                self.symComp = range(k0, k)
            self.basis = coo_matrix((V[:kk], (I[:kk], J[:kk])), shape=(2 * (dimSkew + dim), self.affineDim))
        else:
            self.basis = np.zeros([2 * (dimSkew + dim), self.affineDim])

            k = 0
            if translation:
                k0 = k
                for i in range(dim):
                    self.basis[i + dim ** 2 ,k] = 1
                    k += 1
                self.transComp = range(k0 ,k)
            if skew_sym:
                k0 = k
                for i in range(dim):
                    for j in range(i + 1, dim):
                        self.basis[i * dim + j, k] = u
                        self.basis[j * dim + i, k] = -u
                        k += 1
                self.rotComp = range(k0 ,k)
            if scale:
                k0 = k
                for i in range(dim):
                    self.basis[i * dim + i, k] = 1. / np.sqrt(dim)
                k += 1
                self.simComp = range(k0, k)
            if diagonal:
                k0 = k
                for i in range(dim - 1):
                    uu = np.sqrt((1 - 1.0 / (i + 2))) / (i + 1.)
                    self.basis[(i + 1) * dim + (i + 1), k] = np.sqrt(1 - 1.0 / (i + 2))
                    for j in range(i + 1):
                        self.basis[j * dim + j, k] = -uu
                    k += 1
                self.diagComp = range(k0, k)
            if sym:
                k0 = k
                for i in range(dim):
                    for j in range(i + 1, dim):
                        self.basis[i * dim + j, k] = u
                        self.basis[j * dim + i, k] = u
                        k += 1
                self.symComp = range(k0, k)



    def getTransforms(self, Afft):
        if Afft is not None:
            Tsize = Afft.shape[0]
            dim2 = self.dim**2
            A = [np.zeros([Tsize, self.dim, self.dim]), np.zeros([Tsize, self.dim])]
            if self.affineDim > 0:
                AB = np.zeros((Tsize, self.basis.shape[0]))
                for t in range(Tsize):
                    AB[t, :] = self.basis.dot(Afft[t,:])
                A[0] = AB[:,0:dim2].reshape([Tsize, self.dim,self.dim])
                A[1] = AB[:,dim2:dim2+self.dim]
            return A
        else:
            return None

#     def getExponential(self, A):
#         if self.dim==3 and self.affCode==3:
#             #t = np.sqrt(A[0,1]**2+A[0,2]**2 + A[1,2]**2)
#             t = np.sqrt((A**2).sum()/2)
#             R = np.eye(3)
#             if t > 1e-10:
#                 R += ((1-np.cos(t))/(t**2)) * (np.dot(A,A)) + (np.sin(t)/t)*A
#         else:
#             R = np.eye(self.dim) + A
#         return R
#
#     def gradExponential(self, A, p, x):
#         dR = np.dot(p.T, x)
#         if self.dim==3 and self.affCode==3:
#             t = np.sqrt(A[0,1]**2+A[0,2]**2 + A[1,2]**2)
#             #s2 = np.sqrt(2.)
#             if t > 1e-10:
#                 st = np.sin(t)
#                 ct = np.cos(t)
#                 a1 = st/t
#                 a2 = (1-ct)/(t**2)
#                 da1 = (t*ct-st)/(2*t**3)
#                 da2 = (t*st -2*(1-ct))/(2*t**4)
# #                dR = (a1*dR + (da1 * (p*np.dot(x,A)).sum() + da2 * (p*np.dot(x,np.dot(A,A))).sum())*A
# #                      - a2 * (np.dot(np.dot(p,A.T).T,x) + np.dot(p.T,np.dot(x,A))))
#                 dR = (a1*dR + (da1 * (p*np.dot(x,A.T)).sum() + da2 * (p*np.dot(x,np.dot(A,A).T)).sum())*A
#                   + a2 * (np.dot(np.dot(p,A).T,x) + np.dot(p.T,np.dot(x,A.T))))
# #            dA = np.random.normal(size=A.shape)
# #            dR2 = np.zeros([self.dim, self.dim])
# #            u0 = (p*np.dot(x,self.getExponential(A).T)).sum()
# #            ep = 1e-8
# #            for k in range(self.dim):
# #                for l in range(self.dim):
# #                    Atry = np.copy(A)
# #                    Atry[k,l] = Atry[k,l] + ep
# #                    dR2[k,l] = ((p*np.dot(x,self.getExponential(Atry).T)).sum() - u0)/ep
# #            #print t, A, dA
# #            print 'dR:', dR
# #            print 'dR2:', dR2
# #            u1 = (p*np.dot(x,self.getExponential(A+ep*dA).T)).sum()
# #            print (u1-u0)/ep, (dR*dA).sum()
# #                  TEST OK
#         return dR

                

            
            
    def integrateFlow(self, Afft):
        Tsize = Afft.shape[0]
        #dim2 = self.dim**2
        X = [np.zeros([Tsize+1, self.dim, self.dim]), np.zeros([Tsize+1, self.dim])]
        A = self.getTransforms(Afft)
        dt = 1.0/Tsize
        eye = np.eye(self.dim)
        X[0][0,...] = eye
        for t in range(Tsize):
            B = getExponential(dt*A[0][t,...])
            X[0][t+1,...] = np.dot(B,X[0][t,...])
            X[1][t+1,...] = np.dot(B,X[1][t,...] + dt * A[1][t,...])
            #X[1][t+1] = X[1][t,...] + dt * A[1][t]
        return X
            
    def projectLinear(self, XA, coeff):
        dim = self.dim
        dim2 = dim**2
        linDim = self.affineDim - self.dim
        A1 = (self.basis[0:dim2, 0:linDim]*XA.reshape([dim2,1])).sum(axis=0)
        A1 = A1/coeff[0:linDim]
        AB = (A1[np.newaxis, :] * self.basis[0:dim2, 0:linDim]).sum(axis=1)
        return AB.reshape([dim, dim])

def getExponential(A):
    if (A.shape[0]==3) and (np.fabs(A.T+A).max() < 1e-8):
        t = np.sqrt((A ** 2).sum() / 2)
        R = np.eye(3)
        if t > 1e-10:
            R += ((1 - np.cos(t)) / (t ** 2)) * (np.dot(A, A)) + (np.sin(t) / t) * A
    elif (A.shape[0]==2) and (np.fabs(A.T+A).max() < 1e-8):
        ct = np.cos(A[0,1])
        st = np.sin(A[0,1])
        R = np.array([[ct,st],[-st,ct]])
    else:
        R = np.eye(A.shape[0]) + A
    return R

def gradExponential(A, p, x):
    dR = np.dot(p.T, x)
    if (A.shape[0]==3) and (np.fabs(A.T+A).max() < 1e-8):
        t = np.sqrt(A[0,1]**2+A[0,2]**2 + A[1,2]**2)
        if t > 1e-10:
            st = np.sin(t)
            ct = np.cos(t)
            a1 = st/t
            a2 = (1-ct)/(t**2)
            da1 = .5*(t*ct-st)/(t**3)
            da2 = .5*(t*st -2*(1-ct))/(t**4)
            dR = (a1*dR + (da1 * (p*np.dot(x,A.T)).sum() + da2 * (p*np.dot(x,np.dot(A,A).T)).sum())*A
                  + a2 * (np.dot(np.dot(p,A).T,x) + np.dot(p.T,np.dot(x,A.T))))
    return dR

