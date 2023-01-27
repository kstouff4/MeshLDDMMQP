from PointSets import pointSets
from Common import pointEvolution
from Surfaces.surfaces import *
from Common.kernelFunctions import *
from Surfaces.secondOrderFiberMatching import *

def compute(createsurfaces=True):

    if createsurfaces:
        [x,y,z] = np.mgrid[0:200, 0:200, 0:200]/100.
        ay = np.fabs(y-1)
        az = np.fabs(z-1)
        ax = np.fabs(x-0.5)
        s2 = np.sqrt(2)
        c1 = np.sqrt(0.06)
        c2 = np.sqrt(0.045)
        c3 = 0.1

        I1 = np.minimum(c1**2 - (ax**2 + 0.5*ay**2 + az**2), np.minimum((ax**2 + 0.5*ay**2 + az**2)-c2**2, 1+c3-y)) 
        fv1 = Surface() ;
        fv1.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)


        #return fv1

        #s1 = 1.375
        #s2 = 2
        s1 = 1.1
        s2 = 1.2
        p = 1.75
        I1 = np.minimum(c1**p/s1 - ((ax**p + 0.5*ay**p + az**p)), np.minimum((s2*ax**p + s2*0.5*ay**p + s2*az**p)-c2**p/s1, 1+c3/s1-y))  
        fv2 = Surface() ;
        fv2.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)
        
        fv2.vertices[:,1] += 15 - 15/s1

        s1 *= 1.1
        s2 *= 1.2
        I1 = np.minimum(c1**p/s1 - ((ax**p + 0.5*ay**p + az**p)), np.minimum((s2*ax**p + s2*0.5*ay**p + s2*az**p)-c2**p/s1, 1+c3/s1-y))  
        fv3 = Surface() ;
        fv3.Isosurface(I1, value = 0, target=1000, scales=[1, 1, 1], smooth=0.01)
        
        fv3.vertices[:,1] += 15 - 15/s1

        
        fv1.saveVTK('/Users/younes/Development/Results/Fibers/fv1.vtk')
        fv2.saveVTK('/Users/younes/Development/Results/Fibers/fv2.vtk')
        fv3.saveVTK('/Users/younes/Development/Results/Fibers/fv3.vtk')
    else:
        fv1 = Surface(filename='/Users/younes/Development/Results/Fibers/fv1.vtk')
        fv2  = Surface(filename='/Users/younes/Development/Results/Fibers/fv2.vtk')
        fv3  = Surface(filename='/Users/younes/Development/Results/Fibers/fv3.vtk')

    ## Long axis is 0y
    V1 = fv1.vertices/100 - [0,1,1]
    N1 = fv1.computeVertexNormals()
    
    sel0 = V1[:,1] < 0.95*V1[:,1].max()
    I0 = np.nonzero(sel0)
    I0 = I0[0]
    V1 = V1[I0, :]
    N1 = N1[I0, :]


    QNN = (N1[:,0]-0.5)**2 + 0.5*N1[:,1]**2 + N1[:,2]**2
    QVV = (V1[:,0]-0.5)**2 + 0.5*V1[:,1]**2 + V1[:,2]**2
    QNV = (N1[:,0]-0.5)*(V1[:,0]-0.5) + 0.5*N1[:,1]*V1[:,1] + N1[:,2]*V1[:,2]

    #exterior wall
    sel1 = np.logical_and(np.fabs(QVV-0.06) < 0.01, (np.fabs(N1[:,1]) < 0.8))
    I1 = np.nonzero(sel1)
    I1= I1[0]

    V11 = V1[I1, :]
    N11 = N1[I1, :]
    a = np.zeros(len(I1))
    a = (-QNV[I1] - np.sqrt(QNV[I1]**2 - QNN[I1]*(QVV[I1]-0.0525)))/QNN[I1]
    N = V1.shape[0]
    M = V1.shape[0] + len(I1)
    #M = M1 + len(I1)
    y0 = np.zeros([M,3])
    v0 = np.zeros([M,3])
    nz = np.zeros([N, 3])
    nz2 = np.zeros([N, 3])
    y0[0:N, :] = V1 
    #y0[N:M1, :] = V11 + 0.5*a[:,np.newaxis]*N11
    y0[N:M, :] = V11 + a[:,np.newaxis]*N11
    nz[:, 0] = -N1[:, 2]
    nz[:, 2] = N1[:, 0]
    nz2[:,0] = N1[:,0]*N1[:,1]
    nz2[:,1] = -(N1[:,0]**2 + N1[:,2]**2)
    nz2[:,2] = N1[:,2]*N1[:,1]
    nz = nz / (1e-5 + np.sqrt((nz**2).sum(axis=1)[:,np.newaxis]))
    nz2 = nz2 / (1e-5 + np.sqrt((nz2**2).sum(axis=1)[:,np.newaxis]))
    theta = 2*np.pi/3
    psi = 0 #np.pi/12
    c = np.cos(theta)
    s = np.sin(theta)
    c0 = np.cos(psi)
    s0 = np.sin(psi)
    v0[0:N, :] = -c*nz - s*nz2 
    v0[I1,:] *= -1
    #v0[I1, :] += 2*s*nz2[I1,:]
    v0[N:M, :] = nz[I1, :]
    #v0[M1:M, :] = -c*nz[I1,:] + s*nz2[I1,:]
    v0[0:N,:] = c0*v0[0:N, :] - s0*N1
     
    K1 = Kernel(name='laplacian', sigma = 10.0)
    y0 = 100*(y0+[0,1,1])
    pointSets.savePoints('/Users/younes/Development/Results/Fibers/Ellipses/fibers.vtk', y0, vector=v0)

    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=5, sigmaError=1., errorType='measure')
    f = SurfaceMatching(Template=fv1, Target=[fv2,fv3], Fiber=(y0,v0), outputDir='/Users/younes/Development/Results/Fibers/Ellipses',param=sm, testGradient=False,
                        #subsampleTargetSize = 500,
                         maxIter=1000)

    xt, at, yt, vt = pointEvolution.secondOrderFiberEvolution(fv1.vertices, np.zeros(y0.shape), y0, v0, -10 * np.ones([10, y0.shape[0]]), K1)
    fvDef = Surface(surf=fv1)
    for k in range(xt.shape[0]):
        fvDef.updateVertices(xt[k, ...])
        fvDef.saveVTK('/Users/younes/Development/Results/Fibers/fvDef'+str(k)+'.vtk')

        #return
    f.optimizeMatching()


    return f, f23
