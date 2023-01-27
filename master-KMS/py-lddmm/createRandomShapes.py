from base.surfaces import Surface
from base.pointSets import savelmk
import numpy as np
import scipy.linalg as la
import vtk
from base import diffeo

def compute():
    fv0 = Surface(filename='/Users/younes/Development/Data/sculptris/shape1.obj')
    fv0.updateVertices(10*(fv0.vertices-fv0.vertices.mean()))

    J = np.random.choice(fv0.vertices.shape[0], 50, replace=False)

    center = fv0.vertices.sum(axis=0)/fv0.vertices.shape[0]
    
    a = np.sign(fv0.vertices[:,0] - center[0] - 5) + 1
    a = a[:,np.newaxis]
    #print(fv0.normGrad(a))
    for k in range(500):
        a += 0.0001 * fv0.laplacian(a, weighted=False)
    #print(fv0.normGrad(a))

    fv0.saveVTK('/Users/younes/Development/Data/sculptris/Dataset/template_surf.vtk')
    img = fv0.compute3DVolumeImage(xmin=-50, xmax=50, origin = np.array([0,0,0]))[0]
    diffeo.gridScalars(data=img).saveImg(
        '/Users/younes/Development/Data/sculptris/Dataset/template_img.vtk')

    #print(fv0.surfArea())
    npx = np.arange(-50, 51, 1, dtype=int)
    ln = len(npx)
    npgrid = np.meshgrid(npx, npx, npx)
    x = npgrid[0].ravel()
    y = npgrid[1].ravel()
    z = npgrid[2].ravel()

    points = vtk.vtkPoints()
    for i in range(len(x)):
        points.InsertNextPoint(x[i],y[i],z[i])

    # #grid = numpy_support.numpy_to_vtk(npgrid.ravel(), deep=True) # ,deep=0 ,array_type=None)
    # grid = vtk.vtkRectilinearGrid()
    # grid.SetDimensions(len(npx) ,len(npx) ,len(npx))
    # grid.SetXCoordinates(x)
    # grid.SetYCoordinates(x)
    # grid.SetZCoordinates(x)

    
    for l in range(100):
        A = 0.25 * np.random.randn(3 ,3)
        R = la.expm((A - A.T) / 2)
        b = 10 * np.random.randn(1 ,3)
        for s in ('left', 'right'):
            n = -.25*np.random.uniform(size=(fv0.vertices.shape[0],1))
            for k in range(100):
                n += 0.0001 * fv0.laplacian(n, weighted=False)
            fv1 = Surface(surf=fv0)
            for k in range(1000):
                 fv1.updateVertices(fv1.vertices + 0.003 * (n) * fv1.meanCurvatureVector())
            fv1.smooth()
            if s == 'left':
                fv1.updateVertices(np.dot(fv1.vertices,R) + b)
            else:
                v = np.copy(fv1.vertices)
                m = np.min(v[:,2])
                v[:,2] = 2*m - v[:,2]
                v = np.dot(v,R) + b
                fv1.updateVertices(v)
                fv1.flipFaces()



            img0, origin, orig2 = fv1.compute3DVolumeImage(xmin=0, xmax=100)# origin = np.array([0,0,0]))
            I = np.nonzero(img0)
            print(origin, orig2)
            m0 = I[0].mean()
            m1 = I[1].mean()
            m2 = I[2].mean()
            #M = (fv1.vertices.mean(axis=0) + 50 - np.array((m0,m1,m2))).astype(int)
            M = (origin - orig2).astype(int)
            #img = np.roll(img0, [M[0],M[1],M[2]], axis=[0,1,2] )
            img = img0
            # for i in range(I[0].shape[0]):
            #     img[I[0]-M[0],I[1]-M[1],I[2]-M[2]] = img0[I[0], I[1], I[2]]
            fv1.saveVTK('/Users/younes/OneDrive - Johns Hopkins University/RESEARCH/Data/Dataset/Original Surfaces/subject_' + s + '_surf'+str(l+1)+'.vtk')
            lmk = fv1.vertices[J,:]
            savelmk(lmk, '/Users/younes/OneDrive - Johns Hopkins University/RESEARCH/Data/Dataset/landmarks/subject_' + s + '_lmk'+str(l+1)+'.lmk')
            diffeo.gridScalars(data=img, origin=orig2).saveImg('/Users/younes/OneDrive - Johns Hopkins University/RESEARCH/Data/Dataset/subject_' + s + '_img'+str(l+1)+'.vtk')
            print('subject_'+s+'_img'+str(l+1)+'.vtk')




if __name__=="__main__":
    compute()