from Surfaces.surfaces import *
from Common.kernelFunctions import *
from affineRegistration import *
from Surfaces.surfaceMultiPhase import *

def compute():

    ## Build Two colliding ellipses

    fv01 = Surface(filename = '/Users/younes/Development/Data/multishape/biocard/Atlas_hippo_L_separate.byu')
    fv02 = Surface(filename = '/Users/younes/Development/Data/multishape/biocard/Atlas_amyg_L_separate.byu')
    fv03 = Surface(filename = '/Users/younes/Development/Data/multishape/biocard/Atlas_ent_L_up_separate.byu')

    name = '0186193_1_6_'

    fv1 = Surface(filename = '/Users/younes/Development/Data/multishape/biocard/danielData/'+name+'hippo_L_qc_pass1_daniel2.byu')
    fv2 = Surface(filename = '/Users/younes/Development/Data/multishape/biocard/danielData/'+name+'amyg_L_qc_pass1_daniel2.byu')
    fv3 = Surface(filename = '/Users/younes/Development/Data/multishape/biocard/danielData/'+name+'ec_L_qc_pass1_daniel2.byu')


    vert0 = np.vstack((fv01.vertices, fv02.vertices, fv03.vertices))
    vert = np.vstack((fv1.vertices, fv2.vertices, fv3.vertices))
    R0, T0 = rigidRegistration(surfaces = (vert, vert0),  verb=True, temperature=10., annealing=True)
    fv1.updateVertices(np.dot(fv1.vertices, R0.T) + T0)
    fv2.updateVertices(np.dot(fv2.vertices, R0.T) + T0)
    fv3.updateVertices(np.dot(fv3.vertices, R0.T) + T0)


    ## Object kernel
    K1 = Kernel(name='gauss', sigma = 6.5)
    ## Background kernel
    K2 = Kernel(name='laplacian', sigma = 1.0, order=3)

    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, KparDiffOut=K2, sigmaDist=1., sigmaError=1., errorType='varifold')
    f = (SurfaceMatching(Template=(fv01,fv02,fv03), Target=(fv1,fv2,fv3), outputDir='/Users/younes/Development/Results/biocard_varifold_stitched'+name,param=sm, mu=.01,regWeightOut=1, testGradient=True,
                         typeConstraint='stitched', maxIter_cg=1000, maxIter_al=100, affine='none', rotWeight=0.1))
    f.optimizeMatching()


    return f

if __name__=="__main__":
    compute()

