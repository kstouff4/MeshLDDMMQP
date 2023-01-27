from Common import loggingUtils
from Common.kernelFunctions import *
from Surfaces.surfaceMultiPhase import *

def compute():

    outputDir = '/cis/home/younes/Development/Results/biocardSliding10test'
    #outputDir = '/cis/home/younes/Development/Results/biocardStitched10'

    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info')
    else:
        loggingUtils.setup_default_logging()
    #path = '/Volumes/project/biocard/data/phase_1_surface_mapping_new_structure/'
    path = '/cis/home/younes/MorphingData/biocard/'
    #path = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/6_mappings_baseline_template_all/0_template_to_all/' ;
    #path2 = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/amygdala/6_mappings_baseline_template_all/0_template_to_all/' ;
    #path = '/cis/home/younes/MorphingData/Biocard/'
    sub1 = '0186193_1_6'
    #sub2 = '1449400_1_L'
    sub2 = '1229175_2_4'
    f0 = []
    f0.append(surfaces.Surface(filename =path + 'Atlas_hippo_L_separate.byu'))
    f0.append(surfaces.Surface(filename =path + 'Atlas_amyg_L_separate.byu'))
    f0.append(surfaces.Surface(filename =path + 'Atlas_ent_L_up_separate.byu'))
    f1 = []
    f1.append(surfaces.Surface(filename =path + 'danielData/0186193_1_6_hippo_L_qc_pass1_daniel2_reg.vtk'))
    f1.append(surfaces.Surface(filename =path + 'danielData/0186193_1_6_amyg_L_qc_pass1_daniel2_reg.vtk'))
    f1.append(surfaces.Surface(filename =path + 'danielData/0186193_1_6_ec_L_qc_pass1_daniel2_reg.vtk'))

    #f0[0].smooth()
    #f0[1].smooth()
    #f1[0].smooth()
    #f1[1].smooth()


    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = 10.)
    ## Background kernel
    K2 = Kernel(name='laplacian', sigma = 1.)
    # f0[0].vertices[:,1] += 2. ;
    # f1[0].vertices[:,1] += 1. ;
    # f0[0].vertices[:,2] += 2.0 ;
    # f1[0].vertices[:,2] += 1.0 ;

    f0[0].computeCentersAreas()
    f1[0].computeCentersAreas()

    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, KparDiffOut=K2, sigmaDist=2.5, sigmaError=1, errorType='measure')
    f = (SurfaceMatching(Template=f0, Target=f1, outputDir=outputDir,param=sm, mu=1,regWeightOut=1., testGradient=True,
                         typeConstraint='sliding', maxIter_cg=1000, maxIter_al=100, affine='none', rotWeight=0.1))
    f.optimizeMatching()


    return f

if __name__=="__main__":
    compute()
