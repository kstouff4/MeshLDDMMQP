from Common import loggingUtils
from Common.kernelFunctions import *
from Surfaces import surfaceMatching, surfaces, surfaceTimeSeries as match


#import secondOrderMatching as match

def compute(Atrophy=False):
    if Atrophy:
        pass
    else:
        pass

    #outputDir = '/Users/younes/Development/Results/biocardTS/spline'
    outputDir = '/Users/younes/Development/Results/biocardTS/piecewise'
    #outputDir = '/cis/home/younes/MorphingData/twoBallsStitched'
    #outputDir = '/Users/younes/Development/Results/tight_stitched_rigid2_10'
    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info', stdOutput = True)
    else:
        loggingUtils.setup_default_logging()

    rdir = '/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/6_mappings_baseline_template_all/0_template_to_all/' ;
    sub = '2840698'
    #sub = '2729611'
    fv0 = surfaces.Surface(filename='/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/4_create_population_based_template/newTemplate.byu')
    if sub == '2840698':
        fv1 = surfaces.Surface(filename=rdir + '2840698_2_4_hippo_L_reg.byu_10_6.5_2.5.byu')
        fv2 = surfaces.Surface(filename=rdir + '2840698_3_4_hippo_L_reg.byu_10_6.5_2.5.byu')
        fv3 = surfaces.Surface(filename=rdir + '2840698_4_8_hippo_L_reg.byu_10_6.5_2.5.byu')
        fv4 = surfaces.Surface(filename=rdir + '2840698_5_6_hippo_L_reg.byu_10_6.5_2.5.byu')
    if sub == '2729611':
        fv1 = surfaces.Surface(filename=rdir + sub + '_1_4_hippo_L_reg.byu_10_6.5_2.5.byu')
        fv2 = surfaces.Surface(filename=rdir + sub + '_2_4_hippo_L_reg.byu_10_6.5_2.5.byu')
        fv3 = surfaces.Surface(filename=rdir + sub + '_3_7_hippo_L_reg.byu_10_6.5_2.5.byu')
        fv4 = surfaces.Surface(filename=rdir + sub + '_4_6_hippo_L_reg.byu_10_6.5_2.5.byu')

    ## Object kernel
    K1 = Kernel(name='laplacian', sigma = 6.5, order=4)


    sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=2.5, sigmaError=1., errorType='varifold')
    if Atrophy:
        f = match.SurfaceMatching(Template=fv0, Targets=(fv1,fv2,fv3,fv4), outputDir=outputDir, param=sm, regWeight=.1,
                                affine='none', testGradient=True, affineWeight=.1,  maxIter_cg=1000, mu=0.0001)
    else:
       f = match.SurfaceMatching(Template=fv0, Targets=(fv1,fv2,fv3,fv4), outputDir=outputDir, param=sm, regWeight=.1,
                                affine='none', testGradient=True, affineWeight=.1,  maxIter=1000)
 
    #, affine='none', rotWeight=0.1))
    f.optimizeMatching()


    return f

if __name__=="__main__":
    compute(True)

