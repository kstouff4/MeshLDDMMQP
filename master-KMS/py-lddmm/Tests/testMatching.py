from Surfaces.surfaces import *
from Common.kernelFunctions import *
from Surfaces.surfaceMatching import *

def main():
    #    fv1 = byufun.readbyu('/Volumes/project/bipolar/AmygdalaShape/1892b87d/1892b87d_l_amyg_reg.byu')
    #fv0 = byufun.readbyu('/Volumes/project/bipolar/AmygdalaShape/12aaa4b6/12aaa4b6_l_amyg_reg.byu')
    #fv0 = Surface(filename='/Users/younes/Development/Data/biocard/shape_analysis/8_6_12/erc/2_qc_flipped_registered/1939956_1_ec_mask_R_qc_reg.byu')
    #fv1 = Surface(filename='/Users/younes/Development/Data/biocard/shape_analysis/8_6_12/erc/2_qc_flipped_registered/2340951_1_ec_mask_R_qc_reg.byu')
    fv0 = Surface(filename='/Users/younes/Development/Data/biocard/testERC/rec1.byu')
    fv1 = Surface(filename='/Users/younes/Development/Data/biocard/testERC/1_to_3_rec_rigidTarget.byu')

    K1 = Kernel(name='gauss', sigma = 6.5)

    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=2.5, sigmaError=1., errorType='current')
    f = (SurfaceMatching(Template=fv0, Target=fv1, outputDir='Results/surfaceMatchingERC',param=sm, testGradient=False,
                         maxIter=1000))
    f.optimizeMatching()
    return f

if __name__=="__main__":
    main()

