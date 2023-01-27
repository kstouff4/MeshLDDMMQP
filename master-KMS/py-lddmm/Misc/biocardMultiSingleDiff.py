from Surfaces.surfaces import *
from Common.kernelFunctions import *
from Surfaces.surfaceMatching import *
from Common import pointEvolution as evol, loggingUtils


def compute():

    outputDir = '/Users/younes/Development/Results/biocardMultiLargeKernel'
    #outputDir = '/cis/home/younes/MorphingData/twoBallsStitched'
    #outputDir = '/Users/younes/Development/Results/tight_stitched_rigid2_10'
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
    f0.append(surfaces.Surface(filename = path+'Atlas_hippo_L_separate.byu'))
    f0.append(surfaces.Surface(filename = path+'Atlas_amyg_L_separate.byu'))
    f0.append(surfaces.Surface(filename = path+'Atlas_ent_L_up_separate.byu'))
    fv0 = Surface()
    fv0.concatenate(f0)
    f1 = []
    f1.append(surfaces.Surface(filename = path+'danielData/0186193_1_6_hippo_L_qc_pass1_daniel2_reg.vtk'))
    f1.append(surfaces.Surface(filename = path+'danielData/0186193_1_6_amyg_L_qc_pass1_daniel2_reg.vtk'))
    f1.append(surfaces.Surface(filename = path+'danielData/0186193_1_6_ec_L_qc_pass1_daniel2_reg.vtk'))
    fv1 = Surface()
    fv1.concatenate(f1)


    K1 = Kernel(name='laplacian', sigma = 2.5)
    
    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=2.5, sigmaError=1, errorType='varifold') 
                              #,internalCost='h1')
    f = (SurfaceMatching(Template=fv0, Target=fv1, outputDir=outputDir,param=sm, testGradient=True,
                         regWeight=0.1, maxIter=2000, affine='none', rotWeight=0.1, internalWeight=10.0))
    f.optimizeMatching()

    if len(f0) > 1:
        A = [np.zeros([f.Tsize, f.dim, f.dim]), np.zeros([f.Tsize, f.dim])]
        (xt, Jt)  = evol.landmarkDirectEvolutionEuler(f.x0, f.at, f.param.KparDiff, affine=A,
                                                          withJacobian=True)
        nu = f.fv0ori*f.fvInit.computeVertexNormals()
        v = f.v[0,...]
        displ = np.zeros(f.npt)
        dt = 1.0 /f.Tsize ;
        fvDef = surfaces.Surface(f.fvInit)
        AV0 = fvDef.computeVertexArea()
        for kk in range(f.Tsize+1):
            nv0 = 0
            for (ll,fv) in enumerate(f0):
                nv = nv0 + fv.vertices.shape[0]
                fvDef = surfaces.Surface(fv)
                fvDef.updateVertices(np.squeeze(xt[kk, nv0:nv, :]))
                AV = fvDef.computeVertexArea()
                AV = (AV[0]/AV0[0][nv0:nv,:])
                vf = surfaces.vtkFields() ;
                vf.scalars.append('Jacobian') ;
                vf.scalars.append(np.exp(Jt[kk, nv0:nv]))
                vf.scalars.append('Jacobian_T') ;
                vf.scalars.append(AV[:,0])
                vf.scalars.append('Jacobian_N') ;
                vf.scalars.append(np.exp(Jt[kk, nv0:nv])/(AV[:,0]))
                vf.scalars.append('displacement')
                vf.scalars.append(displ[nv0:nv])
                if kk < f.Tsize:
                    kkm = kk
                else:
                    kkm = kk-1
                vf.vectors.append('velocity') ;
                vf.vectors.append(f.v[kkm,nv0:nv])
                fvDef.saveVTK2(f.outputDir +'/'+ f.saveFile+str(ll)+'_'+str(kk)+'.vtk', vf)
                nv0 = nv
            if kk < f.Tsize:
                nu = f.fv0ori*f.fvDef.computeVertexNormals()
                v = f.v[kk,...]
                displ += dt * (v*nu).sum(axis=1)

    return f

if __name__=="__main__":
    compute()
