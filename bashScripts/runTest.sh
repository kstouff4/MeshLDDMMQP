#!/bin/bash
# Usage: ./runTest.sh

cd ../pythonFunctions

# Tested Parameters for Test Data Set 
sigmaKernel=0.5 # Scale of Laplacian kernel controlling regularization of estimated velocity field 
sigmaIm=1 # Scale of feature kernel in varifold norm; tested with euclidean kernel (Id matrix)
sigmaDist=0.2 # Scale of spatial kernel in varifold norm; Gaussian kernel used as default 
sigmaError=4000 # Coefficient (inverse) for calibrating matching cost (varifold norm) against regularization of velocity field 
ss=-1 # isotropic scaling of template to match support of target; -1 to estimate scale based on differences in area between template and target
rig=1 # number of iterations in which to estimate rigid (rotation + translation) transformation only without diffeomorphism; rigid will be fixed following this number of iterations
thresh=0.01 # constraint on estimated densities for template in feature space of target given as percentile of distribution of densities for target (e.g. alpha > 1% values of alpha in target)
iters=20 # iterations of alternation between QP and LDDMM

resTemp=100
resTarg=50

saveDir="../Results/"
tempDir="../TestImages/"

mkdir $saveDir

fileTemp=$tempDir"Allen_10_anno_16bit_ap_ax2_sl484_mesh100_rz.vtk"
fileTarg=$tempDir"202202221440_60988212_VMSC01601_mesh50_rz_probRNA_t10parts.vtk"

python3 -c "import cleanAtlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$saveDir',diffeo=True,iters=$iters,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()"
