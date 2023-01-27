#!/bin/bash

cd /cis/home/kstouff4/Documents/MeshRegistration/

newV="False" # set to True if using rescaleUnits() function from Younes

# Parameters to use in the setting of rescaleUnits()
sigmaKernel=0.1 # 0.01 orriginall 
sigmaIm=1 # 0.2 original
sigmaDist=0.5 # 0.02 original
sigmaError=500 #0.5

if [[ $newV == "False" ]]; then
    sigmaKernel=0.01
    sigmaIm=0.2
    sigmaDist=0.05
    sigmaError=0.05 # with normalizing weights to 0 or 1
fi

res=100
prefix="AllenToMerfishCellTypes/"
saveDir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/AtlasEstimation/"
targDir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/MERFISH/"
tempDir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/"

mkdir $saveDir$prefix

sd=$saveDir$prefix

fileTemp="${tempDir}Allen_10_anno_16bit_ap_ax2_sl889_mesh${res}_rz.vtk"
fileTarg="${targDir}cell_S1R1_mesh${res}.vtk"
echo $(date) >> $sd/AllentoMERFISHres${res}_NonRigid_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt
python3 -c "import atlasEstimation as ae; ae.solveThetaNonRigid('$fileTemp','$fileTarg','$sd',iters=10,K2='Id',diffeo=True,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newV=$newV); quit()" >> $sd/AllentoMERFISHres${res}_NonRigid_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt
echo $(date) >> $sd/AllentoMERFISHres${res}_NonRigid_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt

fileTarg="${targDir}cell_S1R2_mesh${res}.vtk"
echo $(date) >> $sd/AllentoMERFISHres${res}_NonRigid_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt
python3 -c "import atlasEstimation as ae; ae.solveThetaNonRigid('$fileTemp','$fileTarg','$sd',iters=10,K2='Id',diffeo=True,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newV=$newV); quit()" >> $sd/AllentoMERFISHres${res}_NonRigid_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt
echo $(date) >> $sd/AllentoMERFISHres${res}_NonRigid_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt

fileTemp="${tempDir}Allen_10_anno_16bit_ap_ax2_sl674_mesh${res}_rz.vtk"
fileTarg="${targDir}cell_S2R1_mesh${res}.vtk"
echo $(date) >> $sd/AllentoMERFISHres${res}_NonRigid_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt
python3 -c "import atlasEstimation as ae; ae.solveThetaNonRigid('$fileTemp','$fileTarg','$sd',iters=10,K2='Id',diffeo=True,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newV=$newV); quit()" >> $sd/AllentoMERFISHres${res}_NonRigid_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt
echo $(date) >> $sd/AllentoMERFISHres${res}_NonRigid_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt

fileTarg="${targDir}cell_S2R2_mesh${res}.vtk"
echo $(date) >> $sd/AllentoMERFISHres${res}_NonRigid_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt
python3 -c "import atlasEstimation as ae; ae.solveThetaNonRigid('$fileTemp','$fileTarg','$sd',iters=10,K2='Id',diffeo=True,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newV=$newV); quit()" >> $sd/AllentoMERFISHres${res}_NonRigid_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt
echo $(date) >> $sd/AllentoMERFISHres${res}_NonRigid_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt

