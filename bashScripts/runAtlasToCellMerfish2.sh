#!/bin/bash

while getopts c:g:a:n: flag
do
    case "${flag}" in
        c) cellProbs=${OPTARG};;
        g) geneProbs=${OPTARG};;
        a) atlas=${OPTARG};;
        n) nissl=${OPTARG};;
    esac
done

cd /cis/home/kstouff4/Documents/MeshRegistration/

newV="True" # set to True if using rescaleUnits() function from Younes

# Parameters to use in the setting of rescaleUnits()
sigmaKernel=0.15 # 0.01 orriginall 
sigmaIm=1.02 # 0.2 original
sigmaDist=0.2 # 0.02 original
sigmaError=399 #0.5
ss=-1
rig=1
thresh=0.05

if [[ $newV == "False" ]]; then
    sigmaKernel=0.01
    sigmaIm=0.2
    sigmaDist=0.05
    sigmaError=0.05 # with normalizing weights to 0 or 1
fi

res=100
prefix="AllenToMerfishCellTypeProbs/"
saveDir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/AtlasEstimation/"
targDir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/MERFISH/"
tempDir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/"

mkdir $saveDir$prefix
mkdir $saveDir$prefix"res${res}/"

sd=$saveDir$prefix"res${res}/params${sigmaKernel}$sigmaIm$sigmaDist$sigmaError$ss$thresh/"
mkdir $sd

if [[ $cellProbs == "True" ]]; then

allen="sl889"
merfish="S1R1"
fileTemp="${tempDir}Allen_10_anno_16bit_ap_ax2_${allen}_mesh${res}_rz.vtk"
fileTarg="${targDir}cell_${merfish}_mesh${res}.vtk"
fileTarg="${tempDir}Yongsoo/KimLabDevCCFv001_Annotations_ASL_Oriented_10um_ax0_${allen}_mesh100_rz.vtk"
echo $(date) >> $sd/Allen${allen}_to_Yongsoo${allen}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
python3 -c "import atlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/Allen${allen}_to_Yongsoo${allen}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
echo $(date) >> $sd/Allen${allen}_to_Yongsoo${allen}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt

fileTemp="${tempDir}Allen_10_anno_16bit_ap_ax2_${allen}_mesh${res}_rz.vtk"
merfish="S1R2"
fileTarg="${targDir}cell_${merfish}_mesh${res}.vtk"
echo $(date) >> $sd/Allen${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
python3 -c "import atlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/Allen${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
echo $(date) >> $sd/Allen${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt

allen="sl674"
merfish="S2R1"
fileTemp="${tempDir}Allen_10_anno_16bit_ap_ax2_${allen}_mesh${res}_rz.vtk"
fileTarg="${targDir}cell_${merfish}_mesh${res}.vtk"
fileTarg="${tempDir}Yongsoo/KimLabDevCCFv001_Annotations_ASL_Oriented_10um_ax0_${allen}_mesh100_rz.vtk"
#echo $(date) >> $sd/Allen${allen}_to_Yongsoo${allen}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
#python3 -c "import atlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/Allen${allen}_to_Yongsoo${allen}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
#echo $(date) >> $sd/Allen${allen}_to_Yongsoo${allen}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt

merfish="S2R2"
fileTarg="${targDir}cell_${merfish}_mesh${res}.vtk"
echo $(date) >> $sd/Allen${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
python3 -c "import atlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/Allen${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
echo $(date) >> $sd/Allen${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt

fi
#######################################################
# YOUNGSOO
if [[ $atlas == "True" ]]; then
ss=-1
allen="sl889"
merfish="S1R1_mesh${res}_newImage"
fileTemp="${tempDir}Yongsoo/KimLabDevCCFv001_Annotations_ASL_Oriented_10um_ax0_${allen}_mesh100_rz.vtk"
fileTarg="${targDir}cell_${merfish}.vtk"
echo $(date) >> $sd/Yongsoo${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
python3 -c "import atlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/Yongsoo${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
echo $(date) >> $sd/Yongsoo${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt

ss=-0.95
merfish="S1R2"
fileTarg="${targDir}cell_${merfish}_mesh${res}.vtk"
#echo $(date) >> $sd/Yongsoo${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt
#python3 -c "import atlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss); quit()" >> $sd/Yongsoo${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt
#echo $(date) >> $sd/Yongsoo${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt

ss=-1
allen="sl674"
merfish="S2R1_mesh${res}_newImage"
fileTemp="${tempDir}Yongsoo/KimLabDevCCFv001_Annotations_ASL_Oriented_10um_ax0_${allen}_mesh100_rz.vtk"
fileTarg="${targDir}cell_${merfish}_mesh${res}.vtk"
#echo $(date) >> $sd/Yongsoo${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt
#python3 -c "import atlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss); quit()" >> $sd/Yongsoo${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt
#echo $(date) >> $sd/Yongsoo${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt

ss=-0.95
merfish="S2R2"
fileTarg="${targDir}cell_${merfish}_mesh${res}.vtk"
#echo $(date) >> $sd/Yongsoo${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt
#python3 -c "import atlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss); quit()" >> $sd/Yongsoo${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt
#echo $(date) >> $sd/Yongsoo${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError.txt