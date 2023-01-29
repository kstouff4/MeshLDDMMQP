#!/bin/bash
# Usage: ./runQP.sh -a True -c True -g False -n False -d True -t allen -r allen


while getopts c:g:a:n:d:t:r: flag
do
    case "${flag}" in
        c) cellProbs=${OPTARG};; # compute cell type probs
        g) geneProbs=${OPTARG};; # compute gene type probs
        a) atlas=${OPTARG};; # compute atlas to atlas
        n) nissl=${OPTARG};; # compute Nissl intensity
        d) diffeo=${OPTARG};; # false --> rigid only
        t) atlasName=${OPTARG};; # allen or kim 
        r) targetName=${OPTARG};; # allen or merfish 
    esac
done


cd /cis/home/kstouff4/Documents/MeshRegistration/MeshLDDMMQP/pythonFunctions
#export PYTHON_PATH=/cis/home/kstouff4/Documents/MeshRegistration/Scripts-KMS/approxCode/:$PYTHON_PATH

newV="True" # set to True if using rescaleUnits() function from Younes

# Parameters to use in the setting of rescaleUnits()
sigmaKernel=0.5 # making bigger than 0.15 for Younes # 0.01 orriginall 
sigmaIm=1.02 # 0.2 original
sigmaDist=0.2 # 0.02 original
sigmaError=400 #0.5
ss=-1
rig=1
thresh=0.1
iters=20

if [[ $newV == "False" ]]; then
    sigmaKernel=0.01
    sigmaIm=0.2
    sigmaDist=0.05
    sigmaError=0.05 # with normalizing weights to 0 or 1
fi

res=100
prefix="AllenToMerfishCellTypeProbs/"
saveDir="/cis/home/kstouff4/Documents/MeshRegistration/Results/"
targDir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/MERFISH/"
tempDir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/"

mkdir $saveDir

if [[ $cellProbs == "True" ]]; then
    echo "Computing Cell Probability Mapping and Cell Densities"
    if [[ $atlasName == "kim" ]]; then
        prefix="KimToMerfishCellTypeProbs/"
        fileTempP="${tempDir}Yongsoo/KimLabDevCCFv001_Annotations_ASL_Oriented_10um_ax0_" #${allen}_mesh100_rz.vtk"
    elif [[ $atlasName == "allen" ]]; then
        prefix="AllenToMerfishCellTypeProbs/"
        fileTempP="${tempDir}Allen_10_anno_16bit_ap_ax2_"
    fi
    mkdir $saveDir$prefix
    mkdir $saveDir$prefix"res${res}/"
    sd=$saveDir$prefix"res${res}/params${sigmaKernel}$sigmaIm$sigmaDist$sigmaError$ss$thresh/"
    mkdir $sd

    allen="sl889"
    fileTemp="${fileTempP}${allen}_mesh${res}_rz.vtk"
    merfish="S1R1"
    fileTarg="${targDir}cell_${merfish}_mesh${res}.vtk"
    echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    python3 -c "import cleanAtlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt

    merfish="S1R2"
    fileTarg="${targDir}cell_${merfish}_mesh${res}.vtk"
    echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    python3 -c "import cleanAtlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    
    allen="sl674"
    fileTemp="${fileTempP}${allen}_mesh${res}_rz.vtk"
    merfish="S2R1"
    fileTarg="${targDir}cell_${merfish}_mesh${res}.vtk"
    echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    python3 -c "import cleanAtlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt

    merfish="S2R2"
    fileTarg="${targDir}cell_${merfish}_mesh${res}.vtk"
    echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    python3 -c "import cleanAtlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
     
fi

#######################################################
# Slide Seq 

if [[ $nissl == "True" ]]; then
    echo "Computing Nissl Values and Densities of Pixels > Threshold"
    res=100
    res2=100
    ss=-1
    thresh=0.05
    sigmaKernel=0.5
    sigmaError=50000 #400 when thresh is 0
    prefix="AllenToSlideSeqNissl/"
    saveDir="/cis/home/kstouff4/Documents/MeshRegistration/Results/"
    targDir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/BROAD/Nissl/Gray/"
    tempDir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/"

    mkdir $saveDir$prefix
    mkdir $saveDir$prefix"res${res}to${res2}/"
    
    if [[ $diffeo == "False" ]]; then
        rig=$iters
        sd=$saveDir$prefix"res${res}to${res2}/params${sigmaKernel}$sigmaIm$sigmaDist$sigmaError$ss${thresh}Rigid/"
        mkdir $sd
    fi

    sd=$saveDir$prefix"res${res}to${res2}/params${sigmaKernel}$sigmaIm$sigmaDist$sigmaError$ss${thresh}"
    mkdir $sd
    allen="sl929"
    merfish="210804_15"
    fileTemp="${tempDir}Allen_10_anno_16bit_ap_ax2_${allen}_mesh${res}_rz.vtk"
    fileTarg="${targDir}Puck_${merfish}_mesh${res2}_rz.vtk"
    echo $(date) >> $sd/Allen${allen}_to_Nissl${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    python3 -c "import cleanAtlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=$diffeo,iters=$iters,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/Allen${allen}_to_Nissl${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    echo $(date) >> $sd/Allen${allen}_to_Nissl${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt

    allen="sl679"
    merfish="210922_04"
    fileTemp="${tempDir}Allen_10_anno_16bit_ap_ax2_${allen}_mesh${res}_rz.vtk"
    fileTarg="${targDir}Puck_${merfish}_mesh${res2}_rz.vtk"
    echo $(date) >> $sd/Allen${allen}_to_Nissl${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    python3 -c "import cleanAtlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=$diffeo,iters=$iters,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/Allen${allen}_to_Nissl${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    echo $(date) >> $sd/Allen${allen}_to_Nissl${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt

    allen="sl479"
    merfish="210830_04"
    fileTemp="${tempDir}Allen_10_anno_16bit_ap_ax2_${allen}_mesh${res}_rz.vtk"
    fileTarg="${targDir}Puck_${merfish}_mesh${res2}_rz.vtk"
    echo $(date) >> $sd/Allen${allen}_to_Nissl${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    python3 -c "import cleanAtlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=$diffeo,iters=$iters,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/Allen${allen}_to_Nissl${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    echo $(date) >> $sd/Allen${allen}_to_Nissl${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
fi
#################################
# Yongsoo to Allen
# Parameters to use in the setting of rescaleUnits()

if [[ $atlas == "True" ]]; then
    echo "Computing Atlas Labels and No Densities"
    sigmaKernel=0.5 #0.15 # 0.01 orriginall 
    sigmaIm=1.02 # 0.2 original
    sigmaDist=0.2 # 0.02 original
    sigmaError=100 #300 #0.5
    ss=-1
    rig=1
    thresh=0

    prefix="AtlasToAtlas/"
    targDir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/MERFISH/"
    tempDir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/"

    mkdir $saveDir$prefix
    mkdir $saveDir$prefix"res${res}/"

    sd=$saveDir$prefix"res${res}/params${sigmaKernel}$sigmaIm$sigmaDist$sigmaError$ss$thresh/"
    mkdir $sd
    
    sls=("sl479" "sl679" "sl929" "sl889" "sl674" "sl384" "sl484" "sl584")
    for allen in ${sls[*]}; do
        fileTarg="${tempDir}Allen_10_anno_16bit_ap_ax2_${allen}_mesh${res}_rz.vtk"
        fileTemp="${tempDir}Yongsoo/KimLabDevCCFv001_Annotations_ASL_Oriented_10um_ax0_${allen}_mesh100_rz.vtk"
        echo $(date) >> $sd/Yongsoo${allen}_to_Allen${allen}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        python3 -c "import cleanAtlasEstimation as ae; ae.solveAtlastoAtlas('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=0); quit()" >> $sd/Yongsoo${allen}_to_Allen${allen}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        echo $(date) >> $sd/Yongsoo${allen}_to_Allen${allen}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    
        echo $(date) >> $sd/Allen${allen}_to_Yongsoo${allen}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        python3 -c "import cleanAtlasEstimation as ae; ae.solveAtlastoAtlas('$fileTarg','$fileTemp','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=0); quit()" >> $sd/Allen${allen}_to_Yongsoo${allen}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        echo $(date) >> $sd/Allen${allen}_to_Yongsoo${allen}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    done    
fi

if [[ $geneProbs == "True" ]]; then
    thresh=0.01
    sigmaKernel=0.5 # 0.5 # smoother diffeomorphisms
    tempDir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/"
    resTemp=100
    resTarg=100 #50
    echo "Computing Gene Probability Mapping and Gene Densities"
    if [[ $targetName == "allen" ]]; then
        resTarg=50
        sigmaKernel=0.5
        sigmaDist=0.2
        sigmaError=4000
        if [[ $atlasName == "kim" ]]; then
            prefix="KimToAllenGeneTypeProbs/"
            fileTempP="${tempDir}Yongsoo/KimLabDevCCFv001_Annotations_ASL_Oriented_10um_ax0_" #${allen}_mesh100_rz.vtk"
        elif [[ $atlasName == "allen" ]]; then
            prefix="AllenToAllenGeneTypeProbs/"
            fileTempP="${tempDir}Allen_10_anno_16bit_ap_ax2_"
        fi
        targDir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/Allen3DMeshes/"
    elif [[ $targetName == "merfish" ]]; then
        sigmaIm=1.01 # changed just for saving in new folder 
        if [[ $atlasName == "kim" ]]; then
            prefix="KimToMerfishGeneTypeProbs/"
            fileTempP="${tempDir}Yongsoo/KimLabDevCCFv001_Annotations_ASL_Oriented_10um_ax0_" #${allen}_mesh100_rz.vtk"
        elif [[ $atlasName == "allen" ]]; then
            prefix="AllenToMerfishGeneTypeProbs/"
            fileTempP="${tempDir}Allen_10_anno_16bit_ap_ax2_"
        fi
        targDir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/MERFISH/"
    fi

    #prefix="AllenToMerfishRNAProbs/"
    saveDir="/cis/home/kstouff4/Documents/MeshRegistration/Results/"

    mkdir $saveDir$prefix
    mkdir $saveDir$prefix"res${resTemp}to${resTarg}/"

    sd=$saveDir$prefix"res${resTemp}to${resTarg}/params${sigmaKernel}$sigmaIm$sigmaDist$sigmaError$ss$thresh/"
    mkdir $sd

    if [[ $targetName == "allen" ]]; then
        allen="sl384"
        tName="202202170855_60988202_VMSC01601"
        fileTemp="${fileTempP}${allen}_mesh${resTemp}_rz.vtk"
        fileTarg="${targDir}${tName}_mesh${resTarg}_rz_probRNA_t10parts.vtk"
        
        echo $(date) >> $sd/$atlasName${allen}_to_Allen${tName}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        python3 -c "import cleanAtlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/$atlasName${allen}_to_Allen${tName}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        echo $(date) >> $sd/$atlasName${allen}_to_Allen${tName}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt

        allen="sl484"
        tName="202202221440_60988212_VMSC01601"
        fileTemp="${fileTempP}${allen}_mesh${resTemp}_rz.vtk"
        fileTarg="${targDir}${tName}_mesh${resTarg}_rz_probRNA_t10parts.vtk"
        
        echo $(date) >> $sd/$atlasName${allen}_to_Allen${tName}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        python3 -c "import cleanAtlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/$atlasName${allen}_to_Allen${tName}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        echo $(date) >> $sd/$atlasName${allen}_to_Allen${tName}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt

        allen="sl584"
        tName="202203011708_60988222_VMSC01601"
        fileTemp="${fileTempP}${allen}_mesh${resTemp}_rz.vtk"
        fileTarg="${targDir}${tName}_mesh${resTarg}_rz_probRNA_t10parts.vtk"
        
        echo $(date) >> $sd/$atlasName${allen}_to_Allen${tName}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        python3 -c "import cleanAtlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/$atlasName${allen}_to_Allen${tName}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        echo $(date) >> $sd/$atlasName${allen}_to_Allen${tName}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    
    elif [[ $targetName == "merfish" ]]; then
        ## KATIE TO REVISE
        echo "Computing Gene Probabilities and Cell Densities"
        # do 1st type of Mesh
        allen="sl889"
        fileTemp="${fileTempP}${allen}_mesh${resTemp}_rz.vtk"
        merfish="S1R2"
        fileTarg="${targDir}MI7_gene_${merfish}_mesh${resTarg}avgRNAProbDist.vtk"
        echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        python3 -c "import cleanAtlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt

        merfish="S1R1"
        fileTarg="${targDir}MI7_gene_${merfish}_mesh${resTarg}avgRNAProbDist.vtk"
        echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        python3 -c "import cleanAtlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt

        allen="sl674"
        fileTemp="${tempDir}Allen_10_anno_16bit_ap_ax2_${allen}_mesh${res}_rz.vtk"
        merfish="S2R1"
        fileTarg="${targDir}MI7_gene_${merfish}_mesh${resTarg}avgRNAProbDist.vtk"
        echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        python3 -c "import cleanAtlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt

        merfish="S2R2"
        fileTarg="${targDir}MI7_gene_${merfish}_mesh${resTarg}avgRNAProbDist.vtk"
        echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        python3 -c "import cleanAtlasEstimation as ae; ae.solveThetaNonRigidNonProbability('$fileTemp','$fileTarg','$sd',diffeo=True,iters=20,sigmaKernel=$sigmaKernel,sigmaError=$sigmaError,sigmaIm=$sigmaIm,sigmaDist=$sigmaDist,newVer=1e-3,rigOnly=$rig,scaleStart=$ss,thresh=$thresh); quit()" >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
        echo $(date) >> $sd/$atlasName${allen}_to_Merfish${merfish}_params$sigmaKernel$sigmaIm$sigmaDist$sigmaError$ss$thresh.txt
    fi
    
    # do regular Mesh
    # do log Mesh
fi

