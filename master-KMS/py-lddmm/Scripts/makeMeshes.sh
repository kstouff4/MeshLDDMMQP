#!/bin/bash

file1="/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1/202202170851_60988201_VMSC01001/detected_transcripts.csv"
ID="60988201"
savedir="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/Allen/"
#python3 -c "import preprocessMERFISHImage_KMS as ppf; ppf.processAllenImageMesh('$file1','$ID');quit()"

file1set="/cis/home/kstouff4/Documents/MeshRegistration/TestImages/Allen/60988223"
savePref="JeanFanSet"
#python3 -c "import preprocessMERFISHImage_KMS as ppf; ppf.processAllenImageMeshSubset(['Baiap2','Slc17a6','Adora2a','Gpr151','Gabbr2','Cpr6','Cckar','Malat1','Reln','Gfap','Map2','Mapt'],'${file1set}_geneList.npz','${file1set}_centers.csv','${file1set}_counts.csv','$savePref');quit()"

allenCSV='/cis/home/kstouff4/Documents/MeshRegistration/TestImages/Allen/60988223_transcriptPoints.npz'
python3 -c "import preprocessMERFISHImage_KMS as ppf; ppf.clusterAllenIntoCells('$file1', '$ID', '$savedir',numNeighbors=5);quit()"

file1="/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1/202203071321_60988228_VMSC01001/detected_transcripts.csv"
ID="60988228"

python3 -c "import preprocessMERFISHImage_KMS as ppf; ppf.clusterAllenIntoCells('$file1', '$ID', '$savedir',numNeighbors=5);quit()"

file1="/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1/202203111525_60988234_VMSC01601/detected_transcripts.csv"
ID="60988234"

python3 -c "import preprocessMERFISHImage_KMS as ppf; ppf.clusterAllenIntoCells('$file1', '$ID', '$savedir',numNeighbors=5);quit()"

