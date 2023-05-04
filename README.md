# MeshLDDMMQP

Description: Repository for mesh-based alternating quadratic program and large deformation diffeomorphic metric mapping (LDDMM) optimization for mapping data sets across modalities and scales. Associated preprint: https://www.biorxiv.org/content/10.1101/2023.03.28.534622v1.

System Requirements:
1) Single GPU 
 - code developed and tested on 12 GB GPU with NVIDIA-SMI 440.33.01, Driver Version 440.33.01, CUDA Version 10.2
2) Python Packages
  - numpy
  - scipy
  - vtk
  - numba
  - pyfftw
  - matplotlib
  - pillow
  - scikit-image
  - nibabel
  - imageio
  - h5py
  - tqdm
  - pandas
  - setuptools
  - pykeops
  - pygalmesh
  - meshpy
  - qpsolvers
3) VTK viewer
  - input and output visualized in Paraview available from kitware (https://www.paraview.org) 
4) Repository Dependencies
  - py-lddmm (https://bitbucket.org/laurent_younes/py-lddmm/src/master/)
    * code tested and developed using version of py-lddmm cloned into this repository (master-KMS) 

Test Data (./TestImages):
* Template = mesh of Allen atlas image, coronal slice 485, at 100 micron resolution (Allen_10_anno_16bit_ap_ax2_sl484_mesh100_rz.vtk)
* Target = mesh of Allen Institute MERFISH, 20 selected spatially discriminating genes, coronal slice corresponding to Allen atlas slice 485, at 50 micron resolution (202202221440_60988212_VMSC01601_mesh50_rz_probRNA_t10parts.vtk)

Instructions for Use:
* ./bashScripts/runTest.sh
  - runs the alternating quadratic program and LDDMM optimization for the template and target mesh-based image varifolds contained in ./TestImages
  - parameters selected corresponding to those used in preprint to generate output seen in Figures 2 and 3
  - description of each parameter follows each parameter declaration in script
  - output files will be written to ./Results
  - estimated time to completion on standard (12 GB) GPU is 2 hours.
