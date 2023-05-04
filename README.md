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
3) VTK viewer
  - input and output visualized in Paraview available from kitware (https://www.paraview.org) 
4) Repository Dependencies
  - py-lddmm (https://bitbucket.org/laurent_younes/py-lddmm/src/master/)
    * code tested and developed using version of py-lddmm cloned into this repository 

Test Data:

Instructions for Use:
