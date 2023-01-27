import ntpath
from numba import jit, prange, int64
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/master/py-lddmm/')
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/master/py-lddmm/base')
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
import vtkFunctions as vtf
import os
from base import loggingUtils
import multiprocessing as mp
from multiprocessing import Pool
import glob
import pandas as pd
import matplotlib
if 'DISPLAY' in os.environ:
    matplotlib.use('qt5Agg')
else:
    matplotlib.use("Agg")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
import cv2
import torch
import numpy.matlib
from pykeops.torch import Vi, Vj
from pykeops.torch import LazyTensor
from skimage.segmentation import watershed
from base.meshes import buildImageFromFullListHR, buildMeshFromCentersCounts, buildMeshFromImageData
from PIL import Image
Image.MAX_IMAGE_PIXELS
Image.MAX_IMAGE_PIXELS=1e10 # forget attack
import nibabel as nib
from qpsolvers import solve_qp
from base.kernelFunctions import Kernel, kernelMatrix
import logging
from base.curveExamples import Circle
from base.surfaces import Surface
from base.surfaceExamples import Sphere, Rectangle
from base import loggingUtils
from base.meshes import Mesh, buildMeshFromMerfishData
from base.affineRegistration import rigidRegistration, rigidRegistration_varifold
from base.meshMatching import MeshMatching, MeshMatchingParam
from base.mesh_distances import varifoldNormDef
import pykeops
pykeops.clean_pykeops()
plt.ion()

from datetime import datetime
import time

import scipy as sp
from scipy import sparse
from scipy.sparse import coo_array

#######################################################################
def collapseLabelsProb(fileMesh,newLabCoords):
    '''
    bin labels (feature indices) of fileMesh into newLabCoords such that newLabCoords[0] = list of inds to group as 0
    
    Ex: collapseLabelsProb('/cis/home/kstouff4/Documents/MeshRegistration/TestImages/MERFISH/cell_S1R1_mesh100.vtk',[[0,1,2,3,4,20,21,23,24,25,26,27,28,29,30,31],[5,6,10,11,12,13,14,15,16,17,18],[7,8,9,32]])
    # glial cells, neurons, other (e.g. endothelial cells, ependymal cells)
    '''
    meshO = Mesh(mesh=fileMesh)
    newNumLab = len(newLabCoords)
    imageNew = np.zeros((meshO.image.shape[0],newNumLab))
    for l in range(newNumLab):
        imageNew[:,l] = np.sum(meshO.image[:,newLabCoords[l]],axis=-1) # should already still sum to probability distribution
    ii = np.reciprocal(np.sum(imageNew,axis=-1),where=(np.sum(imageNew,axis=-1))!=0)
    meshO.updateImage(imageNew*ii[...,None])
    meshO.save(fileMesh.replace('.vtk','_newImage.vtk'))
    return
