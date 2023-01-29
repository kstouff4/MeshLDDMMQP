import ntpath
from numba import jit, prange, int64
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/MeshLDDMMQP/master-KMS/py-lddmm/')
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/MeshLDDMMQP/master-KMS/py-lddmm/base')
import os
from base import loggingUtils
import multiprocessing as mp
from multiprocessing import Pool
from glob import glob
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
import socket
pykeops.set_build_folder("/cis/home/kstouff4/.cache/keops"+pykeops.__version__ + "_" + (socket.gethostname()))
from pykeops.numpy import Genred, LazyTensor

plt.ion()

from datetime import datetime
import time

import scipy as sp
from scipy import sparse
from scipy.sparse import coo_array
import torch
#torch.cuda.empty_cache()

##########################################################################
# General Functions for Estimating Transformations

    
##########################################################################
# Functions for Quadratic Programming Step

def getAandBnonRigid(ftemp,ftarg,Kdist,Alpha,cPrimeInc):
    '''
    Values that will be constant: alpha, gammaTarg, Zeta
    '''
    dtype='float32'
    # Compute A
    a1 = (ftemp.volumes * ftemp.weights)[:, None] * ftemp.image
    a2 = (ftarg.volumes * ftarg.weights)[:, None] * ftarg.image
    K1 = Kdist.applyK(ftemp.centers, a1)
    A_ = (K1[:, :, None] * a1[:, None, :]).sum(axis=0)
    K1 = Kdist.applyK(ftarg.centers, a2, firstVar=ftemp.centers)
    b_ = (K1[:, None, :] * a1[:,:,None]).sum(axis=0)
    
    print('A_', A_.shape, b_.shape)
    print(np.max(A_))
    print(np.max(b_))
    return(A_,b_)

def getAandBnonRigidKS(ftemp,ftarg,Kdist,Alpha,cPrimeInc):
    dtype='float32'
    gammaTemp = np.squeeze(ftemp.volumes)[...,None]
    GammaK = (gammaTemp@gammaTemp.T)*kernelMatrix(Kdist,ftemp.centers)
    c = Alpha * GammaK
    gammaTarg = np.squeeze(ftarg.volumes)[...,None]
    cPrime = cPrimeInc*(gammaTemp@gammaTarg.T)*kernelMatrix(Kdist,ftarg.centers,ftemp.centers)
    
    cT = LazyTensor(c.astype(dtype)[...,None,None])
    cPrimeT = LazyTensor(cPrime.astype(dtype)[...,None,None])
    
    xTemp_i = LazyTensor(ftemp.image.astype(dtype)[:,None,:,None]) #M x 1 x L x 1
    xTemp_j = LazyTensor(ftemp.image.astype(dtype)[None,:,None,:]) #1 x M x 1 x L
    xTarg_j = LazyTensor(ftarg.image.astype(dtype)[None,:,None,:]) # 1 x N x 1 x F
    print("shapes should be: M x 1 x L x 1 and 1 x M x 1 x L")
    print(xTemp_i.shape)
    print(xTemp_j.shape)
    print("shape should be: 1 x N x 1 x F")
    print(xTarg_j.shape)
    KTemp_ij = (cT*xTemp_i*xTemp_j).sum(dim=1)
    KTempTarg_ij = (cPrimeT*xTemp_i*xTarg_j).sum(dim=1)
    
    KTemp_ij = KTemp_ij[None,...].sum(axis=1)
    KTempTarg_ij = KTempTarg_ij[None,...].sum(axis=1)
    
    KTemp_ij = np.squeeze(KTemp_ij)
    KTempTarg_ij = np.squeeze(KTempTarg_ij)
    
    print("dimensions should be " + str(ftemp.imageDim) + " and " + str(ftarg.imageDim))
    print(KTemp_ij.shape)
    print(KTempTarg_ij.shape)
    
    A = KTemp_ij
    b = KTempTarg_ij
    print('A', A.shape, b.shape)
    print(np.max(A))
    print(np.max(b))
    return A,b
    
    

def getAandBprecompute(ftemp,ftarg,Kdist,cPrime):
    '''
    cPrime reflects the product of alpha and gammas for template to target
    '''
    b = np.zeros((ftemp.imageDim,ftarg.imageDim))
    KPrime = kernelMatrix(Kdist,ftarg.centers,ftemp.centers)
    for l0 in range(ftemp.imageDim):
        kk = np.squeeze(ftemp.image[...,l0])[...,None] 
        zetaTarg = ftarg.image[:,None,:]
        ZetaTarg = kk@zetaTarg.T # Feats x C x C'
        b[l0,:] = np.sum(cPrime*KPrime*ZetaTarg,axis=(1,2))
        
    return b

    
def precomputeAandcPrimeRigid(ftemp,ftarg,Kdist,Kim='identity'):
    '''
    alpha = weights (per face)
    gamma = volumes (per face)
    m = centers (per face)
    zeta = image (per face)
    
    kernelMatrix(Kpar, x, firstVar=None, grid=None, diff = False, diff2=False, constant_plane = False)
    
    Returns: A[l_0,l_1] and b[l_0, f]
    '''
    alphaTemp = np.squeeze(ftemp.weights)[...,None]
    gammaTemp = np.squeeze(ftemp.volumes)[...,None]
    alphaTarg = np.squeeze(ftarg.weights)[...,None]
    gammaTarg = np.squeeze(ftarg.volumes)[...,None]
        
    c = (alphaTemp*gammaTemp)@(alphaTemp.T*gammaTemp.T) 
    cPrime = (alphaTemp*gammaTemp)@(alphaTarg.T*gammaTarg.T)
    K = kernelMatrix(Kdist,ftemp.centers)
    
    c = c*K # constant which is same for every label
    A = np.zeros((ftemp.imageDim,ftemp.imageDim))
    for l0 in range(ftemp.imageDim):
        kk = np.squeeze(ftemp.image[...,l0])[...,None]
        zz = ftemp.image[:,None,:]
        Zeta = kk@zz.T
        A[l0,:] = np.sum(c*Zeta,axis=(1,2))
        
    return A,cPrime

def precomputeAlphaCprimeNonRigid(ftemp,ftarg):
    '''
    In non-rigid, the Kernel function is changing both for template and target as well as the volumes gamma.
    Consequently, only the alpha values and the alpha and gamma for the target are the same.
    
    Zeta = based on features is the same for the entire iteration set as it is based on original template features
    '''
    alphaTemp = np.squeeze(ftemp.weights)[...,None]
    alphaTarg = np.squeeze(ftarg.weights)[...,None]
    gammaTarg = np.squeeze(ftarg.volumes)[...,None]
    
    Alpha = alphaTemp@alphaTemp.T
    cPrimeInc = alphaTemp@(alphaTarg.T)
    
    return Alpha,cPrimeInc

###########################################################
# QP Solvers

# Python Solver; Estimate Pi_theta = theta where Pi_theta is a probability distribution 
def solveQPpiEqualsTheta(A,b,numLabels,numFeats,solver="osqp"):
    '''
    Possible solvers (dense): cvxopt, qpoases, quadprog
    Possible solvers (sparse): ecos, gurobi, mosek, osqp, qpswift, scs
    
    Assumes A is an L x L matrix such that A[r,c] = A_lr,lc
    Assumes b is an L x F vector such that b[r,c] = b_lr(f_c)
    
    Returns x = [theta_l0(f_0), theta_l0(f_2),....theta_l1(f_0), theta_l1(f_1).....theta_lL(f_F)]
    '''
    sz = numLabels*numFeats
    
    P = np.zeros((sz,sz))
    q = np.zeros((sz,1))
    lb = np.zeros(sz)
    G = np.eye(sz)*-1 # needs to be all greater than or equal to zero 
    Anew = np.zeros((numLabels,sz))
    Anewnew = np.zeros((sz,sz))
    bnewnew = np.zeros((numLabels,1))
    bnewnew[0:numLabels] = 1
    
    bnew = np.ones((numLabels,1))

    # make P
    blocks = []
    for w in range(numLabels):
        for v in range(numLabels):
            P[w*numFeats:(w+1)*numFeats,v*numFeats:(v+1)*numFeats] = np.eye(numFeats)*A[w,v]
    P = 2*P
    
    # make q
    for lab in range(numLabels):
        bb = b[lab,:]
        q[lab*numFeats:(lab+1)*numFeats] = -2*bb[...,None]
    
    # make Anew
    for r in range(numLabels):
        Anew[r,r*numFeats:(r+1)*numFeats] = 1 # to ensure probability distributions
    
    Anewnew[0:numLabels,:] = Anew

    '''
    # for debugging 
    fig,ax = plt.subplots()
    im = ax.imshow(P)
    fig.colorbar(im,ax=ax)
    fig.savefig('/cis/home/kstouff4/Documents/MeshRegistration/TestImages/AtlasEstimation/P.png',dpi=300)
    '''
    theta = solve_qp(P,np.squeeze(q),A=Anew,b=np.squeeze(bnewnew),G=G,h=lb,solver=solver)
    
    print("shape of theta out of solver is " + str(theta.shape))
    print(theta)
    
    return theta

# Python Solver; Estimate Pi_theta = Dirac at the average value of the features (e.g. \sum_f pi_theta \neq 1)
## This is used for estimating cell densities + probability distribution over cell types
def solveQPpiEqualsDirac(A,b,numLabels,numFeats,thresh,solver="osqp"):
    '''
    Possible solvers (dense): cvxopt, qpoases, quadprog
    Possible solvers (sparse): ecos, gurobi, mosek, osqp, qpswift, scs
    
    Assumes A is an L x L matrix such that A[r,c] = A_lr,lc
    Assumes b is an L x F vector such that b[r,c] = b_lr(f_c)
    
    Returns x = [theta_l0(f_0), theta_l0(f_2),....theta_l1(f_0), theta_l1(f_1).....theta_lL(f_F)]
    '''
    sz = numLabels*numFeats
    print("b shape is ") 
    print(b.shape)
    if (b.shape[0] == b.shape[-1]):
        b = b[...,None]
    
    P = np.zeros((sz,sz))
    q = np.zeros((sz,1))
    if (thresh == 0):
        lb = np.zeros(sz)
        G = np.eye(sz)*-1 # needs to be all greater than or equal to zero 
    else:
        # assume needs to be such that sum over features for label is greater than thresh
        lb = np.zeros(numLabels) - thresh
        G = np.zeros((numLabels,sz))
        for l in range(numLabels):
            G[l,l*numFeats:(l+1)*numFeats] = -1
    #lb = np.zeros(sz) - thresh     # estimate lb as 25th percentile of range of target  
    #G = np.eye(sz)*-1 

    # make P
    blocks = []
    for w in range(numLabels):
        for v in range(numLabels):
            P[w*numFeats:(w+1)*numFeats,v*numFeats:(v+1)*numFeats] = np.eye(numFeats)*A[w,v]
    P = 2*P
    
    # make q
    for lab in range(numLabels):
        bb = b[lab,:]
        q[lab*numFeats:(lab+1)*numFeats] = -2*bb[...,None]
        
    P = sp.sparse.csc_matrix(P)
    G = sp.sparse.csc_matrix(G)
    
    '''
    # for debugging
    fig,ax = plt.subplots()
    im = ax.imshow(P)
    fig.colorbar(im,ax=ax)
    fig.savefig('/cis/home/kstouff4/Documents/MeshRegistration/TestImages/AtlasEstimation/P.png',dpi=300)
    '''
    
    print("sizes of variables are ")
    print(P.shape)
    print(np.squeeze(q).shape)
    print(G.shape)
    print(lb.shape)
    
    theta = solve_qp(P,np.squeeze(q),G=G,h=lb,lb=np.zeros(sz),solver=solver,max_iter=100000,eps_abs=1e-5,verbose=True) #default is 1e-8?
    if (theta is None):
        theta = solve_qp(P,np.squeeze(q),G=G,h=lb,lb=np.zeros(sz),solver="ecos",verbose=True)
    print("shape of theta out of solver is " + str(theta.shape))
    print(theta)
    
    return theta

# LY Solver: iterative algorithm as described in section 6 of arxiv paper 
def solveQPLY():
    '''
        NOT IMPLEMENTED YET (12/6/22)
    '''
    return

###########################################################################
# Functions for Alternating Iteration between Estimating Geometric and Feature Transformations

# Rigid Transformation (rotation + translation) estimated only; not updated with rescaleUnits function yet
def solveThetaRigid(fileTemp,fileTarg,saveDir,iters=1,K2="Id",scaleTemp=1):
    '''
    fileTemp and fileTarg should give mesh's that were created at different resolutions (theoretically template is at a lower resolution than the target)
    
    Here, both are assumed to have discrete labeling
    '''
    totalCost = np.zeros((iters+1,1))
    ftempO = Mesh(mesh=fileTemp) # create mesh with original image
    ftarg = Mesh(mesh=fileTarg)
    ftemp = Mesh(mesh=fileTemp) # create mesh for registration
    
    numLabels = ftemp.imageDim
    numFeats = ftarg.imageDim
    
    # Q: Why is this necessary?
    #ftemp.updateWeights(1 + ftemp.weights*1000)
    #ftarg.updateWeights(1 + ftarg.weights*1000)
    #ftempO.updateWeights(1 + ftempO.weights*1000)
    
    baseTemp = fileTemp.split("/")[-1]
    targTemp = fileTarg.split("/")[-1]
    
    ftemp.updateVertices(ftemp.vertices*scaleTemp)
    ftempO.updateVertices(ftempO.vertices*scaleTemp)
    
    #scale = (ftemp.vertices.max(axis=0) - ftemp.vertices.min(axis=0)).max()
    scale = 1e-3 # passing from microns to mm
    
    sigmaKernel = .05
    sigmaDist = .05
    print("sigmaDist is " + str(sigmaDist))
    print("scale is " + str(scale))
    sigmaError = .01
    sigmaIm = 0.2
    Kdist = Kernel(name='gauss', sigma = sigmaDist)

    if (ftemp.centers.shape[1] == 2):
        R2 = np.eye(2)
        T2 = np.squeeze(np.zeros((2,1)))
        #scale = (ftemp.vertices.max(axis=0) - ftemp.vertices.min(axis=0)).max()
    else:
        R2 = np.eye(3)
        T2 = np.squeeze(np.zeros((3,1)))
        #scale = (ftemp.vertices.max(axis=0) - ftemp.vertices.min(axis=0)).max()
    
    # Q: why do we need to scale image??
    #scale = (ftemp.vertices.max(axis=0) - ftemp.vertices.min(axis=0)).max()
    scaleIm = 1 #(ftemp.image.max(axis=0) - ftemp.image.min(axis=0)).max()
    print("scaleIm is " + str(scaleIm))
    logging.info(f'scale = {scale:.2f}, scale image: {scaleIm:.2f}')
    ftemp.rescaleUnits(scale)
    ftarg.rescaleUnits(scale)
    ftempO.rescaleUnits(scale)
    print(f'weights: {ftemp.weights.sum():.0f} {ftarg.weights.sum():.0f}')

    ftemp.updateVertices(ftemp.vertices/scale)
    ftempO.updateVertices(ftempO.vertices/scale)
    ftarg.updateVertices(ftarg.vertices/scale)
    Kdist = Kernel(name='gauss', sigma=sigmaDist)
    
    vTempO = ftemp.vertices
    vTargO = ftarg.vertices
    
    A,cPrime = precomputeAandcPrimeRigid(ftempO,ftarg,Kdist)
    b = getAandBprecompute(ftempO,ftarg,Kdist,cPrime)
    if (K2 == "Id"):
        thetaVec = solveQPpiEqualsTheta(A,b,numLabels,numFeats) # size L x f
    elif (K2 == "Euc"):
        # only difference is lack of constraint of theta's adding up to 1 
        thetaVec = solveQPpiEqualsDirac(A,b,numLabels,numFeats) # size L x f
    np.savez(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_initialTheta.npz"), theta=thetaVec)
    theta = np.reshape(thetaVec,(numLabels,numFeats))
    ftemp.updateImage(ftempO.image@theta)
    sn = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_initialTheta.png")
    plotTheta(theta,numLabels,numFeats,sn)
    totalCost[0] = np.sum(A*(theta@theta.T)) - 2*np.sum(b*theta)
    print("thetaVec shape " + str(thetaVec.shape))
    print("theta shape " + str(theta.shape))
    
    for i in range(iters):
        # update the vertices to be original ones each time
        print("on iteration " + str(i) + " updating vertices")
        #print(time.time()-start)
        ftemp.updateVertices(vTempO)
        ftempO.updateVertices(vTempO)
        # Q: do rigid registration based on subset of vertices?
        J1 = np.random.choice(ftarg.centers.shape[0], min(2000, ftarg.centers.shape[0]), replace=False) 
        J0 = np.random.choice(ftemp.centers.shape[0], min(2001, ftemp.centers.shape[0]), replace=False)
        print("J1 is " + str(J1))
        print("J0 is " + str(J0))

        Alpha = ftemp.weights[J0,None]@ftarg.weights[J1,None].T
        Gamma = ftemp.volumes[J0,None]@ftarg.volumes[J1,None].T
        Zeta = ftemp.image[J0,:]@ftarg.image[J1,:].T
        K = Alpha*Gamma*Zeta
        print("kernel shape " + str(K.shape))
        print(np.unique(K))
        K = 1 + K # add 1 to give more weight to geometry; ends up beign a constant in QP part 
        #weights = ((ftarg.image[J1, None, :] - ftemp.image[None, J0, :])**2)
        print("before rigid registration calculation ")
        #print(time.time()-start)
        R2, T2 = rigidRegistration_varifold((ftemp.centers[J0, :],ftarg.centers[J1, :]), weights=K, ninit=4,
                                        sigma = sigmaDist) # finds matrix 
        print("after rigid registration calculation")
                
        # if ftemp is first 
        ftemp.updateVertices(np.dot(ftemp.vertices, R2.T) + T2)
        ftempO.updateVertices(np.dot(ftempO.vertices, R2.T) + T2)
        
        #A,cPrime = precomputeAandcPrimeRigid(ftemp,ftarg,Kdist)
        b = getAandBprecompute(ftempO,ftarg,Kdist,cPrime)
        if (K2 == "Id"):
            thetaVec = solveQPpiEqualsTheta(A,b,numLabels,numFeats) # size L x f
        elif (K2 == "Euc"):
            # only difference is lack of constraint of theta's adding up to 1 
            thetaVec = solveQPpiEqualsDirac(A,b,numLabels,numFeats) # size L x f
        #thetaVec = solveQPpiEqualsTheta(A,b,numLabels,numFeats) # size L x f
        theta = np.reshape(thetaVec,(numLabels,numFeats)) # pi_theta = dirac on this OR pi_theta = this vector
        ftemp.updateImage(ftempO.image@theta)
        #print("end of iteration " + str(i))
        #print(time.time()-start)
 
        totalCost[i+1] = np.sum(A*(theta@theta.T)) - 2*np.sum(b*theta) # assume no cost for rigid
    
    print("scale")
    print(scale)
    print("R2")
    print(R2)
    print("T2")
    print(T2)
    sn = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_finalTheta" + str(iters) + ".png")
    plotTheta(theta,numLabels,numFeats,sn)
    ftempO.save(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_geomDeform_" + str(iters) + ".vtk"))
    ftemp.save(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_geomDeform+featureMapped" + str(iters) + ".vtk"))
    ftarg.save(saveDir + targTemp.replace(".vtk","_rescaled.vtk"))
    np.savez(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_" + str(iters) + ".npz"), theta=theta,R2=R2,T2=T2,scale=scale)
    fig,ax = plt.subplots()
    ax.plot(totalCost)
    ax.set_xlabel('iters')
    ax.set_ylabel('theta cost')
    fig.savefig(sn.replace("Theta.png","Cost.png"),dpi=300)
    print("total Cost is " + str(totalCost))
    print("theta is " + str(theta))
    print("A is " + str(A))
    print("b is " + str(b))
    sn = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_finalTheta.png")
    plotTheta(theta,numLabels,numFeats,sn)
    
    return 

# Estima
def solveThetaNonRigid(fileTemp,fileTarg,saveDir,diffeo=False,iters=1,K2="Id",sigmaKernel=0.05,sigmaError=0.01,sigmaIm=0.2,sigmaDist=0.05,newVer=False):
    '''
    fileTemp and fileTarg should give mesh's that were created at different resolutions (theoretically template is at a lower resolution than the target)
    
    Here, both are assumed to have discrete labeling
    '''
    saveTemp = fileTemp.split('/')[-1]
    saveTemp = saveTemp.split('.')[0]
    saveTarg = fileTarg.split('/')[-1]
    saveTarg = saveTarg.split('.')[0]
    
    totalCost = np.zeros((iters+1,1))
    ftempO = Mesh(mesh=fileTemp) # create mesh with original image
    ftarg = Mesh(mesh=fileTarg)
    ftemp = Mesh(mesh=fileTemp) # create mesh for registration
    
    numLabels = ftemp.imageDim
    numFeats = ftarg.imageDim
    
    baseTemp = fileTemp.split("/")[-1]
    targTemp = fileTarg.split("/")[-1]
    
    if (newVer):
        scale = 1e-3 # rescale microns to mm
        ftemp.rescaleUnits(scale)
        ftarg.rescaleUnits(scale)
        ftempO.rescaleUnits(scale)
        print(f'weights: {ftemp.weights.sum():.0f} {ftarg.weights.sum():.0f}')
        
        
    else:
        # Q: Why is this necessary?
        ftemp.updateWeights(1.0*(ftemp.weights > 0))
        ftarg.updateWeights(1.0*(ftarg.weights > 0))
        ftempO.updateWeights(1.0 * (ftempO.weights > 0))
        scale = (ftemp.vertices.max(axis=0) - ftemp.vertices.min(axis=0)).max()
        ftemp.updateVertices(ftemp.vertices/scale)
        ftempO.updateVertices(ftempO.vertices/scale)
        ftarg.updateVertices(ftarg.vertices/scale)

    
    if (ftemp.centers.shape[1] == 2):
        R2 = np.eye(2)
        T2 = np.squeeze(np.zeros((2,1)))
        #scale = (ftemp.vertices.max(axis=0) - ftemp.vertices.min(axis=0)).max()
    else:
        R2 = np.eye(3)
        T2 = np.squeeze(np.zeros((3,1)))
        #scale = (ftemp.vertices.max(axis=0) - ftemp.vertices.min(axis=0)).max()
    
    # Katie Added: Center both images around 0
    ftemp.updateVertices(ftemp.vertices-np.mean(ftemp.vertices))
    ftempO.updateVertices(ftempO.vertices-np.mean(ftempO.vertices))
    ftarg.updateVertices(ftarg.vertices-np.mean(ftarg.vertices))
    
    # now rescale weights to be between 0 to 1 indicating is a measure of relative mass per volume compared with max
    ftemp_minMass = np.min(ftemp.weights)
    ftemp_maxMass = np.max(ftemp.weights)
    ftarg_minMass = np.min(ftarg.weights)
    ftarg_maxMass = np.max(ftarg.weights)
    
    '''
    ftemp.updateWeights((ftemp.weights)/(ftemp_maxMass))
    ftempO.updateWeights(ftemp.weights)
    ftarg.updateWeights((ftarg.weights)/(ftarg_maxMass))
    '''
    # Scaling to 1 along one axis for interpretability
    #scale = (ftemp.vertices.max(axis=0) - ftemp.vertices.min(axis=0)).max()
    scaleIm = 1 #+(ftemp.image.max(axis=0) - ftemp.image.min(axis=0)).max()
    print("scaleIm is " + str(scaleIm))
    logging.info(f'scale = {scale:.2f}, scale image: {scaleIm:.2f}')
    
    Kdist = Kernel(name='gauss', sigma=sigmaDist)
    
    vTempO = ftemp.vertices
    vTargO = ftarg.vertices
    
    Alpha,cPrimeInc = precomputeAlphaCprimeNonRigid(ftempO,ftarg)
    A,b = getAandBnonRigid(ftemp,ftarg,Kdist,Alpha,cPrimeInc)
    if (K2 == "Id"):
        thetaVec = solveQPpiEqualsTheta(A,b,numLabels,numFeats) # size L x f
    elif (K2 == "Euc"):
        # only difference is lack of constraint of theta's adding up to 1 
        thetaVec = solveQPpiEqualsDirac(A,b,numLabels,numFeats) # size L x f
    np.savez(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_initialTheta.npz"), theta=thetaVec)
    theta = np.reshape(thetaVec,(numLabels,numFeats))
    ftemp.updateImage(ftempO.image@theta)
    sn = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_initialTheta.png")
    plotTheta(theta,numLabels,numFeats,sn)
    totalCost[0] = np.sum(A*(theta@theta.T)) - 2*np.sum(b*theta)
    print("thetaVec shape " + str(thetaVec.shape))
    print("theta shape " + str(theta.shape))
    
    sn = saveDir + baseTemp.replace(".vtk","_rescaled.vtk")
    ftempO.save(sn)

    
    for i in range(iters):
        # update the vertices to be original ones each time
        print("on iteration " + str(i) + " updating vertices")
        #print(time.time()-start)
        ftemp.updateVertices(vTempO)
        ftempO.updateVertices(vTempO)
        
        # try just estimating rigid and scale on first iteration 
        if (i < 3):
            # Q: do rigid registration based on subset of vertices?
            J1 = np.random.choice(ftarg.centers.shape[0], min(2000, ftarg.centers.shape[0]), replace=False) 
            J0 = np.random.choice(ftemp.centers.shape[0], min(2001, ftemp.centers.shape[0]), replace=False)
            print("J1 is " + str(J1))
            print("J0 is " + str(J0))

            AlphaSub = ftemp.weights[J0,None]@ftarg.weights[J1,None].T
            GammaSub = ftemp.volumes[J0,None]@ftarg.volumes[J1,None].T
            ZetaSub = ftemp.image[J0,:]@ftarg.image[J1,:].T
            K = AlphaSub*GammaSub*ZetaSub
            print("K for rigid before add 1 is ")
            print(K)
            K = 1 + K # add 1 to give more weight to geometry; ends up beign a constant in QP part 
            #weights = ((ftarg.image[J1, None, :] - ftemp.image[None, J0, :])**2)
            print("before rigid registration calculation ")
            #print(time.time()-start)
            R2, T2 = rigidRegistration_varifold((ftemp.centers[J0, :],ftarg.centers[J1, :]), weights=K, ninit=4,
                                            sigma = sigmaDist) # finds matrix 
            print("after rigid registration calculation")
            print(R2)
            print(T2)

            # if ftemp is first 
            ftemp.updateVertices(np.dot(ftemp.vertices, R2.T) + T2)
            ftempO.updateVertices(np.dot(ftempO.vertices, R2.T) + T2)
            
            sn = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_rigid.vtk")
            snF = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_rigidFeats.vtk")

            ftempO.save(sn)
            ftemp.save(snF)
        
            print("before scale estimate ")
            # estimate scale as best out of trying 10 (from 0.5 to 1.5)
            KparIm=Kernel(name='euclidean', sigma=sigmaIm)
            scaleEstimate = getObjScale(ftemp,ftarg,Kdist,K)
            #scaleEstimate2 = getScaleTrial(ftemp,ftarg,Kdist,KparIm)
            print("scale 1 is " + str(scaleEstimate))
            #print("scale 2 is " + str(scaleEstimate2))
            ftemp.updateVertices(scaleEstimate*ftemp.vertices)
            ftempO.updateVertices(scaleEstimate*ftempO.vertices)
            print("after scale estimate")
            
            sn = sn.replace(".vtk","Scale.vtk")
            snF = snF.replace("_rigidFeats.vtk","_rigidScaleFeats.vtk")
            ftempO.save(sn)
            ftemp.save(snF)
            
        if (diffeo and i > 1):
            '''
            Estimate diffeomorphism after estimating rigid + scale
            '''
            # update Vertices after estimation
            K1 = Kernel(name='laplacian', sigma = sigmaKernel, order=3)
            pk_type = 'float32'
            affineSt = 'none'

            sm = MeshMatchingParam(timeStep=0.1, algorithm='bfgs', KparDiff=K1, KparDist=('gauss', sigmaDist),
                       KparIm=('euclidean', sigmaIm), sigmaError=sigmaError)
            sm.KparDiff.pk_dtype = pk_type
            sm.KparDist.pk_dtype = pk_type
            sm.KparIm.pk_type = pk_type
            f = MeshMatching(Template=ftemp, Target=ftarg, outputDir='/cis/home/kstouff4/Documents/MeshRegistration/TestImages/AtlasEstimation/'+saveTemp+'_'+saveTarg+'affine' + affineSt + '/',param=sm,
                    testGradient=False,  maxIter=200,
                 affine=affineSt, rotWeight=.01, transWeight = 100.,
                    scaleWeight=10., affineWeight=10.)

            f.optimizeMatching()
            newV = f.fvDef.vertices
            scaleEstimated = f.Afft
            ftemp.updateVertices(newV)
            ftempO.updateVertices(newV)
            #plt.ioff()
            #plt.show()
        
        A,b = getAandBnonRigid(ftempO,ftarg,Kdist,Alpha,cPrimeInc) # ftempO because need to have original feature set
        if (K2 == "Id"):
            thetaVec = solveQPpiEqualsTheta(A,b,numLabels,numFeats) # size L x f
        elif (K2 == "Euc"):
            # only difference is lack of constraint of theta's adding up to 1 
            thetaVec = solveQPpiEqualsDirac(A,b,numLabels,numFeats) # size L x f
        #thetaVec = solveQPpiEqualsTheta(A,b,numLabels,numFeats) # size L x f
        theta = np.reshape(thetaVec,(numLabels,numFeats)) # pi_theta = dirac on this OR pi_theta = this vector
        ftemp.updateImage(ftempO.image@theta)
 
        totalCost[i+1] = np.sum(A*(theta@theta.T)) - 2*np.sum(b*theta) # assume no cost for rigid
    
    print("scale Estimated")
    print(scaleEstimated)
    print("R2")
    print(R2)
    print("T2")
    print(T2)
    sn = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_finalTheta" + str(iters) + ".png")
    plotTheta(theta,numLabels,numFeats,sn)
    
    # return weights to original 
    ftempO.updateWeights(ftempO.weights*ftemp_maxMass)
    ftemp.updateWeights(ftemp.weights*ftemp_maxMass)
    ftarg.updateWeights(ftarg.weights*ftarg_maxMass)
    
    ftempO.save(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_geomDeform_" + str(iters) + ".vtk"))
    ftemp.save(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_geomDeform+featureMapped" + str(iters) + ".vtk"))
    ftarg.save(saveDir + targTemp.replace(".vtk","_rescaled.vtk"))
    np.savez(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_" + str(iters) + ".npz"), theta=theta,R2=R2,T2=T2,scale=scale)
    fig,ax = plt.subplots()
    ax.plot(totalCost)
    ax.set_xlabel('iters')
    ax.set_ylabel('theta cost')
    fig.savefig(sn.replace("Theta.png","Cost.png"),dpi=300)
    print("total Cost is " + str(totalCost))
    print("theta is " + str(theta))
    print("A is " + str(A))
    print("b is " + str(b))
    sn = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_finalTheta.png")
    plotTheta(theta,numLabels,numFeats,sn)
    
    return theta

# Estimate Scale 1x Based on Ratio of Volumes (scaleStart < 0 with abs(scaleStart) indicating percentage of estimated scale to start with)
# Estimate Rotation + Translation for first rigOnly iterations
# Estimate Diffeomorphism alternately with Pi_Theta for rigOnly to iters 
## Pi_theta estimated using Dirac implementation with assumption that dirac reflects weights*probability distribution over features; target weights used and template weights assumed not to exist 
def solveThetaNonRigidNonProbability(fileTemp,fileTarg,saveDir,diffeo=False,iters=1,sigmaKernel=0.05,sigmaError=0.01,sigmaIm=0.2,sigmaDist=0.05,newVer=1e-3,rigOnly=25,scaleStart=0.5,thresh=0):
    '''
    fileTemp and fileTarg should give mesh's that were created at different resolutions (theoretically template is at a lower resolution than the target)
    
    Here, both are assumed to have discrete labeling
    
    The atlas is assumed to be constant density (with the exception of boundaries)
    
    newVer = what to scale vertex coordinates by
    rigOnly = number of iterations to do just compute rigid orientation with 
    scaleStart = what to start scale with or estimate scale as in terms of percent of expected difference in volume (< 0 indicates to estimate scale by volume differences)
    '''
    saveTemp = fileTemp.split('/')[-1]
    saveTemp = saveTemp.split('.')[0]
    saveTarg = fileTarg.split('/')[-1]
    saveTarg = saveTarg.split('.')[0]
    
    totalCost = np.zeros((iters+1,1))
    ftempO = Mesh(mesh=fileTemp) # create mesh with original image
    ftarg = Mesh(mesh=fileTarg)
    ftemp = Mesh(mesh=fileTemp) # create mesh for registration
    
    # if 1 feature, put it all into alpha weight
    if (ftarg.image.shape[-1] == ftarg.image.shape[0] or ftarg.imageDim == 1):
        print("weights shape")
        print(ftarg.weights.shape)
        iNew = np.zeros((ftarg.image.shape[0],2))
        iNew[:,0] = 0.5
        iNew[:,1] = 0.5 
        ftarg.updateWeights(ftarg.weights*np.squeeze(ftarg.image))
        ftarg.updateImage(iNew) # if only have 1 feature
        print("ftarg image shape")
        print(ftarg.image.shape)
        print(ftarg.imageDim)
        print("ftarg weights shape")
        print(ftarg.weights.shape)
    
    numLabels = ftemp.imageDim
    numFeats = ftarg.imageDim
    
    baseTemp = fileTemp.split("/")[-1]
    targTemp = fileTarg.split("/")[-1]
    
    if (newVer is not None):
        scale = newVer # rescale microns to mm
        ftemp.rescaleUnits(scale)
        ftarg.rescaleUnits(scale)
        ftempO.rescaleUnits(scale)
        print(f'weights: {ftemp.weights.sum():.0f} {ftarg.weights.sum():.0f}')
        
    else:
        # Q: Why is this necessary?
        ftemp.updateWeights(1.0*(ftemp.weights > 0))
        ftarg.updateWeights(1.0*(ftarg.weights > 0))
        ftempO.updateWeights(1.0 * (ftempO.weights > 0))
        scale = (ftemp.vertices.max(axis=0) - ftemp.vertices.min(axis=0)).max()
        ftemp.updateVertices(ftemp.vertices/scale)
        ftempO.updateVertices(ftempO.vertices/scale)
        ftarg.updateVertices(ftarg.vertices/scale)

    
    if (ftemp.centers.shape[1] == 2):
        R2 = np.eye(2)
        T2 = np.squeeze(np.zeros((2,1)))
    else:
        R2 = np.eye(3)
        T2 = np.squeeze(np.zeros((3,1)))
    
    # Katie Added: Center both images around 0
    ftemp.updateVertices(ftemp.vertices-np.mean(ftemp.vertices))
    ftempO.updateVertices(ftempO.vertices-np.mean(ftempO.vertices))
    ftarg.updateVertices(ftarg.vertices-np.mean(ftarg.vertices))
    
        
    sn = saveDir + baseTemp.replace(".vtk","_rescaled.vtk")
    ftempO.save(sn)
    ftarg.save(saveDir + targTemp.replace(".vtk","_rescaled.vtk"))
    
    ftempO.updateJacobianFactor(ftempO.volumes)
    ftemp.updateJacobianFactor(ftemp.volumes)
    #tempVertRescale = np.copy(ftemp.vertices)
    
    # scale vertices of template according to scaleStart (just for optimization purposes)
    # Assume scaleStart is slightly less than where you need to go so that template doesn't set weights to equal zero 
    if (scaleStart < 0):
        scaleMat = np.eye(2)
        maxTarg = np.max(ftarg.vertices,axis=0)
        minTarg = np.min(ftarg.vertices,axis=0)
        maxTemp = np.max(ftemp.vertices,axis=0)
        minTemp = np.min(ftemp.vertices,axis=0)
        # estimate isotropic scale
        #volTarg=(maxTarg[0]-minTarg[0])*(maxTarg[1]-minTarg[1])
        #volTemp=(maxTemp[0]-minTemp[0])*(maxTemp[1]-minTemp[1])
        volTarg = np.sum(ftarg.volumes)
        volTemp = np.sum(ftemp.volumes)
        scaleMat = scaleMat*np.sqrt(volTarg/volTemp)
        scaleMat = scaleMat*np.abs(scaleStart)
        
        print("scaleMat is ")
        print(scaleMat)
        ftemp.updateVertices(ftemp.vertices@scaleMat)
        ftempO.updateVertices(ftempO.vertices@scaleMat)
    else:
        ftemp.updateVertices(ftemp.vertices*scaleStart)
        ftempO.updateVertices(ftempO.vertices*scaleStart)
        
    # rescale weights to represent density on 0 to 1 scale (representative of density of tissue)
    ftemp_minMass = np.min(ftemp.weights)
    ftemp_maxMass = np.max(ftemp.weights)
    ftemp.updateWeights((ftemp.weights)/(ftemp_maxMass))
    ftempO.updateWeights(ftemp.weights)
    
    # 12/4/22 --> keep ftempO as all ones so don't have impact 
    ftempO.updateWeights(np.zeros_like(ftemp.weights)+1)
    
    # Scaling to 1 along one axis for interpretability
    #scale = (ftemp.vertices.max(axis=0) - ftemp.vertices.min(axis=0)).max()
    scaleIm = 1 #+(ftemp.image.max(axis=0) - ftemp.image.min(axis=0)).max()
    print("scaleIm is " + str(scaleIm))
    logging.info(f'scale = {scale:.2f}, scale image: {scaleIm:.2f}')
    
    Kdist = Kernel(name='gauss', sigma=sigmaDist)
    
    vTempO = np.copy(ftemp.vertices)
    vTargO = np.copy(ftarg.vertices)
    
    # KATIE: need to change this portion because nothing can be precomputed since alpha's in atlas are changing??
    # NEVERMIND, alphas are not changing 
    Alpha,cPrimeInc = precomputeAlphaCprimeNonRigid(ftempO,ftarg) 
    A,b = getAandBnonRigid(ftemp,ftarg,Kdist,Alpha,cPrimeInc)
    Ao,bo = getAandBnonRigidKS(ftemp,ftarg,Kdist,Alpha,cPrimeInc)
    
    print("sum of weights after initialization")
    print(np.sum(ftemp.weights))
    print(np.sum(ftarg.weights))
    
    # Initialize weights as mean of target weights
    # initialize pi_theta as uniform over target regions for each atlas region 
    newInitial = np.zeros((ftempO.image.shape[0],numFeats))
    newInitial = newInitial + 1.0/numFeats # uniform distribution for each cell type
    newWeights = np.mean(ftarg.weights) # equal weights per volume based on target (assume each triangle has avg density)
    # alternative = np.sum(ftarg.weights*ftarg.volumes)/np.sum(ftarg.volumes) = total cells / total vol
    ftemp.updateImage(newInitial)
    ftemp.updateWeights(np.zeros_like(ftemp.weights)+newWeights)
    
    print("sum of weights after")
    print(np.sum(ftemp.weights))
    
    sn = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_initialTheta.png")
    #plotTheta(theta,numLabels,numFeats,sn)
    #totalCost[0] = np.sum(A*(theta@theta.T)) - 2*np.sum(b*theta)
    
    # Option for lower bounding estimated weights
    # get threshold based on percentile
    if (thresh == 0):
        threshCompute = 0
    else:
        print("thresh being used is " + str(thresh))
        threshCompute = np.quantile(ftarg.weights,thresh) # make threshold based on 
        print("thresh compute used is " + str(threshCompute))
        print("min and max are " + str(np.min(ftarg.weights)) + ", " + str(np.max(ftarg.weights)))

    J1 = np.random.choice(ftarg.centers.shape[0], min(2000, ftarg.centers.shape[0]), replace=False) 
    J0 = np.random.choice(ftemp.centers.shape[0], min(2001, ftemp.centers.shape[0]), replace=False)
    for i in range(iters):
        # update the vertices to be original ones each time
        # KATIE: maybe try to eliminate this and see if can do better but might get stuck (maybe do this for first few?)
        print("on iteration " + str(i) + " updating vertices")
        #print(time.time()-start)
        ftemp.updateVertices(vTempO)
        ftempO.updateVertices(vTempO)
        
        # try just estimating rigid and scale on first iteration --> doesn't work since reset vertices to initial
        if (i < rigOnly):
            # Q: do rigid registration based on subset of vertices?

            AlphaSub = ftemp.weights[J0,None]@ftarg.weights[J1,None].T
            GammaSub = ftemp.volumes[J0,None]@ftarg.volumes[J1,None].T
            ZetaSub = ftemp.image[J0,:]@ftarg.image[J1,:].T
            K = AlphaSub*GammaSub*ZetaSub
            print("Alpha, Gamma, or Zeta dominating")
            print(AlphaSub)
            print(GammaSub)
            print(ZetaSub)
            #K = 1 + K # add 1 to give more weight to geometry; ends up beign a constant in QP part 
            print("K used for rigid registration")
            print(K)
            R2, T2 = rigidRegistration_varifold((ftemp.centers[J0, :],ftarg.centers[J1, :]), weights=K, ninit=4,
                                            sigma = sigmaDist) # finds matrix 
            print("after rigid registration calculation")
            print(R2)
            print(T2)

            # if ftemp is first 
            ftemp.updateVertices(np.dot(ftemp.vertices, R2.T) + T2)
            ftempO.updateVertices(np.dot(ftempO.vertices, R2.T) + T2)
            
            sn = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_rigid.vtk")
            snF = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_rigidFeats.vtk")

            ftempO.save(sn)
            ftemp.save(snF)
        
            print("before scale estimate ")
            # estimate scale as best out of trying 10 (from 0.5 to 1.5)
            KparIm=Kernel(name='euclidean', sigma=sigmaIm)
            if (scaleStart > 0):
                scaleEstimate = getObjScale(ftemp,ftarg,Kdist,K)
                print("scale 1 is " + str(scaleEstimate))
                print("scale actual is " + str(scaleEstimate*scaleStart))
                ftemp.updateVertices(scaleEstimate*ftemp.vertices)
                ftempO.updateVertices(scaleEstimate*ftempO.vertices)
                print("after scale estimate")

                sn = sn.replace(".vtk","Scale.vtk")
                snF = snF.replace("_rigidFeats.vtk","_rigidScaleFeats.vtk")
                ftempO.save(sn)
                ftemp.save(snF)
            
            if (i == rigOnly-1):
                vTempO = ftempO.vertices
            
        if (diffeo and i > rigOnly-1):
            '''
            Estimate diffeomorphism after estimating rigid + scale
            '''
            # update Vertices after estimation
            if i == rigOnly:
                K1 = Kernel(name='laplacian', sigma = sigmaKernel, order=3)
                pk_type = 'float32'
                affineSt = 'none'

                sm = MeshMatchingParam(timeStep=0.1, algorithm='bfgs', KparDiff=K1, KparDist=('gauss', sigmaDist),
                       KparIm=('euclidean', sigmaIm), sigmaError=sigmaError)
                sm.KparDiff.pk_dtype = pk_type
                sm.KparDist.pk_dtype = pk_type
                sm.KparIm.pk_type = pk_type
                f = MeshMatching(Template=ftemp, Target=ftarg, outputDir='/cis/home/kstouff4/Documents/MeshRegistration/TestImages/AtlasEstimation/'+saveTemp+'_'+saveTarg+'affine' + affineSt + '/',param=sm,
                        testGradient=False,  maxIter=200,
                     affine=affineSt, rotWeight=.01, transWeight = 100.,
                        scaleWeight=10., affineWeight=10.)
            else:
                print('restarting matching')
                f.fv0.updateImage(ftemp.image)
                f.fv0.updateWeights(ftemp.weights)
                f.fvDef.updateImage(ftemp.image)
                f.fvDef.updateWeights(ftemp.weights)
                f.reset = True

            f.optimizeMatching()
            newV = f.fvDef.vertices
            scaleEstimated = f.Afft
            ftemp.updateVertices(newV)
            ftempO.updateVertices(newV)
        
        A,b = getAandBnonRigid(ftempO,ftarg,Kdist,Alpha,cPrimeInc) # ftempO because need to have original feature set
        Ao,bo = getAandBnonRigidKS(ftempO,ftarg,Kdist,Alpha,cPrimeInc)
        
        thetaVec = solveQPpiEqualsDirac(A,b,numLabels,numFeats,threshCompute) # size L x f
        theta = np.reshape(thetaVec,(numLabels,numFeats))
        convImage = ftempO.image@theta # number of cells x number of features
        if (convImage.shape[-1] == convImage.shape[0]):
            convImage = convImage[...,None]
        convWeights = np.sum(convImage,axis=-1)
        ii = np.reciprocal(convWeights,where=convWeights!=0)
        ftemp.updateImage(convImage*ii[...,None])
        #ftemp.updateImage(convImage/convWeights[...,None])
        ftemp.updateWeights(ftempO.weights*convWeights)
 
        totalCost[i+1] = np.sum(A*(theta@theta.T)) - 2*np.sum(b*theta) # assume no cost for rigid
    
    #print(np.linalg.det(scaleEstimated))
    print("R2")
    print(R2)
    print("T2")
    print(T2)
    sn = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_finalTheta" + str(iters) + ".png")
    plotTheta(theta,numLabels,numFeats,sn)
    
    ftempO.save(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_geomDeform_" + str(iters) + ".vtk"))
    ftemp.save(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_geomDeform+featureMapped" + str(iters) + ".vtk"))
    np.savez(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_" + str(iters) + ".npz"), theta=theta,R2=R2,T2=T2,scale=scale)
    fig,ax = plt.subplots()
    ax.plot(totalCost)
    ax.set_xlabel('iters')
    ax.set_ylabel('theta cost')
    fig.savefig(sn.replace("Theta.png","Cost.png"),dpi=300)
    print("total Cost is " + str(totalCost))
    print("theta is " + str(theta))
    print("A is " + str(A))
    print("b is " + str(b))
    sn = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_finalTheta.png")
    plotTheta(theta,numLabels,numFeats,sn)
    
    # save atlas undeformed with just features 
    ftempFeat = Mesh(mesh=fileTemp) # create mesh for registration
    ftempFeat.updateWeights(ftemp.weights)
    ftempFeat.updateImage(ftemp.image)
    ftempFeat.updateVertices(vTempO)
    ftempFeat.save(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_featureMapped" + str(iters) + ".vtk"))
    
    return theta

def solveAtlastoAtlas(fileTemp,fileTarg,saveDir,diffeo=False,iters=1,sigmaKernel=0.05,sigmaError=0.01,sigmaIm=0.2,sigmaDist=0.05,newVer=1e-3,rigOnly=25,scaleStart=0.5,thresh=0):
    '''
    fileTemp and fileTarg should give mesh's that were created at different resolutions (theoretically template is at a lower resolution than the target)
    
    Here, both are assumed to have discrete labeling
    
    The atlas is assumed to be constant density (with the exception of boundaries)
    
    newVer = what to scale vertex coordinates by
    rigOnly = number of iterations to do just compute rigid orientation with 
    scaleStart = what to start scale with or estimate scale as in terms of percent of expected difference in volume (< 0 indicates to estimate scale by volume differences)
    '''
    saveTemp = fileTemp.split('/')[-1]
    saveTemp = saveTemp.split('.')[0]
    saveTarg = fileTarg.split('/')[-1]
    saveTarg = saveTarg.split('.')[0]
    
    totalCost = np.zeros((iters+1,1))
    ftempO = Mesh(mesh=fileTemp) # create mesh with original image
    ftarg = Mesh(mesh=fileTarg)
    ftemp = Mesh(mesh=fileTemp) # create mesh for registration
    
    numLabels = ftemp.imageDim
    numFeats = ftarg.imageDim
    
    baseTemp = fileTemp.split("/")[-1]
    targTemp = fileTarg.split("/")[-1]
    
    if (newVer is not None):
        scale = newVer # rescale microns to mm
        ftemp.rescaleUnits(scale)
        ftarg.rescaleUnits(scale)
        ftempO.rescaleUnits(scale)
        print(f'weights: {ftemp.weights.sum():.0f} {ftarg.weights.sum():.0f}')
        
    else:
        # Q: Why is this necessary?
        ftemp.updateWeights(1.0*(ftemp.weights > 0))
        ftarg.updateWeights(1.0*(ftarg.weights > 0))
        ftempO.updateWeights(1.0 * (ftempO.weights > 0))
        scale = (ftemp.vertices.max(axis=0) - ftemp.vertices.min(axis=0)).max()
        ftemp.updateVertices(ftemp.vertices/scale)
        ftempO.updateVertices(ftempO.vertices/scale)
        ftarg.updateVertices(ftarg.vertices/scale)

    
    if (ftemp.centers.shape[1] == 2):
        R2 = np.eye(2)
        T2 = np.squeeze(np.zeros((2,1)))
    else:
        R2 = np.eye(3)
        T2 = np.squeeze(np.zeros((3,1)))
    
    # Katie Added: Center both images around 0
    ftemp.updateVertices(ftemp.vertices-np.mean(ftemp.vertices))
    ftempO.updateVertices(ftempO.vertices-np.mean(ftempO.vertices))
    ftarg.updateVertices(ftarg.vertices-np.mean(ftarg.vertices))
    
        
    sn = saveDir + baseTemp.replace(".vtk","_rescaled.vtk")
    ftempO.save(sn)
    ftarg.save(saveDir + targTemp.replace(".vtk","_rescaled.vtk"))
    
    ftemp.updateJacobianFactor(ftempO.volumes)
    ftempO.updateJacobianFactor(ftempO.volumes)
    
    # scale vertices of template according to scaleStart (just for optimization purposes)
    # Assume scaleStart is slightly less than where you need to go so that template doesn't set weights to equal zero 
    if (scaleStart < 0):
        scaleMat = np.eye(2)
        maxTarg = np.max(ftarg.vertices,axis=0)
        minTarg = np.min(ftarg.vertices,axis=0)
        maxTemp = np.max(ftemp.vertices,axis=0)
        minTemp = np.min(ftemp.vertices,axis=0)
        # estimate isotropic scale
        #volTarg=(maxTarg[0]-minTarg[0])*(maxTarg[1]-minTarg[1])
        #volTemp=(maxTemp[0]-minTemp[0])*(maxTemp[1]-minTemp[1])
        volTarg = np.sum(ftarg.volumes)
        volTemp = np.sum(ftemp.volumes)
        scaleMat = scaleMat*np.sqrt(volTarg/volTemp)
        scaleMat = scaleMat*np.abs(scaleStart)
        
        print("scaleMat is ")
        print(scaleMat)
        ftemp.updateVertices(ftemp.vertices@scaleMat)
        ftempO.updateVertices(ftempO.vertices@scaleMat)
    else:
        ftemp.updateVertices(ftemp.vertices*scaleStart)
        ftempO.updateVertices(ftempO.vertices*scaleStart)
        
    # rescale weights to be 0 or 1 depending on whether there is tissue in both template and target
    ftemp.updateWeights(1.0*(ftemp.weights > 0))
    ftempO.updateWeights(1.0*(ftempO.weights > 0))
    ftarg.updateWeights(1.0*(ftarg.weights > 0))
    
    # Scaling to 1 along one axis for interpretability
    #scale = (ftemp.vertices.max(axis=0) - ftemp.vertices.min(axis=0)).max()
    scaleIm = 1 #+(ftemp.image.max(axis=0) - ftemp.image.min(axis=0)).max()
    print("scaleIm is " + str(scaleIm))
    logging.info(f'scale = {scale:.2f}, scale image: {scaleIm:.2f}')
    
    Kdist = Kernel(name='gauss', sigma=sigmaDist)
    
    vTempO = np.copy(ftemp.vertices)
    vTargO = np.copy(ftarg.vertices)
    
    # KATIE: need to change this portion because nothing can be precomputed since alpha's in atlas are changing??
    # NEVERMIND, alphas are not changing 
    Alpha,cPrimeInc = precomputeAlphaCprimeNonRigid(ftempO,ftarg) 
    A,b = getAandBnonRigid(ftemp,ftarg,Kdist,Alpha,cPrimeInc)
    
    print("sum of weights after initialization")
    print(np.sum(ftemp.weights))
    print(np.sum(ftarg.weights))
    
    # Initialize weights as mean of target weights
    # initialize pi_theta as uniform over target regions for each atlas region 
    newInitial = np.zeros((ftempO.image.shape[0],numFeats))
    newInitial = newInitial + 1.0/numFeats # uniform distribution for each cell type
    ftemp.updateImage(newInitial)
    
    print("sum of weights after")
    print(np.sum(ftemp.weights))
    
    sn = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_initialTheta.png")
    #plotTheta(theta,numLabels,numFeats,sn)
    #totalCost[0] = np.sum(A*(theta@theta.T)) - 2*np.sum(b*theta)

   
    for i in range(iters):
        # update the vertices to be original ones each time
        # KATIE: maybe try to eliminate this and see if can do better but might get stuck (maybe do this for first few?)
        print("on iteration " + str(i) + " updating vertices")
        #print(time.time()-start)
        ftemp.updateVertices(vTempO)
        ftempO.updateVertices(vTempO)
        
        # try just estimating rigid and scale on first iteration --> doesn't work since reset vertices to initial
        if (i < rigOnly):
            # Q: do rigid registration based on subset of vertices?
            J1 = np.random.choice(ftarg.centers.shape[0], min(2000, ftarg.centers.shape[0]), replace=False) 
            J0 = np.random.choice(ftemp.centers.shape[0], min(2001, ftemp.centers.shape[0]), replace=False)

            AlphaSub = ftemp.weights[J0,None]@ftarg.weights[J1,None].T
            GammaSub = ftemp.volumes[J0,None]@ftarg.volumes[J1,None].T
            ZetaSub = ftemp.image[J0,:]@ftarg.image[J1,:].T
            K = AlphaSub*GammaSub*ZetaSub
            print("Alpha, Gamma, or Zeta dominating")
            print(AlphaSub)
            print(GammaSub)
            print(ZetaSub)
            #K = 1 + K # add 1 to give more weight to geometry; ends up beign a constant in QP part 
            print("K used for rigid registration")
            print(K)
            R2, T2 = rigidRegistration_varifold((ftemp.centers[J0, :],ftarg.centers[J1, :]), weights=K, ninit=4,
                                            sigma = sigmaDist) # finds matrix 
            print("after rigid registration calculation")
            print(R2)
            print(T2)

            # if ftemp is first 
            ftemp.updateVertices(np.dot(ftemp.vertices, R2.T) + T2)
            ftempO.updateVertices(np.dot(ftempO.vertices, R2.T) + T2)
            
            sn = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_rigid.vtk")
            snF = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_rigidFeats.vtk")

            ftempO.save(sn)
            ftemp.save(snF)
        
            print("before scale estimate ")
            # estimate scale as best out of trying 10 (from 0.5 to 1.5)
            KparIm=Kernel(name='euclidean', sigma=sigmaIm)
            if (scaleStart > 0):
                scaleEstimate = getObjScale(ftemp,ftarg,Kdist,K)
                print("scale 1 is " + str(scaleEstimate))
                print("scale actual is " + str(scaleEstimate*scaleStart))
                ftemp.updateVertices(scaleEstimate*ftemp.vertices)
                ftempO.updateVertices(scaleEstimate*ftempO.vertices)
                print("after scale estimate")

                sn = sn.replace(".vtk","Scale.vtk")
                snF = snF.replace("_rigidFeats.vtk","_rigidScaleFeats.vtk")
                ftempO.save(sn)
                ftemp.save(snF)
            
            if (i == rigOnly-1):
                vTempO = ftempO.vertices
            
        if (diffeo and i > rigOnly-1):
            '''
            Estimate diffeomorphism after estimating rigid + scale
            '''
            # update Vertices after estimation
            if i == rigOnly:
                K1 = Kernel(name='laplacian', sigma = sigmaKernel, order=3)
                pk_type = 'float32'
                affineSt = 'none'

                sm = MeshMatchingParam(timeStep=0.1, algorithm='bfgs', KparDiff=K1, KparDist=('gauss', sigmaDist),
                       KparIm=('euclidean', sigmaIm), sigmaError=sigmaError)
                sm.KparDiff.pk_dtype = pk_type
                sm.KparDist.pk_dtype = pk_type
                sm.KparIm.pk_type = pk_type
                f = MeshMatching(Template=ftemp, Target=ftarg, outputDir='/cis/home/kstouff4/Documents/MeshRegistration/TestImages/AtlasEstimation/'+saveTemp+'_'+saveTarg+'affine' + affineSt + '/',param=sm,
                        testGradient=False,  maxIter=200,
                     affine=affineSt, rotWeight=.01, transWeight = 100.,
                        scaleWeight=10., affineWeight=10.)
            else:
                print('restarting matching')
                f.fv0.updateImage(ftemp.image)
                f.fv0.updateWeights(ftemp.weights)
                f.fvDef.updateImage(ftemp.image)
                f.fvDef.updateWeights(ftemp.weights)
                f.reset = True

            f.optimizeMatching()
            newV = f.fvDef.vertices
            scaleEstimated = f.Afft
            ftemp.updateVertices(newV)
            ftempO.updateVertices(newV)
        
        A,b = getAandBnonRigid(ftempO,ftarg,Kdist,Alpha,cPrimeInc) # ftempO because need to have original feature set
        Ao,bo = getAandBnonRigidKS(ftempO,ftarg,Kdist,Alpha,cPrimeInc)
        
        thetaVec = solveQPpiEqualsTheta(A,b,numLabels,numFeats) # size L x f
        theta = np.reshape(thetaVec,(numLabels,numFeats))
        convImage = ftempO.image@theta # number of cells x number of features
        ftemp.updateImage(convImage)
 
        totalCost[i+1] = np.sum(A*(theta@theta.T)) - 2*np.sum(b*theta) # assume no cost for rigid
    
    print("scale Estimated")
    print(scaleEstimated)
    #print(np.linalg.det(scaleEstimated))
    print("R2")
    print(R2)
    print("T2")
    print(T2)
    sn = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_finalTheta" + str(iters) + ".png")
    plotTheta(theta,numLabels,numFeats,sn)
    
    ftempO.save(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_geomDeform_" + str(iters) + ".vtk"))
    ftemp.save(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_geomDeform+featureMapped" + str(iters) + ".vtk"))
    np.savez(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_" + str(iters) + ".npz"), theta=theta,R2=R2,T2=T2,scale=scale)
    fig,ax = plt.subplots()
    ax.plot(totalCost)
    ax.set_xlabel('iters')
    ax.set_ylabel('theta cost')
    fig.savefig(sn.replace("Theta.png","Cost.png"),dpi=300)
    print("total Cost is " + str(totalCost))
    print("theta is " + str(theta))
    print("A is " + str(A))
    print("b is " + str(b))
    sn = saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_finalTheta.png")
    plotTheta(theta,numLabels,numFeats,sn)
    
    # save atlas undeformed with just features 
    ftempFeat = Mesh(mesh=fileTemp) # create mesh for registration
    ftempFeat.updateWeights(ftemp.weights)
    ftempFeat.updateImage(ftemp.image)
    ftempFeat.updateVertices(vTempO)
    ftempFeat.save(saveDir + baseTemp.replace(".vtk","_to_") + targTemp.replace(".vtk","_featureMapped" + str(iters) + ".vtk"))
    
    return theta


############################################################################
# Functions for Visualizing 

def plotTheta(theta,numLabels,numFeats,savename):
    '''
    Plot Bar graph assuming theta[l,f]
    Assume theta in the form of L x F
    '''
    f,ax = plt.subplots(numLabels,1,figsize=(6,10))
    for l in range(numLabels):
        ax[l].bar(np.arange(numFeats),theta[l,:])
        ax[l].set_ylim(0,1)
    f.suptitle("Distribution of Features per Label")
    f.savefig(savename,dpi=300)
    return
    

############################################################################
