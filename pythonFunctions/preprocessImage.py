import ntpath
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/MeshLDDMMQP/master-KMS/py-lddmm/')
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/MeshLDDMMQP/master-KMS/py-lddmm/base')
import os
from base import loggingUtils
import multiprocessing as mp
from multiprocessing import Pool
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
import cv2
from skimage.segmentation import watershed
from base.meshes import buildImageFromFullListHR, buildMeshFromCentersCounts, buildMeshFromImageData, buildMeshFromCentersCountsMinMax
from PIL import Image
Image.MAX_IMAGE_PIXELS
Image.MAX_IMAGE_PIXELS=1e10 # forget attack
import nibabel as nib
import re
import writeImageFunctions as wi

###########################################################################
# Functions for formatting Allen Dataset
def getHemiBrain(file1,axis=0):
    '''
    get left adn right (assume left brain is on right of screen)
    '''
    im = nib.load(file1)
    img = np.asanyarray(im.dataobj)
    print("image shape")
    print(img.shape)
    
    if (axis == 0):
        cutoff = int(img.shape[0]/2)
        imgNewR = img[0:cutoff,...]
        imgNewL = img[cutoff:,...]
    elif (axis == 1):
        cutoff = int(img.shape[1]/2)
        imgNewR = img[:,0:cutoff,...]
        imgNewL = img[:,cutoff:,...]
    elif (axis == 2):
        cutoff = int(img.shape[2]/2)
        imgNewR = img[:,:,0:cutoff,...]
        imgNewL = img[:,:,cutoff:,...]
    if (re.search('nii',file1)):
        iNew = nib.Nifti1Image(imgNewL,im.affine)
        nib.save(iNew,file1.replace(".nii","_hemiLeft.nii"))
        
        iNew = nib.Nifti1Image(imgNewR,im.affine)
        nib.save(iNew,file1.replace(".nii","_hemiRight.nii"))
        
                        
    else:
        fileMap = nib.AnalyzeImage.make_file_map()
        saveBase = file1.replace(".img","_hemiRight")
        fileMap['image'].fileobj = saveBase + '.img'
        fileMap['header'].fileobj = saveBase + '.hdr'
        totImage = nib.AnalyzeImage(imgNewR,im.affine,file_map=fileMap)
        nib.save(totImage,saveBase + '.img')

        fileMap = nib.AnalyzeImage.make_file_map()
        saveBase = file1.replace(".img","_hemiLeft")
        fileMap['image'].fileobj = saveBase + '.img'
        fileMap['header'].fileobj = saveBase + '.hdr'
        totImage = nib.AnalyzeImage(imgNewL,im.affine,file_map=fileMap)
        nib.save(totImage,saveBase + '.img')
    return
def getWhiteAndGray(file1):
    '''
    Assume grey and white are hard-coded as set of values (min = 1)
    Grey = 1 (X), White = 2 (Y), CSF = 3 (Z)
    '''
    im = nib.load(file1)
    img = np.asanyarray(im.dataobj)
    
    whiteTracts = [42, 17, 1016, 665, 900, 848, 117, 125, 158, 911, 93, 794, 798, 413, 633, 697, 237, 326, 78, 1123, 728, 776, 784, 6, 924, 1092, 190, 908, 940, 603, 436, 449, 443, 690, 802, 595, 611] # includes white matter in superior colliculus
    vents = [81, 129, 140, 145, 153, 164]
    imgNew = np.copy(img)
    for w in whiteTracts:
        imgNew[imgNew == w] = -2
    for v in vents:
        imgNew[imgNew == v] = -3
    imgNew[imgNew > 0] = -1 
    imgNew = -1*imgNew # 1 = grey, 2 = white, 3 = CSF, 0 = background
    
    fileMap = nib.AnalyzeImage.make_file_map()
    saveBase = file1.replace(".img","_greyWhite")
    fileMap['image'].fileobj = saveBase + '.img'
    fileMap['header'].fileobj = saveBase + '.hdr'
    totImage = nib.AnalyzeImage(imgNew,im.affine,file_map=fileMap)
    nib.save(totImage, saveBase + '.img')
    return

def splitCompartment(file1,valToSplit,suff,axis=0):
    '''
    split compartments along axis
    save with suff (i.e. split grey)
    '''
    
    im = nib.load(file1)
    img = np.asanyarray(im.dataobj)
    print("img shape is " + str(img.shape))
    
    imgNew = np.copy(img)
    xcoords = np.where(img == valToSplit)
    newVal = np.max(img)+1
    
    if (axis == 0):
        x0 = xcoords[0]
        midpnt = int(np.round((np.max(x0)+np.min(x0))/2))
        n = imgNew[midpnt:,...]
        n[n == valToSplit] = newVal
        imgNew[midpnt:,...] = n
    elif (axis == 1):
        x1 = xcoords[1]
        midpnt = int(np.round((np.max(x1)+np.min(x1))/2))
        n = imgNew[midpnt:,...]
        n[n == valToSplit] = newVal
        imgNew[midpnt:,...] = n

    fileMap = nib.AnalyzeImage.make_file_map()
    saveBase = file1.replace(".img",suff)
    fileMap['image'].fileobj = saveBase + '.img'
    fileMap['header'].fileobj = saveBase + '.hdr'
    totImage = nib.AnalyzeImage(imgNew,im.affine,file_map=fileMap)
    nib.save(totImage, saveBase + '.img')

    return

def makeSampleImage(file1,saveBase):
    '''
    Makes image with same header and size as file1 but 10% less
    Assumes 3D shape
    '''
    
    im = nib.load(file1)
    img = np.asanyarray(im.dataobj)
    print("image shape is " + str(img.shape))
    
    imgNewSingle = np.zeros_like(img)
    x0off = int(np.round(img.shape[0]*0.05))
    x1off = int(np.round(img.shape[1]*0.05))
    x2off = int(np.round(img.shape[2]*0.05))
    
    x0divide = int((img.shape[0]-2*x0off)/2)
    x1divide = int((img.shape[1]-2*x1off)/2)
    
    imgNewSingle[x0off:-x0off:,x1off:x1off+x1divide,:] = 1
    imgNewSingle[x0off:-x0off,x1off+x1divide:-x1off,:] = 2
    
    imgNewDouble = np.zeros_like(img)
    imgNewDouble[x0off:-x0off,x1off+x1divide:-x1off,:] = 1
    imgNewDouble[x0off:x0off+x0divide,x1off:x1off+x1divide,:] = 2
    imgNewDouble[x0off+x0divide:-x0off,x1off:x1off+x1divide,:] = 3
    
    fileMap = nib.AnalyzeImage.make_file_map()
    fileMap['image'].fileobj = saveBase + '_split1.img'
    fileMap['header'].fileobj = saveBase + '_split1.hdr'
    totImage = nib.AnalyzeImage(imgNewSingle,im.affine,file_map=fileMap)
    nib.save(totImage, saveBase + '_split1.img')

    fileMap = nib.AnalyzeImage.make_file_map()
    fileMap['image'].fileobj = saveBase + '_split2.img'
    fileMap['header'].fileobj = saveBase + '_split2.hdr'
    totImage = nib.AnalyzeImage(imgNewDouble,im.affine,file_map=fileMap)
    nib.save(totImage, saveBase + '_split2.img')
    
    imTotS = np.zeros((4,4,4)) # have each box be 100 microns by 100 microns
    imTotD = np.zeros((4,4,4))
    imTotS[:,0:2,:] = 1
    imTotS[:,2:,:] = 2
    imTotD[:,2:,:] = 1
    imTotD[0:2,0:2,:] = 2
    imTotD[2:,0:2,:] = 3
    
    fileMap = nib.AnalyzeImage.make_file_map()
    fileMap['image'].fileobj = saveBase + '_split1_4x4.img'
    fileMap['header'].fileobj = saveBase + '_split1_4x4.hdr'
    totImage = nib.AnalyzeImage(imTotS,im.affine,file_map=fileMap)
    nib.save(totImage, saveBase + '_split1_4x4.img')
    
    fileMap = nib.AnalyzeImage.make_file_map()
    fileMap['image'].fileobj = saveBase + '_split2_4x4.img'
    fileMap['header'].fileobj = saveBase + '_split2_4x4.hdr'
    totImage = nib.AnalyzeImage(imTotD,im.affine,file_map=fileMap)
    nib.save(totImage, saveBase + '_split2_4x4.img')

    return

def makeSampleImageRotate(file1,saveBase):
    '''
    Makes image with same header and size as file1 but 10% less
    Assumes 3D shape
    '''
    
    im = nib.load(file1)
    img = np.asanyarray(im.dataobj)

    imTotS = np.zeros((4,4,4)) # have each box be 100 microns by 100 microns
    imTotD = np.zeros((4,4,4))
    imTotS[0:2,:,:] = 1
    imTotS[2:,:,:] = 2
    imTotD[0:2,:,:] = 1
    imTotD[2:,0:2,:] = 2
    imTotD[2:,2:,:] = 3
    
    fileMap = nib.AnalyzeImage.make_file_map()
    fileMap['image'].fileobj = saveBase + '_split1_4x4_rotate.img'
    fileMap['header'].fileobj = saveBase + '_split1_4x4_rotate.hdr'
    totImage = nib.AnalyzeImage(imTotS,im.affine,file_map=fileMap)
    nib.save(totImage, saveBase + '_split1_4x4_rotate.img')
    
    fileMap = nib.AnalyzeImage.make_file_map()
    fileMap['image'].fileobj = saveBase + '_split2_4x4_rotate.img'
    fileMap['header'].fileobj = saveBase + '_split2_4x4_rotate.hdr'
    totImage = nib.AnalyzeImage(imTotD,im.affine,file_map=fileMap)
    nib.save(totImage, saveBase + '_split2_4x4_rotate.img')

    return

###########################################################################
# General Functions
def readCSVImage(file1, radius=30.):
    # file1 = arg[0]
    # outdir = arg[1]
    # print(outdir)
    # if not os.path.exists(outdir):
    #     os.mkdir(outdir)
    print('reading ' + file1)
    df = pd.read_csv(file1)
    print('done')
    x_ = df['global_x'].to_numpy()
    y_ = df['global_y'].to_numpy()

    ugenes, inv = np.unique(df['gene'], return_inverse=True)
    img1 = buildImageFromFullListHR(x_, y_, inv, radius=radius)
    return img1

def makeMeshForCSV(file1,savedir):
    '''
    file1 = direct path to input CSV file (datadir + '/0_origdata/mouse2/202204181456_60988941_VMSC01601/detected_transcripts.csv')
    savedir = where to save meshes to
    '''
    loggingUtils.setup_default_logging(stdOutput=True)
    #file1 = homedir + '/202202170851_60988201_VMSC01001/detected_transcripts.csv'
    #file1 = datadir + '/0_origdata/mouse2/202204181456_60988941_VMSC01601/detected_transcripts.csv'
    print('building image')
    img, info = readCSVImage(file1, 15.)
    img2 = img.sum(axis=2)
    minx = info[0]
    miny = info[1]
    spacing = info[2]
    cm = img2.max()
    print(f'maximum counts {cm:.0f}')
    imgout = ((255 / np.log(1+cm)) * np.log(1+img2)).astype(np.uint8)
    print('running watershed')
    seg = (1+watershed(-img2)) * (img2>0)
    nlab = seg.max() + 1
    print(f'number of labels: {nlab}')
    print(seg.shape)
    segout = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for j in range(3):
        col = np.zeros(nlab, dtype=int)
        col[1:] = 1+np.random.choice(255, nlab-1)
        print(col.shape, col[seg].shape)
        segout[:, :, j] = col[seg]
    iio.imwrite(savedir + "/img.png", imgout)
    print(segout.shape)
    cv2.imwrite(savedir + "/seg.png", segout.astype(np.int8))


    threshold = 50
    centers = np.zeros((nlab, 2))
    cts = np.zeros((nlab, img.shape[2]), dtype=int)
    nb = np.zeros(nlab, dtype=int)
    x0 = np.linspace(minx+spacing/2, minx + img.shape[0]*spacing - spacing/2, img.shape[0])
    y0 = np.linspace(miny+spacing/2, miny + img.shape[1]*spacing - spacing/2, img.shape[1])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            centers[seg[i,j], 0] += x0[i]
            centers[seg[i,j], 1] += y0[j]
            nb[seg[i,j]] += 1
            cts[seg[i,j],:] += img[i,j,:].astype(int)
    nbc = cts.sum(axis=1)
    centers = centers[nbc>=threshold, :]
    cts = cts[nbc >=threshold, :]
    centers /= nb[nbc>=threshold, None]
    print(f'number of cells: {centers.shape[0]}')
    df = pd.DataFrame(data = {'centers_x':centers[:,0], 'centers_y':centers[:,1]})
    df.to_csv(homedir + '/centers.csv')
    df = pd.DataFrame(data=cts)
    df.to_csv(homedir + '/counts.csv')
    fv = buildMeshFromCentersCounts(centers, cts, resolution=50, radius = None, weights=None, threshold = 10)
    fv.saveVTK(homedir + '/mesh50.vtk')
    fv = buildMeshFromCentersCounts(centers, cts, resolution=100, radius = None, weights=None, threshold = 10)
    fv.saveVTK(homedir + '/mesh100.vtk')
    fv = buildMeshFromCentersCounts(centers, cts, resolution=200, radius = None, weights=None, threshold = 10)
    fv.saveVTK(homedir + '/mesh200.vtk')
    return

    #plt.imshow(imgout)
    #plt.show()
def makeMeshForCartoon(file1,savedir,dx0=0,dx1=0,sliceInd=-1,ax=-1,threshold=0,subsetVals=None,removeZeros=True,backVal=0,rgb=False):
    '''
    file1 = direct path to input image file
    savedir = where to save meshes to
    dx0,dx1 = spacing (default is to make mesh centered at 0; if are = 0, then default to Laurent's way
    threshold is based on the values in genes
    Assume all of the values are integer valued?
    '''
    loggingUtils.setup_default_logging(stdOutput=True)
    rz = ''
    
    # read in image file
    if (file1.split(".")[-1] == "img" or re.search("nii",file1)):
        im = nib.load(file1)
        img = np.asanyarray(im.dataobj).astype('float32')
        print("img shape is " + str(img.shape))
        if (sliceInd >= 0):
            if (ax == 0):
                #img = np.squeeze(img)
                #img = np.transpose(img,axes=(1,2,0))
                img = img[sliceInd,...]
            elif (ax == 1):
                img = img[:,sliceInd,...]
            elif (ax == 2):
                if (len(img.shape) > 2):
                    img = img[:,:,sliceInd,...]
                else:
                    img = img[:,:,sliceInd]
        img = np.squeeze(img)
    else:
        im = Image.open(file1)
        img = np.asanyarray(im)
        if (len(img.shape) > 2):
            if (img.shape[2] > 3):
                img = img[...,0:3]
            if (np.sum(img[...,0] - img[...,1]) == 0):
                img = img[...,0]
            #img = np.squeeze(img)
            #img = img[...,None]
    # assume all background pixels are less than or equal to what is given
    if (backVal != 0 and not rgb):
        print("subtracting " + str(backVal))
        img = img - backVal
    # make into one hot encoding
    if (not rgb):
        if (not removeZeros):
            vals = int(np.max(img)) # don't include 0 as count
            extendImg = np.zeros((img.shape[0],img.shape[1],vals))
            indsToLabs = np.zeros((vals,2))
            indsToLabs[:,0] = np.arange(vals)
            indsToLabs[:,1] = np.arange(vals) + 1
            for i in range(vals):
                extendImg[...,i] = (img == i+1) # labels correspond to 1 less than in original image
        else:
            print("new img shape " + str(img.shape))
            vals = np.unique(img)
            valSize = len(vals)-1 # don't include 0 
            extendImg = np.zeros((img.shape[0],img.shape[1],valSize))
            indsToLabs = np.zeros((valSize,2))
            indsToLabs[:,0] = np.arange(valSize)
            indsToLabs[:,1] = vals[1:]

            for i in range(valSize):
                extendImg[...,i] = (img == vals[i+1])

            #extendImgSum = np.sum(extendImg,axis=(0,1))
            #ids = np.where(extendImgSum > 0)
            #extendImg = extendImg[...,ids]
            #indsToLabs = indsToLabs[ids,:]
            #indsToLabs[:,0] = np.arange(indsToLabs.shape[0])
            rz = '_rz'
        weightsNew = (img > 0).astype('float32')
    else:
        extendImg = img
        weightsNew = (np.sum(img,axis=-1) < backVal*3).astype('float32') # weigh each pixel if not background (assume white background)
    if (dx0 == 0 or dx1 == 0):
        xi = np.linspace(0, 1, extendImg.shape[0])
        yi = np.linspace(0, 1, extendImg.shape[1])
    else:
        xi = np.arange(extendImg.shape[0])*dx0
        yi = np.arange(extendImg.shape[1])*dx1
        xi = xi - np.mean(xi)
        yi = yi - np.mean(yi)
        
    (x, y) = np.meshgrid(yi, xi)
    ng = extendImg.shape[2]
    cts = extendImg.reshape((x.size, ng))
    if subsetVals is not None:
        extendImg = extendImg[:, subsetVals]
    
    centers = np.zeros((x.size, 2))
    centers[:, 0] = np.ravel(x) # bounding_box[0] + bounding_box[1]*x/img.shape[0]
    centers[:, 1] = np.ravel(y) #bounding_box[2] + bounding_box[3]*y/img.shape[1]
    weightsNew = np.ravel(weightsNew) # don't include tissue 
    
    # remove background pixels
    inds = weightsNew > 0
    print("shapes of centers and counts before")
    print(centers.shape)
    print(cts.shape)
    print(weightsNew.shape)
    print("shapes of centers and counts after removing background")
    centers = centers[inds,:]
    cts = cts[inds,:]
    weightsNew = weightsNew[inds]
    print(centers.shape)
    print(cts.shape)
    print(weightsNew.shape)
    print(centers)
        
    baseName = file1.split("/")[-1]
    baseName = baseName.split(".")[0]
    if (sliceInd >= 0):
        baseName = baseName + "_ax" + str(ax) + "_sl" + str(sliceInd)
    
    fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=100, radius = None, weights=weightsNew, threshold =threshold,minx=-6200,miny=-6000,maxx=6200,maxy=6000)
    fv.saveVTK(savedir + baseName + '_mesh100' + rz + '.vtk')
    fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=200, radius = None, weights=weightsNew, threshold=threshold, minx=-6200,miny=-6000,maxx=6200,maxy=6000)
    fv.saveVTK(savedir + baseName + '_mesh200' + rz + '.vtk')
    fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=50, radius = None, weights=weightsNew, threshold=threshold, minx=-6200,miny=-6000,maxx=6200,maxy=6000)
    fv.saveVTK(savedir + baseName + '_mesh50' + rz + '.vtk')
    '''
    fv = buildMeshFromCentersCounts(centers, cts, resolution=15, radius = None, weights=weightsNew, threshold = 1e-10, minx=-4200,miny=-5000,maxx=4200,maxy=5000)
    fv.saveVTK(savedir + baseName + '_mesh15' + rz + '.vtk')
    '''
    #fv = buildMeshFromCentersCounts(centers, cts, resolution=200, radius = None, weights=weightsNew, threshold = 1e-10)
    #fv.saveVTK(savedir + baseName + '_mesh200' + rz + '.vtk')
    if (not rgb):
        np.savez(savedir+baseName+'_indsToLabs.npz',indsToLabs=indsToLabs)
    return

def makeMeshForCartoon3D(file1,savedir,dx0=0,dx1=0,dx2=0, sliceFirst=-1,sliceLast=-1,ax=-1,threshold=0,subsetVals=None,removeZeros=True,backVal=0):
    '''
    file1 = direct path to input image file
    savedir = where to save meshes to
    dx0,dx1 = spacing (default is to make mesh centered at 0; if are = 0, then default to Laurent's way
    threshold is based on the values in genes
    Assume all of the values are integer valued?
    '''
    loggingUtils.setup_default_logging(stdOutput=True)
    rz = ''
    
    # read in image file
    if (file1.split(".")[-1] == "img" or re.search("nii",file1)):
        im = nib.load(file1)
        img = np.asanyarray(im.dataobj).astype('float32')
        print("img shape is " + str(img.shape))
        if (sliceFirst >= 0):
            if (ax == 0):
                #img = np.squeeze(img)
                #img = np.transpose(img,axes=(1,2,0))
                img = img[sliceFirst:sliceLast,...]
            elif (ax == 1):
                img = img[:,sliceFirst:sliceLast,...]
            elif (ax == 2):
                if (len(img.shape) > 2):
                    img = img[:,:,sliceFirst:sliceLast,...]
                else:
                    img = img[:,:,sliceFirst:sliceLast]
        img = np.squeeze(img)
    else:
        im = Image.open(file1)
        img = np.asanyarray(im)
        if (len(img.shape) > 2):
            if (img.shape[2] > 3):
                img = img[...,0:3]
            if (np.sum(img[...,0] - img[...,1]) == 0):
                img = img[...,0]
            #img = np.squeeze(img)
            #img = img[...,None]
    # assume all background pixels are less than or equal to what is given
    if (backVal > 0):
        print("subtracting " + str(backVal))
        img = img - backVal
    # make into one hot encoding
    if (not removeZeros):
        vals = int(np.max(img)) # don't include 0 as count
        extendImg = np.zeros((img.shape[0],img.shape[1],img.shape[2],vals))
        indsToLabs = np.zeros((vals,2))
        indsToLabs[:,0] = np.arange(vals)
        indsToLabs[:,1] = np.arange(vals) + 1
        for i in range(vals):
            extendImg[...,i] = (img == i+1) # labels correspond to 1 less than in original image
    else:
        print("new img shape " + str(img.shape))
        vals = np.unique(img)
        valSize = len(vals)-1 # don't include 0 
        extendImg = np.zeros((img.shape[0],img.shape[1],img.shape[2],valSize)).astype('float32')
        indsToLabs = np.zeros((valSize,2))
        indsToLabs[:,0] = np.arange(valSize)
        indsToLabs[:,1] = vals[1:]

        for i in range(valSize):
            extendImg[...,i] = (img == vals[i+1])
            
        #extendImgSum = np.sum(extendImg,axis=(0,1))
        #ids = np.where(extendImgSum > 0)
        #extendImg = extendImg[...,ids]
        #indsToLabs = indsToLabs[ids,:]
        #indsToLabs[:,0] = np.arange(indsToLabs.shape[0])
        rz = '_rz'
    weightsNew = (img > 0).astype('float32')
    
    if (dx0 == 0 or dx1 == 0 or dx2 == 0):
        xi = np.linspace(0, 1, extendImg.shape[0])
        yi = np.linspace(0, 1, extendImg.shape[1])
        zi = np.linspace(0, 1, extendImg.shape[2])
    else:
        xi = np.arange(extendImg.shape[0])*dx0
        yi = np.arange(extendImg.shape[1])*dx1
        zi = np.arange(extendImg.shape[2])*dx2
        xi = xi - np.mean(xi)
        yi = yi - np.mean(yi)
        zi = zi - np.mean(zi)
        
    (x, y, z) = np.meshgrid(yi, xi, zi)
    ng = extendImg.shape[3]
    cts = extendImg.reshape((x.size, ng))
    if subsetVals is not None:
        extendImg = extendImg[..., subsetVals]
    
    centers = np.zeros((x.size, 3))
    centers[:, 0] = np.ravel(x) # bounding_box[0] + bounding_box[1]*x/img.shape[0]
    centers[:, 1] = np.ravel(y) #bounding_box[2] + bounding_box[3]*y/img.shape[1]
    centers[:, 2] = np.ravel(z)
    weightsNew = np.ravel(weightsNew) # don't include tissue 
    
    # remove background pixels
    inds = weightsNew > 0
    print("shapes of centers and counts before")
    print(centers.shape)
    print(cts.shape)
    print(weightsNew.shape)
    print("shapes of centers and counts after removing background")
    centers = centers[inds,:]
    cts = cts[inds,:]
    weightsNew = weightsNew[inds]
    print(centers.shape)
    print(cts.shape)
    print(weightsNew.shape)
    print(centers)
        
    baseName = file1.split("/")[-1]
    baseName = baseName.split(".")[0]
    if (sliceFirst >= 0):
        baseName = baseName + "_ax" + str(ax) + "_sl" + str(sliceFirst) + "_to_" + str(sliceLast)
    
    fv = buildMeshFromCentersCounts3D(centers, cts, resolution=100, radius = None, weights=weightsNew, threshold = 1e-10,minx=-6200,miny=-6000,maxx=6200,maxy=6000,minz=-12000,maxz=12000)
    fv.saveVTK(savedir + baseName + '_mesh100' + rz + '.vtk')
    fv = buildMeshFromCentersCounts3D(centers, cts, resolution=200, radius = None, weights=weightsNew, threshold = 1e-10, minx=-6200,miny=-6000,maxx=6200,maxy=6000,minz=-12000,maxz=12000)
    fv.saveVTK(savedir + baseName + '_mesh200' + rz + '.vtk')
    np.savez(savedir+baseName+'_indsToLabs.npz',indsToLabs=indsToLabs)
    
    return

def makeMeshForCellTypes(csvMeta, csvCell,savedir,threshold=0,subsetVals=None,removeZeros=False):
    '''
    file1 = direct path to csv file with 
    savedir = where to save meshes to
    dx0,dx1 = spacing (default is to make mesh centered at 0; if are = 0, then default to Laurent's way
    threshold is based on the values in genes
    Assume all of the values are integer valued?
    '''
    loggingUtils.setup_default_logging(stdOutput=True)
    rz = ''
    
    meta = np.genfromtxt(csvMeta,delimiter=',')
    xi = meta[1:,3]
    yi = meta[1:,4]
    
    cells = np.genfromtxt(csvCell,delimiter=',')
    img = cells[1:,4]
    print(img.shape)
    print(np.unique(img))
    
    
    # make into one hot encoding
    if (not removeZeros):
        vals = int(np.max(img)) # don't include 0 as count
        extendImg = np.zeros((img.shape[0],vals))
        indsToLabs = np.zeros((vals,2))
        indsToLabs[:,0] = np.arange(vals)
        indsToLabs[:,1] = np.arange(vals) + 1
        for i in range(vals):
            extendImg[...,i] = (img == i+1) # labels correspond to 1 less than in original image
    else:
        print("new img shape " + str(img.shape))
        vals = np.unique(img)
        valSize = len(vals)-1 # don't include 0 
        extendImg = np.zeros((img.shape[0],valSize))
        indsToLabs = np.zeros((valSize,2))
        indsToLabs[:,0] = np.arange(valSize)
        indsToLabs[:,1] = vals[1:]

        for i in range(valSize):
            extendImg[...,i] = (img == vals[i+1])
            
        #extendImgSum = np.sum(extendImg,axis=(0,1))
        #ids = np.where(extendImgSum > 0)
        #extendImg = extendImg[...,ids]
        #indsToLabs = indsToLabs[ids,:]
        #indsToLabs[:,0] = np.arange(indsToLabs.shape[0])
        rz = '_rz'
    weightsNew = (img > 0).astype('float32') # should be 1 for each count 
    print("confirming " + str(np.sum(weightsNew)) + " is the same as number cells " + str(img.shape[0]))
    
    ng = extendImg.shape[-1]
    cts = extendImg
    if subsetVals is not None:
        cts = extendImg[:, subsetVals]
    
    centers = np.zeros((img.shape[0], 2))
    centers[:, 0] = np.ravel(xi) # bounding_box[0] + bounding_box[1]*x/img.shape[0]
    centers[:, 1] = np.ravel(yi) #bounding_box[2] + bounding_box[3]*y/img.shape[1]
    weightsNew = np.ravel(weightsNew) # don't include tissue 
    
    # remove background pixels
    inds = weightsNew > 0
    print("shapes of centers and counts before")
    print(centers.shape)
    print(cts.shape)
    print(weightsNew.shape)
    print("shapes of centers and counts after removing background")
    centers = centers[inds,:]
    cts = cts[inds,:]
    weightsNew = weightsNew[inds]
    print(centers.shape)
    print(cts.shape)
    print(weightsNew.shape)
    print(centers)
        
    baseName = csvCell.split("/")[-1]
    baseName = baseName.split(".")[0]
      
    fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=100, radius = None, weights=weightsNew, threshold = 1e-10,minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="centers")
    fv.saveVTK(savedir + baseName + '_mesh100' + rz + '.vtk')
    fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=200, radius = None, weights=weightsNew, threshold = 1e-10, minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="centers")
    fv.saveVTK(savedir + baseName + '_mesh200' + rz + '.vtk')
    fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=50, radius = None, weights=weightsNew, threshold = 1e-10, minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="centers")
    fv.saveVTK(savedir + baseName + '_mesh50' + rz + '.vtk')
    '''
    fv = buildMeshFromCentersCounts(centers, cts, resolution=15, radius = None, weights=weightsNew, threshold = 1e-10, minx=-4200,miny=-5000,maxx=4200,maxy=5000)
    fv.saveVTK(savedir + baseName + '_mesh15' + rz + '.vtk')
    '''
    #fv = buildMeshFromCentersCounts(centers, cts, resolution=200, radius = None, weights=weightsNew, threshold = 1e-10)
    #fv.saveVTK(savedir + baseName + '_mesh200' + rz + '.vtk')
    np.savez(savedir+baseName+'_indsToLabs.npz',indsToLabs=indsToLabs)
    return

def makeMeshForMERFISHGenes(csvMeta, csvGene,savedir,subsetVals):
    '''
    file1 = direct path to csv file with 
    savedir = where to save meshes to
    dx0,dx1 = spacing (default is to make mesh centered at 0; if are = 0, then default to Laurent's way
    threshold is based on the values in genes
    Assume all of the values are integer valued?
    # important genes are: 382, 
    '''
    loggingUtils.setup_default_logging(stdOutput=True)
    rz = ''
    
    meta = np.genfromtxt(csvMeta,delimiter=',')
    xi = meta[1:,3]
    yi = meta[1:,4]
    
    # modify this so reads in from correct file 
    cells = np.genfromtxt(csvGene,delimiter=',')
    img = cells[1:,:]
    
    with open(csvGene) as fi:
        first_line = fi.readline()
        f = first_line.split(',')
        f = np.asarray(f)
    geneOrder = f
    print(geneOrder)
    geneNames = geneOrder
    if subsetVals is not None:
        geneNames = []
        genes = subsetVals.split(',')
        indGenes = []
        for s in genes:
            print("looking for " + s )
            x = np.where(geneOrder == s)
            if len(x[0]) > 0:
                print("index is " ) 
                print(x)
                indGenes.append(x[0][0])
                geneNames.append(s)
            else:
                print(s + " not found ")
        print("Confirm number of indices equals number of genes in subsetVals")
        print(len(genes))
        print(len(indGenes))
        img = img[:,indGenes]
        img = np.squeeze(img)
    
    print("shape should be points x genes")
    print(img.shape)
    print(np.unique(img))
    
    # make 2 types of meshes: probability distribution over selected genes with weight = number of geneCounts
    # second = cells = density and get average RNA counts 
    # third = cells = density and get average log RNA counts 
    
    

    weightsNew = (img > 0).astype('float32') # should be 1 for each count 
    weightsNew = np.ones((img.shape[0],1))
    
    centers = np.zeros((img.shape[0], 2))
    centers[:, 0] = np.ravel(xi) # bounding_box[0] + bounding_box[1]*x/img.shape[0]
    centers[:, 1] = np.ravel(yi) #bounding_box[2] + bounding_box[3]*y/img.shape[1]
    weightsNew = np.ravel(weightsNew) # don't include tissue 
        
    baseName = csvGene.split("/")[-1]
    baseName = baseName.split(".")[0]
    
    cts = img
    weightsSum = np.sum(img,axis=-1)
    print("ensure coordinates are nonzero")
    print(np.max(xi))
    print(np.max(yi))
    print(centers.shape)
    
    print("difference in sizes")
    print(weightsNew.shape)
    print(weightsSum.shape)
    
    ctsSub = cts[weightsSum > 0,...] # select only those with counts
    centersSub = centers[weightsSum > 0,...]
    weightsSub = np.ones((ctsSub.shape[0],1))
    ctsNormal = ctsSub/(np.sum(ctsSub,axis=-1)[...,None]) # turn into probability distributions for each, so sum of RNA counts = 1
    
    # first type (keep all triangles with at least 1 rna count)
    '''
    # old way: 
    fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=100, radius = None, weights=weightsSum, threshold = 1e-10,minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="cts",gType="alpha")
    fv.saveVTK(savedir + baseName + '_mesh100' + rz + 'probRNA.vtk')
    fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=200, radius = None, weights=weightsSum, threshold = 1e-10, minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="cts",gType="alpha")
    fv.saveVTK(savedir + baseName + '_mesh200' + rz + 'probRNA.vtk')
    fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=50, radius = None, weights=weightsSum, threshold = 1e-10, minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="cts",gType="alpha")
    fv.saveVTK(savedir + baseName + '_mesh50' + rz + 'probRNA.vtk')
    '''
    
    
    # second type (keep all triangles with at least 1 cell)
    '''
    fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=100, radius = None, weights=weightsNew, threshold = 1e-10,minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="centers",gType="alpha")
    fv.saveVTK(savedir + baseName + '_mesh100' + rz + 'avgRNA.vtk')
    fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=200, radius = None, weights=weightsNew, threshold = 1e-10, minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="centers",gType="alpha")
    fv.saveVTK(savedir + baseName + '_mesh200' + rz + 'avgRNA.vtk')
    fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=50, radius = None, weights=weightsNew, threshold = 1e-10, minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="centers",gType="alpha")
    fv.saveVTK(savedir + baseName + '_mesh50' + rz + 'avgRNA.vtk')
    '''
    
    # third type
    '''
    fv = buildMeshFromCentersCountsMinMax(centers, np.log(cts+1)/np.log(10), resolution=100, radius = None, weights=weightsNew, threshold = 1e-10,minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="centers",gType="alpha")
    fv.saveVTK(savedir + baseName + '_mesh100' + rz + 'avgLog10RNA.vtk')
    fv = buildMeshFromCentersCountsMinMax(centers, np.log(cts+1)/np.log(10), resolution=200, radius = None, weights=weightsNew, threshold = 1e-10, minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="centers",gType="alpha")
    fv.saveVTK(savedir + baseName + '_mesh200' + rz + 'avgLog10RNA.vtk')
    fv = buildMeshFromCentersCountsMinMax(centers, np.log(cts+1)/np.log(10), resolution=50, radius = None, weights=weightsNew, threshold = 1e-10, minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="centers",gType="alpha")
    fv.saveVTK(savedir + baseName + '_mesh50' + rz + 'avgLog10RNA.vtk')
    '''
    
    # fourth type = normalize by # cells = total sum of mRNA, threshold at least 1 cell
    fv = buildMeshFromCentersCountsMinMax(centersSub, ctsNormal, resolution=100, radius = None, weights=weightsSub, threshold = 1e-10,minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="centers",gType="alpha")
    fv.updateImNames(geneNames)
    fv.saveVTK(savedir + baseName + '_mesh100' + rz + 'avgRNAProbDist.vtk')
    fv = buildMeshFromCentersCountsMinMax(centersSub, ctsNormal, resolution=200, radius = None, weights=weightsSub, threshold = 1e-10, minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="centers",gType="alpha")
    fv.updateImNames(geneNames)
    fv.saveVTK(savedir + baseName + '_mesh200' + rz + 'avgRNAProbDist.vtk')
    fv = buildMeshFromCentersCountsMinMax(centersSub, ctsNormal, resolution=50, radius = None, weights=weightsSub, threshold = 1e-10, minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="centers",gType="alpha")
    fv.updateImNames(geneNames)
    fv.saveVTK(savedir + baseName + '_mesh50' + rz + 'avgRNAProbDist.vtk')
    

    return

def makeMeshForSlideSeq(geneMatrix,barcodeLoc,barcodeTSV,savedir,precompute=False,subsetVals=None,rnaCount=False,back=-1):
    '''
    geneMatrix is mtx file (space-separated) with gene indexes as rows and col 0 = gene index, col 1 = barcode index, col 2 = gene count
    barcodeLoc is txt file (tab-separated) with rows as barcodes, col 2 = x coord (microns), col 3 = y coord (microns)
    assume for rnaCount that take log of values 
    '''
    
    rz = ''
    loggingUtils.setup_default_logging(stdOutput=True)
    if (not precompute):
        # make locations from barcodeLoc
        dictLoc = dict()
        file1 = open(barcodeLoc,'r')
        for line in file1:
            info = line.split('\t')
            dictLoc[info[1]] = [float(info[2]),float(info[3])]
        file1.close()
        with open(barcodeTSV) as f:
            lines = [line.rstrip() for line in f]
        centers = np.zeros((len(lines),2)).astype('float32')
        i = 0
        for l in lines:
            centers[i,:] = dictLoc[l]
            i = i+1    
        gB = np.genfromtxt(geneMatrix,comments='%',delimiter=' ')
        numGenes = int(gB[0,0])
        numBarCodes = int(gB[0,1])
        cts = np.zeros((numBarCodes,numGenes))
        gB = gB[1:,:]
        for g in range(numGenes):
            b = gB[gB[:,0] == g+1]
            cts[b[:,1].astype(int)-1,g] = b[:,2]
        #centers = np.genfromtxt(barcodeLoc,delimiter='\t')
        #centers = centers[:,2:]
        #centers = np.unique(centers,axis=0)
        np.savez(barcodeLoc.replace("_barcode_matching.txt","_centers_and_counts.npz"),centers=centers,cts=cts)
        return
    else:
        params = np.load(barcodeLoc.replace("_barcode_matching.txt","_centers_and_counts.npz"))
        centers = params['centers']
        cts = params['cts']
    
    #if (rnaCount):
        #cts = np.log(cts+1)
    print("shape of centers is " + str(centers.shape))
    print("shape of cts is " + str(cts.shape))
    weightsNew = np.sum(cts,axis=-1) # weights = gene counts
    print("shape of weights new is " + str(weightsNew.shape))
    
    # try to remove background
    xi = centers[:,0]
    yi = centers[:,1]
    print("max and min bounds")
    print(np.max(xi))
    print(np.min(xi))
    print(np.max(yi))
    print(np.min(yi))
    print("shapes are " + str(centers.shape))
    bounds = np.zeros((4,2))
    bounds[0,:] = [np.min(xi),np.min(yi)]
    bounds[1,:] = [np.min(yi),np.max(yi)]
    bounds[2,:] = [np.max(xi),np.min(yi)]
    bounds[3,:] = [np.max(yi),np.max(yi)]
    
    inds0 = centers == bounds[0,:]
    inds0 = np.squeeze(inds0[:,0])
    inds1 = centers == bounds[1,:]
    inds1 = np.squeeze(inds1[:,0])
    inds2 = centers == bounds[2,:]
    inds2 = np.squeeze(inds2[:,0])
    inds3 = centers == bounds[3,:]
    inds3 = np.squeeze(inds3[:,0])
    
    inds0 = centers[:,0] == np.min(xi)
    inds1 = centers[:,1] == np.min(yi)
    inds2 = centers[:,0] == np.max(xi)
    inds3 = centers[:,1] == np.max(yi)
    print("inds0 shape is " + str(inds0.shape))
    print("sum of inds ")
    print(np.sum(inds0))
    print(np.sum(inds1))
    print(np.sum(inds2))
    print(np.sum(inds3))
    if (back > 0):
        weightsThresh = np.sum(cts[inds0,back]) + np.sum(cts[inds1,back]) + np.sum(cts[inds2,back]) + np.sum(cts[inds3,back])
        weightsThresh = weightsThresh / (np.sum(inds0) + np.sum(inds1) + np.sum(inds2) + np.sum(inds3))
        inds = cts[:,back] < weightsThresh
    else:
        weightsThresh = np.log(np.sum(weightsNew[inds0])) + np.log(np.sum(weightsNew[inds1])) + np.log(np.sum(weightsNew[inds2])) + np.log(np.sum(weightsNew[inds3]))
        # threshold weights based on average total gene count 
        weightsThresh = weightsThresh / (np.sum(inds0) + np.sum(inds1) + np.sum(inds2) + np.sum(inds3))
        inds = np.log(weightsNew) > weightsThresh
        
    cts = cts[inds,:]
    centers = centers[inds,:]
    weightsNew = weightsNew[inds]
    print("weightsThresh shape is " + str(weightsThresh.shape))
    print("weightsThresh is " + str(weightsThresh))
    
    baseName = barcodeLoc.split("/")[-1]
    baseName = baseName.split(".")[0]
    baseName = baseName.replace("_barcode_matching.txt","")
    
    if (subsetVals is not None):
        totalGeneCt = np.sum(cts,axis=-1)
        ctsNew = cts[:,subsetVals]
        if (type(subsetVals) == int):
            ctsNew = cts[...,None]
        cts = np.zeros((ctsNew.shape[0],ctsNew.shape[-1]+1))
        cts[:,0:-1] = np.log(ctsNew)/np.log(10)
        cts[:,-1] = np.log(totalGeneCt)/np.log(10)
        print("cts shape is " + str(cts.shape))
        weightsNew = np.sum(cts,axis=-1)
        print("min and max of weightsNew " + str(np.min(weightsNew)) + ", " + str(np.max(weightsNew)))
        indsToLabs = np.zeros((cts.shape[-1]-1,2))
        indsToLabs[:,0] = np.arange(cts.shape[-1]-1)
        indsToLabs[:,1] = subsetVals
        np.savez(savedir+baseName+'_indsToLabs.npz',indsToLabs=indsToLabs)
        baseName = baseName + str(subsetVals)
    elif rnaCount:
        # take the 10 most varying genes (0.9995 quantile)
        # assume we are in gene counts
        totalGeneCt = np.sum(cts,axis=-1)
        varTot = np.var(cts,axis=0)
        indsV = varTot > np.quantile(varTot,0.9995)
        ctsNew = cts[:,indsV]
        cts = np.zeros((ctsNew.shape[0],ctsNew.shape[-1]+1))
        cts[:,0:-1] = np.log(ctsNew+1)/np.log(10)
        cts[:,-1] = np.log(totalGeneCt+1)/np.log(10)
        print("cts shape is " + str(cts.shape))
        weightsNew = np.sum(cts,axis=-1)
        print("min and max of weightsNew " + str(np.min(weightsNew)) + ", " + str(np.max(weightsNew)))
        indsToLabs = np.zeros((cts.shape[-1]-1,2))
        indsToLabs[:,0] = np.arange(cts.shape[-1]-1)
        indsToLabs[:,1] = subsetVals
        np.savez(savedir+baseName+'_indsToLabs.npz',indsToLabs=indsToLabs)
        baseName = baseName + 'variance0.9995' + str(indsV)
        
    
    
    if (not rnaCount):
        fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=100, radius = None, weights=weightsNew, threshold =0.1,minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="cts")
        fv.saveVTK(savedir + baseName + '_mesh100' + rz + 'thresh10th_nobg.vtk')
        fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=200, radius = None, weights=weightsNew, threshold = 0.1, minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="cts")
        fv.saveVTK(savedir + baseName + '_mesh200' + rz + 'thresh10th_nobg.vtk')
        fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=50, radius = None, weights=weightsNew, threshold = 0.1, minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="cts")
        fv.saveVTK(savedir + baseName + '_mesh50' + rz + 'thresh10th_nobg.vtk')
    else:
        fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=100, radius = None, weights=weightsNew, threshold =0.1,minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="centers")
        fv.saveVTK(savedir + baseName + '_mesh100' + rz + 'thresh10th_nobg_logRNA.vtk')
        fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=200, radius = None, weights=weightsNew, threshold = 0.1, minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="centers")
        fv.saveVTK(savedir + baseName + '_mesh200' + rz + 'thresh10th_nobg_logRNA.vtk')
        fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=50, radius = None, weights=weightsNew, threshold = 0.1, minx=np.min(xi)-1000,miny=np.min(yi)-1000,maxx=np.max(xi)+1000,maxy=np.max(yi)+1000,norm="centers")
        fv.saveVTK(savedir + baseName + '_mesh50' + rz + 'thresh10th_nobg_logRNA.vtk')
        
    return

def makeMeshForSSNisslThresh(ccNPZ,savedir,baseName,threshold=0.01,gray=False):
    '''
    Reads in npz that has stored counts and RGB values
    Treat the weights as number of cells (e.g. number of pixels)
    Treat the RGB values each as own mrna count
    threshold should indicate number of pixels per triangle (remove if less than 2??)
    '''

    info = np.load(ccNPZ)
    centers = info['centersRZ']
    cts = info['countsRZ'] # 0 where there is no Nissl signal 
    xi = centers[:,0]
    yi = centers[:,1]
    
    if (gray):
        cts = info['grayRavRZ']
        if (cts.shape[-1] == cts.shape[0]):
            cts = cts[...,None]
    
    weightsNew = np.ones(centers.shape[0])
    
    fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=100, radius = None, weights=weightsNew, threshold =threshold,minx=np.min(xi)-10,miny=np.min(yi)-10,maxx=np.max(xi)+10,maxy=np.max(yi)+10,norm="centers",gType="alpha")
    fv.saveVTK(savedir + baseName + '_mesh100_rz.vtk')
    fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=25, radius = None, weights=weightsNew, threshold = threshold, minx=np.min(xi)-10,miny=np.min(yi)-10,maxx=np.max(xi)+10,maxy=np.max(yi)+10,norm="centers",gType="alpha")
    fv.saveVTK(savedir + baseName + '_mesh25_rz.vtk')
    fv = buildMeshFromCentersCountsMinMax(centers, cts, resolution=50, radius = None, weights=weightsNew, threshold = threshold, minx=np.min(xi)-10,miny=np.min(yi)-10,maxx=np.max(xi)+10,maxy=np.max(yi)+10,norm="centers",gType="alpha")
    fv.saveVTK(savedir + baseName + '_mesh50_rz.vtk')
    
    return
    
def makeMeshSlideSeqSub(ccNPZ,subsetVals,featuresTSV,savedir,prefix,threshold=0.008):
    '''
    Make Mesh based on thresholded background and foreground already (don't elimininate any?) 
    '''
    
    # find inds
    feats=pd.read_csv(featuresTSV,sep='\t',header=None)
    feats = feats.values
    feats = feats[:,1]      
    inds = []
    indsNames = []
    for s in subsetVals:
        x = np.where(feats == s)
        if (len(x[0]) > 0):
            indsNames.append(s)
            inds.append(x[0][0])
    info = np.load(ccNPZ)
    centers = info['centers']
    counts = info['cts']
    cts = counts[:,inds]
    print("names found ")
    print(indsNames)
    
    baseName = ccNPZ.split("/")[-1]
    baseName = baseName.replace("_centers_and_counts","")

    fv = buildMeshFromCentersCounts(centers,cts,resolution=50,radius=None,weights=None,imNames=indsNames,thresh='rho',threshold=50*25*threshold)
    fv.saveVTK(savedir + baseName + '_mesh50_thresh' + str(threshold) + '_genes' + prefix + 'beadsAsCellDens.vtk')
    fv = buildMeshFromCentersCounts(centers,cts,resolution=100,radius=None,weights=None,imNames=indsNames,thresh='rho',threshold=100*50*threshold)
    fv.saveVTK(savedir + baseName + '_mesh100_thresh' + str(threshold) + '_genes' + prefix + 'beadsAsCellDens.vtk')

    fv = buildMeshFromCentersCounts(centers,cts,resolution=200,radius=None,weights=None,imNames=indsNames,thresh='rho',threshold=200*100*threshold)
    fv.saveVTK(savedir + baseName + '_mesh200_thresh' + str(threshold) + '_genes' + prefix + 'beadsAsCellDens.vtk')
    return
                                    
                                    
def makeSlideSeqImage(ccNPZ,subsetVals=None,var=0,maxVal=0):
    '''
    Read in centers and gene counts for each center (of bead location)
    Make image and save as analyze image with total counts of all genes
    '''
    
    info = np.load(ccNPZ)
    centers = info['centers']
    counts = info['counts']
    
    minx = round(np.min(centers[:,0])) - 5
    maxx = round(np.max(centers[:,0])) + 5
    miny = round(np.min(centers[:,1])) - 5
    maxy = round(np.min(centers[:,1])) + 5
    
    valCount = 100
    if subsetVals is not None:
        valCount = len(subsetVals)
        
    subsetToPlot = counts[:,0:100]
    strPrint = "_first10"
    
    if subsetVals is not None:
        subsetToPlot = counts[:,subsetVals]
        strPrint = str(subsetVals)
    elif var > 0:
        varVals = np.std(counts,axis=0)
        inds = np.where(varVals > np.quantile(varVals,var))
        valCount = len(inds)
        subsetToPlot = counts[:,inds]
        strPrint = "_var" + str(var) + "quantile"
    elif maxVal > 0:
        varVals = np.max(counts,axis=0)
        inds = np.where(varVals > np.quantile(varVals,maxVal))
        valCount = len(inds)
        subsetToPlot = counts[:,inds]
        strPrint = "_maxVal" + str(maxVal) + "quantile"
    
    arrayPlot = np.zeros((int(maxx - minx),int(maxy-miny),valCount))
    x = round(centers[:,0])
    y = round(centers[:,1])
    arrayPlot[x,y] = subsetToPlot
    
    # make image of sum
    np.savez(ccNPZ.replace("_centers_and_counts.npz",strPrint + ".npz"),arrayPlot=arrayPlot)
    im = np.sum(arrayPlot,axis=-1)
    im = im[...,None]
    
    fileMap = nib.AnalyzeImage.make_file_map()
    saveBase = ccNPZ.replace("_centers_and_counts.npz",strPrint)
    fileMap['image'].fileobj = saveBase + '.img'
    fileMap['header'].fileobj = saveBase + '.hdr'
    totImage = nib.AnalyzeImage(im,np.eye(4),file_map=fileMap)
    nib.save(totImage, saveBase + '.img')

    return

def makeSlideSeqMesh(geneMatrix,barcodeLoc,barcodeTSV,savedir,precompute=False,xbeads=(700,4400),ybeads=(700,4000),thresholdBeadDensity=(0.008),thresholdGene=(21984,3.0)):
    '''
    Different mechanisms for negating background:
    bead = apply a threshold to the bead density (weights based on bead density then)
    
    To do:
    choose subset based on names by reading in featureTSV and looking for gene name (without capitals)
    threshold background vs. foreground based on density of triangles = number of beads 
    '''
    rz = ''
    loggingUtils.setup_default_logging(stdOutput=True)
    if (not precompute):
        # make locations from barcodeLoc
        dictLoc = dict()
        file1 = open(barcodeLoc,'r')
        for line in file1:
            info = line.split('\t')
            dictLoc[info[1]] = [float(info[2]),float(info[3])]
        file1.close()
        with open(barcodeTSV) as f:
            lines = [line.rstrip() for line in f]
        centers = np.zeros((len(lines),2)).astype('float32')
        i = 0
        for l in lines:
            centers[i,:] = dictLoc[l]
            i = i+1    
        gB = np.genfromtxt(geneMatrix,comments='%',delimiter=' ')
        numGenes = int(gB[0,0])
        numBarCodes = int(gB[0,1])
        cts = np.zeros((numBarCodes,numGenes))
        gB = gB[1:,:]
        for g in range(numGenes):
            b = gB[gB[:,0] == g+1]
            cts[b[:,1].astype(int)-1,g] = b[:,2]
        #centers = np.genfromtxt(barcodeLoc,delimiter='\t')
        #centers = centers[:,2:]
        #centers = np.unique(centers,axis=0)
        np.savez(barcodeLoc.replace("_barcode_matching.txt","_centers_and_counts.npz"),centers=centers,cts=cts)
    else:
        params = np.load(barcodeLoc.replace("_barcode_matching.txt","_centers_and_counts.npz"))
        centers = params['centers']
        cts = params['cts']
    
    print("starting number of beads " + str(centers.shape[0]))
    # Eliminate those out of bounds
    indsInBounds = (centers[:,0] > xbeads[0])*(centers[:,0] < xbeads[1])*(centers[:,1] > ybeads[0])*(centers[:,1] < ybeads[1])
    centersInBounds = centers[indsInBounds,:]
    ctsInBounds = cts[indsInBounds,:]
    
    print("after eliminating those out of bounds " + str(centers.shape[0]))

    #if (back == 'bead'):
    return

############################################################
# Allen 3D Meshes following Mutual Information Calculation

def makeGeneSubsetAllenMeshes(dirName,geneFile,saveLoc,num=20):
    '''
    Example: dirName = '/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1/*/'
    geneFile = '/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/MI_Results/all_MI_nobound_ordered.npz'
    '''
    fils = glob(dirName + 'detected_transcripts.npz')
    gFile = np.load(geneFile,allow_pickle=True)
    gIndSub = gFile['geneInds'][-num:]
    gNameSub = gFile['genes'][-num:]
    
    for i in range(len(fils)):
        fname = fils[i]
        fpref = fname.split('/')[-2]
        info = np.load(fname)
        c = info['coordsTot']
        g = info['geneInd']
        if (np.max(g) < 701):
            print("missing genes")
            continue
        t = g == gIndSub[0]
        for j in range(1,len(gIndSub)):
            t += g == gIndSub[j]
        cSub = c[t,...]
        gSub = g[t,...]
        wi.makeAllen3DVTK(cSub,gSub,saveLoc+fpref+'.vtk')
        gVals,inv = np.unique(gSub,return_inverse=True)
        cts = np.zeros((gSub.shape[0],len(gVals)))
        cts[np.arange(gSub.shape[0]),inv] = 1
        gIndSubSort,invG = np.unique(gIndSub,return_inverse=True)
        print("lengths should be same: " + str(len(gVals)) + str(len(gIndSubSort)))
        gNameSubSort = gNameSub[invG,...]

        # normalizing by counts and centers should be the same here
        # alpha should be the number of mRNA per volume 
        # threshold by zeta so that keep all triangles with at least one mRNA
        xi = cSub[:,0]
        yi = cSub[:,1]
        fv = buildMeshFromCentersCountsMinMax(cSub[:,0:2], cts, resolution=100, radius = None, weights=np.ones(cSub.shape[0]), threshold = 10,minx=np.min(xi)-100,miny=np.min(yi)-100,maxx=np.max(xi)+100,maxy=np.max(yi)+100,norm="cts",gType="alpha")
        fv.updateImNames(gNameSubSort)
        fv.saveVTK(saveLoc + fpref + '_mesh100_rz_probRNA_t10parts.vtk')
        fv = buildMeshFromCentersCountsMinMax(cSub[:,0:2], cts, resolution=50, radius = None, weights=np.ones(cSub.shape[0]), threshold = 10, minx=np.min(xi)-100,miny=np.min(yi)-100,maxx=np.max(xi)+100,maxy=np.max(yi)+100,norm="cts",gType="alpha")
        fv.updateImNames(gNameSubSort)
        fv.saveVTK(saveLoc + fpref + '_mesh50_rz_probRNA_t10parts.vtk')
    return

def makeBarSeqGeneMesh(npzName,geneName='/cis/home/kstouff4/Documents/SpatialTranscriptomics/BarSeq/Genes/geneList.npz',maxV=114):
    info = np.load(npzName)
    geneID = info['nu_X']
    coords = info['X']
    saveLoc = npzName.replace('.npz','')
    print("coords shape, ", coords.shape)
    
    geneList = np.load(geneName,allow_pickle=True)
    gNameSubSort = geneList

    xi = coords[:,0]
    yi = coords[:,1]
    cts = np.zeros((coords.shape[0],maxV))
    cts[np.arange(geneID.shape[0]),(geneID-1).astype(int)] = 1
    fv = buildMeshFromCentersCountsMinMax(coords[:,0:2], cts, resolution=100, radius = None, weights=np.ones(coords.shape[0]), threshold = 10,minx=np.min(xi)-100,miny=np.min(yi)-100,maxx=np.max(xi)+100,maxy=np.max(yi)+100,norm="cts",gType="alpha")
    fv.updateImNames(gNameSubSort)
    fv.saveVTK(saveLoc + '_mesh100_rz_probRNA_t10.vtk')
    fv = buildMeshFromCentersCountsMinMax(coords[:,0:2], cts, resolution=50, radius = None, weights=np.ones(coords.shape[0]), threshold = 10, minx=np.min(xi)-100,miny=np.min(yi)-100,maxx=np.max(xi)+100,maxy=np.max(yi)+100,norm="cts",gType="alpha")
    fv.updateImNames(gNameSubSort)
    fv.saveVTK(saveLoc + '_mesh50_rz_probRNA_t5.vtk')
    
    return

    
                 
