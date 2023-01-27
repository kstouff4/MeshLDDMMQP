import ntpath
from sys import path as sys_path
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/master/py-lddmm/')
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/master/py-lddmm/base')
sys_path.append('/cis/home/kstouff4/Documents/SurfaceTools/')
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
from base.meshes import buildImageFromFullListHR, buildMeshFromCentersCounts
from sklearn.neighbors import kneighbors_graph

from sknetwork.clustering import Louvain, get_modularity
from sknetwork.linalg import normalize
from sknetwork.utils import get_membership

import vtkFunctions as vtkf

testRun = False
homedir = '/Users/younes/Development/Data/Merfish/allen_data2'
homedir = '/cis/home/kstouff4/Documents/MeshRegistration/TestImages/Allen'
if not os.path.exists(homedir):
    os.mkdir(homedir)
#mouses = glob(homedir + '/0_origdata/mouse2')
#datadir = '/Users/younes/Development/Data/Merfish/allen_data2'

def f(file1, ID, radius=30.):
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
    z_ = df['global_z'].to_numpy()
    centers = np.zeros((x_.shape[0],3))
    centers[:,0] = x_
    centers[:,1] = y_
    numz = len(np.unique(z_))
    centers[:,2] = z_*(25.0/numz)

    ugenes, inv = np.unique(df['gene'], return_inverse=True) # inverse = occurrences of elements 
    img1 = buildImageFromFullListHR(x_, y_, inv, radius=radius)
    np.savez(homedir + '/' + ID + '_geneList.npz',genes=ugenes)
    np.savez(homedir + '/' + ID + '_transcriptPoints.npz',centers=centers,geneIndex=inv)
    return img1

def f3D(file1, ID, radius=30.):
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
    z_ = df['global_z'].to_numpy()
    centers = np.zeros((x_.shape[0],3))
    centers[:,0] = x_
    centers[:,1] = y_
    numz = len(np.unique(z_))
    centers[:,2] = z_*(25.0/numz)

    ugenes, inv = np.unique(df['gene'], return_inverse=True) # inverse = occurrences of elements 
    np.savez(homedir + '/' + ID + '_geneList.npz',genes=ugenes)
    np.savez(homedir + '/' + ID + '_transcriptPoints.npz',centers=centers,geneIndex=inv)
    
    return centers, inv

#if __name__ == '__main__':
def processAllenImageMesh(file1,ID):
    loggingUtils.setup_default_logging(stdOutput=True)
    #file1 = homedir + '/202202170851_60988201_VMSC01001/detected_transcripts.csv'
    #file1 = datadir + '/0_origdata/mouse2/202204181456_60988941_VMSC01601/detected_transcripts.csv'
    #file1 = '/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1//detected_transcripts.csv'
    #ID = '60988223'
    print('building image')
    img, info = f(file1, ID, 15.)
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
    iio.imwrite(homedir + "/" + ID + "_img.png", imgout)
    print(segout.shape)
    cv2.imwrite(homedir + "/" + ID + "_seg.png", segout.astype(np.int8))
    np.savez(file1.replace('.csv','_seg.npz'),img=img,seg=seg,nlab=nlab)


    threshold = 50
    centers = np.zeros((nlab, 2))
    cts = np.zeros((nlab, img.shape[2]), dtype=int)
    print("cts size is " + str(cts.shape))
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
    df.to_csv(homedir + '/' + ID + '_centers.csv')
    df = pd.DataFrame(data=cts)
    df.to_csv(homedir + '/' + ID + '_counts.csv')
    fv = buildMeshFromCentersCounts(centers, cts, resolution=50, radius = None, weights=None, threshold = 10)
    fv.saveVTK(homedir + '/' + ID + '_mesh50.vtk')
    fv = buildMeshFromCentersCounts(centers, cts, resolution=100, radius = None, weights=None, threshold = 10)
    fv.saveVTK(homedir + '/' + ID + '_mesh100.vtk')
    fv = buildMeshFromCentersCounts(centers, cts, resolution=200, radius = None, weights=None, threshold = 10)
    fv.saveVTK(homedir + '/' + ID + '_mesh200.vtk')
    return

    #plt.imshow(imgout)
    #plt.show()
def processAllenImageMeshSubset(subsetNamesO,geneListNPZ,centersCSV,countsCSV,savePref):
    
    counts = np.genfromtxt(countsCSV,delimiter=',')
    counts = counts[1:,1:]
    x = np.load(geneListNPZ,allow_pickle=True)
    genes = x['genes']
    subsetInds = []
    subsetNames = []
    for name in subsetNamesO:
        t = np.where(genes == name)
        if (len(t[0]) > 0):
            subsetInds.append(t[0][0])
            subsetNames.append(name)
    cts = counts[:,subsetInds]

    data = np.genfromtxt(centersCSV,delimiter=',')
    centers = data[1:,1:]
    #centers[:, 0] = data.loc[:, coordinate_columns[0]]
    #centers[:, 1] = data.loc[:, coordinate_columns[1]]
    #cts = ctsSub.to_numpy()
    ID = centersCSV.split('/')[-1].replace('_centers.csv','')
    ID = ID + savePref
    fv = buildMeshFromCentersCounts(centers, cts, resolution=50, radius = None, weights=None, threshold = 0, imNames=subsetNames)
    fv.saveVTK(homedir + '/' + ID + '_mesh50.vtk')
    fv = buildMeshFromCentersCounts(centers, cts, resolution=100, radius = None, weights=None, threshold = 0, imNames=subsetNames)
    fv.saveVTK(homedir + '/' + ID + '_mesh100.vtk')
    fv = buildMeshFromCentersCounts(centers, cts, resolution=200, radius = None, weights=None, threshold = 0, imNames=subsetNames)
    fv.saveVTK(homedir + '/' + ID + '_mesh200.vtk')
    return

def clusterAllenIntoCells(detectedCSV, ID, savedir, numNeighbors=3):
    '''
    Get Cell centers only based on clustering with KNN
    '''
    #data = np.genfromtxt(centersCSV,delimiter=',')
    #centers = data[1:,1:]
    centers, geneInd = f3D(detectedCSV, ID)
    #data = np.load(centersNPZ)
    #centers = data['centers']
    
    #ID = centersNPZ.split('/')[-1].replace('_transcriptPoints.npz','')
    
    A = kneighbors_graph(centers,numNeighbors)
    louvain = Louvain()
    labels = louvain.fit_transform(A)
    #labels_unique, counts = np.unique(labels, return_counts=True)
    cells, cellAssignment = np.unique(labels, return_inverse=True) # inverse = occurrences of elements 
    numCells = len(cells)
    cells, geneCounts = np.unique(labels,return_counts=True)
    average = normalize(get_membership(labels).T)
    print("shape of average should be number of cells by number of transcripts")
    print(average.shape)
    print("values should just be 0 or 1")
    print(np.unique(average))
    position_aggregate = average.dot(centers)
    np.savez(savedir + ID + '_clusteringNN' + str(numNeighbors) + '.npz',cellCenters=position_aggregate, cellAssignment=cellAssignment,geneCounts=geneCounts)
    vtkf.writeVTK(position_aggregate,[geneCounts,np.log(1+geneCounts)/np.log(10)],['GENE_COUNTS','LOG_GENE_COUNTS'],savedir + ID + '_clusteringNN' + str(numNeighbors) + '.vtk',polyData=None)
    
    cellsByGene = np.zeros((numCells,np.max(geneInd)+1))
    genes = (average > 0).multiply(geneInd[None,...]+1)
    for c in range(np.max(geneInd)+1):
        #cellsByGene[c,np.squeeze(geneInd[average[c,:]])] += 1 # add 1 count for every selected point
        cellsByGene[:,c] = np.squeeze(np.sum(genes == (c+1),axis=-1))
    np.savez(savedir + ID + '_cellsByGene.npz',cellsByGene=cellsByGene)
    
    return
    
    