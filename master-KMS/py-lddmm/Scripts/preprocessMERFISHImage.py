import ntpath
from sys import path as sys_path
#sys_path.append('..')
#sys_path.append('../base')
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/master/py-lddmm/')
sys_path.append('/cis/home/kstouff4/Documents/MeshRegistration/master/py-lddmm/base')
import os
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


testRun = False
homedir = '/Users/younes/Development/Data/Merfish/allen_data2'
homedir = '/cis/home/kstouff4/Documents/MeshRegistration/TestImages/Allen'
if not os.path.exists(homedir):
    os.mkdir(homedir)
#mouses = glob(homedir + '/0_origdata/mouse2')
#datadir = '/Users/younes/Development/Data/Merfish/allen_data2'

def f(file1, radius=30.):
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

<<<<<<< HEAD
    ugenes, inv = np.unique(df['gene'], return_inverse=True) # inverse = occurrences of elements 
=======
    ugenes, indx, inv = np.unique(df['gene'], return_index=True, return_inverse=True)
>>>>>>> origin
    img1 = buildImageFromFullListHR(x_, y_, inv, radius=radius)
    genes = df['gene'].iloc[indx]
    return img1 + (genes,)

<<<<<<< HEAD
#if __name__ == '__main__':
def processAllenImageMesh(file1,ID):
    loggingUtils.setup_default_logging(stdOutput=True)
    #file1 = homedir + '/202202170851_60988201_VMSC01001/detected_transcripts.csv'
    #file1 = datadir + '/0_origdata/mouse2/202204181456_60988941_VMSC01601/detected_transcripts.csv'
    #file1 = '/cis/home/kstouff4/Documents/SpatialTranscriptomics/Mouse/Mouse1_20220506/zipfiles1//detected_transcripts.csv'
    #ID = '60988223'
=======
if __name__ == '__main__':
    file1 = homedir + '/202202170851_60988201_VMSC01001/detected_transcripts.csv'
    #file1 = homedir + '/202202170915_60988203_VMSC00401/detected_transcripts.csv'
    threshold = 100
    spacing = 10
>>>>>>> origin
    print('building image')
    img, info, genes = f(file1, spacing)
    print(genes)
    img2 = img.sum(axis=2)
    minx = info[0]
    miny = info[1]
    spacing = info[2]
    cm = img2.max()
    imgout = ((255 / cm) * img2).astype(np.uint8)
    print('running watershed')
    seg = (1+watershed(-img2)) * (img2>threshold)
    nlab = seg.max()
    print(f'number of labels: {nlab}')
    print(seg.shape)
    segout = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for j in range(3):
        col = np.zeros(nlab+1, dtype=int)
        col[1:] = 1+np.random.choice(255, nlab)
        print(col.shape, col[seg].shape)
        segout[:, :, j] = col[seg]
    iio.imwrite(homedir + "/" + ID + "_img.png", imgout)
    print(segout.shape)
    cv2.imwrite(homedir + "/" + ID + "_seg.png", segout.astype(np.int8))


    centers = np.zeros((nlab, 2))
    cts = np.zeros((nlab, img.shape[2]), dtype=int)
    print("cts size is " + str(cts.shape))
    nb = np.zeros(nlab, dtype=int)
    x0 = np.linspace(minx+spacing/2, minx + img.shape[0]*spacing - spacing/2, img.shape[0])
    y0 = np.linspace(miny+spacing/2, miny + img.shape[1]*spacing - spacing/2, img.shape[1])
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if seg[i,j] > 1:
                centers[seg[i,j]-1, 0] += x0[i]
                centers[seg[i,j]-1, 1] += y0[j]
                nb[seg[i,j]-1] += 1
                cts[seg[i,j]-1,:] += img[i,j,:].astype(int)
    nbc = cts.sum(axis=1)
    select = (nbc >= threshold) & (nb <= 10)
    centers = centers[select, :]
    cts = cts[select, :]
    centers /= nb[select, None]
    print(f'number of cells: {centers.shape[0]}')
    df = pd.DataFrame(data = {'centers_x':centers[:,0], 'centers_y':centers[:,1]})
<<<<<<< HEAD
    df.to_csv(homedir + '/' + ID + '_centers.csv')
    df = pd.DataFrame(data=cts)
    df.to_csv(homedir + '/' + ID + '_counts.csv')
=======
    df.to_csv(homedir + '/centers.csv')
    df = pd.DataFrame(data=cts, columns=genes)
    df.to_csv(homedir + '/counts.csv')
>>>>>>> origin
    fv = buildMeshFromCentersCounts(centers, cts, resolution=50, radius = None, weights=None, threshold = 10)
    fv.saveVTK(homedir + '/' + ID + '_mesh50.vtk')
    fv = buildMeshFromCentersCounts(centers, cts, resolution=100, radius = None, weights=None, threshold = 10)
    fv.saveVTK(homedir + '/' + ID + '_mesh100.vtk')
    fv = buildMeshFromCentersCounts(centers, cts, resolution=200, radius = None, weights=None, threshold = 10)
    fv.saveVTK(homedir + '/' + ID + '_mesh200.vtk')
    return

    #plt.imshow(imgout)
    #plt.show()
