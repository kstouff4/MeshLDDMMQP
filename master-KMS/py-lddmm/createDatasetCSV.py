import pandas as pd
import os
from os import path
from glob import glob
from base.shapeAnalysisPipeline import Pipeline
from base import loggingUtils

def makeFileList(dataDir):
    files = glob(dataDir+ '/*.vtk')
    dct = {'id':[], 'path_left':[], 'path_right':[], 'path_left_lmk':[], 'path_right_lmk':[], 'visit_number':[]}
    id_left = []
    id_right = []
    path_left = []
    path_right = []
    path_left_lmk = []
    path_right_lmk = []
    N = len(files)
    for f in files:
        dir, fn = path.split(f)
        n, e = path.splitext(fn)
        j = n.find('img') + 3
        id = n[j:]
        if 'left' in n:
            id_left.append(id)
            path_left.append(f)
            path_left_lmk.append(dir + '/landmarks/'+n[:(j-3)]+'lmk'+id+'.lmk')
        elif 'right' in n:
            id_right.append(id)
            path_right.append(f)
            path_right_lmk.append(dir + '/landmarks/'+n[:(j-3)]+'lmk'+id+'.lmk')
            #path_right_lmk.append('none')



    # dfl = pd.DataFrame({'id':id_left, 'path_left':path_left, 'path_left_lmk':path_left_lmk}, index=id_left)
    # dfr = pd.DataFrame({'id':id_right, 'path_right':path_right, 'path_right_lmk':path_right_lmk}, index=id_right)
    dfl = pd.DataFrame({'path_left':path_left, 'path_left_lmk':path_left_lmk}, index=id_left)
    dfr = pd.DataFrame({'path_right':path_right, 'path_right_lmk':path_right_lmk}, index=id_right)

    df = pd.concat((dfl, dfr), axis=1, sort=True)


    df.to_csv(dataDir+'/fileList.csv')




if __name__=="__main__":
    dataDir =  os.getenv('HOME') + "/OneDrive - Johns Hopkins University/RESEARCH/DATA/Dataset"

    #makeFileList(dataDir)
    pip = Pipeline(dataDir+'/fileList.csv', dataDir+'/Pipeline')
    loggingUtils.setup_default_logging(pip.dirOutput, fileName='info.txt', stdOutput = True)
    pip.Step1_Isosurface()
    pip.Step2_Rigid()
    pip.Step3_Template()
    pip.Step4_Registration()