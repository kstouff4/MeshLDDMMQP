#! /usr/bin/env python
import csv
import argparse
from Common import loggingUtils
from Common.kernelFunctions import *
from Surfaces import surfaceMatching, surfaces


def runLongitudinalSurface(template, targetList, minL=3,atrophy=False, resultDir='.'):
    if atrophy:
        pass
    else:
        pass

    
    with open(targetList,'r') as csvf:
        rdr = list(csv.DictReader(csvf,delimiter=',',fieldnames=('lab','isleft','id','filename')))
        files = []
        previousLab = 0
        currentFile = []
        for row in rdr:
            if int(row['lab']) == previousLab:
                if int(row['isleft']) == 1:
                    if len(row['filename']) > 0:
                        currentFile += [row['filename']]
            else:
                if len(currentFile) >= minL:
                    files +=[currentFile]
                if int(row['isleft']) == 1:
                    if len(row['filename']) > 0:
                        currentFile = [row['filename']]
                else:
                    currentFile = [] ;
                previousLab = int(row['lab'])
    print len(files)
    return                  

    fv0 = surfaces.Surface(filename=template)
    K1 = Kernel(name='laplacian', sigma = 6.5, order=4)
    sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=2.5, sigmaError=1., errorType='varifold')

    #files = [files[1],files[5],files[8]]
    #files = [files[9]]
    #selected = range(len(files)) 
    selected = (83,86,90) 
    logset = False
    for k in selected:
        s = files[k]
        fv = []
        print s[0]
        for fn in s:
                try:
                    #fv += [surfaces.Surface(filename=fn+'.byu')]
                    fv += [surfaces.Surface(filename=fn + '.byu')]
                except NameError as e:
                    print e
  

        outputDir = resultDir +str(k)
        info_outputDir = outputDir
        if __name__ == "__main__" and (not logset):
            loggingUtils.setup_default_logging(info_outputDir, fileName='info', stdOutput=True)
        else:
            loggingUtils.setup_default_logging(fileName='info')

        try:
            if atrophy:
                f = match.SurfaceMatching(Template=fv0, Targets=fv, outputDir=outputDir, param=sm, regWeight=.1,
                                            affine='euclidean', testGradient=False, affineWeight=.1,  maxIter_cg=50, maxIter_al=50, mu=0.0001)
            else:
                f = match.SurfaceMatching(Template=fv0, Targets=fv, outputDir=outputDir, param=sm, regWeight=.1,
                                        affine='euclidean', testGradient=False, affineWeight=.1,  maxIter=1000)
        except NameError:
            print 'exception'
 
        try:
            f.optimizeMatching()
        except NameError:
            print 'Exception'
 

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='runs longitudinal surface matching based on an input file')
    parser.add_argument('template', metavar='template', type = str, help='template')
    parser.add_argument('targetList', metavar='targetlist', type = str, help='file containing the list of targets')
    parser.add_argument('--results', metavar = 'resultDir', type = str, dest = 'resultDir', default = '.', help='Output directory')
    args = parser.parse_args()
    
    #template: /cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/4_create_population_based_template/newTemplate.byu'
    #targetList: '/cis/home/younes/MATLAB/shapeFun/CA_STUDIES/BIOCARD/filelist.txt'
    #Results: '/cis/home/younes/Results/biocardTS/withAtrophy'
    
    
    runLongitudinalSurface(args.template, args.targetList, atrophy=True, resultDir=args.resultDir)
