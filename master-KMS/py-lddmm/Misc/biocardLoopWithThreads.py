import csv
import argparse
from Common import loggingUtils
from Surfaces.surfaces import *
from Common.kernelFunctions import *
from Surfaces import surfaceMatching, surfaces, surfaceTimeSeries as match
import threading
import Queue

def threadfun(q):
    while True:
        print str(q.qsize())+' jobs left'
        os.environ['OMP_NUM_THREADS'] = '4'
        f = q.get()
        try:
            f.optimizeMatching()
            #break
        except NameError:
            print 'Exception'
        q.task_done()

def runLongitudinalSurface(template, targetList, minL=3,atrophy=False, resultDir='.'):
    if atrophy:
        pass
    else:
        pass

    max_proc = 12
    
    with open('/cis/home/younes/MATLAB/shapeFun/CA_STUDIES/BIOCARD/filelist.txt','r') as csvf:
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
                
    info_outputDir = '/cis/home/younes/Results/biocardTS/infoDir'
    #outputDir = '/cis/home/younes/MorphingData/twoBallsStitched'
    #outputDir = '/Users/younes/Development/Results/tight_stitched_rigid2_10'

    fv0 = surfaces.Surface(filename='/cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/4_create_population_based_template/newTemplate.byu')
    K1 = Kernel(name='laplacian', sigma = 6.5, order=4)
    sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=2.5, sigmaError=1., errorType='varifold')

    q = Queue.Queue()
    #files = [files[1],files[5],files[8]]
    selected = range(len(files)) 
    #selected = (1,9) 
    for k in selected:
        s = files[k]
        fv = []
        print s[0]
        for fn in s:
                try:
                    fv += [surfaces.Surface(filename=fn + '.byu')]
                except NameError as e:
                    print e
  

        outputDir = '/cis/home/younes/Results/biocardTS/piecewise_NA_'+str(k)
        if __name__ == "__main__":
            loggingUtils.setup_default_logging(info_outputDir, fileName='info', stdOutput=(k == selected[0]))
        else:
            loggingUtils.setup_default_logging(fileName='info')

        try:
            if atrophy:
                f = match.SurfaceMatching(Template=fv0, Targets=fv, outputDir=outputDir, param=sm, regWeight=.1,
                                            affine='euclidean', testGradient=False, affineWeight=.1,  maxIter_cg=1000, mu=0.0001)
            else:
                f = match.SurfaceMatching(Template=fv0, Targets=fv, outputDir=outputDir, param=sm, regWeight=.1,
                                        affine='euclidean', testGradient=False, affineWeight=.1,  maxIter=1000)
        except NameError:
            print 'exception'
 
        #, affine='none', rotWeight=0.1))
        q.put(f)

    for k in range(max_proc):
        w = threading.Thread(target=threadfun, args=(q,))
        w.setDaemon(True)
        w.start()
        #f.optimizeMatching()

    q.join()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='runs longitudinal surface matching based on an input file')
    parser.add_argument('template', metavar='template', type = str, help='template')
    parser.add_argument('targetList', metavar='targetlist', type = str, help='file containing the list of targets')
    parser.add_argument('--results', metavar = 'resultDir', type = str, dest = 'resultDir', default = '.', help='Output directory')
    args = parser.parse_args()
    
    # /cis/project/biocard/data/2mm_complete_set_surface_mapping_10212012/hippocampus/4_create_population_based_template/newTemplate.byu /cis/home/younes/MATLAB/shapeFun/CA_STUDIES/BIOCARD/filelist.txt --results /cis/home/younes/Results/biocardTS/withAtrophy
    
    
    runLongitudinalSurface(args.template, args.targetList, atrophy=True, resultDir=args.resultDir)
