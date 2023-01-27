import csv
from base import loggingUtils, surfaces
from base.kernelFunctions import *
from base import surfaceMatching, surfaceTimeSeries as match


def runLongitudinalSurface(minL=3, atrophy=False):
    if atrophy:
        pass
    else:
        pass

    
    with open('/cis/home/younes/MATLAB/shapeFun/CA_STUDIES/PREDICT/predictFiles.txt','r') as csvf:
        rdr = list(csv.DictReader(csvf,delimiter=',',fieldnames=('filename','cap','group')))
        files = []
        group = np.zeros(len(rdr), dtype=long)
        cap = np.zeros(len(rdr))
        k=0
        for row in rdr:
           cap[k] = row['cap']
           group[k] = row['group']
           files += [row['filename']]
           k+=1
    print(files[0:5])
    print(cap[0:5])
    I1 = np.nonzero(np.logical_and(group==2,cap>0))[0][0:3]
    I2 = np.nonzero(np.logical_and(group==3,cap>0))[0][0:4]
    I3 = np.nonzero(np.logical_and(group==4,cap>0))[0][0:3]
    I = np.concatenate((I1, I2, I3))
    cap = cap[I]
    
    outputDir = '/Users/younes/Development/Results/predictTS/piecewiseAtrophy'
    #outputDir = '/cis/home/younes/MorphingData/twoBallsStitched'
    #outputDir = '/Users/younes/Development/Results/tight_stitched_rigid2_10'
    if __name__ == "__main__":
        loggingUtils.setup_default_logging(outputDir, fileName='info')
    else:
        loggingUtils.setup_default_logging(fileName='info')

    fv0 = surfaces.Surface(filename='/cis/project/predict/phase_2/07_surface_meshes_mapping.20130520_valid_surfaces_only/putamen/4_create_population_based_template/newTemplate.byu')
    K1 = Kernel(name='laplacian', sigma = 6.5, order=4)
    sm = surfaceMatching.SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=2.5, sigmaError=1., errorType='varifold')

    It = np.argsort(cap)
    fv = []
    j = 0
    t = []
    for i0 in range(len(It)):
        i = It[i0]
        print(i, files[i], cap[i])
        try:
            fv += [surfaces.Surface(filename=files[I[i]] + '.byu')]
            t += [cap[i]]
            j += 1
        except NameError as e:
            print(e)

    t = np.array(t)
    t = 1 +  4 * (t-t.min())/(t.max() - t.min())
    print(t)
    if atrophy:
        f = match.SurfaceMatching(Template=fv0, Targets=fv, outputDir=outputDir, param=sm, regWeight=.1, times = t,
                                        affine='euclidean', testGradient=True, affineWeight=.1,  maxIter_cg=1000, mu=0.0001)
    else:
        f = match.SurfaceMatching(Template=fv0, Targets=fv, outputDir=outputDir, param=sm, regWeight=.1, times=t,
                                        affine='euclidean', testGradient=True, affineWeight=.1,  maxIter=1000)
    f.optimizeMatching()


if __name__=="__main__":
    runLongitudinalSurface(atrophy=True)
