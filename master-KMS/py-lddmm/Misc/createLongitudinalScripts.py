#! /usr/bin/env python
import os.path
import glob
#import argparse
import subprocess



def createLongitudinalSurfaceScripts(minL=3):

    #Directory where the data is. One subfolder per subject.
    targetDir = '/cis/home/younes/MorphingData/TimeseriesResults/'
    #python directory for registration
    source = '/cis/home/younes/github/registration/registration'
    allDir = glob.glob(targetDir+'*')
    for d in allDir:
        nscan = 0 
        while os.path.exists(d+'/imageOutput_time_{0:d}_channel_0.vtk'.format(nscan)):
            nscan += 1
        print d, nscan
        if nscan >= minL:
            #Creates script in this subfolder 
            shname = targetDir+'scripts/s'+ os.path.basename(d) +'.sh'
            with open(shname, 'w') as fname:
                fname.write('#!/bin/bash\n')
                fname.write('#$ -cwd\n')
                fname.write('#$ -j y\n')
                fname.write('#$ -S /bin/bash\n')
                fname.write('#$ -pe orte 8\n')
                fname.write('#$ -M laurent.younes@jhu.edu\n')
                fname.write('#$ -o /dev/null\n')
                fname.write('cd '+ source +'\n')
                
                ### Switch comments according to program
                ## L2TimesSeries: first order LDDMM. Use --atrophy keyword to enforce atrophy constraint
                ## L2TimeSeriesSecondOrder: spline LDDMM. Use --geodesic keyword to run geodesic regression
                ## --rescale matches the template volume to the baseline
                #fname.write('python L2TimeSeries.py ' +  os.path.basename(d) + ' --atrophy --rescale\n')
                fname.write('python L2TimeSeriesSecondOrder.py ' +  os.path.basename(d) + '--geodesic  --rescale\n')
            cstr = "qsub  " + shname
            print cstr
            #comment out this line to create the scrpits without starting them
            subprocess.call(cstr, shell=True)


 

if __name__=="__main__":
    createLongitudinalSurfaceScripts()
