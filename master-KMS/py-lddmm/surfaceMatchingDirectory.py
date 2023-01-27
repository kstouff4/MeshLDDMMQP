#!/opt/local/bin/python2.7
from os import path
import glob
import argparse
from base.kernelFunctions import *
from surfaceMatching import *


def main():
    parser = argparse.ArgumentParser(description='runs surface matching registration over directories (relative to the template)')
    parser.add_argument('template', metavar='template', type = str, help='template')
    parser.add_argument('dirIn', metavar='dirIn', type = str, help='input directory')
    parser.add_argument('--sigmaKernel', metavar='sigmaKernel', type=float, dest='sigmaKernel', default = 6.5, help='kernel width') 
    parser.add_argument('--sigmaDist', metavar='sigmaDist', type=float, dest='sigmaDist', default = 2.5, help='kernel width (error term); (default = 2.5)') 
    parser.add_argument('--sigmaError', metavar='sigmaError', type=float, dest='sigmaError', default = 1.0, help='std error; (default = 1.0)') 
    parser.add_argument('--typeError', metavar='typeError', type=str, dest='typeError', default = 'measure', help='type error term (default: measure)') 
    parser.add_argument('--pattern', metavar = 'pattern', type = str, dest='pattern', default='*.byu', help='Regular expression for files to process (default: *.byu)')
    parser.add_argument('--dirOut', metavar = 'dirOut', type = str, dest = 'dirOut', default = '', help='Output directory')
    parser.add_argument('--tmpOut', metavar = 'tmpOut', type = str, dest = 'tmpOut', default = '', help='info files directory')
    parser.add_argument('--cont', action = 'store_true', dest='cont', default=False,help='continue previous run (do not update already registered files)')
    args = parser.parse_args()

    if args.dirOut == '':
        args.dirOut = args.dirIn

    if args.tmpOut == '':
        args.tmpOut = args.dirOut + '/tmp'
    if not os.path.exists(args.dirOut):
        os.makedirs(args.dirOut)
    if not os.path.exists(args.tmpOut):
        os.makedirs(args.tmpOut)


    files = glob.glob(args.dirIn+'/'+ args.pattern)
    tmpl = surfaces.Surface(filename=args.template)
    K1 = Kernel(name='gauss', sigma = args.sigmaKernel)
    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=args.sigmaDist, sigmaError=args.sigmaError, errorType=args.typeError)
    tmpl.Simplify(target=1000)

    for name in files:
        print('Processing', name)
        fv = surfaces.Surface(filename=name)
        f = SurfaceMatching(Template=tmpl, Target=fv, outputDir=args.tmpOut,param=sm, testGradient=False,
                            maxIter=1000, affine= 'none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)

        f.optimizeMatching()
        u = path.split(name)
        [nm,ext] = path.splitext(u[1])
        f.fvDef.savebyu(args.dirOut+'/'+nm+'Def.byu')

        #ff.close()
if __name__=="__main__":
    main()
