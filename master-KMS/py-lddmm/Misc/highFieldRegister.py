#!/opt/local/bin/python2.7
from os import path
import argparse
from Common.kernelFunctions import *
from affineRegistration import *
from Surfaces.surfaceMatching import *


def main():
    parser = argparse.ArgumentParser(description='runs surface matching registration and exports labels from subregions')
    parser.add_argument('target', metavar='target', type = str, help='target')
    parser.add_argument('highfield', metavar='highfield', type = str, help='highfield (global)')
    parser.add_argument('highfield_parts', metavar='highfield_parts', nargs='+', type = str, help='highfield segments (list)')
    parser.add_argument('--sigmaKernel', metavar='sigmaKernel', type=float, dest='sigmaKernel', default = 6.5, help='kernel width') 
    parser.add_argument('--sigmaDist', metavar='sigmaDist', type=float, dest='sigmaDist', default = 2.5, help='kernel width (error term); (default = 2.5)') 
    parser.add_argument('--sigmaError', metavar='sigmaError', type=float, dest='sigmaError', default = 1.0, help='std error; (default = 1.0)') 
    parser.add_argument('--typeError', metavar='typeError', type=str, dest='typeError', default = 'measure', help='type error term (default: measure)') 
    parser.add_argument('--dirOut', metavar = 'dirOut', type = str, dest = 'dirOut', default = '', help='Output directory')
    parser.add_argument('--tmpOut', metavar = 'tmpOut', type = str, dest = 'tmpOut', default = '', help='info files directory')
    parser.add_argument('--initialRotation', metavar = 'initRot', type = float, dest = 'initRot', default = (0,0,0), nargs=3, help='theta and phi for initial rotation')
    parser.add_argument('--flip', action = 'store_true', dest = 'flip', default = False, help='flip before rigid registration')
    
    args = parser.parse_args()

    if args.dirOut == '':
        args.dirOut = '.'

    if args.tmpOut == '':
        args.tmpOut = args.dirOut + '/tmp'
    if not os.path.exists(args.dirOut):
        os.makedirs(args.dirOut)
    if not os.path.exists(args.tmpOut):
        os.makedirs(args.tmpOut)


    targ = surfaces.Surface(filename=args.target)
    hf = surfaces.Surface(filename=args.highfield)
    hfSeg = []
    for name in args.highfield_parts:
        hfSeg.append(surfaces.Surface(filename=name))
    nsub = len(hfSeg)
    nvSeg = np.zeros(nsub)
    
    for k in range(nsub):
        nvSeg[k] = np.int_(hfSeg[k].vertices.shape[0]) ;
    


    # Find Labels
    nv = hf.vertices.shape[0] 
    #print 'vertices', nv, nvSeg
    dist = np.zeros([nv,nsub])
    for k in range(nsub):
        dist[:,k] = ((hf.vertices.reshape(nv, 1, 3) - hfSeg[k].vertices.reshape(1, nvSeg[k], 3))**2).sum(axis=2).min(axis=1)
    hfLabel = 1+dist.argmin(axis=1) ;
    K1 = Kernel(name='gauss', sigma = args.sigmaKernel)
    sm = SurfaceMatchingParam(timeStep=0.1, KparDiff=K1, sigmaDist=args.sigmaDist, sigmaError=args.sigmaError, errorType=args.typeError)

    psi, phi, th = np.array(args.initRot) * (np.pi/180)
    cth = np.cos(th)
    cphi = np.cos(phi)
    cpsi = np.cos(psi)
    sth = np.sin(th)
    sphi = np.sin(phi)
    spsi = np.sin(psi)
    Rinit = np.array([[cphi, -sphi*cpsi,sphi*spsi],[cth*sphi,cth*cphi*cpsi,-cth*cphi*spsi],[sth*sphi,sth*cphi*cpsi,cth*cpsi]]) 
    #print Rinit
    u = path.split(args.highfield)
    [nm,ext] = path.splitext(u[1])
    hf.vertices = np.dot(hf.vertices, Rinit.T) 
    hf.saveVTK(args.dirOut+'/'+nm+'Init.vtk', scalars=hfLabel, scal_name='Labels')

    R0, T0 = rigidRegistration(surfaces = (hf.vertices, targ.vertices),  rotWeight=1.0, rotationOnly=True, flipMidPoint=args.flip, verb=False, temperature=1., annealing=True, translationOnly=False)
    hf.vertices = np.dot(hf.vertices, R0.T) + T0
    #print hfLabel
    hf.saveVTK(args.dirOut+'/'+nm+'.vtk', scalars=hfLabel, scal_name='Labels')

    print 'Starting Matching'
    f = SurfaceMatching(Template=hf, Target=targ, outputDir=args.tmpOut,param=sm, testGradient=False,
                        maxIter=1000, affine= 'none', rotWeight=1., transWeight = 1., scaleWeight=10., affineWeight=100.)
    f.optimizeMatching()
    u = path.split(args.target)
    [nm,ext] = path.splitext(u[1])
    f.fvDef.savebyu(args.dirOut+'/'+nm+'Def.byu')
    f.fvDef.saveVTK(args.dirOut+'/'+nm+'Def.vtk', scalars=hfLabel, scal_name='Labels')
    nvTarg = targ.vertices.shape[0]
    closest = np.int_(((f.fvDef.vertices.reshape(nv, 1, 3) - targ.vertices.reshape(1, nvTarg, 3))**2).sum(axis=2).argmin(axis=0))
    targLabel = np.zeros(nvTarg) ;
    for k in range(nvTarg):
        targLabel[k] = hfLabel[closest[k]]
    targ.saveVTK(args.dirOut+'/'+nm+'.vtk', scalars=targLabel, scal_name='Labels')
        

if __name__=="__main__":
    main()
