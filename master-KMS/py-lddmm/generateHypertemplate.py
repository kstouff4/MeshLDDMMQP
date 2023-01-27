import glob
import argparse
import numpy as np
from base import surfaces

try:
    from vtk import *
    gotVTK = True
except ImportError:
    print('could not import VTK functions')
    gotVTK = False


def main():
    if not gotVTK:
        raise Exception('Cannot run function without VTK')

    parser = argparse.ArgumentParser(description='Computes hypertemplate: chooses surface from a directory with average volume and retriangulates it as a smoother volume')
    parser.add_argument('dirIn', metavar='dirIn', type = str, help='input directory')
    parser.add_argument('fileout', metavar='fileout', type = str, help='output file') 
    parser.add_argument('--pattern', metavar = 'pattern', type = str, dest='pattern', default='*.byu', help='Regular expression for files to process (default: *.byu)')
    parser.add_argument('--targetSize', metavar = 'targetSize', type = int, dest = 'targetSize', default = 1000, help='targeted number of vertices')
    parser.add_argument('--imageDisc', metavar = 'imageDisc', type = int, dest = 'disc', default = 100, help='discretization step for triangulation')
    parser.add_argument('--smooth', metavar = 'smooth', type = float, dest = 'smooth', default = 0.1, help='smoothing parameter before decimation')

    args = parser.parse_args()

    sf = surfaces.Surface()
    files = glob.glob(args.dirIn+'/'+args.pattern)
    z = np.zeros(len(files))
    k=0
    for name in files:
        fv = surfaces.Surface(filename = name)
        z[k] = np.fabs(fv.surfVolume())
        print(name, z[k])
        k+=1

    mean = z.sum() / z.shape[0]
    #print mean
    k0 = np.argmin(np.fabs(z-mean))
    fv = surfaces.Surface(filename = files[k0])
    print('keeping ' + files[k0])
    minx = fv.vertices[:,0].min() 
    maxx = fv.vertices[:,0].max() 
    miny = fv.vertices[:,1].min() 
    maxy = fv.vertices[:,1].max() 
    minz = fv.vertices[:,2].min() 
    maxz = fv.vertices[:,2].max()

    delta = np.array([maxx-minx, maxy-miny, maxz-minz]).max()

    dx = delta/ args.disc
    #dy = (maxy-miny)/ args.disc
    #dz = (maxz-minz)/ args.disc
    resol = [dx, dx, dx] 
    origin = [minx-10*dx, miny-10*dx, minz-10*dx] 

    g = fv.toPolyData()
    h = vtkImplicitPolyDataDistance()
    h.SetInput(g)

    grd = np.mgrid[(minx-10*dx):(maxx+10*dx):dx, miny-10*dx:maxy+10*dx:dx, minz-10*dx:maxz+10*dx:dx]
    img = np.zeros([grd.shape[1], grd.shape[2], grd.shape[3]])
    for k1 in range(img.shape[0]):
        for k2 in range(img.shape[1]):
            for k3 in range(img.shape[2]):
                img[k1,k2,k3] = h.EvaluateFunction(grd[:,k1,k2,k3])

                #[u,v] = path.splitext(args.fileout) ;
    #diffeo.gridScalars(data=img,resol=resol,origin=origin).saveVTK(u+'.vtk')

    fv.Isosurface(img, 0., target = args.targetSize, smooth=args.smooth)
    fv.vertices = np.array([minx, miny, minz]) - 10*dx + dx * fv.vertices

    fv.savebyu(args.fileout) 


if __name__=="__main__":
    main()
