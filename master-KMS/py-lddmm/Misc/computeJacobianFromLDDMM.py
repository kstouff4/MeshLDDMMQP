#!/opt/local/bin/python2.7
import sys
from Common import diffeo


def main():
    if not (len(sys.argv) == 3):
        print 'Usage: computeJacobianFromLDDMM LDDMM_map_file.vtk result.vtk'
        sys.exit(0)
    v = diffeo.Diffeomorphism(filename=sys.argv[1])
    u = diffeo.jacobianDeterminant(v.data, resol = v.resol, periodic=True)
    diffeo.gridScalars(u, resol=v.resol).saveVTK(sys.argv[2])

if __name__=="__main__":
    main()
