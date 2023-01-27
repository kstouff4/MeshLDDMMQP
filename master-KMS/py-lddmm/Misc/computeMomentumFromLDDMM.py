#!/opt/local/bin/python2.6
import sys
from Common import diffeo
import numpy as np

def main():
    if not (len(sys.argv) == 5):
        print 'Usage: computeMomentumFromLDDMM LDDMM_Template.vtk LDDMM_Deformed_Target.vtk LDDMM_map_Jacobian.vtk result.vtk'
        sys.exit(0)
    I0 = diffeo.gridScalars(fileName=sys.argv[1])
    I1 = diffeo.gridScalars(fileName=sys.argv[2])
    jac = diffeo.gridScalars(fileName=sys.argv[3])
    u = np.multiply(jac.data, I1.data - I0.data)
    diffeo.gridScalars(u).saveVTK(sys.argv[4])
    I0.saveVTK('I0.vtk')
    I1.saveVTK('I1.vtk')
    jac.saveVTK('jac.vtk')

if __name__=="__main__":
    main()
