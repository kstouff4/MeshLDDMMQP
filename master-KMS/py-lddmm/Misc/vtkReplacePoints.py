from Surfaces import surfaces
from vtk import *
import argparse

def main():
    parser = argparse.ArgumentParser(description='replaces points from a vtk file by those read from a byu or vtk file')
    parser.add_argument('vtkFile', metavar='vtkFile', type = str, help='original vtk file')
    parser.add_argument('pointFile', metavar='pointFile', type = str, help='input file with new points')
    parser.add_argument('outputFile', metavar='outputFile', type = str, help='output file')
    args = parser.parse_args()

    u = vtkPolyDataReader()
    u.SetFileName(args.vtkFile)
    u.Update()
    v = u.GetOutput()

    npoints = int(v.GetNumberOfPoints())
    nfaces = int(v.GetNumberOfPolys())

    fv = surfaces.Surface(filename=args.pointFile)

    if ((fv.vertices.shape[0] != npoints) or (fv.faces.shape[0] != nfaces)):
        print 'surfaces are not compatible'
        return

    z = v.GetPoints()

    for k in range(npoints):
        z.SetPoint(k, fv.vertices[k,:])

    a = vtkPolyDataWriter()
    a.SetFileName(args.outputFile)
    a.SetInput(v)
    a.Write()

if __name__=="__main__":
    main()



