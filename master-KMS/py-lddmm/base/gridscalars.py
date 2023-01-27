import os
import pickle
import sys
import struct
import numpy as np
from imageio import imread, imwrite
import nibabel


try:
    from vtk import vtkStructuredPointsReader
    import vtk.util.numpy_support as v2n
    gotVTK = True
except ImportError:
    print('could not import VTK functions')
    gotVTK = False


class GridScalars:
    # initializes either form a previous array (data) or from a file
    def __init__(self, grid=None, dim = 3, resol = 1., origin= 0., force_axun=False, withBug=False):
        if type(grid) is np.ndarray:
            self.data = np.copy(grid)
            self.affine = np.eye(dim+1)
            if type(resol) == float:
                self.resol = resol *np.ones(dim)
            else:
                self.resol = np.copy(resol)
            if type(origin) == float:
                self.origin = origin*np.ones(dim)
            else:
                self.origin = np.copy(origin)
            self.dim = dim
        elif type(grid) is str:
            self.dim = dim
            if type(resol) == float:
                self.resol = resol *np.ones(dim)
            else:
                self.resol = np.copy(resol)
            if type(origin) == float:
                self.origin = origin*np.zeros(dim)
            else:
                self.origin = np.copy(origin)
            self.read(grid, force_axun, withBug)
        elif issubclass(type(grid), GridScalars):
            self.data = np.copy(grid.data)
            self.resol = np.copy(grid.resol)
            self.origin = np.copy(grid.origin)
            self.dim = grid.dim
        else:
            self.data = np.empty(0)
            self.centers = np.empty(0)
            self.faces = np.empty(0)
            self.surfel = np.empty(0)
            self.component = np.empty(0)



    def read(self, fileName, force_axun=False, withBug=False):
            if (self.dim == 1):
                with open(fileName, 'r') as ff:
                    ln0 = ff.readline()
                    while (len(ln0) == 0) | (ln0=='\n'):
                        ln0 = ff.readline()
                    ln = ln0.split()
                    nb = int(ln[0])
                    self.data = np.zeros(nb)
                    j = 0
                    while j < nb:
                        ln = ff.readline().split()
                        for u in ln:
                            self.data[j] = u
                            j += 1
            elif (self.dim==2):
                img=imread(fileName).astype(float)
                if (img.ndim == 3):
                    img = img.sum(axis=2)/(img.shape[2]+1)
                self.data = img
            elif (self.dim == 3):
                (nm, ext) = os.path.splitext(fileName)
                if ext=='.hdr' or ext=='.img':
                    self.loadAnalyze(fileName, force_axun= force_axun, withBug=withBug)
                elif ext =='.vtk':
                    self.readVTK(fileName)
            else:
                print("get_image: unsupported input dimensions")
                return
            self.data = np.squeeze(self.data)
            if len(self.data.shape) > self.dim:
                self.data = np.mean(self.data, axis=0)

    def save(self, filename):
        with open(filename, 'w') as f:
            pickle.dump(self, f)

    def load(self, filename):
        with open(filename, 'r') as f:
            pickle.load(self, f)

    def saveImg(self, filename, normalize=False):
        saveImage(self.data, filename, normalize=normalize, origin=self.origin, resol=self.resol)

    # Reads from vtk file
    def readVTK(self, filename):
        if gotVTK:
            u = vtkStructuredPointsReader()
            u.SetFileName(filename)
            u.Update()
            v = u.GetOutput()
            dim = np.zeros(3)
            dim = v.GetDimensions()
            self.origin = v.GetOrigin()
            self.resol = v.GetSpacing()
            self.data = np.ndarray(shape=dim, order='F', buffer = v.GetPointData().GetScalars())
            self.dim = 3
        else:
            raise Exception('Cannot run readVTK without VTK')

    # Saves in vtk file
    # def saveVTK(self, filename, scalarName='scalars_', title='lddmm data'):
    #     with open(filename, 'w') as ff:
    #         ff.write('# vtk DataFile Version 2.0\n'+title+'\nBINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS {0: d} {1: d} {2: d}\n'.format(self.data.shape[0], self.data.shape[1], self.data.shape[2]))
    #         nbVox = np.array(self.data.shape).prod()
    #         ff.write('ORIGIN {0: f} {1: f} {2: f}\nSPACING {3: f} {4: f} {5: f}\nPOINT_DATA {6: d}\n'.format(self.origin[0], self.origin[1], self.origin[2], self.resol[0], self.resol[1], self.resol[2], nbVox))
    #         ff.write('SCALARS '+scalarName+' double 1\nLOOKUP_TABLE default\n')
    #         if sys.byteorder[0] == 'l':
    #             tmp = self.data.byteswap()
    #             tmp.tofile(ff)
    #         else:
    #             self.data.T.tofile(ff)
    #
    # # Saves in analyze file
    # def saveAnalyze(self, filename):
    #     nibabel.AnalyzeImage(self.data, None).to_filename(filename)
        # [nm, ext] = os.path.splitext(filename)
        # with open(nm+'.hdr', 'wb') as ff:
        #     x = 348
        #     self.header = np.zeros(250, dtype=int)
        #     ff.write(struct.pack('i', x))
        #     self.header[33] = self.data.shape[0]
        #     self.header[34] = self.data.shape[1]
        #     self.header[35] = self.data.shape[2]
        #     self.header[53] = 16
        #     self.header[57:60] = self.resol
        #     self.header[178] = 1
        #     frmt = 28*'B'+'i'+'h'+2*'B'+8*'h'+12*'B'+4*'h'+16*'f'+2*'i'+168*'B'+8*'i'
        #     ff.write(struct.pack(frmt, *self.header.tolist()))
        # with open(nm+'.img', 'wb') as ff:
        #     print(self.data.max())
        #     #array.array('f', self.data[::-1,::-1,::-1].T.flatten()).tofile(ff)
        #     array.array('f', self.data.T.flatten()).tofile(ff)
        #     #uu = self.data[::-1,::-1,::-1].flatten()
        #     #print uu.max()
        #     #uu.tofile(ff)

    # Reads in analyze file
    def loadAnalyze(self, filename, force_axun=False, withBug=False):
        if not force_axun and not withBug:
            img = nibabel.load(filename)
            self.data = img.get_fdata()
        else:
            [nm, ext] = os.path.splitext(filename)
            with open(nm+'.hdr', 'rb') as ff:
                frmt = 28*'B'+'i'+'h'+2*'B'+8*'h'+12*'B'+4*'h'+16*'f'+2*'i'+168*'B'+8*'i'
                lend = '<'
                ls = struct.unpack(lend+'i', ff.read(4))
                x = int(ls[0])
                #print 'x=',x
                if not (x == 348):
                    lend = '>'
                ls = struct.unpack(lend+frmt, ff.read())
                self.header = np.array(ls)
                #print ls

                sz = ls[33:36]
                #print sz
                datatype = ls[53]
                #print  "Datatype: ", datatype
                self.resol = ls[57:60]
                self.hist_orient = ls[178]
                if force_axun:
                    self.hist_orient = 0
                if withBug:
                    self.hist_orient = 0
                print("Orientation: ", int(self.hist_orient))
            #
            #
            with open(nm+'.img', 'rb') as ff:
                nbVox = sz[0]*sz[1]*sz[2]
                s = ff.read()
                #print s[0:30]
                if datatype == 2:
                    ls2 = struct.unpack(lend+nbVox*'B', s)
                elif datatype == 4:
                    ls2 = struct.unpack(lend+nbVox*'h', s)
                elif datatype == 8:
                    ls2 = struct.unpack(lend+nbVox*'i', s)
                elif datatype == 16:
                    ls2 = struct.unpack(lend+nbVox*'f', s)
                elif datatype == 32:
                    ls2 = struct.unpack(lend+2*nbVox*'f', s)
                    print('Warning: complex input not handled')
                elif datatype == 64:
                    ls2 = struct.unpack(lend+nbVox*'d', s)
                else:
                    print('Unknown datatype')
                    return

                #ls = np.array(ls)
                #print ls
            #print 'size:', sz
            self.data = np.float_(ls2)
            self.data.resize(sz[::-1])
            #self.data = img.get_fdata().astype(float)
            #print 'size:', self.data.shape
            self.data = self.data.T
            #self.resol = [img.affine[0,0], img.affine[1,1], img.affine[2,2]]
            #self.affine = img.affine
            #return
            #self.data = self.data[::-1,::-1,::-1]
            #print 'size:', self.data.shape
            #print self.resol, ls[57]
            if self.hist_orient == 1:
                # self.resol = [ls[57],ls[58], ls[59]]
                self.resol = [ls[58],ls[57], ls[59]]
                #self.data = self.data.swapaxes(1,2)
                #self.data = self.data[::-1,::-1,::-1].swapaxes(1,2)
                #print self.resol
                #print self.data.shape
            elif self.hist_orient == 2:
                self.resol = [ls[58],ls[59], ls[57]]
                self.data = self.data[::-1,::-1,::-1].swapaxes(0,1).swapaxes(1,2)
            elif self.hist_orient == 3:
                self.resol = [ls[57],ls[58], ls[59]]
                self.data  = self.data[:, ::-1, :]
            elif self.hist_orient == 4:
                self.resol = [ls[58],ls[57], ls[59]]
                self.data = self.data[:,  ::-1, :].swapaxes(0,1)
            elif self.hist_orient == 5:
                self.resol = [ls[58],ls[59], ls[57]]
                self.data = self.data[:,::-1,:].swapaxes(0,1).swapaxes(1,2)
            else:
                self.resol = [ls[57],ls[58], ls[59]]
                if withBug:
                    self.data  = self.data[::-1,::-1,::-1]
                    #self.saveAnalyze('Data/foo.hdr')
        return


    def zeroPad(self, h):
        d = np.copy(self.data)
        self.data = np.zeros([d.shape[0] + 2, d.shape[1]+2, d.shape[2]+2])
        self.data[1:d.shape[0]+1, 1:d.shape[1]+1, 1:d.shape[2]+1] = d


def saveImage(img, filename, normalize=False, origin = (0,0,0), resol = (1.,1.,1.)):
    if len(img.shape) == 2:
        if normalize:
            src = 255 * (img - img.min()) / (img.max()-img.min())
        else:
            src = img
        imwrite(filename+'.png', src.astype(np.uint8))
    else:
        [u, v] = os.path.splitext(filename)
        if v=='.vtk':
            saveVTK(img, filename, origin=origin, resol=resol)
        else:
            saveAnalyze(img, filename)

def saveVTK(img, filename, scalarName='scalars_', title='lddmm data', origin = (0,0,0), resol = (1.,1.,1.)):
    with open(filename, 'w') as ff:
        ff.write('# vtk DataFile Version 2.0\n'+title+
                 '\nBINARY\nDATASET STRUCTURED_POINTS\nDIMENSIONS {0: d} {1: d} {2: d}\n'.format(img.shape[0], img.shape[1], img.shape[2]))
        nbVox = np.array(img.shape).prod()
        ff.write('ORIGIN {0: f} {1: f} {2: f}\nSPACING {3: f} {4: f} {5: f}\nPOINT_DATA {6: d}\n'.format(origin[0], origin[1], origin[2], resol[0], resol[1], resol[2], nbVox))
        ff.write('SCALARS '+scalarName+' double 1\nLOOKUP_TABLE default\n')
        if sys.byteorder[0] == 'l':
            tmp = img.byteswap()
            tmp.tofile(ff)
        else:
            img.T.tofile(ff)

# Saves in analyze file
def saveAnalyze(self, filename):
    nibabel.AnalyzeImage(self.data, None).to_filename(filename)


