def _get_type(a):
    if 'float' in str(a.dtype):
        return 'float'
    if 'int' in str(a.dtype):
        return 'int'
    return None


class vtkFields:
    def __init__(self, data_type, size, scalars = None, vectors = None, normals = None,
                 tensors = None, fields = None):
        if scalars is None:
            self.scalars = dict()
        else:
            self.scalars = scalars
        if vectors is None:
            self.vectors = dict()
        else:
            self.vectors = vectors

        if normals is None:
            self.normals = dict()
        else:
            self.normals = normals
        if tensors is None:
            self.tensors = dict()
        else:
            self.tensors = tensors
        if fields is None:
            self.fields = dict()
        else:
            self.fields = fields
        self.data_type = data_type
        self.size = size

    def write(self, fvtkout):
        fvtkout.write(f'\n{self.data_type} {self.size}')
        if len(self.scalars):
            names = self.scalars.keys()
            for n in names:
                a = self.scalars[n]
                tp = _get_type(a)
                fvtkout.write(f'\nSCALARS {n} {tp} 1\nLOOKUP_TABLE default')
                for ll in range(a.shape[0]):
                    fvtkout.write(f'\n {a[ll]}')
        if len(self.vectors):
            names = self.vectors.keys()
            for n in names:
                a = self.vectors[n]
                tp = _get_type(a)
                fvtkout.write(f'\nVECTORS {n} {tp}')
                for kk in range(a.shape[0]):
                    for ll in range(3):
                        fvtkout.write(f'\n {a[kk,ll]}')
        if len(self.normals):
            a = self.normals['NORMALS']
            tp = _get_type(a)
            fvtkout.write(f'\nNORMALS normals {tp}')
            for kk in range(a.shape[0]):
                for ll in range(3):
                    fvtkout.write(f'\n {a[kk, ll]}')
        if len(self.tensors):
            names = self.tensors.keys()
            for n in names:
                a = self.tensors[n]
                tp = _get_type(a)
                fvtkout.write(f'\nTENSORS {n} {tp}')
                for kk in range(a.shape[0]):
                    for ll in range(3):
                        fvtkout.write(f'\n {a[ll, 0]} {a[ll, 1]} {a[ll, 2]}')
                    fvtkout.write('\n')
        if len(self.fields):
            names = self.fields.keys()
            nf = len(self.fields)
            fvtkout.write(f'\nFIELD field {nf}')
            for n in names:
                a = self.fields[n]
                tp = _get_type(a)
                fvtkout.write(f'\n{n} {a.shape[1]} {a.shape[0]} {tp}')
                for kk in range(a.shape[0]):
                    fvtkout.write('\n')
                    for ll in range(a.shape[1]):
                        fvtkout.write(f'{a[kk, ll]} ')


