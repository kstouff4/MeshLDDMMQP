from os import path
import glob
import argparse
import numpy as np
from .base import pointSets, surfaces
from .base.affineRegistration import rigidRegistration, saveRigid


def main():
    parser = argparse.ArgumentParser(description='runs rigid registration over directories (relative to the first left surface)')
    parser.add_argument('dirIn', metavar='dirIn', type = str, help='input directory')
    parser.add_argument('--dirOut', metavar = 'dirOut', type = str, dest = 'dirOut', default = '', help='Output directory')
    parser.add_argument('--lr-pattern', metavar = 'left/right pattern', dest='lr_pattern', type = str, nargs = 2, default = ('_L', '_R'), help='uses alternate pattern instead of L/R to differentiate left and right files')
    parser.add_argument('--use-landmarks', metavar = 'dirLmk', type = str, dest = 'dirLmk', default = '', help='Uses landmak files in dirLmk directory') 
    parser.add_argument('--continue', metavar = 'continue', type = bool, dest = 'dirLmk', default = '', help='Uses landmak files in dirLmk directory') 
    parser.add_argument('--cont', action = 'store_true', dest='cont', default=False,help='continue previous run (do not update already registered files)')
    args = parser.parse_args()

    if args.dirOut == '':
        args.dirOut = args.dirIn
    if args.dirLmk == '':
        lmk = False
    else:
        lmk = True

    left_pattern = '*'+args.lr_pattern[0]+'*'
    right_pattern = '*'+args.lr_pattern[1]+'*'

    left = glob.glob(args.dirIn+'/'+left_pattern+'.byu')
    right = glob.glob(args.dirIn+'/'+right_pattern+'.byu')
    if (len(left) > 0):
        tmpl = surfaces.Surface(filename=left[0])
        if lmk:
            for name in left:
                u = path.split(name)
                [nm,ext] = path.splitext(u[1])
                if path.exists(args.dirLmk+'/'+nm+'.lmk'):
                    tmpl = surfaces.Surface(filename=left[0])
                    tmplLmk, foo = pointSets.loadlmk(args.dirLmk + '/' + nm + '.lmk')
                    R0, T0 = rigidRegistration(surfaces = (tmplLmk, tmpl.vertices),  translationOnly=True, verb=False, temperature=10., annealing=True)
                    tmplLmk = tmplLmk + T0
                    cLmk = float(tmpl.vertices.shape[0]) / tmplLmk.shape[0]
                    break
    # if lmk:
    #     leftLmk = glob.glob(args.dirLmk+'/'+left_pattern+'.lmk')
    #     rightLmk = glob.glob(args.dirLmk+'/'+right_pattern+'.lmk')
    #     if len(leftLmk) > 0:
    #         tmplLmk, foo = pointSets.loadlmk(leftLmk[0])
    #         R0, T0 = rigidRegistration(surfaces = (tmplLmk, tmpl.vertices),  translationOnly=True, verb=False, temperature=10., annealing=True)
    #         tmplLmk = tmplLmk + T0
    #         cLmk = float(tmpl.vertices.shape[0]) / tmplLmk.shape[0] 
    
    #ff = open('suspicious.txt', 'w')
    for (kk, name) in enumerate(left):
        print('Processing ', name)
        u = path.split(name)
        [nm,ext] = path.splitext(u[1])
        if args.cont & (path.exists(args.dirOut+'/'+nm+'_rig.dat')):
            print('found', args.dirOut+'/'+nm+'_rig.dat')
        if (args.cont == False) | (path.exists(args.dirOut+'/'+nm+'_rig.dat')==False):
            sf = surfaces.Surface(filename = name)
            if lmk:
                nmLmk = args.dirLmk+'/'+nm+'.lmk'
                try:
                    y, foo = pointSets.loadlmk(nmLmk)
                    R0, T0 = rigidRegistration(surfaces = (y, sf.vertices),  translationOnly=True, verb=False, temperature=10., annealing=True)
                    y = y+T0
                    (R0, T0) = rigidRegistration(landmarks=(y, tmplLmk, 1.), flipMidPoint=False, rotationOnly=False, verb=False, temperature=10., annealing=True, rotWeight=1.)
                    yy = np.dot(sf.vertices, R0.T) + T0
                    yyl = np.dot(y, R0.T) + T0
                    pointSets.savelmk(yyl, args.dirOut + '/' + nm + '_reg.lmk')
                    (R, T) = rigidRegistration(surfaces = (yy, tmpl.vertices), landmarks=(yyl, tmplLmk, cLmk), flipMidPoint=False, rotationOnly=True, verb=False, temperature=1., annealing=False, rotWeight=1.)
                    T += np.dot(T0, R.T)
                    R = np.dot(R, R0)
                    #R = R0
                    #T = T0
                    # if ((R-np.diag([1,1,1]))**2).sum() > 1:
                    #     print R, T
                    #     ff.write(nm+'\n')
                    yyl = np.dot(y, R.T) + T
                    pointSets.savelmk(yyl, args.dirOut + '/' + nm + '_reg.lmk')
                    #print R,T
                except IOError:
                    (R, T) = rigidRegistration(surfaces=(sf.vertices, tmpl.vertices), rotationOnly=True, verb=False, temperature=10., annealing=True, rotWeight=1.)
            else:
                (R, T) = rigidRegistration(surfaces=(sf.vertices, tmpl.vertices), rotationOnly=False, verb=False, temperature=10., annealing=True, rotWeight=1.)
            saveRigid(args.dirOut+'/'+nm+'_rig.dat', R, T)
            sf.vertices = np.dot(sf.vertices, R.T) + T
            sf.savebyu(args.dirOut+'/'+nm+'_reg.byu')

    for (kk, name) in enumerate(right):
        print('Processing ', name)
        u = path.split(name)
        [nm,ext] = path.splitext(u[1])
        if args.cont & (path.exists(args.dirOut+'/'+nm+'_rig.dat')):
            print('found', args.dirOut+'/'+nm+'_rig.dat')
        if (args.cont==False) | (path.exists(args.dirOut+'/'+nm+'_rig.dat')==False):
            sf = surfaces.Surface(filename = name)
            if lmk:
                nmLmk = args.dirLmk+'/'+nm+'.lmk'
                try:
                    y, foo = pointSets.loadlmk(nmLmk)
                    R0, T0 = rigidRegistration(surfaces = (y, sf.vertices),  translationOnly=True, verb=False, temperature=10., annealing=True)
                    y = y+T0
                    #print T0, y.min(), y.max()
                    (R0, T0) = rigidRegistration(landmarks=(y, tmplLmk, 1.), flipMidPoint=False, rotationOnly=False, verb=False, temperature=10., annealing=True, rotWeight=1.)
                    yy = np.dot(sf.vertices, R0.T) + T0
                    yyl = np.dot(y, R0.T) + T0
                    (R, T) = rigidRegistration(surfaces = (yy, tmpl.vertices), landmarks=(yyl, tmplLmk, cLmk), flipMidPoint=False, rotationOnly=True, verb=False, temperature=1., annealing=False, rotWeight=1.)
                    T += np.dot(T0, R.T)
                    R = np.dot(R, R0)
                    #R = R0
                    #T = T0
                    # if ((R-np.diag([-1,1,1]))**2).sum() > 1:
                    #     print R, T
                    #     ff.write(nm+'\n')
                        #print R, T
                    yyl = np.dot(y, R.T) + T
                    pointSets.savelmk(yyl, args.dirOut + '/' + nm + '_reg.lmk')
                except IOError:
                    (R, T) = rigidRegistration(surfaces = (sf.vertices, tmpl.vertices), flipMidPoint=True, rotationOnly=True, verb=False, temperature=10., annealing=True, rotWeight=1.)
            else:
                (R, T) = rigidRegistration(surfaces = (sf.vertices, tmpl.vertices), flipMidPoint=True, rotationOnly=False, verb=False, temperature=10., annealing=True, rotWeight=1.)
            sf.faces = sf.faces[:, [0,2,1]]
            saveRigid(args.dirOut+'/'+nm+'_rig.dat', R, T)
            sf.vertices = np.dot(sf.vertices, R.T) + T
            sf.savebyu(args.dirOut+'/'+nm+'_reg.byu')

        #ff.close()
if __name__=="__main__":
    main()
