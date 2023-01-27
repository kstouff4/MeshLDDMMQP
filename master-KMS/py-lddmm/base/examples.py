import numpy as np
from .curves import Curve
from .surfaces import Surface
from matplotlib import pyplot as plt

def peanut(r = (1,1), delta = 0.5, sigma = .25, theta = 0, tau = (0,0), d = 100):
    [x, y] = np.mgrid[0:2*d, 0:2*d] / d
    y = y - 1
    x = x - 1
    dx = delta*np.cos(theta)
    dy = delta*np.sin(theta)
    s2 = 2*sigma*sigma
    I1 = r[0]*np.exp(- ((x-dx)**2 + (y-dy)**2)/s2) + r[1]*np.exp(- ((x+dx)**2 + (y+dy)**2)/s2)
    val = (r[0]+r[1])*np.exp(- (dx**2 + dy**2)/s2)*0.75
    #plt.imshow(I1>val)
    #plt.show()
    print(I1.min(), I1.max(), val)
    fv = Curve()
    fv.Isocontour(I1, value=val, target=150, scales=[1, 1])
    fv.updateVertices(fv.vertices + d*np.array(tau))

    return fv


def flowers(n=6):
    [x, y, z] = np.mgrid[0:200, 0:200, 0:200] / 100.
    ay = np.fabs(y - 1)
    az = np.fabs(z - 1)
    ax = np.fabs(x - 0.5)
    s2 = np.sqrt(2)
    c1 = np.sqrt(0.06)
    c2 = np.sqrt(0.03)
    c3 = 0.1

    th = np.arctan2(ax, az)
    I1 = c1 ** 2 - (ax ** 2 + 0.5 * ay ** 2 + az ** 2) * (1 + 0.25 * np.cos(n * th))
    # I2 = -(ax ** 2 + 0.5 * ay ** 2 + az ** 2) * (1+0.5*np.cos(6*th))  + c2 ** 2
    I2 = -(ax ** 2 + 0.5 * ay ** 2 + az ** 2) + c2 ** 2
    fvTop = Surface()
    fvTop.Isosurface(I1, value=0, target=3000, scales=[1, 1, 1], smooth=-0.01)
    fvTop = fvTop.truncate((np.array([0, 1, 0, 95]), np.array([0, -1, 0, -105])))

    fvBottom = Surface()
    fvBottom.Isosurface(I2, value=0, target=3000, scales=[1, 1, 1], smooth=-0.01)
    fvBottom = fvBottom.truncate((np.array([0, 1, 0, 95]), np.array([0, -1, 0, -105])))
    return fvBottom,fvTop

def waves(w0=(0,0), r=(1,0.5), d=25):
    [x, y, z] = np.mgrid[0:2*d, 0:2*d, 0:2*d] / d - 1
    az = z - w0[1]*np.sin(w0[0]*np.pi*x)
    fv = Surface()
    fv.Isosurface(az, value=0, target=-1, scales=[1, 1, 1], smooth=-0.01)
    fv = fv.truncate((np.array([0, 1, 0, d - r[1]*d]), np.array([0, -1, 0, -d - r[1]*d]),
                                  np.array([1, 0, 0, d - r[0]*d]), np.array([-1, 0, 0, -d - r[0]*d])))
    return fv

def waves2(w0=(0,0), w1=(1,0.25), r=(1,0.5), delta=10, d=25):
    fvBottom = waves(w0,r,d)
    fvTop = waves(w1, r, d)
    fvTop.updateVertices(fvTop.vertices + np.array([0,0,d*(w0[1]+w1[1])+delta]))


    return fvBottom,fvTop


def bumps(centers1 = ([-0.1,-0.5], [.5,0], [-0.1,.5]),
          scale1 = (.25, .25, .25),
          weights1 = (.5, .5, .5), d=25):
    [x, y, z] = np.mgrid[0:2*d, 0:2*d, 0:2*d] / d - 1
    I = np.zeros(x.shape)
    for (c,s,w) in zip(centers1, scale1, weights1):
        I += w*np.exp(-((x-c[0])**2 + (y-c[1])**2)/(s*s))
    az = z-I
    fv1 = Surface()
    fv1.Isosurface(az, value=0,  target=-1, scales=[1, 1, 1], smooth=-0.01)
    return fv1

def bumps2(centers1 = ([-0.1,-0.5], [.5,0], [-0.1,.5]),
          scale1 = (.25, .25, .25),
          weights1 = (.5, .5, .5),
          centers2 = ([.5,-.5], [-.5, .5], [.5, .5]),
          scale2 = (.25,.25,.25),
          weights2 = (.5,.5,.5),
          d=25):
    fv1 = bumps(centers1, scale1, weights1, d)
    fv2 = bumps(centers2, scale2, weights2, d)
    return fv1,fv2

def ellipsoid_smallDef():
    [x, y, z] = np.mgrid[0:200, 0:200, 0:200] / 100.
    y = y - 1
    z = z - 1
    s2 = np.sqrt(2)

    I1 = .06 - ((x - .50) ** 2 + 0.5 * y ** 2 + z ** 2)
    fv1 = Surface()
    fv1.Isosurface(I1, value=0, target=2000, scales=[1, 1, 1], smooth=0.01)

    # return fv1

    u = (z + y) / s2
    v = (z - y) / s2
    I1 = np.maximum(0.05 - (x - .6) ** 2 - 0.5 * y ** 2 - z ** 2, 0.03 - (x - .50) ** 2 - 0.5 * y ** 2 - z ** 2)
    # I1 = .06 - ((x-.50)**2 + 0.75*y**2 + z**2)
    # I1 = .095 - ((x-.7)**2 + v**2 + 0.5*u**2)
    fv2 = Surface()
    fv2.Isosurface(I1, value=0, target=2000, scales=[1, 1, 1], smooth=0.01)
    return fv1, fv2

def ellipsoid_largeDef():
    [x, y, z] = np.mgrid[0:200, 0:200, 0:200] / 100.
    y = y - 1
    z = z - 1
    s2 = np.sqrt(2)

    I1 = .06 - ((x - .50) ** 2 + 0.5 * y ** 2 + z ** 2)
    fv1 = Surface()
    fv1.Isosurface(I1, value=0, target=2000, scales=[1, 1, 1], smooth=0.01)

    # return fv1

    u = (z + y) / s2
    v = (z - y) / s2
    I1 = np.maximum(0.05 - (x - .6) ** 2 - 0.5 * y ** 2 - z ** 2, 0.03 - (x - .50) ** 2 - 0.5 * y ** 2 - z ** 2)
    # I1 = .06 - ((x-.50)**2 + 0.75*y**2 + z**2)
    # I1 = .095 - ((x-.7)**2 + v**2 + 0.5*u**2)
    fv2 = Surface()
    fv2.Isosurface(I1, value=0, target=2000, scales=[1, 1, 1], smooth=0.01)
    return fv1, fv2


def ellipsoid(a=.5, b=.5, c=.5, d=25):
    [x, y, z] = np.mgrid[0:2*d, 0:2*d, 0:2*d] / d - 1
    I1 = 1 - (x/a)**2 - (y/b)**2 - (z/c)**2
    fv = Surface()
    #fv.Isosurface(I1, value=0, target=1000, scales=[1, 1, 1], smooth=0.01)
    fv.Isosurface_ski(I1, value=0, step=5)
    print('vertices',fv.vertices.shape[0])
    return fv

def split_ellipsoid(a=.5, b=.5, c=.5, d=25):
    [x, y, z] = np.mgrid[0:2*d, 0:2*d, 0:2*d] / d - 1
    I1 = (1 - (x/a)**2 - (y/b)**2 - (z/c)**2)
    fv = Surface()
    #fv.Isosurface(I1, value=0, target=1000, scales=[1, 1, 1], smooth=0.01)
    fv.Isosurface_ski(I1, value=0, step=1)
    fvTop = fv.truncate((np.array([0, 0, 1, d *(1+ 0.25*c)]),))
    fvBottom = fv.truncate((np.array([0, 0, -1, -d*(1-0.25*c)]),))

    return fvBottom, fvTop
