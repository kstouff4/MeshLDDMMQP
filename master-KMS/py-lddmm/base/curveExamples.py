import numpy as np
from .curves import Curve, remesh

class Rectangle(Curve):
    def __init__(self, corners = None, npt = 25, spacing = None):
        super().__init__()
        if corners is None:
            self.corners = ([0,0], [1,0], [0,1])
        else:
            self.corners = corners

        ul = np.array(self.corners[0])
        ur = np.array(self.corners[1])
        ll = np.array(self.corners[2])
        d = ul.shape[0]
        self.vertices = np.zeros((4,d))
        self.vertices[0, :] = ul
        self.vertices[1, :] = ur
        self.vertices[2, :] = ur + ll - ul
        self.vertices[3, :] = ll

        self.faces = np.zeros((4,d), dtype=int)
        self.faces[0, :] = [0,1]
        self.faces[1, :] = [1,2]
        self.faces[2, :] = [2,3]
        self.faces[3, :] = [3,0]
        self.weights = np.ones(self.vertices.shape[0])
        self.component = np.zeros(self.faces.shape[0], dtype=int)
        self.computeCentersLengths()


class Circle(Curve):
    def __init__(self, center = (0.0, 0.0), radius=1.0, targetSize = 100):
        super().__init__()
        t = np.arange(0., 2*np.pi, 0.05)
        x0 = center[0]
        y0 = center[1]
        x = x0+radius*np.cos(t)
        y = y0+radius*np.sin(t)
        v = np.zeros([t.shape[0],2])
        v[:,0] = x
        v[:,1] = y
        v = remesh(v, N=targetSize)
        N = v.shape[0]
        f = np.zeros([N,2], dtype=int)
        f[0:N-1,0] = range(0, N-1)
        f[0:N-1,1] = range(1, N)
        f[N-1,:] = [N-1,0]
        self.center = center
        self.radius = radius
        self.vertices = v
        self.faces = f
        self.weights = np.ones(self.vertices.shape[0])
        self.component = np.zeros(self.faces.shape[0], dtype=int)
        self.computeCentersLengths()


class Ellipse(Curve):
    def __init__(self, center = (0.0, 0.0), a=1.0, b = 1.0, targetSize = 100):
        super().__init__()
        t = np.arange(0., 2*np.pi, 0.05)
        x0 = center[0]
        y0 = center[1]
        x = x0+a*np.cos(t)
        y = y0+a*np.sin(t)
        v = np.zeros([t.shape[0],2])
        v[:,0] = x
        v[:,1] = y
        v = remesh(v, N=targetSize)
        N = v.shape[0]
        f = np.zeros([N,2], dtype=int)
        f[0:N-1,0] = range(0, N-1)
        f[0:N-1,1] = range(1, N)
        f[N-1,:] = [N-1,0]
        self.center = center
        self.a = a
        self.b = b
        self.vertices = v
        self.faces = f
        self.weights = np.ones(self.vertices.shape[0])
        self.component = np.zeros(self.faces.shape[0], dtype=int)
        self.computeCentersLengths()




class Cardioid(Curve):
    def __init__(self, center = (0.0, 0.0), a=1.5, b=2, c = 0.7, targetSize = 100):
        super().__init__()
        t = np.arange(0., 2*np.pi, 0.05)
        x0 = center[0]
        y0 = center[1]
        x = x0+a*np.cos(t) * (1-c*np.cos(t))
        y = y0+b*np.sin(t) * (1-c*np.cos(t))
        v = np.zeros([t.shape[0],2])
        v[:,0] = x
        v[:,1] = y
        v = remesh(v, N=targetSize)
        N = v.shape[0]
        f = np.zeros([N,2], dtype=int)
        f[0:N-1,0] = range(0, N-1)
        f[0:N-1,1] = range(1, N)
        f[N-1,:] = [N-1,0]
        self.center = center
        self.a = a
        self.b = b
        self.c = c
        self.vertices = v
        self.faces = f
        self.weights = np.ones(self.vertices.shape[0])
        self.component = np.zeros(self.faces.shape[0], dtype=int)
        self.computeCentersLengths()


