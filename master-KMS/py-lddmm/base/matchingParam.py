from . import kernelFunctions as kfun


## Parameter class for matching
#      timeStep: time discretization
#      KparDiff: object kernel: if not specified, use typeKernel with width sigmaKernel
#      KparDist: kernel in current/measure space: if not specified, use gauss kernel with width sigmaDist
#      sigmaError: normlization for error term
#      errorType: 'measure' or 'current'
#      typeKernel: 'gauss' or 'laplacian'
class MatchingParam:
    def __init__(self, timeStep = .1, algorithm = 'bfgs', Wolfe = False, KparDiff = None, KparDist = None,
                 sigmaError=1.0, errorType = 'measure', vfun = None):
        self.timeStep = timeStep
        self.sigmaKernel = 6.5
        self.orderKernel = 3
        self.sigmaDist = 2.5
        self.orderKDist = 3
        self.typeKDist = 'gauss'
        self.sigmaError = sigmaError
        self.typeKernel = 'gauss'
        self.errorType = errorType
        self.vfun = vfun
        self.algorithm = algorithm
        self.wolfe = Wolfe

        if type(KparDiff) in (list,tuple):
            self.typeKernel = KparDiff[0]
            self.sigmaKernel = KparDiff[1]
            if self.typeKernel == 'laplacian' and len(KparDiff) > 2:
                self.orderKernel = KparDiff[2]
            self.KparDiff = kfun.Kernel(name = self.typeKernel, sigma = self.sigmaKernel, order=self.orderKernel)
        else:
            self.KparDiff = KparDiff

        if type(KparDist) in (list,tuple):
            self.typeKDist = KparDist[0]
            self.sigmaDist = KparDist[1]
            if self.typeKernel == 'laplacian' and len(KparDist) > 2:
                self.orderKdist = KparDist[2]
            self.KparDist = kfun.Kernel(name = self.typeKDist, sigma = self.sigmaDist, order=self.orderKDist)
        else:
            self.KparDist = KparDist
