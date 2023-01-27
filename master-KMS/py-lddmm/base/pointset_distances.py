import numpy as np

def L2Norm0(x1):
    return (x1.points**2).sum()

def L2NormDef(xDef, x1):
    return -2*(xDef*x1).sum() + (xDef**2).sum()

def L2NormGradient(xDef,x1):
    return 2*(xDef-x1)


# Measure norm of fv1
def measureNorm0(fv1, KparDist):
    cr2 = fv1.weights[:, None]
    return (cr2*KparDist.applyK(fv1.points, cr2)).sum()


# Computes |fvDef|^2 - 2 fvDef * fv1 with measure dot produuct
def measureNormDef(fvDef, fv1, KparDist):
    cr1 = fvDef.weights[:, None]
    cr2 = fv1.weights[:, None]
    obj = ((cr1 * KparDist.applyK(fvDef.points, cr1)).sum()
           - 2 * (cr1 * KparDist.applyK(fv1.points, cr2, firstVar=fvDef.points)).sum())
    return obj


# Returns |fvDef - fv1|^2 for measure norm
def measureNorm(fvDef, fv1, KparDist):
    return measureNormDef(fvDef, fv1, KparDist) + measureNorm0(fv1, KparDist)


# Returns gradient of |fvDef - fv1|^2 with respect to vertices in fvDef (measure norm)
def measureNormGradient(fvDef, fv1, KparDist):
    cr1 = fvDef.weights[:, None]
    cr2 = fv1.weights[:, None]

    dz1 = (KparDist.applyDiffKT(fvDef.points, cr1, cr1) -
                       KparDist.applyDiffKT(fv1.points, cr1, cr2, firstVar=fvDef.points))

    return 2 * dz1


