import numpy as np
import logging
from .linesearch import line_search_wolfe

# Class running nonlinear conjugate gradient
# opt is an optimizable class that must provide the following functions:
#   getVariable(): current value of the optimzed variable
#   objectiveFun(): value of the objective function
#   updateTry(direction, step, [acceptThreshold]) computes a temporary variable by moving the current one in the direction 'dircetion' with step 'step'
#                                                 the temporary variable is not stored if the objective function is larger than acceptThreshold (when specified)
#                                                 This function should not update the current variable
#   acceptVarTry() replace the current variable by the temporary one
#   getGradient(coeff) returns coeff * gradient; the result can be used as 'direction' in updateTry
#
# optional functions:
#   startOptim(): called before starting the optimization
#   startOfIteration(): called before each iteration
#   endOfIteration() called after each iteration
#   endOptim(): called once optimization is completed
#   dotProduct(g1, g2): returns a list of dot products between g1 and g2, where g1 is a direction and g2 a list of directions
#                       default: use standard dot product assuming that directions are arrays
#   addProd(g0, step, g1): returns g0 + step * g1 for directions g0, g1
#   copyDir(g0): returns a copy of g0
#   randomDir(): Returns a random direction
# optional attributes:
#   gradEps: stopping theshold for small gradient
#   gradCoeff: normalizaing coefficient for gradient.
#
# verb: for verbose printing
# TestGradient evaluate accracy of first order approximation (debugging)
# epsInit: initial gradient step

def __dotProduct(x,y):
    res = []
    for yy in y:
        res.append((x*yy).sum())
    return res

def __addProd(x,y,a):
    return x + a*y

def __prod(x, a):
    return a*x

def __copyDir(x):
    return np.copy(x)

def __stopCondition():
    return True



def cg(opt, verb = True, maxIter=1000, TestGradient = False, epsInit=0.01, sgdPar=None, Wolfe=True):

    if (hasattr(opt, 'getVariable')==False or hasattr(opt, 'objectiveFun')==False or hasattr(opt, 'updateTry')==False
            or hasattr(opt, 'acceptVarTry')==False or hasattr(opt, 'getGradient')==False):
        logging.error('Error: required functions are not provided')
        return

    # if Wolfe and hasattr(opt, 'dotProduct_euclidean'):
    #     dotProduct = opt.dotProduct_euclidean   #
    # elif not Wolfe and hasattr(opt, 'dotProduct'):
    #     dotProduct = opt.dotProduct
    # else:
    #     dotProduct = __dotProduct

    elif hasattr(opt, 'dotProduct'):
        dotProduct = opt.dotProduct
    else:
        dotProduct = __dotProduct


    if hasattr(opt, 'startOptim'):
        opt.startOptim()

    if hasattr(opt, 'gradEps'):
        gradEps = opt.gradEps
    else:
        gradEps = None

    if hasattr(opt, 'cgBurnIn'):
        cgBurnIn = opt.cgBurnIn
    else:
        cgBurnIn = 10
    
    if hasattr(opt, 'gradCoeff'):
        gradCoeff = opt.gradCoeff
    else:
        gradCoeff = 1.0

    if hasattr(opt, 'restartRate'):
        restartRate = opt.restartRate
    else:
        restartRate = 100

    if hasattr(opt, 'epsMax'):
        epsMax = opt.epsMax
    else:
        epsMax = 1.

    if sgdPar is None:
        sgd = False
    else:
        sgd = True
        sgdBurnIn = sgdPar[0]
        sgdRate = sgdPar[1]

    if sgd:
        restartRate = 1
        TestGradient = False

    eps = epsInit
    epsMin = 1e-10
    opt.converged = False

    if hasattr(opt, 'reset') and opt.reset:
        opt.obj = None

    obj = opt.objectiveFun()
    opt.objIni = obj
    opt.reset = False
    #obj = opt.objectiveFun()
    logging.info('iteration 0: obj = {0: .5f}'.format(obj))
    # if (obj < 1e-10):
    #     return opt.getVariable()


    gval = None
    obj_old = None
    skipCG = 0
    noUpdate = 0
    it = 0
    while it < maxIter:
        if it % restartRate == 0:
            skipCG = 1

        if hasattr(opt, 'startOfIteration'):
            opt.startOfIteration()

        if opt.reset:
            opt.obj = None
            obj = opt.objectiveFun()
            obj_old = None

        if noUpdate==0:
            if gval is None:
                grd = opt.getGradient(gradCoeff)
            else:
                grd = gval

        if TestGradient:
            if hasattr(opt, 'randomDir'):
                dirfoo = opt.randomDir()
            else:
                dirfoo = np.random.normal(size=grd.shape)
            epsfoo = 1e-8
            objfoo = opt.updateTry(dirfoo, epsfoo, obj-1e10)
            objfoo2 = opt.updateTry(dirfoo, 0, obj-1e10)
            [grdfoo] = opt.dotProduct(grd, [dirfoo])
            logging.info('Test Gradient: %.6f %.6f' %((objfoo - obj)/epsfoo, -grdfoo * gradCoeff ))
        if sgd:
            eps = epsInit / (1 + sgdRate*max(0, it - sgdBurnIn))
            objTry = opt.updateTry(grd, eps, obj+1e10)
            opt.acceptVarTry()
            obj = objTry
            if verb | (it == maxIter):
                logging.info('iteration {0:d}: obj = {1:.5f}, eps = {2:.5f}'.format(it+1, obj, eps))

            if hasattr(opt, 'endOfIteration'):
                opt.endOfIteration()
            it += 1
        else:
            if it == 0 or it == cgBurnIn:
                [grdOld2] = opt.dotProduct(grd, [grd])
                grd2= grdOld2
                grdTry = np.sqrt(max(1e-20, grdOld2))
                if hasattr(opt, 'copyDir'):
                    oldDir = opt.copyDir(grd)
                    grdOld = opt.copyDir(grd)
                    dir0 = opt.copyDir(grd)
                else:
                    oldDir = np.copy(grd)
                    grdOld = np.copy(grd)
                    dir0 = np.copy(grd)
                beta = 0
            else:
                [grd2, grd12] = opt.dotProduct(grd, [grd, grdOld])

                if skipCG == 0:
                    beta = max(0, (grd2 - grd12)/grdOld2)
                else:
                    beta = 0

                grdOld2 = grd2
                grdTry = np.sqrt(np.maximum(1e-20,grd2 + beta * grd12))

                if hasattr(opt, 'addProd'):
                    dir0 = opt.addProd(grd, oldDir, beta)
                else:
                    dir0 = grd + beta * oldDir

                if hasattr(opt, 'copyDir'):
                    oldDir = opt.copyDir(dir0)
                    grdOld = opt.copyDir(grd)
                else:
                    oldDir = np.copy(dir0)
                    grdOld = np.copy(grd)

            if it == 0 or it == cgBurnIn:
                if gradEps is None:
                    gradEps = max(0.001 * np.sqrt(grd2), 0.0001)
                else:
                    # gradEps = max(min(gradEps, 0.0001 * np.sqrt(grd2)), 0.00001)
                    gradEps = max(min(gradEps, 0.001 * np.sqrt(grd2)), 0.0001)  # original

            noUpdate = 0
            epsBig = epsMax / (grdTry)
            if eps > epsBig:
                eps = epsBig
            _eps = eps
            __Wolfe = True
            if Wolfe:
                objTry = opt.updateTry(dir0, eps, obj)
                eps, fc, gc, phi_star, old_fval, gval = line_search_wolfe(opt, dir0, gfk=grd, old_fval=obj,
                                   old_old_fval=obj_old, c1=1e-4, c2=0.9, amax=None,
                                   maxiter=10)
                if eps is not None:
                    obj_old = obj
                    opt.acceptVarTry()
                    obj = phi_star
                else:
                    eps = _eps
                    logging.info('Wolfe search unsuccessful')
                    __Wolfe = False

            if not Wolfe or not __Wolfe:
                eps = _eps
                gval = None
                objTry = opt.updateTry(dir0, eps, obj)
                badGradient = 0
                if objTry > obj:
                    #fprintf(1, 'iteration %d: obj = %.5f, eps = %.5f\n', it, objTry, eps) ;
                    epsSmall = np.maximum(1e-6/(grdTry), epsMin)
                    #print 'Testing small variation, eps = {0: .10f}'.format(epsSmall)
                    objTry0 = opt.updateTry(dir0, epsSmall, obj)
                    if objTry0 > obj:
                        if (skipCG == 1) | (beta < 1e-10):
                            logging.info('iteration {0:d}: obj = {1:.5f}, eps = {2:.5f}, beta = {3:.5f}, gradient: {4:.5f}'.format(it+1, obj, eps, beta, np.sqrt(grd2)))
                            logging.info('Stopping Gradient Descent: bad direction')
                            break
                        else:
                            if verb:
                                logging.info('Disabling CG: bad direction')
                                badGradient = 1
                                noUpdate = 1
                    else:
                        while (objTry > obj) and (eps > epsMin):
                            eps = eps / 2
                            objTry = opt.updateTry(dir0, eps, obj)
                            #opt.acceptVarTry()

                            #print 'improve'
                ## reducing step if improves
                if noUpdate == 0:
                    contt = 1
                    while contt==1:
                        objTry2 = opt.updateTry(dir0, .5*eps, objTry)
                        if objTry > objTry2:
                            eps = eps / 2
                            objTry=objTry2
                        else:
                            contt=0


                # increasing step if improves
                    contt = 10
                    #eps0 = eps / 4
                    while contt>=1 and eps<epsBig:
                        objTry2 = opt.updateTry(dir0, 1.25*eps, objTry)
                        if objTry > objTry2:
                            eps *= 1.25
                            objTry=objTry2
                            contt -= 1
                        else:
                            contt=0
                    obj_old = obj
                    opt.acceptVarTry()
                    obj = objTry

                #print obj+obj0, objTry+obj0
                smallVariation = 0
                if (np.fabs(obj-obj_old) < 1e-7):
                # if (np.fabs(obj-objTry) < 1e-6): # original
                    if (skipCG==1): # or (beta < 1e-10) :
                        if it > cgBurnIn:
                            logging.info(
                                'iteration {0:d}: obj = {1:.5f}, eps = {2:.5f}, eps/epsMax = {5:.5f}, beta = {3:.5f}, gradient: {4:.5f}'.format(
                                    it + 1 ,obj ,eps ,beta ,np.sqrt(grd2) ,eps / epsBig))
                            logging.info('Stopping Gradient Descent: small variation')
                            opt.converged = True
                            break
                        else:
                            skipCG = 0
                        if hasattr(opt, 'endOfIteration'):
                            opt.endOfIteration()
                    else:
                        if verb:
                            logging.info('Disabling CG: small variation')
                        smallVariation = 1
                        eps = 1.0

                skipCG = badGradient or smallVariation
            #logging.info('Obj Fun CG: ' + str(opt.objectiveFun(force=True)))
            if verb | (it == maxIter):
                logging.info('iteration {0:d}: obj = {1:.5f}, eps = {2:.5f}, eps/epsMax = {5:.5f},  beta = {3:.5f}, gradient: {4:.5f}'.format(it+1, obj, eps, beta, np.sqrt(grd2), eps/epsBig))

            if np.sqrt(grd2) <gradEps and it>cgBurnIn:
                logging.info('Stopping Gradient Descent: small gradient')
                opt.converged = True
                if hasattr(opt, 'endOfProcedure'):
                    opt.endOfProcedure()
                break
            eps = np.minimum(100*eps, epsMax)

            if noUpdate==0 and hasattr(opt, 'endOfIteration'):
                opt.endOfIteration()
                if hasattr(opt ,'cgBurnIn'):
                    cgBurnIn = opt.cgBurnIn
            if noUpdate==0:
                it += 1

    if it == maxIter and hasattr(opt, 'endOfProcedure'):
        opt.endOfProcedure()

    if hasattr(opt, 'endOptim'):
        opt.endOptim()

    return opt.getVariable()

def linearcg(op, b, iterMax=100, x0=None, param = None, verb=False):
    if x0 is None:
        x = np.zeros(b.shape)
    else:
        x = x0
    if param is None:
        z = op(x) - 2*b
    else:
        z = op(x, param) - 2*b

    ener = (z*x).sum()
    oldEner = ener
    f=0
    mu = 0
    r = -z - b
    for i2 in range(iterMax):
        if i2 == 0:
            mu = (r*r).sum()
            if mu < 1e-10:
                return x
            p = r
        else:
            muold = mu
            mu = (r*r).sum()
            beta = mu/muold
            p = r + beta * p

        if param is None:
            q = op(p)
        else:
            q = op(p, param)

        u = (p*q).sum()
        alpha = mu / u
        x += alpha * p
        r -= alpha * q
        z += alpha * q
        ener = (z*x).sum()
        if param is None:
            error = ((op(x)-b)**2).sum()
        else:
            error = ((op(x,param)-b)**2).sum()
        if verb and (i2%10) == 0:
            logging.info('iter {0:d} ener = {1:.6f} error = {2:.15f}'.format(i2+1, ener, error))
        if alpha * np.mean((np.fabs(p))) < 1e-10:
            f = 1
            return x

        if np.fabs(ener - oldEner) < (1e-20)*np.fabs(oldEner):
            if verb:
                logging.info('iter {0:d} ener = {1:.6f} error = {2:.15f}'.format(i2+1, ener, error))
            f = 1
            return x
        oldEner = ener

    if verb:
        logging.info('iter {0:d} ener = {1:.6f} error = {2:.15f}'.format(i2+1, ener, error))
    return x

