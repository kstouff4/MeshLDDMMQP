import logging

# A simple implementation of gradient descent with
# heuristic changing of descent step (eps)
def descend(opt, verb = True, maxIter=1000, TestGradient = False, epsInit=10.):

    eps = epsInit
    gradCoeff = 1.0
    epsDoubleIterations = 10

    obj = opt.objectiveFun()
    logging.info("***** iteration 0, energy: %e, eps: %e" % (obj, epsInit))

    for it in range(maxIter):
        if hasattr(opt, 'startOfIteration'):
            opt.startOfIteration()
        grd = opt.getGradient(gradCoeff)

        if (it % epsDoubleIterations) == 0:
            eps *= 2.

        objTry = 1e30
        while (objTry > obj and eps>1e-20):
            objTry = opt.updateTry(grd, eps, obj)
            eps *= .5

        opt.acceptVarTry()
        obj = objTry
        if hasattr(opt, 'endOfIteration'):
            opt.endOfIteration()
        logging.info("***** iteration %d, energy: %e, eps: %e" % (it, obj, eps))
