from scipy.optimize.linesearch import  scalar_search_wolfe2, scalar_search_wolfe1


def line_search_wolfe(opt, pk, gfk=None, old_fval=None,
                       old_old_fval=None, c1=0.001, c2=0.9, amax=None,
                       maxiter=100, euclidean = True):
    """Find alpha that satisfies strong Wolfe conditions.

    Parameters
    ----------
    f : callable f(x,*args)
        Objective function.
    myfprime : callable f'(x,*args)
        Objective function gradient.
    xk : ndarray
        Starting point.
    pk : ndarray
        Search direction.
    gfk : ndarray, optional
        Gradient value for x=xk (xk being the current parameter
        estimate). Will be recomputed if omitted.
    old_fval : float, optional
        Function value for x=xk. Will be recomputed if omitted.
    old_old_fval : float, optional
        Function value for the point preceding x=xk.
    args : tuple, optional
        Additional arguments passed to objective function.
    c1 : float, optional
        Parameter for Armijo condition rule.
    c2 : float, optional
        Parameter for curvature condition rule.
    amax : float, optional
        Maximum step size
    extra_condition : callable, optional
        A callable of the form ``extra_condition(alpha, x, f, g)``
        returning a boolean. Arguments are the proposed step ``alpha``
        and the corresponding ``x``, ``f`` and ``g`` values. The line search
        accepts the value of ``alpha`` only if this
        callable returns ``True``. If the callable returns ``False``
        for the step length, the algorithm will continue with
        new iterates. The callable is only called for iterates
        satisfying the strong Wolfe conditions.
    maxiter : int, optional
        Maximum number of iterations to perform.

    Returns
    -------
    alpha : float or None
        Alpha for which ``x_new = x0 + alpha * pk``,
        or None if the line search algorithm did not converge.
    fc : int
        Number of function evaluations made.
    gc : int
        Number of gradient evaluations made.
    new_fval : float or None
        New function value ``f(x_new)=f(x0+alpha*pk)``,
        or None if the line search algorithm did not converge.
    old_fval : float
        Old function value ``f(x0)``.
    new_slope : float or None
        The local slope along the search direction at the
        new value ``<myfprime(x_new), pk>``,
        or None if the line search algorithm did not converge.


    Notes
    -----
    Uses the line search algorithm to enforce strong Wolfe
    conditions. See Wright and Nocedal, 'Numerical Optimization',
    1999, pp. 59-61.
    """

    fc = [0]
    gc = [0]
    gval = [None]
    gval_alpha = [None]


    def phi(alpha):
        fc[0] += 1
        return opt.updateTry(pk, alpha)


    def derphi(alpha):
        gc[0] += 1
        gval[0] = opt.getGradient(opt.gradCoeff, update=[pk, alpha])  # store for later use
        gval_alpha[0] = alpha
        if euclidean:
            return -opt.dotProduct_euclidean(gval[0], [pk])[0]
        else:
            return -opt.dotProduct(gval[0], [pk])[0]


    if gfk is None:
        gfk = opt.getGradient(opt.gradCoeff)
    if euclidean:
        derphi0 = -opt.dotProduct_euclidean(gfk, [pk])[0]
    else:
        derphi0 = -opt.dotProduct(gfk, [pk])[0]



    # alpha_star, phi_star, old_fval = scalar_search_wolfe1(
    #     phi, derphi, old_fval, old_old_fval, derphi0, c1, c2)
    # # if alpha_star is None:
    alpha_star, phi_star, old_fval, derphi_star = scalar_search_wolfe2(
        phi, derphi, old_fval, old_old_fval, derphi0, c1, c2, amax,
        None, maxiter=maxiter)

    # if derphi_star is None:
    #     warn('The line search algorithm did not converge', LineSearchWarning)
    # else:
    #     # derphi_star is a number (derphi) -- so use the most recently
    #     # calculated gradient used in computing it derphi = gfk*pk
    #     # this is the gradient at the next step no need to compute it
    #     # again in the outer loop.
    #     derphi_star = gval[0]


    return alpha_star, fc[0], gc[0], phi_star, old_fval, gval[0]
