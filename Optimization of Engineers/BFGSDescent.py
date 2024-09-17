# Optimization for Engineers - Dr.Johannes Hild
# global BFGS descent

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + t_k * d_k
# d_k is the BFGS direction. If a descent direction check fails, d_k is set to steepest descent and the inverse BFGS matrix is reset.
# t_k results from Wolfe-Powell

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# t = WolfePowellSearch(f, x, d) from WolfePowellSearch.py

# Test cases:
# myObjective = noHessianObjective()
# x0 = np.array([[-0.01], [0.01]])
# xmin = BFGSDescent(myObjective, x0, 1.0e-6, 1)
# should return
# xmin close to [[0.26],[-0.21]] with the inverse BFGS matrix being close to [[0.0078, 0.0005], [0.0005, 0.0080]]


import numpy as np
import WolfePowellSearch as WP


def matrnr():
    # set your matriculation number here
    matrnr = 23358926
    return matrnr


def BFGSDescent(f, x0: np.array, eps=1.0e-3, verbose=0):
    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start BFGSDescent...')

    countIter = 0
    x = x0
    n = x0.shape[0]
    E = np.eye(n)
    B = E
    # INCOMPLETE CODE STARTS

    grad_f = f.gradient(x)
    while np.linalg.norm(grad_f) > eps:
        countIter += 1
        d = -B @ grad_f

        if grad_f.T @ d >= 0:
            d = -grad_f
            B = E

        t = WP.WolfePowellSearch(f, x, d)
        grad_x = t * d
        x = x + grad_x
        grad_g = f.gradient(x) - grad_f
        grad_f = f.gradient(x)

        if grad_g.T @ grad_x <= 0:
            B = E
        else:
            # lemma 6.6
            r = grad_x - B @ grad_g
            B = B + ((r @ grad_x.T + grad_x @ r.T) / (grad_g.T @ grad_x)) - (((r.T @ grad_g) / (grad_g.T @ grad_x)**2) * (grad_x @ grad_x.T))

    # INCOMPLETE CODE ENDS
    if verbose:
        gradx = f.gradient(x)
        print('BFGSDescent terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(gradx), 'and the inverse BFGS matrix is')
        print(B)

    return x
