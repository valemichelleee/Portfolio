# Optimization for Engineers - Dr.Johannes Hild
# Newton descent

# Purpose: Find xmin to satisfy norm(gradf(xmin))<=eps
# Iteration: x_k = x_k + d_k
# d_k is the Newton direction

# Input Definition:
# f: objective class with methods .objective() and .gradient() and .hessian()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# d = PrecCGSolver(A,b) from PrecCGSolver.py

# Test cases:
# myObjective = bananaValleyObjective()
# x0 = np.array([[0], [1]])
# xmin = NewtonDescent(myObjective, x0, 1.0e-6, 1)
# should return
# xmin close to [[1],[1]]

import numpy as np
import PrecCGSolver as PCG


def matrnr():
    # set your matriculation number here
    matrnr = 23358926
    return matrnr


def NewtonDescent(f, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start NewtonDescent...')

    countIter = 0
    x = x0

    # INCOMPLETE CODE STARTS
    grad_f = f.gradient(x)
    B = f.hessian(x)

    while (np.linalg.norm(grad_f)) > eps:
        countIter += 1
        d = PCG.PrecCGSolver(B, -grad_f)
        x = x + d
        grad_f = f.gradient(x)
        B = f.hessian(x)


    # INCOMPLETE CODE ENDS

    if verbose:
        gradx = f.gradient(x)
        print('NewtonDescent terminated after ', countIter, ' steps with norm of gradient =', np.linalg.norm(gradx))

    return x
