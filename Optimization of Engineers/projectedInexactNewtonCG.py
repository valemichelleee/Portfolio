# Optimization for Engineers - Dr.Johannes Hild
# projected inexact Newton descent

# Purpose: Find xmin to satisfy norm(xmin - P(xmin - gradf(xmin)))<=eps
# Iteration: x_k = P(x_k + t_k * d_k)
# d_k starts as a steepest descent step and then CG steps are used to improve the descent direction until negative curvature is detected or a full Newton step is made.
# t_k results from projected backtracking

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# P: box projection class with method .project() and .activeIndexSet()
# x0: column vector in R ** n(domain point)
# eps: tolerance for termination. Default value: 1.0e-3
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# xmin: column vector in R ** n(domain point)

# Required files:
# dH = projectedHessApprox(f, P, x, d) from projectedHessApprox.py
# t = projectedBacktrackingSearch(f, P, x, d) from projectedBacktrackingSearch.py

# Test cases:
# p = np.array([[1], [1]])
# myObjective = simpleValleyObjective(p)
# a = np.array([[1], [1]])
# b = np.array([[2], [2]])
# myBox = projectionInBox(a, b)
# x0 = np.array([[2], [2]], dtype=float)
# eps = 1.0e-3
# xmin = projectedInexactNewtonCG(myObjective, myBox, x0, eps, 1)
# should return xmin close to [[1],[1]]

import numpy as np
import projectedBacktrackingSearch as PB
import projectedHessApprox as PHA

def matrnr():
    # set your matriculation number here
    matrnr = 23358926
    return matrnr


def projectedInexactNewtonCG(f, P, x0: np.array, eps=1.0e-3, verbose=0):

    if eps <= 0:
        raise TypeError('range of eps is wrong!')

    if verbose:
        print('Start projectedInexactNewtonCG...')

    countIter = 0
    xp = P.project(x0)
    # INCOMPLETE CODE STARTS

    rhs = np.linalg.norm(xp - P.project(xp - f.gradient(xp)))
    eta = min([0.5, np.sqrt(rhs)]) * rhs

    while rhs > eps:
        countIter += 1
        flag = False
        x = xp.copy()
        r = f.gradient(xp)
        d = -r.copy()

        while np.linalg.norm(r) > eta:
            dA = PHA.projectedHessApprox(f, P, xp, d)
            rho = d.T @ dA
            if rho <= eps * np.linalg.norm(d)**2:
                flag = True
                break

            t = np.linalg.norm(r)**2 / rho
            x = x + t * d
            r_old = r.copy()
            r = r_old + t * dA
            beta = np.linalg.norm(r)**2 / np.linalg.norm(r_old)**2
            d = -r + beta * d

        if flag:
            dp = -f.gradient(xp)
        else:
            dp = x - xp

        tp = PB.projectedBacktrackingSearch(f, P, xp, dp)
        xp = P.project(xp + tp * dp)
        rhs = np.linalg.norm(xp - P.project(xp - f.gradient(xp)))
        eta = min([0.5, np.sqrt(rhs)]) * rhs

    # INCOMPLETE CODE ENDS
    if verbose:
        gradx = f.gradient(xp)
        stationarity = np.linalg.norm(xp - P.project(xp - gradx))
        print('projectedInexactNewtonCG terminated after ', countIter, ' steps with stationarity =', np.linalg.norm(stationarity))

    return xp

