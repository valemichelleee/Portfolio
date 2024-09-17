# Optimization for Engineers - Dr.Johannes Hild
# Preconditioned Conjugate Gradient Solver

# Purpose: PregCGSolver finds y such that norm(A * y - b) <= delta using incompleteCholesky as preconditioner

# Input Definition:
# A: real valued matrix nxn
# b: column vector in R ** n
# delta: positive value, tolerance for termination. Default value: 1.0e-6.
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# x: column vector in R ^ n(solution in domain space)

# Required files:
# L = incompleteCholesky(A, 1.0e-3, delta) from IncompleteCholesky.py
# y = LLTSolver(L, r) from LLTSolver.py

# Test cases:
# A = np.array([[4, 1, 0], [1, 7, 0], [ 0, 0, 3]], dtype=float)
# b = np.array([[5], [8], [3]], dtype=float)
# delta = 1.0e-6
# x = PrecCGSolver(A, b, delta, 1)
# should return x = [[1], [1], [1]]

# A = np.array([[484, 374, 286, 176, 88], [374, 458, 195, 84, 3], [286, 195, 462, -7, -6], [176, 84, -7, 453, -10], [88, 3, -6, -10, 443]], dtype=float)
# b = np.array([[1320], [773], [1192], [132], [1405]], dtype=float)
# delta = 1.0e-6
# x = PrecCGSolver(A, b, delta, 1)
# should return approx x = [[1], [0], [2], [0], [3]]


import numpy as np
import incompleteCholesky as IC
import LLTSolver as LLT


def matrnr():
    # set your matriculation number here
    matrnr = 23358926
    return matrnr


def PrecCGSolver(A: np.array, b: np.array, delta=1.0e-6, verbose=0):

    if verbose:
        print('Start PrecCGSolver...')

    countIter = 0

    L = IC.incompleteCholesky(A)
    x = LLT.LLTSolver(L, b)
    r = A @ x - b
    # INCOMPLETE CODE STARTS
    d = -LLT.LLTSolver(L, r)

    while np.linalg.norm(r) > delta:
        countIter += 1
        d_tilde = A @ d
        rho = d.T @ d_tilde
        t = (r.T @ LLT.LLTSolver(L, r)) / rho
        x = x + t * d
        r_old = r
        r = r_old + t * d_tilde
        beta = (r.T @ LLT.LLTSolver(L, r)) / (r_old.T @ LLT.LLTSolver(L, r_old))
        d = -(LLT.LLTSolver(L, r)) + beta * d

    # INCOMPLETE CODE ENDS

    if verbose:
        print('precCGSolver terminated after ', countIter, ' steps with norm of residual being ', np.linalg.norm(r))

    return x