# Optimization for Engineers - Dr.Johannes Hild
# Augmented Lagrangian objective

# Purpose: Provides .objective() and .gradient() of the augmented Lagrangian mapping x -> A = f(x) + alpha*h(x)+ 0.5*gamma*h(x)**2

# Input Definition:
# f: objective class with methods .objective() and .gradient(), objective
# h: objective class with methods .objective() and .gradient(), equality constraint
# alpha: real value, current guess for Lagrangian multiplier for h.
# gamma: positive value, penalty parameter.

# Output Definition:
# .objective(): real number, evaluation of augmentedLagrangianObjective at x
# .gradient(): real column vector in R^n, evaluation of the gradient at x

# Required files:
# <none>

# Test cases:
# A = np.array([[2, 0], [0, 2]], dtype=float)
# B = np.array([[0], [0]], dtype=float)
# C = 1
# myObjective = quadraticObjective(A, B, C)
# D = np.array([[2, 0], [0, 2]], dtype=float)
# E = np.array([[0], [0]], dtype=float)
# F = -1
# myConstraint = quadraticObjective(D, E, F)
# x0 = np.array([[2],[2]])
# alpha = -1
# gamma = 10
# myAugLag = augmentedLagrangianObjective(myObjective, myConstraint, alpha, gamma)
# should return
# myAugLag.objective(x0) close to 247
# myAugLag.gradient(x0) close to [[280], [280]]


import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23358926
    return matrnr


class augmentedLagrangianObjective:

    def __init__(self, f, h, alpha, gamma):
        if gamma <= 0:
            raise TypeError('range of gamma is wrong!')

        self.f = f
        self.h = h
        self.alpha = alpha
        self.gamma = gamma

    def objective(self, x: np.array):
        myObjective = self.f.objective(x)
        # INCOMPLETE CODE STARTS

        myObjective += self.alpha * self.h.objective(x) + 0.5 * self.gamma * (self.h.objective(x)**2)

        # INCOMPLETE CODE ENDS

        return myObjective

    def gradient(self, x: np.array):
        myGradient = self.f.gradient(x)
        # INCOMPLETE CODE STARTS

        myGradient += self.alpha * self.h.gradient(x) + self.gamma * self.h.objective(x) * self.h.gradient(x)

        # INCOMPLETE CODE ENDS

        return myGradient
