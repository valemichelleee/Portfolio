# Optimization for Engineers - Dr.Johannes Hild
# Wolfe-Powell line search

# Purpose: Find t to satisfy f(x+t*d)<=f(x) + t*sigma*gradf(x).T@d and gradf(x+t*d).T@d >= rho*gradf(x).T@d

# Input Definition:
# f: objective class with methods .objective() and .gradient()
# x: column vector in R ** n(domain point)
# d: column vector in R ** n(search direction)
# sigma: value in (0, 1 / 2), marks quality of decrease. Default value: 1.0e-3
# rho: value in (sigma, 1), marks quality of steepness. Default value: 1.0e-2
# verbose: bool, if set to true, verbose information is displayed

# Output Definition:
# t: t is set, such that t satisfies both Wolfe - Powell conditions

# Required files:
# < none >

# Test cases:
# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.01], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=1

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-1.2], [1]])
# d = np.array([[0.1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=16

# p = np.array([[0], [1]])
# myObjective = simpleValleyObjective(p)
# x = np.array([[-0.2], [1]])
# d = np.array([[1], [1]])
# sigma = 1.0e-3
# rho = 1.0e-2
# t = WolfePowellSearch(myObjective, x, d, sigma, rho, 1)
# should return t=0.25

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23358926
    return matrnr


# Define the objective class
class QuadraticObjective:
    def objective(self, x):
        return np.dot(x.T, x)

    def gradient(self, x):
        return 2 * x


# Define the initial point x0 and search direction d0
x0 = np.array([[-0.001], [-0.001]])
d0 = np.array([[0.5], [0.5]])
sigma = 0.25
rho = 0.75

def WolfePowellSearch(f, x: np.array, d: np.array, sigma=1.0e-3, rho=1.0e-2, verbose=0):
    fx = f.objective(x)
    gradx = f.gradient(x)
    descent = gradx.T @ d

    if descent >= 0:
        raise TypeError('descent direction check failed!')

    if sigma <= 0 or sigma >= 0.5:
        raise TypeError('range of sigma is wrong!')

    if rho <= sigma or rho >= 1:
        raise TypeError('range of rho is wrong!')

    if verbose:
        print('Start WolfePowellSearch...')

    def WP1(ft, s):
        isWP1 = ft <= fx + s*sigma*descent
        return isWP1

    def WP2(gradft: np.array):
        isWP2 = gradft.T @ d >= rho*descent
        return isWP2

    t = 1
    # INCOMPLETE CODE STARTS

    if not WP1(f.objective(x + t * d), t):
        #backtracking
        t /= 2
        while not WP1(f.objective(x + t * d), t):
            t /= 2
        t_minus = t
        t_plus = 2 * t
    elif WP2(f.gradient(x + t * d)):
        return t
    else:
        #fronttracking
        t *= 2
        while WP1(f.objective(x + t * d), t):
            t *= 2
        t_minus = t/2
        t_plus = t

    while not WP2(f.gradient(x + t * d)):
        #refining
        t = (t_minus + t_plus) / 2
        if WP1(f.objective(x + t * d), t):
            t_minus = t
        else:
            t_plus = t
    t = t_minus

    # INCOMPLETE CODE ENDS

    if verbose:
        xt = x + t * d
        fxt = f.objective(xt)
        gradxt = f.gradient(xt)
        print('WolfePowellSearch terminated with t=', t)
        print('Wolfe-Powell: ', fxt, '<=', fx+t*sigma*descent, ' and ', gradxt.T @ d, '>=', rho*descent)

    return t

# Define the objective instance
objective = QuadraticObjective()

# Perform Wolfe-Powell search
t_star = WolfePowellSearch(objective, x0, d0, sigma, rho, verbose=1)
print("Optimal step size t*:", t_star)

