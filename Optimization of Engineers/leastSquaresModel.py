# Optimization for Engineers - Dr.Johannes Hild
# Least squares model objective

# Purpose: Provides .residual() and .jacobian() of the least squares mapping p -> 0.5*sum_k (model(xData_k,p)-fData_k)**2

# Input Definition:
# model: objective class with methods .objective() and .gradient() for data evaluation
# and .setParameters() and .parameterGradient()
# p: column vector in R**m (parameter space)
# xData: matrix in R**nxN (measure points). xData[:,k].reshape((n,1)) returns the k-th measure point as column vector.
# fData: row vector in R**1xN (measure results). fData[:,k] returns the k-th measure result as a scalar.

# Output Definition:
# residual(): column vector in R**N, the k-th entry is model(xData_k,p)-fData_k
# jacobian(): matrix in R**Nxm, the [k,j]-th entry returns: partial derivative with respect to p_j of (model(xData_k,p)-fData_k)

# Required files:
# <none>

# Test cases:
# p0 = np.array([[2],[3]])
# myObjective =  simpleValleyObjective(p0)
# xk = np.array([[0, 0, 1, 2], [1, 2, 3, 4]])
# fk = np.array([[2, 3, 2.54, 4.76]])
# myErrorVector = leastSquaresModel(myObjective, xk, fk)
# should return
# myErrorVector.residual(p0) close to [[2], [3], [10], [20]]
# myErrorVector.jacobian(p0) = [[0, 1], [1, 1], [4, 1],  [9, 1]]

import numpy as np


def matrnr():
    # set your matriculation number here
    matrnr = 23358926
    return matrnr


class leastSquaresModel:

    def __init__(self, model, xData: np.array, fData: np.array):
        self.model = model
        self.xData = xData
        self.fData = fData
        self.N = fData.shape[1]
        self.n = xData.shape[0]

    def residual(self, p: np.array):
        self.model.setParameters(p)
        myResidual = np.zeros((self.N, 1))

        # INCOMPLETE CODE STARTS

        for k in range(self.N):
            x_k = self.xData[:, k].reshape((self.n, 1))
            myResidual[k] = self.model.objective(x_k) - self.fData[:, k]

        # INCOMPLETE CODE ENDS

        return myResidual

    def jacobian(self, p: np.array):
        self.model.setParameters(p)
        myJacobian = np.zeros((self.N, p.shape[0]))

        # INCOMPLETE CODE STARTS

        for k in range(self.N):
            x_k = self.xData[:, k].reshape((self.n, 1))
            self.model.objective(x_k)
            myJacobian[k, :] = self.model.parameterGradient(x_k).reshape((1, -1))

        # INCOMPLETE CODE ENDS

        return myJacobian
