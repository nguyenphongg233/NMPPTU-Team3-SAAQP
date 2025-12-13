import numpy as np
from autograd import grad
import autograd.numpy as np1
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
import time
from scipy.optimize import BFGS,SR1
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint


class GDA:

    def __init__(self, sigma=0.1, lamda=1.0, K=10, bounds=None, cons=None,log = False):
        self.sigma = sigma
        self.lamda = lamda
        self.K = K
        self.bounds = bounds #if bounds is not None else Bounds([0, 0], [np.inf, np.inf])
        self.cons = cons #if cons is not None else ({'type': 'ineq', 'fun': lambda x: np.array([1]), 'jac': lambda x: np.array([2])},)
        self.log = log

    def set_sigma(self, sigma):
        self.sigma = sigma
    def set_lamda(self, lamda):
        self.lamda = lamda
    def set_K(self, K):
        self.K = K
    def set_bounds(self, lower, upper):
        self.bounds = Bounds(lower, upper)
    def set_constraints(self, constraint):
        self.cons = constraint
    def set_log(self, log):
        self.log = log

    def rosen(self,x,y):
        """The Rosenbrock function"""
        return np.sqrt(np.sum((x-y)**2))

    def projection(self, y,n):
        x = np.random.rand(1,n).tolist()[0]
        res = minimize(self.rosen, x, args=(y), jac="2-point",hess=BFGS(),
                    constraints=self.cons,method='trust-constr', options={'disp': False},bounds = self.bounds)
        return res.x
        
    def gda(self, x_0, max_iters, f, f_dx,n,alpha = 0,mu0 = None):
        # (initial_point, max_iters, objective_function, gradient_function, dimension)
        # GDA algorithm
        sigma = self.sigma
        lda = self.lamda
        #K = np.random.rand(1,1)
        K = self.K
        val = []
        point = []

        point.append(x_0)
        mut = mu0
        val.append(f(x_0) if mu0 is None else f(x_0,mut))

        x_pre = x_0
        for t in range(max_iters):
            y = x_0 - lda*f_dx(x_0) if mu0 is None else x_0 - lda*f_dx(x_0,mut)
            x_pre = x_0.copy()
            # skip the projection if not need
            if self.bounds or self.cons: x_0 = self.projection(y,n)
            else: x_0 = y
            value = 0
            if mu0 is not None:
                expected_decrease = sigma*(np.dot(f_dx(x_pre,mut).T,x_pre - x_0))
                value = f(x_0,mut) - f(x_pre,mut) + expected_decrease
            else:
                expected_decrease = sigma*(np.dot(f_dx(x_pre).T,x_pre - x_0))
                value = f(x_0) - f(x_pre) + expected_decrease

            if value <= 0:
                lda = lda
            else:
                lda = K*lda

            if mu0 is not None:
                mut = mut * np.exp(-alpha * t)
            point.append(x_0)
            val.append(f(x_0,mut) if mu0 is not None else f(x_0))
            if self.log:
                print(f"Iteration {t+1}/{max_iters}, fval: {val[-1]}, lda: {lda}, x: {x_0}")

        return point,val      