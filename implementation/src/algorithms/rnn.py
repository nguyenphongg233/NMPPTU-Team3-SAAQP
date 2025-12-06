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


class RNN:
    def __init__(self, A=None, b=None, step=0.05, n_steps=100, log=False):
        self.A = np.array(A) if A is not None else None
        self.b = np.array(b).reshape(-1) if b is not None else None

        self.h = np.array([step])
        self.n_steps = n_steps
        self.log = log

    def set_step(self, step):
        self.h = step

    def set_n_steps(self, n_steps):
        self.n_steps = n_steps

    def set_log(self, log):
        self.log = log

    def Phi(self, s):
        if s > 0:
            return 1
        elif s == 0:
            return np.random.rand(1)
        return 0

   
    def ode_solve_G(self, z0, G):
        z = z0

        for _ in range(self.n_steps):
            h = self.h

            k1 = h * G(z)
            k2 = h * G(z + 0.5 * h)
            k3 = h * G(z + 0.5 * h)
            k4 = h * G(z + h)
            k = (1/6)*(k1+2*k2+2*k3+k4)
            z = z + k
            z=np.array([z0]).reshape(-1,1)
        return z

    def build_G(self, f_dx, g_i, grad_gi):

        def G(x_vec):
            x = x_vec.reshape(-1)

            gi_val = g_i(x)
            gi_grad = grad_gi(x).reshape(-1, 1)

            cxt = 1 - self.Phi(gi_val)
            Px = self.Phi(gi_val) * gi_grad

            if self.A is not None:
                Ax_b = float(self.A @ x - self.b)
                eq_term = (2 * self.Phi(Ax_b) - 1) * self.A.T
                eq_term = eq_term.reshape(-1, 1)
            else:
                eq_term = 0

            Fx = (-cxt * f_dx(x)).reshape(-1, 1)

            return Fx - Px - eq_term

        return G

  
    def rnn(self, x0, max_iters, f, f_dx, g_i, grad_gi):
        xt = x0.reshape(-1, 1)
        history = [xt.reshape(1, -1).tolist()[0]]

        G = self.build_G(f_dx, g_i, grad_gi)

        for t in range(max_iters):
            xt = self.ode_solve_G(xt, G)
            history.append(xt.reshape(1, -1).tolist()[0])

            if self.log:
                print(f"[RNN] iteration {t+1}/{max_iters}, f_val = {f(xt.reshape(-1))}")

        return history, xt.reshape(1, -1)
