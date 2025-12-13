import numpy as np
from scipy.optimize import Bounds

class Nesterov:
    def __init__(self, lr=0.1, strongly_convex=False, mu=0, bounds=None, log=False):
        """
        Nesterov's accelerated gradient descent.
        
        Arguments:
            lr (float): Step size (estimate of 1/L).
            strongly_convex (bool): Use strongly convex variant if True.
            mu (float): Strong convexity parameter (required if strongly_convex=True).
            bounds (scipy.optimize.Bounds): Optional constraints.
            log (bool): Enable logging.
        """
        self.lr = lr
        self.strongly_convex = strongly_convex
        self.mu = mu
        self.bounds = bounds
        self.log = log
        
        if self.strongly_convex:
            if self.mu <= 0:
                raise ValueError(f"Mu must be > 0 for strongly_convex=True, got {mu}")
            kappa = (1 / self.lr) / self.mu
            self.momentum = (np.sqrt(kappa) - 1) / (np.sqrt(kappa) + 1)
        else:
            self.momentum = 0 # Initial placeholder
            self.alpha = 1.0

    def projection(self, x):
        """Simple projection if bounds are present."""
        if self.bounds is not None:
            return np.clip(x, self.bounds.lb, self.bounds.ub)
        return x

    def nesterov(self, x_0, max_iters, f, f_dx, n, mu0=None):
        """
        Run Nesterov Accelerated Gradient Descent.
        
        x_0: Initial point
        max_iters: Number of iterations
        f: Loss function f(w)
        f_dx: Gradient function f'(w)
        n: Dimension of x (unused, kept for API consistency)
        mu0: Optional argument for smoothing (unused here)
        """
        
        val = []
        points = []
        
       
        x = x_0.copy()
        y = x_0.copy() 
        
        # Initial logging
        curr_loss = f(x)
        val.append(curr_loss)
        points.append(x.copy())
        
        if self.log:
            print(f"Nesterov Start: fval={curr_loss:.5f}, lr={self.lr}")

        for t in range(max_iters):
            grad = f_dx(y)
        
            # x_{k+1} = y_k - lr * grad(y_k)
            x_new = y - self.lr * grad
            #x_new = self.projection(x_new) 

            if not self.strongly_convex:
               
                alpha_new = 0.5 * (1 + np.sqrt(1 + 4 * self.alpha ** 2))
                self.momentum = (self.alpha - 1) / alpha_new
                self.alpha = alpha_new
            
            # y_{k+1} = x_{k+1} + momentum * (x_{k+1} - x_k)
            
            y_new = x_new + self.momentum * (x_new - x)
            #y_new = self.projection(y_new)

            
            x = x_new.copy()
            y = y_new.copy()

            curr_loss = f(x)
            val.append(curr_loss)
            points.append(x.copy())

            if self.log and t % 100 == 0:
                print(f"Iter {t}/{max_iters}, fval: {curr_loss:.5f}")

        return points, val