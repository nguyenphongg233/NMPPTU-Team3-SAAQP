# from IPython import display
# display.Image("8143e0da24f78ea9d7e6.jpg")
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

# dimension n (can be changed as needed)
n = 70
# Build A_mat dynamically: first n/2 entries are 1, last n/2 entries are 3
A_mat = np.array([ [1] * (n//2) + [3] * (n - n//2) ])
b_vec = np.array([16])

# RK4 method
def ode_solve_G(z0, G, n_dim):
    """
    Simplest RK4 ODE initial value solver
    """
    n_steps = 500
    z = z0
    h = np.array([0.05])
    for i_step in range(n_steps):
        k1 = h*G(z)
        k2 = h * (G((z+h/2)))
        k3 = h * (G((z+h/2)))
        k4 = h * (G((z+h)))
        k = (1/6)*(k1+2*k2+2*k3+k4)
        z = z.reshape(n_dim, 1)
        z = z + k
    return z

def f(x):
    """
    Objective function f(x) as defined in the problem.
    x: vector of variables
    q: vector of parameters
    """
    tmp = n*[1]
    q = np1.array(tmp)
    return -np1.exp(-np1.sum((x**2) / (q**2)))

def g_i(x):
    """
    Inequality constraint g_i(x) - dimension agnostic
    """
    squared_terms = x**2
    g_x_i = np1.sum(squared_terms) - 20
    return g_x_i

def derivative_g_i(x):
    """
    Derivative of the inequality constraint g_i(x) with respect to x.
    """
    return 2 * x

def g3(x):
    x_np = np.array(x)
    A = A_mat
    b = b_vec
    val = np.dot(A, x_np) - b
    return val[0]

f_dx = grad(f)

cons = ({'type': 'eq',
          'fun' : lambda x: np.array([g3(x)]),
          'jac' : lambda x: A_mat},
        {'type': 'ineq',
          'fun' : lambda x: np.array([-g_i(x)]),
          'jac' : lambda x: np.array([-grad(g_i)(x)])})

def rosen(x,y):
    """The Rosenbrock function"""
    return np.sqrt(np.sum((x-y)**2))

def find_min(y,n):
    x = np.random.rand(1,n).tolist()[0]
    res = minimize(rosen, x, args=(y), jac="2-point",hess=BFGS(),
                constraints=cons,method='trust-constr', options={'disp': False})
    return res.x

def run_nonsmooth1(x, max_iters, f, f_dx,n,alpha,mu0):
    res = []
    val = []
    lda = 1
    sigma = 0.1
    mut = mu0
    K = np.random.rand(1,1)
    res.append(x)
    val.append(f(x))
    x_pre = x
    for t in range(max_iters):
        y = x - lda*f_dx(x)
        x_pre = x.copy()
        x = find_min(y,n)
        if f(x) - f(x_pre) + sigma*(np.dot(f_dx(x_pre).T,x_pre - x)) <= 0:
            lda = lda
        else:
            lda = K*lda
        res.append(x)
        val.append(f(x))
    return res,x

def Phi(s):
    if s > 0:
        return 1
    elif s == 0:
        return np.random.rand(1)
    return 0

def G(x, n_dim):
    """Neural network gradient field"""
    gx = [g_i(x)]
    g_dx = [grad(g_i)]
    c_xt = 1.
    Px = np.zeros((n_dim, 1))

    for (i,j) in zip(gx, g_dx):
        c_xt *= (1-Phi(i))
        Px += np.array([Phi(i)*j(x)]).reshape(n_dim, 1)
    
    A = A_mat
    b = b_vec
    c_xt *= (1-Phi(np.abs(A@(x) - b)))
    eq_constr_dx = ((2*Phi(A@(x)-b)-1)*A.T)
    
    return np.array([-c_xt*f_dx(x)]).reshape(n_dim, 1) - Px - eq_constr_dx

def run_nonsmooth(x0, max_iters, n_dim):
    xt = x0
    res = []
    res.append(xt.tolist())
    for t in range(max_iters):
        xt = ode_solve_G(xt, lambda z: G(z, n_dim), n_dim)
        res.append(xt.reshape(1, n_dim).tolist()[0])
    return res, xt.reshape(1, n_dim)



def main_GDA(num, max_iters, n, alpha, mu0):
    print("\n" + "="*80)
    print("RUNNING GDA ALGORITHM")
    print("="*80)
    sol_all1 = []
    for i in range(num):
        x0 = np.random.rand(1, n)
        print(f"Initial random x0: {x0[:,:3]}...")
        x0 = find_min(x0, n)
        print(f"Feasible x0: {x0[:3]}...")
        print(f"f(x0) = {f(x0):.8f}")
        
        start_time = time.time()
        _, xt = run_nonsmooth1(x0, max_iters, f, f_dx, n, alpha, mu0)
        elapsed = time.time() - start_time
        
        print(f"\nResult GDA - x*: {xt[:3]}...")
        print(f"f(x*) = {f(xt):.8f}")
        print(f"Value -ln(-f(x*)) of GDA: {-np.log(-f(xt)):.8f}")
        print(f"Time: {elapsed:.4f}s")
        sol_all1.append(xt)
    return sol_all1

def main_RNN(num, max_iters, n_dim):
    print("\n" + "="*80)
    print("RUNNING RNN ALGORITHM")
    print("="*80)
    sol_all = []
    for i in range(num):
        x0 = np.random.rand(1, n_dim)
        print(f"Initial random x0: {x0[:,:3]}...")
        x0 = find_min(x0, n_dim)
        print(f"Feasible x0: {x0[:3]}...")
        print(f"f(x0) = {f(x0):.8f}")
        
        start_time = time.time()
        _, xt = run_nonsmooth(x0, max_iters, n_dim)
        elapsed = time.time() - start_time
        
        print(f"\nResult RNN - x*: {xt[0,:3]}...")
        print(f"f(x*) = {f(xt):.8f}")
        print(f"Value -ln(-f(x*)) of RNN: {-np.log(-f(xt)):.8f}")
        print(f"Time: {elapsed:.4f}s")
        sol_all.append(xt)
    return sol_all

if __name__ == '__main__':
    print("="*80)
    print("Example 04: GDA vs RNN Comparison")
    print("="*80)
    
    num = 1  # Number of starting points
    max_iters = 500  # Maximum number of iterations
    max_iters_rnn = 10*max_iters
    n_dim = 70  # Size of x (can be changed)
    
    n = n_dim
    A_mat = np.array([ [1] * (n//2) + [3] * (n - n//2) ])
    b_vec = np.array([16])
    
    alpha = np.random.rand(1)  # Alpha parameter
    mu0 = np.random.rand(1)  # Mu0 parameter

    print(f"\nParameters:")
    print(f"  - Dimension (n): {n}")
    print(f"  - Max iterations: {max_iters}")
    print(f"  - A_mat shape: {A_mat.shape}")
    print(f"  - A_mat: {A_mat[0]}")
    print(f"  - b_vec: {b_vec}\n")

    result_GDA = main_GDA(num, max_iters, n, alpha, mu0)

    result_RNN = main_RNN(num, max_iters_rnn, n_dim)

    print("\n" + "="*80)
    
