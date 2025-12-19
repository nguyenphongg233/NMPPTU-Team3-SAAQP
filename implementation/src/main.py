import numpy as np
from autograd import grad
import autograd.numpy as anp 
from numpy import linalg as LA
from scipy.optimize import minimize
import random
from scipy.optimize import BFGS, SR1
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint, LinearConstraint
import time
import os
import warnings

# suppress non-critical warnings printed to the terminal
warnings.filterwarnings('ignore')


np.random.seed(42)

print("=" * 60)

n = 100  # dimension
#a = np.array([[0.6733, 0.3428, 0.1369, 0.1149, 0.8319, 0.9215, 0.646, 0.7565, 0.5893, 0.9416]])
a=np.random.rand(1,n)
#beta = np.random.rand(1, 1)
beta=0.741271
alpha_val=3 * np.sqrt(n + 1)*beta**(3/2)
#beta = np.random.uniform(0.4, 0.8)
#alpha_val = 2.0 * np.sqrt(n)
L=4*beta**(3/2)*np.sqrt(n)+3*alpha_val
print(f"L: {L}")
e = np.array([i for i in range(n)])
A = np.random.rand(n, n)
b = np.random.rand(n, 1)
#alpha_val = np.random.rand(1, 1)
alpha_val = np.array([[alpha_val]])
beta = np.array([[beta]])


while 2 * alpha_val <= 3 * (beta)**(3/2) * np.sqrt(n):
    alpha_val = 10*np.random.rand(1, 1)

print(f"Problem dimension: {n}")
print(f"Alpha: {alpha_val[0][0]:.6f}")
print(f"Beta: {beta[0][0]:.6f}")

print()

def f(x):
    return anp.dot(a, x.T) + alpha_val * anp.dot(x, x.T) + \
           (beta / anp.sqrt(1 + beta * anp.dot(x, x.T))) * anp.dot(e, x.T)

'''def g1(x):
    """Constraint function: prod(x) - 1 >= 0, i.e., prod(x) >= 1"""
    return -(anp.prod(x) - 1)'''
def g1(x):
   return -(anp.sum(anp.log(x)) - 0)


g1_dx = grad(g1)
f_dx = grad(f)

nonlinear_constraint = NonlinearConstraint(g1, -np.inf, 0, jac=g1_dx, hess=BFGS())
bounds = Bounds(np.array([0.0001 for i in range(n)]), np.array([np.inf for i in range(n)]))

def rosen(x, y):
    """Euclidean distance function for projection"""
    return np.sqrt(np.sum((x - y)**2))

def find_min(y, n):
    x = np.random.rand(1, n).tolist()[0]
    res = minimize(rosen, x, args=(y), jac="2-point", hess=BFGS(),
                   constraints=[nonlinear_constraint], method='trust-constr',
                   options={'disp': False}, bounds=bounds)
    return res.x

def run_gda(x0, max_iters, f, f_dx, n, sigma, lamda, K, eps):
  
    res = []
    val = []
    lda = lamda

    res.append(x0)
    val.append(f(x0))

    x = x0
    x_pre = x0


    print("Running GDA with backtracking line search:")
    for t in range(max_iters):
        y = x - lda * f_dx(x)
        x_pre = x.copy()
        x = find_min(y, n)

        print(f" Iteration {t+1}: f(x) = {f(x).item():.6f}, Step size (lambda) = {np.asarray(lda).item():.6f}")
        if f(x) - f(x_pre) + sigma * (np.dot(f_dx(x_pre).T, x_pre - x)) <= 0:
            lda = lda
        else:
            lda = K * lda

        res.append(x)
        val.append(f(x))

        if len(val) > 1 and abs(val[-1] - val[-2]) < eps:
            break

    return res, val

def run_gd(x0, max_iters, f, f_dx, n, lamda, eps):
    
    res = []
    val = []
    lda = lamda

    res.append(x0)
    val.append(f(x0))

    x = x0

    print("Running GD with fixed step size: ", np.asarray(lamda).item())

    for t in range(max_iters):
        y = x - lda * f_dx(x)
        x = find_min(y, n)

        res.append(x)
        val.append(f(x))
        print(f" Iteration {t+1}: f(x) = {f(x).item():.6f}")
        # Check convergence
        if len(val) > 1 and abs(val[-1] - val[-2]) < eps:
            break

    return res, val

# ================ Experiment Configuration ================
num_trials = 1  # number of different initialization points
max_iters = 50  # maximum iterations (safety limit)
eps = 0.0001  # convergence tolerance: |f(x_t) - f(x_{t-1})| < eps
sigma = 0.5
lamda_gda=5/L
lamda_gd=1/L
K = np.random.rand(1,1)

# print("" )
# print("Initial single-run (n=100) removed. Use `run_sweep()` or call `run_experiment_for_n()` to execute experiments.")


# ================ Optional: sweep over multiple problem sizes ================
def run_experiment_for_n(local_n, seed=None):
    global n, a, e, A, b, alpha_val, beta, L, bounds, nonlinear_constraint, g1_dx, f_dx, lamda_gda, lamda_gd, K

    if seed is not None:
        np.random.seed(seed)
    
    print('\n' + '=' * 60)
    n = local_n
    a = np.random.rand(1, n)
    beta = 0.741271
    alpha_val = 3 * np.sqrt(n + 1) * beta**(3/2)
    alpha_val = np.array([[alpha_val]])
    beta = np.array([[beta]])

    print(f"n: {n}, Alpha: {alpha_val[0][0]:.6f}, Beta: {beta[0][0]:.6f}")

    e = np.array([i for i in range(n)])
    A = np.random.rand(n, n)
    b = np.random.rand(n, 1)

    L = 4 * beta**(3/2) * np.sqrt(n) + 3 * alpha_val

    # redefine constraints/gradients that depend on n
    g1_dx = grad(g1)
    f_dx = grad(f)
    nonlinear_constraint = NonlinearConstraint(g1, -np.inf, 0, jac=g1_dx, hess=BFGS())
    bounds = Bounds(np.array([0.0001 for i in range(n)]), np.array([np.inf for i in range(n)]))

    lamda_gda = 5 / L
    lamda_gd = 1 / L
    K = np.random.rand(1, 1)

    print(f"Running experiment for n={n}...")
    # single-trial runs for comparison
    x0 = np.random.rand(1, n)
    #print("Initial point generated x0:", x0)
    x0 = find_min(x0, n)

    
    t0 = time.time()
    trajectory_gda, values_gda = run_gda(x0, max_iters, f, f_dx, n, sigma, lamda_gda, K, eps)
    t1 = time.time()
    gda_time = t1 - t0
    gda_final = values_gda[-1] if isinstance(values_gda[-1], (int, float)) else values_gda[-1].item()
    gda_iters = len(values_gda) - 1

    # GD
    x0 = np.random.rand(1, n)
    x0 = find_min(x0, n)
    t0 = time.time()
    trajectory_gd, values_gd = run_gd(x0, max_iters, f, f_dx, n, lamda_gd, eps)
    t1 = time.time()
    gd_time = t1 - t0
    gd_final = values_gd[-1] if isinstance(values_gd[-1], (int, float)) else values_gd[-1].item()
    gd_iters = len(values_gd) - 1

    return {
        'n': n,
        'gda_final': float(gda_final),
        'gda_iters': int(gda_iters),
        'gda_time': float(gda_time),
        'gd_final': float(gd_final),
        'gd_iters': int(gd_iters),
        'gd_time': float(gd_time)
    }


def run_sweep(ns=[10, 20, 50, 100, 200, 500], seed=42):
    print('\n' + '=' * 80)
    print('Sweep: running experiments for n in', ns)
    print('=' * 80)

    results = []
    for local_n in ns:
        print(f"Running n={local_n}...")
        res = run_experiment_for_n(local_n, seed=seed)
        results.append(res)

    # print comparison table
    print('\nComparison table:')
    hdr = f"{'n':>6} | {'GDA f(x*)':>14} | {'#Iter':>6} | {'Time(s)':>8} || {'GD f(x*)':>14} | {'#Iter':>6} | {'Time(s)':>8}"
    print(hdr)
    print('-' * len(hdr))
    for r in results:
        print(f"{r['n']:6d} | {r['gda_final']:14.6f} | {r['gda_iters']:6d} | {r['gda_time']:8.4f} || {r['gd_final']:14.6f} | {r['gd_iters']:6d} | {r['gd_time']:8.4f}")

    print('\nSweep complete.')
    return results


if __name__ == '__main__':
    # run the sweep when script executed directly
    ns = [1000]
    run_sweep(ns)