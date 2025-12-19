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
import warnings

# suppress non-critical warnings printed to the terminal
warnings.filterwarnings('ignore')

# RK4 method
def ode_solve_G(z0, G,epsi_t,mut,n):
    """
    Simplest RK4 ODE initial value solver
    """
    n_steps = 500
    z = z0
    h = np.array([0.01])
    for i_step in range(n_steps):
        k1 = h*G(z,epsi_t,mut,n)
        k2 = h * (G((z+h/2),epsi_t,mut,n))
        k3 = h * (G((z+h/2),epsi_t,mut,n))
        k4 = h * (G((z+h),epsi_t,mut,n))
        k = (1/6)*(k1+2*k2+2*k3+k4)
        z = z + k
    return z
def f(x):
    return (x[0]**2 + x[1]**2 + 3) / (1 + 2*x[0] + 8*x[1])
def g1(x):
    return -x[0]**2 - 2*x[0]*x[1] + 4
def g2(x):
    return -x[0]
def g3(x):
    return -x[1]

g1_dx = grad(g1)
g2_dx = grad(g2)
g3_dx = grad(g3)
g_dx = [g1_dx,g2_dx]
f_dx = grad(f)
bounds = Bounds([0,0],[np.inf,np.inf])
cons = (
        {'type': 'ineq',
          'fun' : lambda x: np.array([-g1(x)]),
          'jac' : lambda x: np.array([-g1_dx(x)])})
def rosen(x,y):
    """The Rosenbrock function"""
    return np.sqrt(np.sum((x-y)**2))
def find_min(y,n):
    x = np.random.rand(1,n).tolist()[0]
    res = minimize(rosen, x, args=(y), jac="2-point",hess=BFGS(),
                constraints=cons,method='trust-constr', options={'disp': False},bounds = bounds)
    return res.x
def run_nonsmooth1(x, max_iters, f, f_dx,n,alpha,mu0):
    res = []
    val = []
    lda = 1 #1e9
    sigma = 0.5 #100
    mut = mu0
    K = 0.7
    res.append(x)
    val.append(f(x))
    x_pre = x
    iter = 0
    for t in range(max_iters):
        y = x - lda*f_dx(x)
        x_pre = x.copy()
        x = find_min(y,n)
        if f(x) - f(x_pre) + sigma*(np.dot(f_dx(x_pre).T,x_pre - x)) <= 0:
            lda = lda
        else:
            lda = K*lda

        iter += 1
        if abs(f(x)-f(x_pre)) <= 1e-6:
            break
        #mut = mut*np.exp(-alpha*t)
        res.append(x)
        val.append(f(x))

    return res,val,iter

def plot_x(sol_all,count,max_iters):
    plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size': 16})
    # Plot each trajectory using its own time axis (handles varying lengths)
    for i in range(min(count, len(sol_all))):
        arr = np.asarray(sol_all[i])
        if arr.ndim == 1:
            # single-point
            arr = arr.reshape(1, -1)
        t = np.arange(arr.shape[0])
        # only label the first plotted series to avoid duplicate legend entries
        label_x1 = r'$x_{1}(t)$' if i == 0 else None
        label_x2 = r'$x_{2}(t)$' if i == 0 else None
        plt.plot(t, arr[:,0], color='red', label=label_x1, linewidth=1)
        plt.plot(t, arr[:,1], color='green', label=label_x2, linewidth=1)
    plt.xlabel('iteration')
    plt.ylabel('x(t)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def print_summary_table(time_run, iteration, value, sol_all):
    """In ra bảng tóm tắt kết quả: Iter | Time | Iterations | Value | x_final"""
    # Prepare header
    headers = ["#", "Time(s)", "Iters", "Value", "x final"]
    # Compute rows
    rows = []
    for i in range(len(time_run)):
        x_final = sol_all[i][-1,:]
        # format x_final compactly (fixed-point, no scientific 'e')
        try:
            x_vals = [float(v) for v in x_final]
        except Exception:
            x_vals = [float(x_final)]
        x_str = '[' + ', '.join(f"{v:.6f}" for v in x_vals) + ']'
        # format value without scientific notation
        val_str = f"{float(value[i]):.6f}"
        rows.append([i+1, f"{time_run[i]:.6f}", f"{iteration[i]}", val_str, x_str])

    # compute column widths
    col_widths = [max(len(str(r[j])) for r in ([headers] + rows)) for j in range(len(headers))]

    # print header
    header_line = ' | '.join(headers[j].ljust(col_widths[j]) for j in range(len(headers)))
    sep_line = '-+-'.join('-'*col_widths[j] for j in range(len(headers)))
    print(header_line)
    print(sep_line)

    # print rows
    for r in rows:
        print(' | '.join(str(r[j]).ljust(col_widths[j]) for j in range(len(r))))
if __name__ == '__main__':
    num = 10 # number of init points
    max_iters = 100 # number of interations
    max_iters1 = 100
    sol_all,sol_all1 = [],[]
    val_all,val_all1 = [],[]
    count = 0
    epsilon = 0.1
    mu0 = np.random.rand(1) # init mu0
    epsi0 = np.random.rand(1) # init epsi0
    alpha = np.random.rand(1) # init alpha
    n = 2 # dimension x
    x_init = np.random.rand(1,n)

    time_run = []
    value = []
    iteration = []

    for i in range(num):
        x0 = np.random.rand(1,n)
        x0 = find_min(x0,n) # init point
        count += 1

        t2 = time.time()
        res1,val1,iter1 = run_nonsmooth1(x0, max_iters1, f, f_dx,n,alpha,mu0)
        e2 = time.time()
        print("GDA: ",e2-t2)
        time_run.append(e2-t2)
        tmp1 = np.array(res1)[:,:]
        sol_all1.append(tmp1)
        print("Value: ",val1[-1])
        value.append(val1[-1])
        iteration.append(iter1)

    # Print summary table instead of individual prints
    print_summary_table(time_run, iteration, value, sol_all1)
    # Plot trajectory
    plot_x(sol_all1,count,max_iters)