import algorithms.gda as gda_module
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

def theta(x,mu):
    if np1.abs(x) >= (mu/2):
        return np1.abs(x)
    else:
        return x**2/mu + mu/4 
def f_mu(x,mu):
    return (np1.exp(theta(x[1]-3,mu)) - 30) / (x[0]**2 + x[2]**2 + 2*(x[3]**2) + 4)
def f(x):
    return (np1.exp(np1.abs(x[1]-3)) - 30) / (x[0]**2 + x[2]**2 + 2*(x[3]**2) + 4)
def g1(x):
    return ((x[0] + x[2])**3 + 2*(x[3])**2) - 10
def g2(x):
    return ((x[1] - 1)**2) - 1
def g3(x):
    x = np.array(x)
    A = np.array([[2,4,1,0]])
    b = np.array([[-1]])
    return (A@(x.T) - b.T).tolist()[0][0] # 
g1_dx = grad(g1)
g2_dx = grad(g2)
g3_dx = grad(g3)

f_dx = grad(f)
f_dx_mu = grad(f_mu)
cons = ({'type': 'eq',
          'fun' : lambda x: np.array([g3(x)]),
          'jac' : lambda x: np.array([2,4,1,0])},
        {'type': 'ineq',
          'fun' : lambda x: np.array([-g1(x)]),
          'jac' : lambda x: np.array([-g1_dx(x)])},
         {'type': 'ineq',
          'fun' : lambda x: np.array([-g2(x)]),
          'jac' : lambda x: np.array([-g2_dx(x)])})

def plot_x(sol_all,count,max_iters):
    t = [i for i in range(max_iters+1)]
    plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size': 16})
    for i in range(count):
        plt.plot(t, sol_all[i][:,0],color='red',label=r'$x_{1}(t)$',linewidth=1)
        plt.plot(t, sol_all[i][:,1],color='green',label=r'$x_{2}(t)$',linewidth=1)
        plt.plot(t, sol_all[i][:,2],color='blue',label=r'$x_{3}(t)$',linewidth=1)
        plt.plot(t, sol_all[i][:,3],color='orange',label=r'$x_{4}(t)$',linewidth=1)
    plt.xlabel('iteration')
    plt.ylabel('x(t)')
    plt.legend([r'$x_{1}(t)$',r'$x_{2}(t)$',r'$x_{3}(t)$',r'$x_{4}(t)$']) #,r'$x_{4}(t)$',r'$x_{5}(t)$',r'$x_{6}(t)$',r'$x_{7}(t)$',r'$x_{8}(t)$',r'$x_{9}(t)$',r'$x_{10}(t)$'])
    plt.show()

    
if __name__ == "__main__":
    num = 20 # number of init points
    max_iters = 1000
    sol_all,sol_all1 = [],[]
    val_all,val_all1 = [],[]

    count = 0
    epsilon = 0.1

    mu0 = np.random.rand(1) # init mu0 
    epsi0 = np.random.rand(1) # init epsi0
    alpha = np.random.rand(1) # init alpha
    n = 4 # dimension x
    x_init = np.random.rand(1,n)
    gda_instance = gda_module.GDA()
    gda_instance.set_bounds([-np.inf,-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf,np.inf])
    gda_instance.set_constraints(cons)
    gda_instance.set_log(True)

    for i in range(num):
        x0 = np.random.rand(1,n)
        x0 = gda_instance.projection(x0,n) # init point
        count += 1

        t2 = time.time()
        res1,val1 = gda_instance.gda(x0, max_iters, f_mu, f_dx_mu,n,alpha,mu0)
        e2 = time.time()
        print("time_run: ",e2-t2,"x_init: ",x0,"f_val: ",f(res1[-1]),"x_end: ",res1[-1],"\n")
        tmp1 = np.array(res1)[:,:]
        sol_all1.append(tmp1)

    plot_x(sol_all1,count,max_iters)