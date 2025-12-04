import algorithms.gda as gda_module
import algorithms.gd as gd_module
import numpy as np
from autograd import grad
import autograd.numpy as np1
from numpy import linalg as LA, sqrt
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
import time
from scipy.optimize import BFGS,SR1
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint
from scipy.optimize import NonlinearConstraint

def f(x):
    return np1.dot(a,x.T) + alpha*np1.dot(x,x.T) + (beta/np1.sqrt(1+beta*np1.dot(x,x.T)))*np1.dot(e,x.T)
def g1(x):
    return 1- np1.prod(x)
def g2(x):
    return 1 - np1.sum(x)
g1_dx = grad(g1)
g2_dx = grad(g2)
f_dx = grad(f)


bounds = Bounds([0,0],[np.inf,np.inf])
cons = (
        {'type': 'ineq',
          'fun' : lambda x: np.array([-g1(x)]),
          'jac' : lambda x: np.array([-g1_dx(x)])})

def plot_x(sol_all,sol_all1,count,max_iters):
    t = [i for i in range(max_iters+1)]
    plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size': 16})
    for i in range(count):
        plt.plot(t, sol_all[i][:,0],color='red',label=r'$x_{1}(t)$',linewidth=1)
        plt.plot(t, sol_all[i][:,1],color='green',label=r'$x_{2}(t)$',linewidth=1)
        plt.plot(t, sol_all1[i][:,0],color='blue',label=r'$x_{1}(t)$',linewidth=1)
        plt.plot(t, sol_all1[i][:,1],color='yellow',label=r'$x_{2}(t)$',linewidth=1)
    plt.xlabel('iteration')
    plt.ylabel('x(t)')
    #plt.legend([r'$x_{1}(t)$',r'$x_{2}(t)$']) #,r'$x_{4}(t)$',r'$x_{5}(t)$',r'$x_{6}(t)$',r'$x_{7}(t)$',r'$x_{8}(t)$',r'$x_{9}(t)$',r'$x_{10}(t)$'])
    plt.legend([r'GDA: $x_{1}(t)$',r'GDA: $x_{2}(t)$',r'GD: $x_{1}(t)$',r'GD: $x_{2}(t)$']) #,r'$x_{4}(t)$',r'$x_{5}(t)$',r'$x_{6}(t)$',r'$x_{7}(t)$',r'$x_{8}(t)$',r'$x_{9}(t)$',r'$x_{10}(t)$'])
    plt.show()
    
if __name__ == "__main__":
    n = 20
    a = np.random.rand(1,n)
    beta = np.random.rand(1,1)
    e = np.array([i for i in range(n)])
    A = np.random.rand(n,n)
    b = np.random.rand(n,1)
    alpha = 10 * np.random.rand(1,1)
    print("alpha:",alpha)
    while 2 * alpha <= 3 * (beta) ** (3/2) * sqrt(n):
        alpha = np.random.rand(1,1)
        print("alpha:",alpha)
    B = A.tolist()
    C = b.reshape(1,n).tolist()[0]
    L = 4 * beta**(3/2) * sqrt(n) + 3 * alpha
    lamda = 1 / L
    lamda0 = 5/L
    print("A:",B)
    print("b:",C)

    nonlinear_constraint = NonlinearConstraint(g1, -np.inf, 0, jac=g1_dx, hess=BFGS())
    bounds = Bounds([0.0001 for i in range(n)], [np.inf for i in range(n)])

    num = 2
    max_iters = 50

    sol_all,sol_all1 = [],[]
    val_all,val_all1 = [],[]
    count = 0

    gda_instance = gda_module.GDA()
    gda_instance.set_bounds([0.0001 for i in range(n)], [np.inf for i in range(n)])
    gda_instance.set_constraints((nonlinear_constraint,))
    gda_instance.set_lamda(lamda0)

    gd_instance = gd_module.GD()
    gd_instance.set_bounds([0.0001 for i in range(n)], [np.inf for i in range(n)])
    gd_instance.set_constraints((nonlinear_constraint,))
    gd_instance.set_lamda(lamda)

    for i in range(num):
        x0 = np.random.rand(1,n)
        x0 = gda_instance.projection(x0,n) # init point
        g1x = g1(x0)
        count += 1

        res,val = gda_instance.gda(x0, max_iters, f, f_dx,n)
        print("GDA-Agorithms : Initial point:",x0,"Final point:",res[-1],"Final value:",val[-1])
        tmp = np.array(res)[:,:]
        sol_all.append(tmp)
        val_all.append(val)

        res1,val1 = gd_instance.gd(x0, max_iters, f, f_dx,n)
        print("GD-Agorithms : Initial point:",x0,"Final point:",res1[-1],"Final value:",val1[-1])
        tmp1 = np.array(res1)[:,:]
        sol_all1.append(tmp1)
        val_all1.append(val1)
    
    plot_x(sol_all,sol_all1,count,max_iters)



