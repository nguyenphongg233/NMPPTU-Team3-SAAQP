import numpy as np
import autograd.numpy as np1
from autograd import grad
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, BFGS
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import algorithms.gda  as gda_module
import algorithms.rnn  as rnn_module

# dimension n (can be changed as needed)
n = 10
# Bu2ild A_mat dynamically: first n/2 entries are 1, last n/2 entries are 3
A_mat = np.array([ [1] * (n//2) + [3] * (n - n//2) ])
b_vec = np.array([16])
def f(x):
    tmp = n*[1]
    q = np1.array(tmp)
    return -np1.exp(-np1.sum((x**2) / (q**2)))

f_dx = grad(f)

def g_i(x):
    squared_terms = x**2
    g_x_i = np1.sum(squared_terms) - 20
    return g_x_i

def derivative_g_i(x):
    i=1
    grad=np.zeros_like(x)
    indice=range(10*(i-1),10*i)
    grad[indice]=2*x[indice]
    return grad

def g3(x):
    val = np.dot(A_mat, x) - b_vec
    return val[0]

cons = (
    {'type': 'eq',
     'fun': lambda x: np.array([g3(x)]),
     'jac': lambda x: A_mat}, 
    {'type': 'ineq',
     'fun': lambda x: np.array([-g_i(x)]), 
     'jac': lambda x: np.array([-grad(g_i)(x)])}
)


def plot_x(sol_all,count,max_iters):
    t = [i for i in range(max_iters+1)]
    plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size': 16})
    for i in range(count):
        if i ==0:
            text_color = 'red'
            text_label = r'$x_{1}(t)$'
        else:
            text_color = 'green'
            text_label = r'$x_{2}(t)$'
        plt.plot(t, sol_all[i][:,0],color=text_color,label=text_label,linewidth=1)
        plt.plot(t, sol_all[i][:,1],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,2],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,3],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,4],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,5],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,6],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,7],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,8],color=text_color,linewidth=1)
        plt.plot(t, sol_all[i][:,9],color=text_color,linewidth=1)
    plt.xlabel('iteration')
    plt.ylabel('x(t)')
    plt.legend([r'$x_{1}(t)$',r'$x_{2}(t)$']) #,r'$x_{4}(t)$',r'$x_{5}(t)$',r'$x_{6}(t)$',r'$x_{7}(t)$',r'$x_{8}(t)$',r'$x_{9}(t)$',r'$x_{10}(t)$'])
    plt.legend()
    plt.show()

def main():
    n = 10
    max_iters = 100
    num_init_points = 1

    print(f"=== Optimization Comparison: GDA vs RNN (n={n}) ===")
    
    # Set up GDA with proper learning rate
    gda_instance=gda_module.GDA()
    gda_instance.set_constraints(cons)
    limit = np.sqrt(20) # hoặc tham số R
    
    gda_instance.set_bounds([-limit]*n, [limit]*n)
    #gda_instance.set_bounds([-np.inf]*n, [np.inf]*n)
    #lamda = 0.5  # Learning rate scaled for this problem
    #gda_instance.set_lamda(lamda)
    
    # Set up RNN with reduced n_steps for fair comparison
    rnn_instance = rnn_module.RNN(A=A_mat, b=b_vec, step=0.05, n_steps=100, log=False)

    sol_gda_all = []
    sol_rnn_all = []
    val_gda_all = []
    val_rnn_all = []
    alpha = np.random.rand(1)  # Alpha parameter
    mu0 = np.random.rand(1)
    count = 0

    for i in range(num_init_points):
        x0 = np.random.rand(1, n)
        x0 = gda_instance.projection(x0, n)
        #count += 1
        res_gda, val_gda = gda_instance.gda(x0.copy(), max_iters, f, f_dx, n)
        final_val_gda = -np.log(-f(res_gda[-1]))
        #final_test=-np.log(-val_gda[-1])
        print("GDA-Agorithms : Initial point:",x0,"Final point:",res_gda[-1],"Final value:",val_gda[-1])
        tmp_gda = np.array(res_gda)[:,:]
        sol_gda_all.append(tmp_gda)
        val_gda_all.append(val_gda)
        print(f"  GDA - f(x*): {val_gda[-1]},{final_val_gda}" )
        res_rnn_hist, xt_rnn = rnn_instance.rnn(x0.copy(), max_iters, f, f_dx, g_i, derivative_g_i)
        tmp_rnn = np.array(res_rnn_hist)
        sol_rnn_all.append(tmp_rnn)
        print(f"  RNN - f(x*): {-np.log(-f(xt_rnn.reshape(-1)))}")
        final_val_rnn = -np.log(-f(xt_rnn.reshape(-1)))
        
        #print(f"  RNN - x*: {xt_rnn}")
        #print(f"  RNN - f(x*): {final_val_rnn}")
        #print(f"  Difference (RNN - GDA): {final_val_rnn - final_val_gda}")
        
        count += 1

        #plot_x(sol_gda_all, sol_rnn_all, count, max_iters)

if __name__ == "__main__":
    main()