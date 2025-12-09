import numpy as np
import autograd.numpy as np1
from autograd import grad
import matplotlib.pyplot as plt
from scipy.optimize import Bounds, BFGS
import sys, os
import time
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


def plot_x(sol_gda_all, sol_rnn_all, count, max_iters):

    t = [i for i in range(max_iters+1)]
    
    n_vars = sol_gda_all[0].shape[1]
    
    # Define distinct colors for each variable (2n colors total)
    gda_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'][:n_vars]
    rnn_colors = ['darkred', 'darkblue', 'darkgreen', 'darkorange', 'darkviolet', 'saddlebrown', 'deeppink', 'dimgray', 'darkolivegreen', 'darkcyan'][:n_vars]
    
    plt.figure(figsize=(14, 10))
    plt.rcParams.update({'font.size': 12})
    
    for i in range(count):
        # Plot all n variables for GDA with distinct colors (dashed lines)
        for j in range(n_vars):
            label = rf'GDA: $x_{{{j+1}}}(t)$' if i==0 else ""
            plt.plot(t, sol_gda_all[i][:, j], 
                    color=gda_colors[j], 
                    label=label,
                    linewidth=1.5, 
                    linestyle='--', 
                    alpha=0.7)
        
        # Plot all n variables for RNN with distinct colors (solid lines)
        for j in range(n_vars):
            label = rf'RNN: $x_{{{j+1}}}(t)$' if i==0 else ""
            plt.plot(t, sol_rnn_all[i][:, j], 
                    color=rnn_colors[j], 
                    label=label,
                    linewidth=1.8, 
                    alpha=0.8)
    
    plt.xlabel('Iteration', fontsize=13)
    plt.ylabel('x(t)', fontsize=13)
   #plt.title(f'Comparison: GDA vs RNN Convergence (all {n_vars} variables)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', ncol=2, fontsize=9)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    n = 10
    max_iters = 50
    num_init_points = 1

    
    gda_instance=gda_module.GDA()
    gda_instance.set_constraints(cons)
    limit = np.sqrt(20)
    
    gda_instance.set_bounds([-limit]*n, [limit]*n)
    #gda_instance.set_bounds([-np.inf]*n, [np.inf]*n)
    lamda = 0.5  
    gda_instance.set_lamda(lamda)
    
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
        
        # GDA execution time
        start_time_gda = time.time()
        res_gda, val_gda = gda_instance.gda(x0.copy(), max_iters, f, f_dx, n)
        end_time_gda = time.time()
        time_gda = end_time_gda - start_time_gda
        
        final_val_gda = -np.log(-f(res_gda[-1]))
        tmp_gda = np.array(res_gda)[:,:]
        sol_gda_all.append(tmp_gda)
        val_gda_all.append(final_val_gda)
        print("GDA-Agorithms : Initial point:",x0,"Final point:",res_gda[-1],"Final value:",final_val_gda)
        print(f"GDA Execution Time: {time_gda:.6f} seconds\n")
        
        # RNN execution time
        start_time_rnn = time.time()
        res_rnn_hist, xt_rnn = rnn_instance.rnn(x0.copy(), max_iters, f, f_dx, g_i, derivative_g_i)
        end_time_rnn = time.time()
        time_rnn = end_time_rnn - start_time_rnn
        
        tmp_rnn = np.array(res_rnn_hist)[:,:]
        final_val_rnn = -np.log(-f(xt_rnn.reshape(-1)))

        sol_rnn_all.append(tmp_rnn)
        val_rnn_all.append(final_val_rnn)
        print("RNN-Agorithms : Initial point:",x0,"Final point:",res_gda[-1],"Final value:",final_val_rnn)
        print(f"RNN Execution Time: {time_rnn:.6f} seconds\n")
        
        count += 1

        #plot_x(sol_gda_all, sol_rnn_all, count, max_iters)

if __name__ == "__main__":
    main()