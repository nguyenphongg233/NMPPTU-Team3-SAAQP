import pandas as pd
import numpy as np
from scipy.special import gamma, psi
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler
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

def _nearest_distances(X, k=1):
    """
    Returns the distance to the kth nearest neighbor for every point in X
    """
    knn = NearestNeighbors(n_neighbors=k, metric='chebyshev')
    knn.fit(X)
    d, _ = knn.kneighbors(X)
    return d[:, -1]

def _entropy(X, k=1):
    """
    Returns the differential entropy of X using kNN estimator.
    """
    r = _nearest_distances(X, k)
    n, d = X.shape
    volume_unit_ball = (np.pi ** (.5 * d)) / gamma(.5 * d + 1)
    
    
    return (d * np.mean(np.log(r + np.finfo(X.dtype).eps)) +
            np.log(volume_unit_ball) + psi(n) - psi(k))

def _mi_dc(x, y, k):
    """
    Calculates Mutual Information between continuous variables x and discrete y.
    """
    y = y.flatten()
    n = x.shape[0]
    classes = np.unique(y)
    knn = NearestNeighbors(n_neighbors=k)
    # distance to kth in-class neighbour
    d2k = np.empty(n)
    # number of points within each point's class
    Nx = []
    for yi in y:
        Nx.append(np.sum(y == yi))

    # find the distance of the kth in-class point
    for c in classes:
        mask = np.where(y == c)[0]
        knn.fit(x[mask, :])
        dists, _ = knn.kneighbors()
        d2k[mask] = dists[:, -1]
        

    # find the number of points within the distance of the kth in-class point
    knn.fit(x)

    m = knn.radius_neighbors(radius=d2k, return_distance=False)
    m = [i.shape[0] for i in m] 
    
    
    MI = psi(n) - np.mean(psi(Nx)) + psi(k) - np.mean(psi(m))
    return MI 



def setup_problem(file_path):
    print("Loading data...")
    
    df = pd.read_csv(file_path)
    
    if 'name' in df.columns:
        df = df.drop('name', axis=1)
    
    labels = df['status'].values
    features = df.drop('status', axis=1).values
    
    scaler = MinMaxScaler((-1, 1)) # not used
   
    n_samples, n_features = features.shape
    print(f"Data Loaded: {n_samples} samples, {n_features} features.")

    print("Calculating Relevancy Vector (p)...")
    p = mutual_info_classif(features, labels).reshape(1, -1)

    print(p)
 
    print("Calculating Redundancy Matrix (Q)...")
    Q = np.zeros((n_features, n_features))
    k = 5 
    
    entropies = []
    for i in range(n_features):
        entropies.append(_entropy(features[:, i].reshape(-1, 1), k))
   
    for i in range(n_features):
        for j in range(n_features):                
            H_sum = entropies[i] + entropies[j]
            
           
            # I(Fi; Fj; Y) = I(Fi; Y) + I(Fj; Y) - I({Fi,Fj}; Y)  
            
            fi = features[:, i].reshape(-1, 1)
            fj = features[:, j].reshape(-1, 1)
            fi_fj = features[:, [i, j]] # Joint features
            
            mi_i = _mi_dc(fi, labels, k)
            mi_j = _mi_dc(fj, labels, k)
            mi_joint = _mi_dc(fi_fj, labels, k)
            
            interaction_info = mi_i + mi_j - mi_joint
            
            # Q_ij = max(0, Interaction / Entropy_Sum)

            Q[i, j] = max(0, interaction_info / H_sum)
        
        if i % 5 == 0:
            print(f"Processed row {i}/{n_features}")

    
    
    return p, Q, features, labels

file_name = 'parkinsons.data'

p, Q, X, y = setup_problem(file_name)

w, v = LA.eig(Q)
#print('W: ', w)
xi = -min(0,min(w))
#print('Xi: ', xi)
Q = xi*np.eye(22) + Q
np.linalg.det(Q)
w, v = LA.eig(Q)
#print(Q)
#print(w)

def plot_solution(n, gda_solution, rnn_solution, number_of_runs, max_iterations):
    t = [i for i in range(max_iterations+1)]
    
    plt.figure(figsize=(8,8))
    plt.rcParams.update({'font.size': 16})

    colors = []
    for i in range(number_of_runs):
        for j in range(n):
            plt.plot(t, gda_solution[i][:,j], color = 'red', label=r'$GDA$', linewidth=1)
        for j in range(n):
            plt.plot(t, rnn_solution[i][:,j], color = 'green', label=r'$RNN$', linewidth=1)
    
    plt.xlabel('iteration')
    plt.ylabel('x(t)')
    gda_line = plt.Line2D([0], [0], color='red', lw=2)
    rnn_line = plt.Line2D([0], [0], color='green', lw=2)

    plt.legend([gda_line, rnn_line], [r'$GDA$', r'$RNN$'])
    plt.show()

def f(x):
    return (np1.dot(np1.dot(x,Q),x.T))/(np1.dot(p,x.T))
# eT.w = 1
def g1(x):
    return np1.sum(x) - 1
f_dx = grad(f)
g1_dx = grad(g1)

n = 22
cons = ({'type': 'eq',
          'fun' : lambda x: np.array([g1(x)]),
          'jac' : lambda x: np.array([g1_dx(x)])})
bounds = Bounds([0 for i in range(n)], [np.inf for i in range(n)])


gda_alg = GDA(lamda = 10, K = 0.8, cons = cons, bounds = bounds)

number_of_runs = 1
max_iters = 30
gda_solution = []
rnn_solution = []

A_eq = np.ones((1, n))
b_eq = np.array([1.0])


def constraints_ineq(x):
    # We sum the violations: sum(max(0, -w_i))
    return np.sum(np.maximum(0, -x))

def grad_constraints_ineq(x):
    x = x.reshape(-1, 1)
    grad = np.zeros_like(x)
   
    violated_indices = np.where(x < 0)[0]
    grad[violated_indices] = -1.0
    return grad
    
rnn_alg = RNN(A = A_eq, b = b_eq, step=0.001, n_steps=500, log=False)

for i in range(number_of_runs):
    x_0 = np.random.rand(1, n)
    x_0 = gda_alg.projection(x_0, n)
    
    sol, val = gda_alg.gda(x_0, max_iters, f, f_dx, n)
    gda_solution.append(np.array(sol)[:,:])
    print(val[-1])

    #x_0 = np.ones(n) / n
    x_0 = np.random.rand(1, n) - 0.5
    history, x_opt = rnn_alg.rnn(x_0, max_iters, f, f_dx, constraints_ineq, grad_constraints_ineq)

    rnn_solution.append(np.array(history)[:, :])
    print(f(x_opt))
   
plot_solution(22, gda_solution, rnn_solution, number_of_runs, max_iters)
