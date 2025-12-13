import scipy
import numpy as np

import numpy.linalg as la
from sklearn.utils.extmath import safe_sparse_dot

import algorithms.gda as gda_module
import algorithms.nesterov as nag_module
import algorithms.gd as gd_module
def safe_sparse_add(a, b):
    if scipy.sparse.issparse(a) and scipy.sparse.issparse(b):
        # both are sparse, keep the result sparse
        return a + b
    else:
        # on of them is non-sparse, convert
        # everything to dense.
        if scipy.sparse.issparse(a):
            a = a.toarray()
            if a.ndim == 2 and b.ndim == 1:
                b.ravel()
        elif scipy.sparse.issparse(b):
            b = b.toarray()
            if b.ndim == 2 and a.ndim == 1:
                b = b.ravel()
        return a + b


def logsig(x):
    """
    Compute the log-sigmoid function component-wise.
    See http://fa.bianp.net/blog/2019/evaluate_logistic/ for more details.
    """
    out = np.zeros_like(x)
    idx0 = x < -33
    out[idx0] = x[idx0]
    idx1 = (x >= -33) & (x < -18)
    out[idx1] = x[idx1] - np.exp(x[idx1])
    idx2 = (x >= -18) & (x < 37)
    out[idx2] = -np.log1p(np.exp(-x[idx2]))
    idx3 = x >= 37
    out[idx3] = -np.exp(-x[idx3])
    return out


def logistic_loss(w, X, y, l2, reg=True):
    """Logistic loss, numerically stable implementation.

    Parameters
    ----------
    x: array-like, shape (n_features,)
        Coefficients

    A: array-like, shape (n_samples, n_features)
        Data matrix

    b: array-like, shape (n_samples,)
        Labels

    Returns
    -------
    loss: float
    """
    z = np.dot(X, w)
    y = np.asarray(y)
    return np.mean((1-y)*z - logsig(z)) + l2/2 * la.norm(w)**2


def logistic_gradient(w, X, y_, l2, normalize=True):
    """
    Gradient of the logistic loss at point w with features X, labels y and l2 regularization.
    If labels are from {-1, 1}, they will be changed to {0, 1} internally
    """
    y = (y_+1) / 2 if -1 in y_ else y_
    activation = scipy.special.expit(safe_sparse_dot(X, w, dense_output=True).ravel())
    grad = safe_sparse_add(X.T.dot(activation - y) / X.shape[0], l2 * w)
    grad = np.asarray(grad).ravel()
    if normalize:
        return grad
    return grad * len(y)

import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import numpy.linalg as la
from sklearn.datasets import load_svmlight_file


sns.set(style="whitegrid", font_scale=1.2, context="talk", palette=sns.color_palette("bright"), color_codes=False)
matplotlib.rcParams['mathtext.fontset'] = 'cm'
dataset = 'mushrooms'
data_path = dataset + '.txt'
if dataset == 'covtype':
    data_path += '.bz2'

if dataset == 'covtype':
    it_max = 10000
elif dataset == 'w8a':
    it_max = 8000
else:
    it_max = 4000

def logistic_smoothness(X):
    return 0.25 * np.max(la.eigvalsh(X.T @ X / X.shape[0]))


data = load_svmlight_file(data_path)
X, y = data[0].toarray(), data[1]

if (np.unique(y) == [1, 2]).all():
    # Loss functions support only labels from {0, 1}
    y -= 1
if (np.unique(y) == [-1, 1]).all():
    y = (y+1) / 2
n, d = X.shape
L = logistic_smoothness(X)
l2 = L / n if dataset == 'covtype' else L / (10 * n)
w0 = np.zeros(d)

# def loss_func(w):
#     return logistic_loss(w, X, y, l2)

# def grad_func(w):
#     return logistic_gradient(w, X, y, l2)


def func(w):
    return logistic_loss(w, X, y, l2)

def grad(w):
    return logistic_gradient(w, X, y, l2)

gda = gda_module.GDA(sigma=1.0/L, lamda=1000, K=0.85, log=False)
#print(X[1])
print("\n--- Running GDA ---")
start = time.time()
w_gda, val_gda = gda.gda(
    x_0 = w0, 
    max_iters = 4000, 
    f = func, 
    f_dx = grad, 
    n = d
)
print(f"GDA Time: {time.time() - start:.2f}s")


gd = gd_module.GD(lamda=1.0/L, log=False)

print("\n--- Running GD ---")
start = time.time()
w_gd, val_gd = gd.gd(
    x_0=np.zeros(d), 
    max_iters=4000, 
    f=func, 
    f_dx=grad, 
    n=d
)
print(f"GD Time: {time.time() - start:.2f}s")


# If you want to use the strongly convex version, set strongly_convex=True and mu=l2
nesterov_solver = nag_module.Nesterov(
    lr=1.0/L,                # Step size 1/L
    strongly_convex=False,   # Set to True if you want to leverage l2 > 0
    mu=l2,                   # Only used if strongly_convex=True
    log=True
)

print("\n--- Running Nesterov ---")
start = time.time()
w_nag, val_nag = nesterov_solver.nesterov(
    x_0=w0, 
    max_iters=4000, 
    f=func, 
    f_dx=grad, 
    n=d
)
print(f"Nesterov Time: {time.time() - start:.2f}s")

val = min(val_gda[-1], val_nag[-1], val_gd[-1])
val_gda = [val1 - val for val1 in val_gda]
val_gd = [val1 - val for val1 in val_gd]
val_nag = [val1 - val for val1 in val_nag]

# --- 3. Visualization ---
plt.figure(figsize=(10, 6))
plt.plot(val_gda, label='GDA (Adaptive)', color='blue', linewidth = 1)
plt.plot(val_nag, label='Nesterov', color='red', linestyle='-.', linewidth = 1)
plt.plot(val_gd, label='GD', color='yellow', linestyle = '--', linewidth = 1)
plt.yscale('log')
#plt.ylabel('Loss (Log Scale)')
#plt.xlabel('Iterations')
#plt.title('GDA_0.9 vs Nesterov vs GD Convergence')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Final Loss GDA: {val_gda[-1]}")
print(f"Final Loss Nesterov: {val_nag[-1]}")
print(f"Final Loss GD: {val_gd[-1]}")