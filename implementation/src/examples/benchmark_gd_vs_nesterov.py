import time
import numpy as np


def run_gd(A_diag, x0, L, tol=1e-8, max_iters=20000):
    x = x0.copy()
    it = 0
    start = time.time()
    while it < max_iters:
        grad = A_diag * x
        if np.linalg.norm(grad) <= tol:
            break
        x -= (1.0 / L) * grad
        it += 1
    elapsed = time.time() - start
    fval = 0.5 * np.sum(A_diag * x * x)
    return fval, it, elapsed, x


def run_nesterov(A_diag, x0, L, tol=1e-8, max_iters=20000):
    x = x0.copy()
    y = x.copy()
    t = 1.0
    it = 0
    start = time.time()
    while it < max_iters:
        grad = A_diag * y
        if np.linalg.norm(grad) <= tol:
            break
        x_next = y - (1.0 / L) * grad
        t_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        y = x_next + ((t - 1.0) / t_next) * (x_next - x)
        x = x_next
        t = t_next
        it += 1
    elapsed = time.time() - start
    fval = 0.5 * np.sum(A_diag * x * x)
    return fval, it, elapsed, x


def prepare_diagonal_problem(n, A_diag=None, low=1.0, high=10.0, seed=0):
    """Prepare diagonal A (as vector of eigenvalues), Lipschitz L and init x0.

    If `A_diag` is provided it is used directly (must be length >= n); otherwise random in [low,high].
    Returns (A_diag, L, x0, rng)
    """
    rng = np.random.default_rng(seed)
    if A_diag is None:
        A_diag = rng.uniform(low, high, size=(n,))
    else:
        A_diag = np.asarray(A_diag).reshape(-1)[:n]
    L = np.max(A_diag)
    x0 = rng.standard_normal(size=(n,))
    return A_diag, L, x0, rng


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    ns = [10, 20, 50, 100, 200, 500]

    results = []

    for n in ns:
        A_diag, L, x0, _ = prepare_diagonal_problem(n, seed=0)

        # Run proposed algorithm (Nesterov accelerated GD)
        f_prop, it_prop, time_prop, xprop = run_nesterov(A_diag, x0, L)

        # Run baseline GD with step = 1 / L
        f_gd, it_gd, time_gd, xgd = run_gd(A_diag, x0, L)

        results.append((n, f_prop, it_prop, time_prop, f_gd, it_gd, time_gd))

    # Print plain text table similar to attached format
    # Header
    print(f"{'n':>6}  {'Algorithm GDA (proposed)':^36}  {'Algorithm GD':^28}")
    print(f"{'':6}  {'f(x*)':>8} {'#Iter':>8} {'Time':>10}  {'f(x*)':>8} {'#Iter':>8} {'Time':>10}")
    print('-' * 90)
    for (n, f_prop, it_prop, time_prop, f_gd, it_gd, time_gd) in results:
        print(f"{n:6d}  {f_prop:8.4f} {it_prop:8d} {time_prop:10.4f}  {f_gd:8.4f} {it_gd:8d} {time_gd:10.4f}")

    # Optionally print results dictionary
    # for row in results:
    #     print(row)
