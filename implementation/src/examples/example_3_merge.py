import numpy as np
from autograd import grad
import autograd.numpy as anp 
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import random
from scipy.optimize import BFGS, SR1
from scipy.optimize import Bounds
from scipy.optimize import NonlinearConstraint, LinearConstraint
import time
import os


np.random.seed(42)

print("=" * 60)

n = 150  # dimension
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

    for t in range(max_iters):
        y = x - lda * f_dx(x)
        x_pre = x.copy()
        x = find_min(y, n)

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

    for t in range(max_iters):
        y = x - lda * f_dx(x)
        x = find_min(y, n)

        res.append(x)
        val.append(f(x))

        # Check convergence
        if len(val) > 1 and abs(val[-1] - val[-2]) < eps:
            break

    return res, val

# ================ Experiment Configuration ================
num_trials = 3  # number of different initialization points
max_iters = 1000  # maximum iterations (safety limit)
eps = 0.0003  # convergence tolerance: |f(x_t) - f(x_{t-1})| < eps
sigma = 0.1
lamda_gda=5/L
lamda_gd=1/L
K = np.random.rand(1,1)

print("=" * 60)
print("Experiment Configuration:")
print("=" * 60)
print(f"Number of trials (different init points): {num_trials}")
print(f"Convergence tolerance (eps): {eps}")
print(f"Maximum iterations (safety): {max_iters}")
print(f"Sigma: {sigma}")
print(f"Lambda GDA (initial step size): {lamda_gda:.6f}")
print(f"Lambda GD (fixed step size): {lamda_gd:.6f}")
print(f"K: {K}")

print()

# ================ Run GDA Algorithm ================
print("=" * 60)
print("Running GDA Algorithm")
print("=" * 60)

gda_trajectories = []
gda_values = []
gda_final_points = []
gda_final_values = []

start_time_gda = time.time()  # Start timing GDA
for trial in range(num_trials):
    trial_start = time.time()

    x0 = np.random.rand(1, n)
    x0 = find_min(x0, n)

    trajectory, values = run_gda(x0, max_iters, f, f_dx, n, sigma, lamda_gda, K, eps)

    actual_iters = len(values) - 1

    trial_end = time.time()
    trial_time = trial_end - trial_start

    gda_trajectories.append(np.array(trajectory))
    gda_values.append(values)
    gda_final_points.append(trajectory[-1])
    final_val = values[-1] if isinstance(values[-1], (int, float)) else values[-1].item()
    gda_final_values.append(final_val)

    x_opt = trajectory[-1]
    print(f"GDA Trial {trial + 1}/{num_trials}: Optimal = {final_val:.6f}, Iters = {actual_iters}, Time = {trial_time:.3f}s")
end_time_gda = time.time()  
print(f"\nGDA Total Runtime: {end_time_gda - start_time_gda:.2f} seconds")

print("\n" + "=" * 60)
print("Running GD Algorithm")
print("=" * 60)

gd_trajectories = []
gd_values = []
gd_final_points = []
gd_final_values = []

start_time_gd = time.time()  # Start timing GD
for trial in range(num_trials):
    trial_start = time.time()

    x0 = np.random.rand(1, n)
    x0 = find_min(x0, n)

    trajectory, values = run_gd(x0, max_iters, f, f_dx, n, lamda_gd, eps)

    actual_iters = len(values) - 1

    trial_end = time.time()
    trial_time = trial_end - trial_start

    gd_trajectories.append(np.array(trajectory))
    gd_values.append(values)
    gd_final_points.append(trajectory[-1])
    final_val = values[-1] if isinstance(values[-1], (int, float)) else values[-1].item()
    gd_final_values.append(final_val)

    x_opt = trajectory[-1]
    print(f"GD Trial {trial + 1}/{num_trials}: Optimal = {final_val:.6f}, Iters = {actual_iters}, Time = {trial_time:.3f}s")
end_time_gd = time.time()  # End timing GD
print(f"\nGD Total Runtime: {end_time_gd - start_time_gd:.2f} seconds")

# ================ Summary Statistics ================
print("\n" + "=" * 60)


print("\nGDA Results:")
print(f"  Average final value: {np.mean(gda_final_values):.6f}")
print(f"  Std dev final value: {np.std(gda_final_values):.6f}")
print(f"  Best final value: {np.min(gda_final_values):.6f}")

print("\nGD Results:")
print(f"  Average final value: {np.mean(gd_final_values):.6f}")
print(f"  Std dev final value: {np.std(gd_final_values):.6f}")
print(f"  Best final value: {np.min(gd_final_values):.6f}")

print("\nComparison:")
gda_avg = np.mean(gda_final_values)
gd_avg = np.mean(gd_final_values)
print(f"  GDA vs GD (avg): {gda_avg:.6f} vs {gd_avg:.6f}")
print(f"  Difference: {abs(gda_avg - gd_avg):.6f}")
if gda_avg < gd_avg:
    print(f"  GDA performs {((gd_avg - gda_avg) / gd_avg * 100):.2f}% better")
else:
    print(f"  GD performs {((gda_avg - gd_avg) / gda_avg * 100):.2f}% better")

# ================ Visualization ================
print("\n" + "=" * 60)
print("Generating visualizations...")
print("=" * 60)

output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'output', 'figures', 'example_03')
os.makedirs(output_dir, exist_ok=True)

# Color schemes
colors_gda = ['#e74c3c', '#c0392b', '#8e44ad']  # Reds and purples for GDA
colors_gd = ['#3498db', '#2980b9', '#16a085']   # Blues and teals for GD

# ================ 1. GDA Trajectory Plot ================
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 14})

iterations = range(max_iters + 1)

# Plot trajectories for selected dimensions
dims_to_plot = [0, 1, 2]  # Plot first 3 dimensions

for trial in range(num_trials):
    trajectory = gda_trajectories[trial]
    for dim in dims_to_plot:
        plt.plot(iterations, trajectory[:, dim],
                color=colors_gda[trial], alpha=0.7, linewidth=2,
                label=f'Trial {trial+1}, dim {dim+1}' if dim == dims_to_plot[0] else '')

plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Component Value', fontsize=16)
plt.title('GDA: Trajectory Convergence\n(Multiple initialization points)', fontsize=18, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gda_trajectories.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'gda_trajectories.png')}")
plt.close()

# ================ 2. GD Trajectory Plot ================
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 14})

for trial in range(num_trials):
    trajectory = gd_trajectories[trial]
    for dim in dims_to_plot:
        plt.plot(iterations, trajectory[:, dim],
                color=colors_gd[trial], alpha=0.7, linewidth=2,
                label=f'Trial {trial+1}, dim {dim+1}' if dim == dims_to_plot[0] else '')

plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Component Value', fontsize=16)
plt.title('GD: Trajectory Convergence\n(Multiple initialization points)', fontsize=18, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gd_trajectories.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'gd_trajectories.png')}")
plt.close()

# ================ 3. GDA Objective Value Convergence ================
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 14})

for trial in range(num_trials):
    plt.plot(iterations, gda_values[trial],
            color=colors_gda[trial], linewidth=2.5, marker='o',
            markersize=4, markevery=5, label=f'Trial {trial+1}')

plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Objective Value f(x)', fontsize=16)
plt.title('GDA: Objective Value Convergence', fontsize=18, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gda_objective_convergence.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'gda_objective_convergence.png')}")
plt.close()

# ================ 4. GD Objective Value Convergence ================
plt.figure(figsize=(12, 8))
plt.rcParams.update({'font.size': 14})

for trial in range(num_trials):
    plt.plot(iterations, gd_values[trial],
            color=colors_gd[trial], linewidth=2.5, marker='s',
            markersize=4, markevery=5, label=f'Trial {trial+1}')

plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Objective Value f(x)', fontsize=16)
plt.title('GD: Objective Value Convergence', fontsize=18, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gd_objective_convergence.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'gd_objective_convergence.png')}")
plt.close()

# ================ 5. GDA vs GD Comparison - Average Convergence ================
plt.figure(figsize=(14, 8))
plt.rcParams.update({'font.size': 14})

# Calculate average objective values across trials
gda_avg_values = np.mean([gda_values[i] for i in range(num_trials)], axis=0)
gd_avg_values = np.mean([gd_values[i] for i in range(num_trials)], axis=0)

gda_std_values = np.std([gda_values[i] for i in range(num_trials)], axis=0)
gd_std_values = np.std([gd_values[i] for i in range(num_trials)], axis=0)

plt.plot(iterations, gda_avg_values, color='#e74c3c', linewidth=3,
         marker='o', markersize=6, markevery=5, label='GDA (avg)')
plt.fill_between(iterations,
                 gda_avg_values - gda_std_values,
                 gda_avg_values + gda_std_values,
                 color='#e74c3c', alpha=0.2)

plt.plot(iterations, gd_avg_values, color='#3498db', linewidth=3,
         marker='s', markersize=6, markevery=5, label='GD (avg)')
plt.fill_between(iterations,
                 gd_avg_values - gd_std_values,
                 gd_avg_values + gd_std_values,
                 color='#3498db', alpha=0.2)

plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Objective Value f(x)', fontsize=16)
plt.title('GDA vs GD: Average Convergence Comparison\n(with standard deviation bands)',
          fontsize=18, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gda_vs_gd_average.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'gda_vs_gd_average.png')}")
plt.close()

# ================ 6. GDA vs GD Comparison - All Trials ================
plt.figure(figsize=(14, 8))
plt.rcParams.update({'font.size': 14})

for trial in range(num_trials):
    plt.plot(iterations, gda_values[trial],
            color=colors_gda[trial], linewidth=2, alpha=0.7,
            linestyle='-', label=f'GDA Trial {trial+1}')
    plt.plot(iterations, gd_values[trial],
            color=colors_gd[trial], linewidth=2, alpha=0.7,
            linestyle='--', label=f'GD Trial {trial+1}')

plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Objective Value f(x)', fontsize=16)
plt.title('GDA vs GD: All Trials Comparison', fontsize=18, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize=11, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gda_vs_gd_all_trials.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'gda_vs_gd_all_trials.png')}")
plt.close()

# ================ 7. Final Value Distribution ================
plt.figure(figsize=(10, 8))
plt.rcParams.update({'font.size': 14})

x_pos = np.arange(num_trials)
width = 0.35

plt.bar(x_pos - width/2, gda_final_values, width,
        color='#e74c3c', alpha=0.8, label='GDA')
plt.bar(x_pos + width/2, gd_final_values, width,
        color='#3498db', alpha=0.8, label='GD')

plt.xlabel('Trial', fontsize=16)
plt.ylabel('Final Objective Value', fontsize=16)
plt.title('GDA vs GD: Final Values by Trial', fontsize=18, fontweight='bold')
plt.xticks(x_pos, [f'{i+1}' for i in range(num_trials)])
plt.legend(loc='best', fontsize=14)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gda_vs_gd_final_values.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'gda_vs_gd_final_values.png')}")
plt.close()

# ================ 8. Convergence Rate Comparison ================
plt.figure(figsize=(14, 8))
plt.rcParams.update({'font.size': 14})

# Calculate convergence to final value (normalized)
for trial in range(num_trials):
    gda_conv = np.abs(np.array(gda_values[trial]) - gda_final_values[trial])
    gd_conv = np.abs(np.array(gd_values[trial]) - gd_final_values[trial])

    plt.semilogy(iterations, gda_conv + 1e-10,
                color=colors_gda[trial], linewidth=2, alpha=0.7,
                linestyle='-', label=f'GDA Trial {trial+1}')
    plt.semilogy(iterations, gd_conv + 1e-10,
                color=colors_gd[trial], linewidth=2, alpha=0.7,
                linestyle='--', label=f'GD Trial {trial+1}')

plt.xlabel('Iteration', fontsize=16)
plt.ylabel('Distance to Final Value (log scale)', fontsize=16)
plt.title('GDA vs GD: Convergence Rate Comparison', fontsize=18, fontweight='bold')
plt.grid(True, alpha=0.3, which='both')
plt.legend(loc='best', fontsize=11, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gda_vs_gd_convergence_rate.png'), dpi=300, bbox_inches='tight')
print(f"Saved: {os.path.join(output_dir, 'gda_vs_gd_convergence_rate.png')}")
plt.close()

print("\n" + "=" * 60)
print("All visualizations completed!")
print(f"Output directory: {output_dir}")
print("=" * 60)
print("\nGenerated figures:")
print("  1. gda_trajectories.png - GDA trajectory convergence")
print("  2. gd_trajectories.png - GD trajectory convergence")
print("  3. gda_objective_convergence.png - GDA objective values")
print("  4. gd_objective_convergence.png - GD objective values")
print("  5. gda_vs_gd_average.png - Average comparison with std dev")
print("  6. gda_vs_gd_all_trials.png - All trials comparison")
print("  7. gda_vs_gd_final_values.png - Final values bar chart")
print("  8. gda_vs_gd_convergence_rate.png - Convergence rate (log scale)")
print("=" * 60)