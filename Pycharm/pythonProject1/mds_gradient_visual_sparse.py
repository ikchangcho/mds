import numpy as np
import matplotlib.pyplot as plt

def compute_gradient(x, D):
    N, d = x.shape  # N points in d-dimensional space
    gradient = np.zeros_like(x)

    for n in range(N):
        for m in range(n, N):
            if D[n, m] > 0:  # Consider only non-zero distances
                diff = x[n] - x[m]
                dist = np.linalg.norm(diff)
                if dist > 0:
                    grad_nm = (dist - D[n, m]) * diff / dist
                    gradient[n] += grad_nm
                    gradient[m] -= grad_nm  # Symmetry: grad_mn = -grad_nm

    return gradient

def stress_function(x, D):
    stress = 0
    for n in range(x.shape[0]):
        for m in range(n + 1, x.shape[0]):
            if D[n, m] > 0:
                stress += (D[n, m] - np.linalg.norm(x[n] - x[m])) ** 2
    return stress

def gradient_descent(x_init, D, alpha=0.01, max_iter=1000, tol=1e-6):
    x = x_init.copy()
    for i in range(max_iter):
        gradient = compute_gradient(x, D)
        x -= alpha * gradient
        stress = stress_function(x, D)
        if i % 100 == 0:
            print(f"Iteration {i}, Stress: {stress}")
        if np.max(np.abs(alpha * gradient)) < tol:
            break
    return x


# Example usage
N = 6  # Number of points
d = 2  # Dimensionality of the space

# Random initial positions
x_init = np.random.rand(N, d)

# Example distance matrix (symmetric, with zeros on the diagonal and some non-zero distances)
D = np.array([
    [0, 1, 2, 3, 4, 5],
    [1, 0, 1, 2, 3, 4],
    [3, 2, 0, 1, 2, 3],
    [3, 2, 1, 0, 1, 2],
    [4, 3, 2, 1, 0, 1],
    [5, 4, 3, 2, 1, 0]
])

halfN = int(N/2)
D_sparse = D.copy()
D_sparse[:halfN, :halfN] = 0
D_sparse[halfN:, halfN:] = 0

# Compute initial stress
initial_stress = stress_function(x_init, D)
initial_stress_sparse = stress_function(x_init, D_sparse)

# Run the gradient descent
x_final = gradient_descent(x_init, D)
x_final_sparse = gradient_descent(x_init, D_sparse)

# Compute final stress
final_stress = stress_function(x_final, D)
final_stress_sparse = stress_function(x_final_sparse, D_sparse)

# Compute discrepancy
discrepancy_squared_sum = 0
for i in range(N):
    for j in range(d):
        discrepancy_squared_sum += (x_final[i, j] - x_final_sparse[i, j]) ** 2
discrepancy_rms = np.sqrt(discrepancy_squared_sum / N)

radius_squared_sum = 0
center_final = np.mean(x_final, axis=0)
centered_x_final = x_final - center_final
for i in range(N):
    for j in range(d):
        radius_squared_sum += centered_x_final[i, j] ** 2
radius_rms = np.sqrt(radius_squared_sum / N)

normalized_discrepancy = discrepancy_rms / radius_rms   # Normalized discrepancy per point per dimension

# Plotting
plt.figure(figsize=(10, 5))
plt. suptitle(f'Discrepancy: {normalized_discrepancy * 100: .1f}%')

# Full distance matrix
plt.subplot(1, 2, 1)
plt.scatter(x_final[:, 0], x_final[:, 1], c='red')
plt.title('Full Distance Matrix')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.text(0.05, 0.95, f'Initial Stress: {initial_stress:.2f}', transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.05, 0.90, f'Final Stress: {final_stress:.2f}', transform=plt.gca().transAxes, verticalalignment='top')

# Sparse distance matrix
plt.subplot(1, 2, 2)
plt.scatter(x_final_sparse[:, 0], x_final_sparse[:, 1], c='blue')
plt.title('Sparse Distance Matrix')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.text(0.05, 0.95, f'Initial Stress: {initial_stress_sparse:.2f}', transform=plt.gca().transAxes, verticalalignment='top')
plt.text(0.05, 0.90, f'Final Stress: {final_stress_sparse:.2f}', transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.show()
plt.savefig('/Users/ik/Pycharm/Project1/discrepancy10.png')
