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
N = 5  # Number of points
d = 2  # Dimensionality of the space

# Random initial positions
x_init = np.random.rand(N, d)

# Example distance matrix (symmetric, with zeros on the diagonal and some non-zero distances)
D = np.array([
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0]
])

# Compute initial stress
initial_stress = stress_function(x_init, D)

# Run the gradient descent
x_final = gradient_descent(x_init, D)
print(x_final)

# Compute final stress
final_stress = stress_function(x_final, D)

# Plotting
plt.figure(figsize=(10, 5))

# Initial positions
plt.subplot(1, 2, 1)
plt.scatter(x_init[:, 0], x_init[:, 1], c='blue', label='Initial')
plt.title('Initial Positions')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.legend()
# Display initial stress
plt.text(0.05, 0.95, f'Stress: {initial_stress:.2f}', transform=plt.gca().transAxes, verticalalignment='top')

# Final positions
plt.subplot(1, 2, 2)
plt.scatter(x_final[:, 0], x_final[:, 1], c='red', label='Final')
plt.title('Final Positions')
plt.xlabel('X1')
plt.ylabel('X2')
plt.grid(True)
plt.legend()
# Display final stress
plt.text(0.05, 0.95, f'Stress: {final_stress:.2f}', transform=plt.gca().transAxes, verticalalignment='top')

plt.tight_layout()
plt.show()