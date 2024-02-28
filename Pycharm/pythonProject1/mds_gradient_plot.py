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


N1 = 5     # Number of antigens
N2 = 5      # Number of antibodies
N = N1 + N2
L = 10      # Length of sequence

# Generate random binary sequences
s = np.random.randint(2, size=(N, L))

# Generate distance matrix (Model 1)
# D = np.zeros(shape=(N, N))
# for n in range(N1):
#     for m in range(N1, N):
#         D[n, m] = np.linalg.norm(s[n] - s[m])
#         D[m ,n] = D[n, m]

# Generate distance matrix (Model 2: Longest common subsequence)
D = np.zeros(shape=(N, N))
for n in range(N1):
    for m in range(N1, N):
        non_equal_positions = [-1] + [i for i in range(L) if s[n, i] != s[m, i]] + [L]
        common_length = [non_equal_positions[i + 1] - non_equal_positions[i] - 1 for i in range(len(non_equal_positions) - 1)]
        D[n, m] = L - max(common_length)
        D[m, n] = D[n, m]

# Obtain stress minimum for d = 1, 2, ... , N-1
min_stress = np.zeros(N-1)
for d in range(1, N):
    print(f'=================={d}th dimension==================')
    iter_per_d = 3
    values = np.zeros(iter_per_d)

    for i in range(iter_per_d):
        print(f'>>> {i + 1}th initial position')
        x_init = np.random.uniform(low=0, high=L, size=(N, d))
        x_final = gradient_descent(x_init, D)
        values[i] = stress_function(x_final, D)

    min_stress[d-1] = np.min(values)

# Plot
plt.plot(np.arange(1, N), min_stress, marker='o')  # 'o' creates circular markers for each point
plt.title(f'{N1} Antigens and {N2} Antibodies, Length {L}')
plt.xlabel('Dimension')
plt.ylabel('Stress')
#plt.grid(True)  # Add grid for better readability
plt.show()

print('Distance Matrix: \n', D)