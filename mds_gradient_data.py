import numpy as np
import matplotlib.pyplot as plt
# Shift + F6 (+ fn)
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
#        stress = stress_function(x, D)
#        if i % 100 == 0:
#            print(f"Iteration {i}, Stress: {stress}")
        if np.max(np.abs(alpha * gradient)) < tol:
            break
    return x

for L in [5, 10, 15, 20, 25, 30]:      # Length of sequence
    max_iter = int(L / 5 * 2)
    for j in range(1, max_iter + 1):
        N1 = 5 * j     # Number of antigens
        N2 = 5 * j     # Number of antibodies
        N = N1 + N2

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
        max_dim = min(L * 2, N + 5)
        min_normalized_stress = np.zeros(max_dim)
        for d in range(1, max_dim + 1):
            iter_per_d = 10
            stress = np.zeros(iter_per_d)

            for i in range(iter_per_d):
                x_init = np.random.uniform(low=0, high=L, size=(N, d))
                x_final = gradient_descent(x_init, D)
                stress[i] = stress_function(x_final, D)

            min_normalized_stress[d - 1] = np.min(stress) / N ** 2
            print(f'Length {L}: N {j}/{max_iter}, Dimension {d}/{max_dim}')

        # Plot
        plt.plot(np.arange(3, max_dim + 1), min_normalized_stress[2:], marker='o')
        plt.title(f'Length = {L}, N1 = {N1}, N2 = {N2}')
        plt.xlabel('Dimension')
        plt.ylabel('Normalized Stress')
    #    plt.grid(True)  # Add grid for better readability
        plt.savefig(f'length{L}_{N1}+{N2}.png')
        np.savetxt(f'length{L}_{N1}+{N2}.csv', min_normalized_stress, delimiter=',', fmt='%.3f')
        np.savetxt(f'distance_length{L}_{N1}+{N2}.csv', D, delimiter=',', fmt='%.2f')
        plt.close()

