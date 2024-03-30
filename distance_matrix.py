import numpy as np

N1 = 1      # Number of antigens
N2 = 1      # Number of antibodies
N = N1 + N2
L = 7      # Length of sequence

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

print(s)
print(D)