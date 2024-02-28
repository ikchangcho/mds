import numpy as np

s1 = [0, 0, 1, 0, 0, 1, 0]
s2 = [0, 1, 1, 0, 0, 0, 0]

non_equal_positions = [-1] + [i for i in range(len(s1)) if s1[i] != s2[i]] + [len(s1)]
common_length = [non_equal_positions[i+1] - non_equal_positions[i] - 1 for i in range(len(non_equal_positions) - 1)]
lcs = max(common_length)

print(non_equal_positions)
print(common_length)
print(lcs)