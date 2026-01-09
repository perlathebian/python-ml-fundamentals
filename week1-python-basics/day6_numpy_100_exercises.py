"""
Day 6: NumPy 100 Exercises
Date: January 9, 2026
Source: https://github.com/rougier/numpy-100
Progress: 70/100 
"""
import numpy as np

#### 61. Find the nearest value from a given value in an array (★★☆)
# Z = np.random.uniform(0, 1, 10)
# z = 0.5
# nearest_val = Z.flat[np.abs(Z-z).argmin()]
# print(Z)
# print(nearest_val)

#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)
# A = np.arange(3).reshape(3,1)
# B = np.arange(3).reshape(1,3)
# it = np.nditer([A, B, None])
# for x, y, z in it:
#     z[...] = x + y
# print(it.operands[2])    


#### 63. Create an array class that has a name attribute (★★☆)
# class NamedArray(np.ndarray):
#     def __new__(cls, array, name='no name'):
#         obj = np.asarray(array).view(cls)
#         obj.name = name
#         return obj
#     def __array_finalize__(self, obj):
#         if obj is None: return
#         self.name = getattr(obj, 'name', 'no name')

# Z = NamedArray(np.arange(10), 'range_10')
# print(Z.name)        

#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)
# Z = np.zeros(10)
# print(Z)
# I = np.random.randint(0, len(Z), 20)
# print(I)
# Z += np.bincount(I, minlength=len(Z))
# print(Z)

# OR
# np.add.at(Z, I, 1)
# print(Z)

#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)
# X = [1,2,3,4,5,6]
# I = [1,3,9,3,4,1]
# F = np.bincount(I, X)
# print(F)

#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)
# Solution 1:
# w, h = 256, 256
# I = np.random.randint(0, 4, (h, w, 3)).astype(np.ubyte)
# colors = np.unique(I.reshape(-1,3), axis = 0)
# n = len(colors)
# print(n)

# Solution 2:
# w, h = 256, 256
# I = np.random.randint(0, 4, (h, w, 3)).astype(np.ubyte)
# I24 = np.dot(I.astype(np.uint32), [1, 256, 256**2])
# n = len(np.unique(I24))
# print(n)


#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)
# A = np.random.randint(0,10, (3,4,3,4))
# Solution 1:
# sum = A.sum(axis=(-2,-1))
# print(sum)
# Solution 2:
# sum_sol2 = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
# print(sum_sol2)

#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)
# D = np.random.uniform(0,1,100)
# S = np.random.randint(0,10,100)
# D_sums = np.bincount(S, weights=D)
# D_counts = np.bincount(S)
# D_means = D_sums / D_counts
# print(D_means)

# OR
# Pandas solution
# import pandas as pd
# print(pd.Series(D).groupby(S).mean())

#### 69. How to get the diagonal of a dot product? (★★★)
# A = np.random.uniform(0,1,(5,5))
# B = np.random.uniform(0,1,(5,5))
# # Soltuion 1:
# print(np.diag(np.dot(A, B)))
# # Solution 2: faster than sol1
# print(np.sum(A * B.T, axis=1))
# # Solution 3: faster than sol1 and sol2
# print(np.einsum("ij,ji->i", A, B))

#### 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)
A =  np.array([1, 2, 3, 4, 5])
num_of_zeros = 3
new_A = np.zeros(len(A) + ((len(A) - 1))*num_of_zeros)
new_A[::num_of_zeros+1] = A
print(new_A)

