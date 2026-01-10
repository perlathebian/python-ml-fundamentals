"""
Day 7: NumPy 100 Exercises
Date: January 10, 2026
Source: https://github.com/rougier/numpy-100
Progress: 87/100 
"""
import numpy as np

#### 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)
# from numpy.lib.stride_tricks import as_strided

# Z = np.arange(1, 15, dtype=np.uint32)
# R = as_strided(Z, (11,4),(4,4))
# print(R)

#### 82. Compute a matrix rank (★★★)
# Z = np.random.uniform(0,1,(10,10))
# U, S, V = np.linalg.svd(Z) 
# threshold = len(S) * S.max() * np.finfo(S.dtype).eps
# rank = np.sum(S > threshold)
# print(rank)

# # OR
# rankk = np.linalg.matrix_rank(Z)
# print(rankk)

#### 83. How to find the most frequent value in an array?
# Z = np.random.randint(0,10,50)
# print(np.bincount(Z).argmax())

#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)
# from numpy.lib.stride_tricks import sliding_window_view

# Z = np.random.randint(0,5,(10,10)) 
# print(Z)
# print(sliding_window_view(Z, window_shape=(3, 3)))

#### 85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)
# class Symmetric(np.ndarray):
#     def __setitem__(self, index, value):
#         i,j = index
#         super(Symmetric, self).__setitem__((i,j), value)
#         super(Symmetric, self).__setitem__((j,i), value)

# def symmetric(Z):
#     return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symmetric)

# S = symmetric(np.random.randint(0,10,(5,5)))
# S[1,0] = 90
# print(S)

#### 86. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)
# p, n = 10, 20
# M = np.ones((p,n,n))
# V = np.ones((p,n,1))
# S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
# print(S)

#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)
from numpy.lib.stride_tricks import sliding_window_view

Z = np.ones((16,16))
k = 4

windows = sliding_window_view(Z, (k, k))
S = windows[::k, ::k, ...].sum(axis=(-2, -1))
print(S)

