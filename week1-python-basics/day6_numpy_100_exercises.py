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
# A =  np.array([1, 2, 3, 4, 5])
# num_of_zeros = 3
# new_A = np.zeros(len(A) + ((len(A) - 1))*num_of_zeros)
# new_A[::num_of_zeros+1] = A
# print(new_A)

#### 71. Consider an array of dimension (5,5,3), how to multiply it by an array with dimensions (5,5)? (★★★)
# A = np.ones((5,5,3))
# B = 2*np.ones((5,5))
# print(A*B[:,:,None])

#### 72. How to swap two rows of an array? (★★★)
# M = np.arange(25).reshape(5,5)
# M[[0,1]] = M[[1,0]]
# print(M)

#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)
# vertices = np.random.randint(0,100,(10,3))
# edges = np.roll(vertices.repeat(2, axis = 1), -1, axis = 1)
# edges = edges.reshape(len(edges)*3, 2)
# edges = np.sort(edges, axis = 1)
# print(edges)
# unique_edges = np.unique(edges.view(dtype=[('v0', edges.dtype), ('v1', edges.dtype)]))
# print(unique_edges)

#### 74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)
# C = np.bincount([1,1,2,3,4,4,6])
# A = np.repeat(np.arange(len(C)), C)
# print(A)

#### 75. How to compute averages using a sliding window over an array? (★★★)
# from numpy.lib.stride_tricks import sliding_window_view

# Z = np.arange(20)
# print(sliding_window_view(Z, window_shape=3).mean(axis=-1))

#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is  shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)
# from numpy.lib.stride_tricks import sliding_window_view

# Z = np.arange(10)
# print(sliding_window_view(Z, window_shape=3))

#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)
# Z = np.random.randint(0,2,10)
# print(Z)
# np.logical_not(Z, out=Z)
# print(Z)

# A = np.random.uniform(-1.0,1.0,10)
# print(A)
# np.negative(A, out=A)
# print(A)

#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)
# P0 = np.random.uniform(-10,10,(10,2))
# P1 = np.random.uniform(-10,10,(10,2))
# p  = np.random.uniform(-10,10,( 1,2))

# def distance_slower(P0, P1, p):
#     T = P1 - P0
#     L = (T**2).sum(axis=1)
#     U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
#     U = U.reshape(len(U),1)
#     D = P0 + U*T - p
#     return np.sqrt((D**2).sum(axis=1))

# print(distance_slower(P0, P1, p))

#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)
# based on distance function from question 78
# P0 = np.random.uniform(-10, 10, (10,2))
# P1 = np.random.uniform(-10,10,(10,2))
# p = np.random.uniform(-10, 10, (10,2))
# print(np.array([distance(P0,P1,p_i) for p_i in p]))

# # Broadcasting
# def distance_points_to_lines(p: np.ndarray, p_1: np.ndarray, p_2: np.ndarray) -> np.ndarray:
#     x_0, y_0 = p.T  
#     x_1, y_1 = p_1.T  
#     x_2, y_2 = p_2.T  

#     dx = x_2 - x_1 
#     dy = y_2 - y_1 

#     cross_term = x_2 * y_1 - y_2 * x_1 

#     numerator = np.abs(
#         dy[np.newaxis, :] * x_0[:, np.newaxis]
#         - dx[np.newaxis, :] * y_0[:, np.newaxis]
#         + cross_term[np.newaxis, :]
#     )
#     denominator = np.sqrt(dx**2 + dy**2)  

#     return numerator / denominator[np.newaxis, :]

# distance_points_to_lines(p, P0, P1)

#### 80. Consider an arbitrary array, write a function that extracts a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)
Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = tuple([slice(start,stop) for start,stop in zip(R_start,R_stop)])
z = tuple([slice(start,stop) for start,stop in zip(Z_start,Z_stop)])
R[r] = Z[z]
print(Z)
print(R)
