"""
Day 5: NumPy 100 Exercises
Date: January 8, 2026
Source: https://github.com/rougier/numpy-100
Progress: 60/100 
"""
import numpy as np

#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)
# dt = [('position', [('x', 'f4'), ('y', 'f4')]), ('color', [('r', 'u1'), ('g', 'u1'), ('b', 'u1')])]
# A = np.zeros(1, dtype=dt)
# print(A)

#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)
# M = np.random.random((10,2))
# X, Y = np.atleast_2d(M[:,0], M[:,1])
# D = np.sqrt((X - X.T)**2 + (Y - Y.T)**2)
# print(D)

#### 53. How to convert a float (32 bits) array into an integer (32 bits) array in place?
# Z = np.arange(10, dtype=np.float32)
# Z[:] = Z.astype(np.int32)

#### 54. How to read the following file? (★★☆)

# 1, 2, 3, 4, 5
# 6,  ,  , 7, 8
#  ,  , 9,10,11

# from io import StringIO
# data = StringIO('''
# 1, 2, 3, 4, 5
# 6,  ,  , 7, 8
#  ,  , 9,10,11
# ''')
# A = np.genfromtxt(data, delimiter=',', filling_values = 0)
# print(A)

#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)
# M = np.arange(9).reshape(3,3)
# for index, value in np.ndenumerate(M):
#     print(index, value)

#### 56. Generate a generic 2D Gaussian-like array (★★☆)
# X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
# D = np.sqrt(X**2, Y**2)
# sigma, mu = 1.0, 0.0
# G = np.exp(-((D-mu)**2)/(2.0*sigma**2))
# print(G)

#### 57. How to randomly place p elements in a 2D array? (★★☆)
# n = 10
# p = 3
# M = np.zeros((n, n))
# np.put(M, np.random.choice(range(n*n), p, replace=False), 1)
# print(M)

#### 58. Subtract the mean of each row of a matrix (★★☆)
# M = np.random.rand(5, 10)
# N = M - M.mean(axis = 1, keepdims=True)
# print(N)

#### 59. How to sort an array by the nth column? (★★☆)
# M = np.random.randint(0, 10, (3,3))
# print(M)
# print(M[M[:,1].argsort()])

#### 60. How to tell if a given 2D array has null columns? (★★☆)
# M = np.random.randint(0,3,(3,10))
# print(np.isnan(M).all(axis=0))
# #OR
# print((~M.any(axis=0)).any())

