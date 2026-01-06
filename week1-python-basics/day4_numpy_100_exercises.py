"""
Day 4: NumPy 100 Exercises
Date: January 6, 2026
Source: https://github.com/rougier/numpy-100
Progress: /100 
"""

import numpy as np

#### 21. Create a checkerboard 8x8 matrix using the tile function (★☆☆)
# M = np.tile(np.array([[0,1],[1,0]]), (4,4))
# print(M)

#### 22. Normalize a 5x5 random matrix (★☆☆)
# M = np.random.random((5,5))
# M_normalized = (M - M.mean())/M.std()
# print(M)
# print(M_normalized)

#### 23. Create a custom dtype that describes a color as four unsigned bytes (RGBA) (★☆☆)
# color = np.dtype([('r', np.ubyte),('g', np.ubyte),('b', np.ubyte),('a', np.ubyte)])
# pixel = np.array((255, 128, 0, 255), dtype=color)
# print(pixel['g'])

#### 24. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product) (★☆☆)
# M = np.matmul(np.ones((5, 3)), np.ones((3, 2)))
#OR
# M = np.ones((5, 3)) @ np.ones((3, 2))
# print(M)

#### 25. Given a 1D array, negate all elements which are between 3 and 8, in place. (★☆☆)
# M = np.arange(11)
# M[(M > 3) & (M < 8)] *= -1
# print(M)

#### 26. What is the output of the following script? (★☆☆)

# Author: Jake VanderPlas

# print(sum(range(5),-1)) --> 9: Python's sum(iterable, start): -1 + 0 + 1 + 2 + 3 + 4 = 9
# from numpy import *  
# print(sum(range(5),-1)) --> after import this uses Numpy's sum(array, axis): 0 + 1 + 2 + 3 + 4 = 10 (along axis = -1)


#### 27. Consider an integer vector Z, which of these expressions are legal? (★☆☆)

# Z**Z --> legal: element-wise operation
# 2 << Z >> 2 --> legal: python read left-to-right so (2 << Z) >> 2
# Z <- Z --> legal: Z < (-Z) element-wsise comparison
# 1j*Z --> legal: complex numbers are supported
# Z/1/1 --> legal: (Z/1)/1
# Z<Z>Z --> illegal: (Z < Z) AND (Z > Z) chained comparison not supported on arrays

#### 28. What are the result of the following expressions? (★☆☆)

# np.array(0) / np.array(0) --> nan: 0/0 undeifned
# np.array(0) // np.array(0) --> 0: integer division floor
# np.array([np.nan]).astype(int).astype(float) --> -9223372036854775808 → -9.223372036854776e+18 (NumPy fills the integer with min possible int)

#### 29. How to round away from zero a float array ? (★☆☆)
# M = np.array([-3.7, -1.2, 1.2, 3.7])
# M_updated = np.copysign(np.ceil(np.abs(M)), M)
#OR (more readable, less efficient)
#M_updated = np.where(M > 0, np.ceil(M), np.floor(M))
# print(M_updated)

#### 30. How to find common values between two arrays? (★☆☆)
# M1 = np.array([0, 1, 2, 3, 4])
# M2 = np.array([2, 3, 4, 5, 6])
# common = np.intersect1d(M1, M2)
# print(common)


#### 31. How to ignore all numpy warnings (not recommended)? (★☆☆)

#### 32. Is the following expressions true? (★☆☆)

# np.sqrt(-1) == np.emath.sqrt(-1)


#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)

#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)

#### 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)

#### 36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)

#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)

#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)

#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)

#### 40. Create a random vector of size 10 and sort it (★★☆)

#### 41. How to sum a small array faster than np.sum? (★★☆)

#### 42. Consider two random arrays A and B, check if they are equal (★★☆)

#### 43. Make an array immutable (read-only) (★★☆)

#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)

#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)

#### 46. Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area (★★☆)

#### 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) (★★☆)

#### 48. Print the minimum and maximum representable values for each numpy scalar type (★★☆)

#### 49. How to print all the values of an array? (★★☆)

#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)

#### 51. Create a structured array representing a position (x,y) and a color (r,g,b) (★★☆)

#### 52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances (★★☆)

#### 53. How to convert a float (32 bits) array into an integer (32 bits) array in place?

#### 54. How to read the following file? (★★☆)

# 1, 2, 3, 4, 5
# 6,  ,  , 7, 8
#  ,  , 9,10,11


#### 55. What is the equivalent of enumerate for numpy arrays? (★★☆)

#### 56. Generate a generic 2D Gaussian-like array (★★☆)

#### 57. How to randomly place p elements in a 2D array? (★★☆)

#### 58. Subtract the mean of each row of a matrix (★★☆)

#### 59. How to sort an array by the nth column? (★★☆)

#### 60. How to tell if a given 2D array has null columns? (★★☆)

#### 61. Find the nearest value from a given value in an array (★★☆)

#### 62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator? (★★☆)

#### 63. Create an array class that has a name attribute (★★☆)

#### 64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)? (★★★)

#### 65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? (★★★)

#### 66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors (★★☆)

#### 67. Considering a four dimensions array, how to get sum over the last two axis at once? (★★★)

#### 68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset  indices? (★★★)

#### 69. How to get the diagonal of a dot product? (★★★)

#### 70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value? (★★★)

#### 71. Consider an array of dimension (5,5,3), how to multiply it by an array with dimensions (5,5)? (★★★)

#### 72. How to swap two rows of an array? (★★★)

#### 73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the  triangles (★★★)

#### 74. Given a sorted array C that corresponds to a bincount, how to produce an array A such that np.bincount(A) == C? (★★★)

#### 75. How to compute averages using a sliding window over an array? (★★★)

#### 76. Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is  shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1]) (★★★)

#### 77. How to negate a boolean, or to change the sign of a float inplace? (★★★)

#### 78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i])? (★★★)

#### 79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])? (★★★)

#### 80. Consider an arbitrary array, write a function that extracts a subpart with a fixed shape and centered on a given element (pad with a `fill` value when necessary) (★★★)

#### 81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]? (★★★)

#### 82. Compute a matrix rank (★★★)

#### 83. How to find the most frequent value in an array?

#### 84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix (★★★)

#### 85. Create a 2D array subclass such that Z[i,j] == Z[j,i] (★★★)

#### 86. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once? (result has shape (n,1)) (★★★)

#### 87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)? (★★★)

#### 88. How to implement the Game of Life using numpy arrays? (★★★)

#### 89. How to get the n largest values of an array (★★★)

#### 90. Given an arbitrary number of vectors, build the cartesian product (every combination of every item) (★★★)

#### 91. How to create a record array from a regular array? (★★★)

#### 92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods (★★★)

#### 93. Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? (★★★)

#### 94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3]) (★★★)

#### 95. Convert a vector of ints into a matrix binary representation (★★★)

#### 96. Given a two dimensional array, how to extract unique rows? (★★★)

#### 97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function (★★★)

#### 98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples (★★★)?

#### 99. Given an integer n and a 2D array X, select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, i.e., the rows which only contain integers and which sum to n. (★★★)

#### 100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X (i.e., resample the elements of an array with replacement N times, compute the mean of each sample, and then compute percentiles over the means). (★★★)