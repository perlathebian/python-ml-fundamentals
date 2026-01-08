"""
Day 4: NumPy 100 Exercises
Date: January 6, 2026
Source: https://github.com/rougier/numpy-100
Progress: 50/100 
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
# global
# defaults = np.seterr(all="ignore")
# Z = np.ones(1) / 0
# back to default
# _ = np.seterr(**defaults)
# context manager: recommended; warnings ignored only inside this block, automatically restored afterward
# with np.errstate(all="ignore"):
#     np.arange(3) / 0

#### 32. Is the following expressions true? (★☆☆)
# np.sqrt(-1) == np.emath.sqrt(-1) --> no: False; nan != 1j

#### 33. How to get the dates of yesterday, today and tomorrow? (★☆☆)
# today = np.datetime64('today')
# yesterday = today - np.timedelta64(1)
# tomorrow  = today + np.timedelta64(1)
# print(today, yesterday, tomorrow)

#### 34. How to get all the dates corresponding to the month of July 2016? (★★☆)
# Z = np.arange('2016-07', '2016-08', dtype='datetime64[D]')
# print(Z)

#### 35. How to compute ((A+B)*(-A/2)) in place (without copy)? (★★☆)
#np.add(A, B, out=B)        # B = A + B
#np.divide(A, 2, out=A)    # A = A / 2
#np.negative(A, out=A)     # A = -A
#np.multiply(A, B, out=A)  # A = A * B

#### 36. Extract the integer part of a random array of positive numbers using 4 different methods (★★☆)
# Z = np.random.uniform(0,10,10)
# print(Z - Z%1)
# print(Z // 1)
# print(np.floor(Z))
# print(Z.astype(int))
# print(np.trunc(Z))

#### 37. Create a 5x5 matrix with row values ranging from 0 to 4 (★★☆)
# broadacsting
# M = np.zeros((5,5))
# M += np.arange(5)
# or without broadcasting
# M = np.tile(np.arange(5), (5,1))
# print(M)

#### 38. Consider a generator function that generates 10 integers and use it to build an array (★☆☆)
# def generate():
#     for x in range(10):
#         yield x
# Z = np.fromiter(generate(),dtype=float,count=-1)
# print(Z)

#### 39. Create a vector of size 10 with values ranging from 0 to 1, both excluded (★★☆)
# np.linspace(0,1,11,endpoint=False)[1:]

#### 40. Create a random vector of size 10 and sort it (★★☆)
# Z = np.random.random(10)
# Z.sort()
# print(Z)

#### 41. How to sum a small array faster than np.sum? (★★☆)
# Z = np.arange(10)
# sum_Z = np.add.reduce(Z)
# print(Z)
# print(sum_Z)

#### 42. Consider two random arrays A and B, check if they are equal (★★☆)
# A = np.random.randint(0,2,5)
# B = np.random.randint(0,2,5)
# equal = np.allclose(A,B)
# print(equal)

#### 43. Make an array immutable (read-only) (★★☆)
# Z = np.zeros(10)
# Z.flags.writeable = False
# Z[0] = 1
# print(Z)

#### 44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates (★★☆)
# Z = np.random.random((10,2))
# X,Y = Z[:,0], Z[:,1]
# R = np.sqrt(X**2+Y**2)
# T = np.arctan2(Y,X)
# print(R)
# print(T)

#### 45. Create random vector of size 10 and replace the maximum value by 0 (★★☆)
# Z = np.random.random(10)
# Z[Z.argmax()] = 0
# print(Z)

#### 46. Create a structured array with `x` and `y` coordinates covering the [0,1]x[0,1] area (★★☆)
# M = np.zeros((5,5), [('x',float),('y',float)])
# M['x'], M['y'] = np.meshgrid(np.linspace(0,1,5),np.linspace(0,1,5))
# print(M)

#### 47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj)) (★★☆)
# X = np.arange(8)
# Y = X + 0.5
# C = 1.0 / np.subtract.outer(X, Y)
# print(np.linalg.det(C))

#### 48. Print the minimum and maximum representable values for each numpy scalar type (★★☆)
# for dtype in [np.int8, np.int32, np.int64]:
#    print(np.iinfo(dtype).min)
#    print(np.iinfo(dtype).max)
# for dtype in [np.float32, np.float64]:
#    print(np.finfo(dtype).min)
#    print(np.finfo(dtype).max)
#    print(np.finfo(dtype).eps)

#### 49. How to print all the values of an array? (★★☆)
# np.set_printoptions(threshold=float("inf"))
# M = np.zeros((40,40))
# print(M)

#### 50. How to find the closest value (to a given scalar) in a vector? (★★☆)
# Z = np.arange(100)
# v = np.random.uniform(0,100)
# index = (np.abs(Z-v)).argmin()
# print(Z[index])

