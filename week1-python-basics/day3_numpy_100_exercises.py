"""
Day 3: NumPy 100 Exercises
Date: January 5, 2026
Source: https://github.com/rougier/numpy-100
Progress: 20/100 
"""

import numpy as np

#### 11. Create a 3x3 identity matrix (★☆☆)
M = np.eye(3)
print(M)

#### 12. Create a 3x3x3 array with random values (★☆☆)
M = np.random.random((3,3,3))
print(M)

#### 13. Create a 10x10 array with random values and find the minimum and maximum values (★☆☆)
M = np.random.random((10,10))
m_min = M.min()
m_max = M.max()
print(m_min, m_max)

#### 14. Create a random vector of size 30 and find the mean value (★☆☆)
z = np.random.random(30)
z_mean = z.mean()
print(z_mean)

#### 15. Create a 2d array with 1 on the border and 0 inside (★☆☆)
n = 5
M = np.ones((n, n))
M[1:-1, 1:-1] = 0
print(M)

#### 16. How to add a border (filled with 0's) around an existing array? (★☆☆)
M = np.ones((3,3))
N = np.pad(M, pad_width=1, mode='constant', constant_values=0)
print(N)

#### 17. What is the result of the following expression? (★☆☆)
# 
# 0 * np.nan  --> nan:Anything multiplied by nan = nan
# np.nan == np.nan --> False: nan is not equal to anything even itself
# np.inf > np.nan --> False: Comparisons with nan are always False
# np.nan - np.nan --> nan: Arithmetic with nan gives nan
# np.nan in set([np.nan]) --> True: Membership check uses identity not equality
# 0.3 == 3 * 0.1 --> False: Floating point precision error
#

#### 18. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal (★☆☆)
M = np.diag([1,2,3,4], k=-1)
print(M)

#### 19. Create a 8x8 matrix and fill it with a checkerboard pattern (★☆☆)
M = np.zeros((8,8), dtype=int)
M[1::2, ::2] = 1
M[::2, 1::2] = 1
print(M)

#### 20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element? (★☆☆)
idx = np.unravel_index(99, (6, 7, 8))
print(idx)

