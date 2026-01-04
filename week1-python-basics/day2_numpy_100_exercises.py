"""
Day 2: NumPy 100 Exercises
Date: January 4, 2026
Source: https://github.com/rougier/numpy-100
Progress: 10/100 
"""

import numpy as np

# ==============================================================================
# EXERCISES 1-10: BASICS
# ==============================================================================

# Exercise 1: Import the numpy package under the name np
# Already done above

# Exercise 2: Print the numpy version and the configuration
print(np.__version__)
np.show_config()

# Exercise 3: Create a null vector of size 10
z = np.zeros(10)
print(z)

# Exercise 4: How to find the memory size of any array
print(z.nbytes)

# Exercise 5: How to get the documentation of the numpy add function from the command line?
help(np.add)

# Exercise 6: Create a null vector of size 10 but the fifth value which is 1
z = np.zeros(10)
z[4] = 1
print(z)

# Exercise 7: Create a vector with values ranging from 10 to 49
z = np.arange(10, 50)
print(z)

# Exercise 8: Reverse a vector (first element becomes last)
z = z[::-1]
print(z)

# Exercise 9: Create a 3x3 matrix with values ranging from 0 to 8
z = np.arange(9).reshape(3,3)
print(z)

# Exercise 10: Find indices of non-zero elements from [1,2,0,0,4,0]
z = np.array([1,2,0,0,4,0])
indices = np.nonzero(z)
print(indices)