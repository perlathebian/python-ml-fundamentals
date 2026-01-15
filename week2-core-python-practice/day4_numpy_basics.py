# NumPy Basics
# Source: https://numpy.org/doc/stable/user/absolute_beginners.html

import numpy as np
import math


# --------------------------------------------------
# Array creation and basic properties (shape)
# --------------------------------------------------

a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a)
print(a.shape)


# --------------------------------------------------
# 1D array creation, indexing, and assignment
# --------------------------------------------------

a = np.array([1, 2, 3, 4, 5, 6])

print(a[0])        # Access first element
a[0] = 10          # Modify element
print(a)


# --------------------------------------------------
# Slicing arrays
# --------------------------------------------------

print(a[:3])       # First three elements


# --------------------------------------------------
# Views vs copies (slicing creates a view)
# --------------------------------------------------

b = a[3:]
print(b)

b[0] = 40          # Modifies original array
print(a)


# --------------------------------------------------
# Multidimensional arrays and indexing
# --------------------------------------------------

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

print(a)
print(a[1, 3])     # Row 1, column 3


# --------------------------------------------------
# Array dimensions and shape
# --------------------------------------------------

print(a.ndim)
print(a.shape)
print(len(a.shape) == a.ndim)


# --------------------------------------------------
# Array size and total elements
# --------------------------------------------------

print(a.size)
print(a.size == math.prod(a.shape))


# --------------------------------------------------
# Data types
# --------------------------------------------------

print(a.dtype)


# --------------------------------------------------
# Creating arrays with predefined values
# --------------------------------------------------

print(np.zeros(2))
print(np.ones(2))
print(np.empty(2))        # Values are uninitialized

print(np.arange(4))
print(np.arange(2, 9, 2))
print(np.linspace(0, 10, num=5))


# --------------------------------------------------
# Specifying data types
# --------------------------------------------------

x = np.ones(2, dtype=np.int64)
print(x)


# --------------------------------------------------
# Sorting arrays
# --------------------------------------------------

arr = np.array([2, 1, 5, 3, 7, 4, 6, 8])
print(np.sort(arr))

# Other useful sorting-related functions:
# - argsort
# - lexsort
# - searchsorted
# - partition


# --------------------------------------------------
# Concatenating arrays
# --------------------------------------------------

a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(np.concatenate((a, b)))

x = np.array([[1, 2],
              [3, 4]])
y = np.array([[5, 6]])

print(np.concatenate((x, y), axis=0))


# --------------------------------------------------
# Working with higher-dimensional arrays
# --------------------------------------------------

array_example = np.array([
    [[0, 1, 2, 3],
     [4, 5, 6, 7]],
    [[0, 1, 2, 3],
     [4, 5, 6, 7]],
    [[0, 1, 2, 3],
     [4, 5, 6, 7]]
])

print(array_example.ndim)
print(array_example.size)
print(array_example.shape)


# --------------------------------------------------
# Reshaping arrays
# --------------------------------------------------

a = np.arange(6)
print(a)

b = a.reshape(3, 2)
print(b)

print(np.reshape(a, shape=(1, 6), order='C'))


# --------------------------------------------------
# Adding new axes (row vs column vectors)
# --------------------------------------------------

a = np.array([1, 2, 3, 4, 5, 6])
print(a.shape)

a2 = a[np.newaxis, :]
print(a2.shape)

row_vector = a[np.newaxis, :]
print(row_vector.shape)

col_vector = a[:, np.newaxis]
print(col_vector.shape)


# --------------------------------------------------
# Using expand_dims
# --------------------------------------------------

b = np.expand_dims(a, axis=1)
print(b.shape)

c = np.expand_dims(a, axis=0)
print(c.shape)


# --------------------------------------------------
# Basic indexing and slicing
# --------------------------------------------------

data = np.array([1, 2, 3])

print(data[1])
print(data[0:2])
print(data[1:])
print(data[-2:])


# --------------------------------------------------
# Boolean indexing and filtering
# --------------------------------------------------

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

print(a[a < 5])

five_up = (a >= 5)
print(a[five_up])

divisible_by_2 = a[a % 2 == 0]
print(divisible_by_2)

c = a[(a > 2) & (a < 11)]
print(c)

five_up = (a > 5) | (a == 5)
print(five_up)


# --------------------------------------------------
# Finding indices with nonzero
# --------------------------------------------------

b = np.nonzero(a < 5)
print(b)

list_of_coordinates = list(zip(b[0], b[1]))

for coord in list_of_coordinates:
    print(coord)

print(a[b])


# --------------------------------------------------
# Nonexistent values example
# --------------------------------------------------

not_there = np.nonzero(a == 42)
print(not_there)
