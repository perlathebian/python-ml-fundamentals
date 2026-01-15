# NumPy Basics
# Source: https://numpy.org/doc/stable/user/absolute_beginners.html

import numpy as np
import pandas as pd
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


# --------------------------------------------------
# Array slicing (1D)
# --------------------------------------------------

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
arr1 = a[3:8]
print(arr1)


# --------------------------------------------------
# Vertical and horizontal stacking
# --------------------------------------------------

a1 = np.array([[1, 1],
               [2, 2]])

a2 = np.array([[3, 3],
               [4, 4]])

print(np.vstack((a1, a2)))
print(np.hstack((a1, a2)))


# --------------------------------------------------
# Splitting arrays
# --------------------------------------------------

x = np.arange(1, 25).reshape(2, 12)
print(x)

print(np.hsplit(x, 3))
print(np.hsplit(x, (3, 4)))


# --------------------------------------------------
# Views vs copies
# --------------------------------------------------

a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

b1 = a[0, :]
print(b1)

b1[0] = 99
print(b1)
print(a)     # Original array modified (view)

b2 = a.copy()  # Independent copy


# --------------------------------------------------
# Elementwise arithmetic
# --------------------------------------------------

data = np.array([1, 2])
ones = np.ones(2, dtype=int)

print(data + ones)
print(data - ones)
print(data * data)
print(data / data)


# --------------------------------------------------
# Reduction operations (sum, max, min)
# --------------------------------------------------

a = np.array([1, 2, 3, 4])
print(a.sum())

b = np.array([[1, 1],
              [2, 2]])

print(b.sum(axis=0))
print(b.sum(axis=1))


# --------------------------------------------------
# Broadcasting
# --------------------------------------------------

data = np.array([1.0, 2.0])
print(data * 1.6)


# --------------------------------------------------
# Aggregations on 1D and 2D arrays
# --------------------------------------------------

data = np.array([1, 2, 3])
print(data.max())
print(data.min())
print(data.sum())

a = np.array([[0.45053314, 0.17296777, 0.34376245, 0.5510652],
              [0.54627315, 0.05093587, 0.40067661, 0.55645993],
              [0.12697628, 0.82485143, 0.26590556, 0.56917101]])

print(a.sum())
print(a.min())
print(a.min(axis=0))


# --------------------------------------------------
# Matrix-style indexing and slicing
# --------------------------------------------------

data = np.array([[1, 2],
                 [3, 4],
                 [5, 6]])

print(data)
print(data[0, 1])
print(data[1:3])
print(data[0:2, 0])

print(data.max())
print(data.min())
print(data.sum())


# --------------------------------------------------
# Axis-based reductions
# --------------------------------------------------

data = np.array([[1, 2],
                 [5, 3],
                 [4, 6]])

print(data.max(axis=0))
print(data.max(axis=1))


# --------------------------------------------------
# Broadcasting with 2D arrays
# --------------------------------------------------

data = np.array([[1, 2],
                 [3, 4]])

ones = np.array([[1, 1],
                 [1, 1]])

print(data + ones)

data = np.array([[1, 2],
                 [3, 4],
                 [5, 6]])

ones_row = np.array([[1, 1]])
print(data + ones_row)


# --------------------------------------------------
# Creating arrays and random numbers
# --------------------------------------------------

print(np.ones((4, 3, 2)))
print(np.ones(3))
print(np.zeros(3))

rng = np.random.default_rng()
print(rng.random(3))
print(rng.random((3, 2)))
print(rng.integers(5, size=(2, 4)))


# --------------------------------------------------
# Finding unique values
# --------------------------------------------------

a = np.array([11, 11, 12, 13, 14, 15, 16, 17, 12, 13, 11, 14, 18, 19, 20])

unique_values = np.unique(a)
print(unique_values)

unique_values, indices = np.unique(a, return_index=True)
print(indices)

unique_values, counts = np.unique(a, return_counts=True)
print(counts)


# --------------------------------------------------
# Unique rows in 2D arrays
# --------------------------------------------------

a_2d = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12],
                 [1, 2, 3, 4]])

print(np.unique(a_2d))
print(np.unique(a_2d, axis=0))

unique_rows, indices, counts = np.unique(
    a_2d, axis=0, return_index=True, return_counts=True
)

print(unique_rows)
print(indices)
print(counts)


# --------------------------------------------------
# Reshaping and transposing
# --------------------------------------------------

data = np.array([[1, 2, 3],
                 [4, 5, 6]])

print(data.reshape(3, 2))
print(data.reshape(2, 3))

arr = np.arange(6).reshape((2, 3))
print(arr)
print(arr.transpose())
print(arr.T)


# --------------------------------------------------
# Reversing arrays
# --------------------------------------------------

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(np.flip(arr))

arr_2d = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12]])

print(np.flip(arr_2d))
print(np.flip(arr_2d, axis=0))
print(np.flip(arr_2d, axis=1))

arr_2d[1] = np.flip(arr_2d[1])
print(arr_2d)

arr_2d[:, 1] = np.flip(arr_2d[:, 1])
print(arr_2d)


# --------------------------------------------------
# Flatten vs ravel
# --------------------------------------------------

x = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

a1 = x.flatten()
a1[0] = 99
print(x)   # unchanged
print(a1)  # copy

a2 = x.ravel()
a2[0] = 98
print(x)   # modified
print(a2)


# --------------------------------------------------
# Saving and loading NumPy arrays
# --------------------------------------------------

a = np.array([1, 2, 3, 4, 5, 6])
np.save('filename', a)
b = np.load('filename.npy')
print(b)

csv_arr = np.array([1, 2, 3, 4, 5, 6, 7, 8])
np.savetxt('new_file.csv', csv_arr)
print(np.loadtxt('new_file.csv'))


# --------------------------------------------------
# NumPy and Pandas interoperability
# --------------------------------------------------

x = pd.read_csv('music.csv', header=0).values
print(x)

x = pd.read_csv('music.csv', usecols=['Artist', 'Plays']).values
print(x)

a = np.array([[-2.58289208,  0.43014843, -1.24082018, 1.59572603],
              [ 0.99027828,  1.17150989,  0.94125714, -0.14692469],
              [ 0.76989341,  0.81299683, -0.95068423, 0.11769564],
              [ 0.20484034,  0.34784527,  1.96979195, 0.51992837]])

df = pd.DataFrame(a)
print(df)

df.to_csv('pd.csv')
np.savetxt('np.csv', a, fmt='%.2f', delimiter=',', header='1,2,3,4')
