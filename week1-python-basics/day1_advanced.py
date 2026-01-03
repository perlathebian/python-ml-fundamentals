# List comprehensions

# Basic transformation
squares = [x**2 for x in range(10)]

# With condition
even_squares = [x**2 for x in range(10) if x % 2 == 0]

# Nested comprehension
matrix = [[i*j for j in range(5)] for i in range(5)]

# Dictionary comprehension 
word_lengths = {word: len(word) for word in ['hello', 'world', 'python']}

# Lambda functions 
# Sort by specific criteria
students = [
    {'name': 'Alice', 'grade': 85},
    {'name': 'Bob', 'grade': 92},
    {'name': 'Charlie', 'grade': 78}
]
sorted_students = sorted(students, key=lambda x: x['grade'], reverse=True)

# Map, filter, reduce (functional programming)
from functools import reduce

numbers = [1, 2, 3, 4, 5]
doubled = list(map(lambda x: x * 2, numbers))
evens = list(filter(lambda x: x % 2 == 0, numbers))
product = reduce(lambda x, y: x * y, numbers)

# 1. Get all numbers divisible by 3 or 5 from 1-100
result1 = [x for x in range(1, 101) if x % 3 == 0 or x % 5 == 0]

# 2. Convert list of strings to uppercase, only if length > 3
words = ['hi', 'hello', 'world', 'test', 'ai']
result2 = [w.upper() for w in words if len(w) > 3]

# 3. Create dict of numbers and their squares from 1-20
result3 = {x: x**2 for x in range(1, 21)}

# 4. Flatten this nested structure using comprehension
nested = [[1, 2], [3, 4], [5, 6, 7]]
result4 = [num for sublist in nested for num in sublist]

# 5. Get names of students with grade >= 80
result5 = [s['name'] for s in students if s['grade'] >= 80]

print("Divisible by 3 or 5:", result1)
print("Uppercase words:", result2)
print("Squares dict:", result3)
print("Flattened:", result4)
print("High performers:", result5)