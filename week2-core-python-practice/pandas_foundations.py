"""
Pandas Foundations
==================

This file demonstrates core Pandas concepts used in Data Science and
Machine Learning, including:

- Series creation and indexing
- Conditional selection and logical operators
- DataFrame operations
- Data cleaning
- apply() and lambda functions
- Joins, merges, and concatenation
"""

import pandas as pd
import numpy as np

# ======================================================
# 1. Pandas Series
# ======================================================

# s = pd.Series([10, 20, 30, 40, 50])
# print(s)
# print('----------------------------------------------')
# print(s.index)
# print('----------------------------------------------')
# print(s.values)
# print('----------------------------------------------')
# print(s.dtype)
# print('----------------------------------------------')
# print(s.name)
# print('----------------------------------------------')

# Naming a Series
# s.name = 'numbers'
# print(s)
# print('----------------------------------------------')
# print(s.name)
# print('----------------------------------------------')

# Indexing
# print(s[0])
# print('----------------------------------------------')
# print(s[2:4])
# print('----------------------------------------------')

# iloc → location-based indexing
# print(s.iloc[3])
# print('----------------------------------------------')

# Multiple values using double brackets
# print(s.iloc[[1, 3, 4]])
# print('----------------------------------------------')

# Changing index labels
# s.name = 'calories'
# index = ['apple', 'banana', 'grapes', 'orange', 'strawberry']
# s.index = index
# print(s)
# print('----------------------------------------------')

# Label-based indexing
# print(s['grapes'])
# print('----------------------------------------------')

# loc → label-based indexing (iloc only works with numbers)
# print(s.loc['grapes'])
# print(s.loc[['grapes', 'apple']])
# print('----------------------------------------------')

# In label-based slicing, start and stop values are included
# print(s['banana':'orange'])
# print('----------------------------------------------')

# ======================================================
# 2. Series from Dictionary
# ======================================================

# fruit_protein = {
#     'Avocado': 2.0,      # grams of protein
#     'Guava': 2.6,
#     'Blackberries': 2.0,
#     'Oranges': 0.9,
#     'Banana': 1.1,
#     'Apples': 0.3,
#     'Kiwi': 1.1,
#     'Pomegranate': 1.7,
#     'Mango': 0.8,
#     'Cherries': 1.0
# }

# s2 = pd.Series(fruit_protein, name='Protein')
# print(s2)

# Conditional Selection
# print(s2 > 1)
# print(s2[s2 > 1])

# Logical operators: and (&), or (|), not (~)
# print((s2 > 0.5) & (s2 < 2))
# print(s2[(s2 > 0.5) & (s2 < 2)])
# print(s2[~(s2 > 1)])

# Modifying a Series
# s2["Mango"] = 2.8
# print(s2)

# Counting non-null values
# ser = pd.Series(['a', np.nan, 1, np.nan, 2])
# print(ser.notnull().sum())

# ======================================================
# 3. DataFrames
# ======================================================

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Alice'],
    'Age': [25, 30, 35, np.nan, 29, 25],
    'Department': ['HR', 'IT', 'Finance', 'IT', 'HR', 'HR'],
    'Salary': [50000, 60000, 70000, 62000, np.nan, 50000]
}

df = pd.DataFrame(data)

# print(df)
# print(df.head())       # first 5 rows
# print(df.head(2))
# print(df.tail())       # last 5 rows
# print(df.tail(2))

# ======================================================
# 4. loc and iloc
# ======================================================

# Fetching rows
# print(df.iloc[1:3])

# Selecting specific columns
# print(df.loc[1:2, ['Age', 'Department']])

# Row and column slicing
# print(df.iloc[1:3, :2])

# Column access
# print(df['Age'])
# print(df[['Age', 'Department']])

# Dropping a column
# df.drop('Age', axis=1, inplace=True)
# print(df)

# Data info
# print(df.shape)
# print(df.info())
# print(df.describe())

# ======================================================
# 5. Broadcasting & Column Operations
# ======================================================

df['Salary'] = df['Salary'] + 5000

# Renaming columns
df.rename(columns={'Department': 'Dept'}, inplace=True)

# Unique values & counts
# print(df['Dept'].unique())
# print(df['Dept'].value_counts())

# Creating new column
df['Promoted Salary'] = df['Salary'] * 10

# ======================================================
# 6. Data Cleaning
# ======================================================

# print(df.isnull().sum())

# Drop rows with NaN values
# print(df.dropna(how='any'))
# print(df.dropna(how='all'))

# Filling missing values
# print(df['Age'].fillna(df['Age'].mean()))
# print(df['Salary'].fillna(df['Salary'].median()))

# Forward fill & backward fill
# print(df['Age'].fillna(method='ffill'))
# print(df['Age'].fillna(method='bfill'))

# Replace values
df['Name'] = df['Name'].replace('Charlie', 'Rose')

# Handling duplicates
df_dup = df[df.duplicated(keep='last')]
# print(df_dup)

df = df.drop_duplicates()

# ======================================================
# 7. apply() and lambda
# ======================================================

# Fixing invalid values
df['Promoted Salary'] = df['Promoted Salary'].apply(
    lambda x: x / 10 if x > 650000 else x
)

# String functions (example only)
name = 'alice_fernqndes'
# df[['first_name', 'last_name']] = df['name'].str.split('_')

def multiplying_age(x):
    return x * 2

df['Age'] = df['Age'].apply(multiplying_age)
df['Age'] = df['Age'].apply(lambda x: x / 2)

# ======================================================
# 8. Joins and Merges
# ======================================================

department_info = {
    'Dept': ['HR', 'IT', 'Finance'],
    'Location': ['New York', 'San Francisco', 'Chicago'],
    'Manager': ['Laura', 'Steve', 'Nina']
}

df2 = pd.DataFrame(department_info)

# Concatenation
# print(pd.concat([df, df2]))
# print(pd.concat([df, df2], axis=1))

# Merge
print(pd.merge(df, df2, on='Dept'))

# ======================================================
# 9. Importing Files (Reference)
# ======================================================

# data = pd.read_csv('path_to_file')
# data.head()
# data.shape
# data.info()

# Convert object to datetime
# df['date'] = pd.to_datetime(data['date'])
# data.info()
