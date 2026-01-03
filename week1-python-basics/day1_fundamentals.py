# Exercises

# 1. Temperature converter
def celsius_to_fahrenheit(celsius):
    return (celsius * 9/5) + 32

# 2. List statistics
def get_stats(numbers):
    return {
        'min': min(numbers),
        'max': max(numbers),
        'avg': sum(numbers) / len(numbers)
    }

# 3. String reverser
def reverse_string(text):
    return text[::-1]

# 4. Even number filter
def filter_evens(numbers):
    return [n for n in numbers if n % 2 == 0]

# 5. Word counter
def count_words(text):
    return len(text.split())

# 6. Factorial calculator
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

# 7. Palindrome checker
def is_palindrome(text):
    clean = text.lower().replace(' ', '')
    return clean == clean[::-1]

# 8. Find duplicates
def find_duplicates(items):
    return list(set([x for x in items if items.count(x) > 1]))

# 9. Flatten nested list
def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]

# 10. Grade calculator
def calculate_grade(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    elif score >= 60: return 'D'
    else: return 'F'

# TEST ALL FUNCTIONS
if __name__ == "__main__":
    print(celsius_to_fahrenheit(0))  # 32.0
    print(get_stats([1, 2, 3, 4, 5]))  # {'min': 1, 'max': 5, 'avg': 3.0}
    print(reverse_string("hello"))  # olleh
    print(filter_evens([1, 2, 3, 4, 5]))  # [2, 4]
    print(count_words("hello world test"))  # 3
    print(factorial(5))  # 120
    print(is_palindrome("racecar"))  # True
    print(find_duplicates([1, 2, 2, 3, 3, 4]))  # [2, 3]
    print(flatten([[1, 2], [3, 4], [5]]))  # [1, 2, 3, 4, 5]
    print(calculate_grade(85))  # B