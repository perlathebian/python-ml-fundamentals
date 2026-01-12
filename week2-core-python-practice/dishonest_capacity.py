
"""
    Calculates the difference between a storage device’s advertised capacity
    and the actual capacity reported by an operating system.

    Storage manufacturers use decimal units (GB/TB), which makes devices
    appear larger than their true usable size.
"""

print('Enter TB or GB for the advertised unit:')
unit = input('Here: ')

# Calculate the amount that the advertised capacity lies:
if unit == 'TB' or unit == 'tb':
    discrepancy = 1000000000000 / 1099511627776
elif unit == 'GB' or unit == 'gb':
    discrepancy = 1000000000 / 1073741824

print('Enter the advertised capacity:')
advertised_capacity = input('Here: ')
advertised_capacity = float(advertised_capacity)

real_capacity = str(round(advertised_capacity * discrepancy, 2))

print('The actual capacity is ' + real_capacity + ' ' + unit)
