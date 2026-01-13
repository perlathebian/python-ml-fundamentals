"""Simulates coin flips and calculates probability of streaks."""

import random

number_of_streaks = 0

for experiment_number in range(10000):

    # Generate 100 coin flips
    flips = []
    for i in range(100):
        if random.randint(0, 1) == 0:
            flips.append('H')
        else:
            flips.append('T')

    # Check for a streak of 6
    for i in range(len(flips) - 5):
        if flips[i:i+6] == ['H'] * 6 or flips[i:i+6] == ['T'] * 6:
            number_of_streaks += 1
            break   # Only count one streak per experiment

# Print result
print('Chance of streak: %s%%' % (number_of_streaks / 100))