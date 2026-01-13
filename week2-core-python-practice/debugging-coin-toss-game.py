"""Interactive coin toss game with validation and retry."""

# import random
# guess = ''
# while guess not in ('heads', 'tails'):
#     print('Guess the coin toss! Enter heads or tails:')
#     guess = input()
# toss = random.randint(0, 1)  # 0 is tails, 1 is heads
# if toss == guess:
#     print('You got it!')
# else:
#     print('Nope! Guess again!')
#     guess = input()
#     if toss == guess:
#         print('You got it!')
#     else:
#         print('Nope. You are really bad at this game.')

# Solution

import random
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

def coin_toss_game():
    valid_guesses = ('heads', 'tails')

    guess = ''
    while guess not in valid_guesses:
        logging.info('Guess the coin toss! Enter heads or tails:')
        guess = input().lower()

    toss = random.choice(valid_guesses)
    logging.info(f'DEBUG: coin toss is {toss}')

    assert guess in valid_guesses, "Invalid guess after validation loop"

    if guess == toss:
        print('You got it!')
        return

    print('Nope! Guess again!')
    guess = input().lower()
    assert guess in valid_guesses, "Second guess must be heads or tails"

    if guess == toss:
        print('You got it!')
    else:
        print('Nope. You are really bad at this game.')

coin_toss_game()