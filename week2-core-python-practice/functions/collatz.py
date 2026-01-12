import sys

def collatz(number):
    try:
        print(number, end = ' ')
        while True:
            if (number % 2) == 0:
                number = number//2
            else:
                number = number*3 + 1

            print(number, end = ' ')    
    except KeyboardInterrupt:
        sys.exit()

print('Print any number here:')
number = input('>>')
number = int(number)
collatz(number)        