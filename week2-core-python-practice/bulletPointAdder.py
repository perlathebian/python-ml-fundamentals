"""
Add a bullet (*) to the beginning of each line from clipboard text.

This script reads text from the system clipboard, splits it into lines,
prepends a '*' to each line, joins the lines back together, copies the
result to the clipboard, and prints it.
"""

# example text to copy to clipboard initially: 
# Lists of animals
# Lists of aquarium life
# Lists of biologists by author abbreviation
# Lists of cultivars

import pyperclip

text = pyperclip.paste()

# Separate lines and add stars.
lines = text.split('\n')
for i in range(len(lines)):  # Loop through all indexes in the "lines" list.
    lines[i] = '* ' + lines[i]  # Add a star to each string in the "lines" list.
text = '\n'.join(lines)
pyperclip.copy(text)
print(text)