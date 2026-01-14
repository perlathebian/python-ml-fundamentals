"""
Alternates the case of characters in text from the clipboard.

This script reads text from the system clipboard, converts it to a string
with alternating lowercase and uppercase characters, copies the result
back to the clipboard, and prints it to the terminal.

Uses pyperclip for clipboard access:
- pyperclip.paste() reads clipboard text
- pyperclip.copy(text) writes text to the clipboard
"""
# need to install pyperclip lib first

import pyperclip

text = pyperclip.paste()  # Get the text off the clipboard.
alt_text = ''  # This string holds the alternating case.
make_uppercase = False
for character in text:
    # Go through each character and add it to alt_text:
    if make_uppercase:
        alt_text += character.upper()
    else:
        alt_text += character.lower()

    # Set make_uppercase to its opposite value:
    make_uppercase = not make_uppercase
pyperclip.copy(alt_text)  # Put the result on the clipboard.
print(alt_text)  # Print the result on the screen too.