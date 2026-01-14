"""
Print a list of lists of strings in a neatly formatted right-justified table.

Parameters:
    tableData (list of lists of str): Each inner list represents a column.

The function:
- Calculates the maximum width of each column.
- Prints each row with strings right-justified according to their column width.
- Assumes all inner lists are of the same length.
"""
def printTable(tableData):
    """
    Print a list of lists of strings in a right-justified table.
    
    Parameters:
        tableData (list of lists of str): Each inner list represents a column.
    """
    
    # Step 1: find maximum width of each column
    colWidths = [0] * len(tableData)  # Initialize list with same number of columns
    for i in range(len(tableData)):   # Loop through each column
        for item in tableData[i]:
            if len(item) > colWidths[i]:
                colWidths[i] = len(item)
   
    
    # Step 2: print rows
    numRows = len(tableData[0])  # assume all inner lists same length
    for row in range(numRows):
        rowStr = ''
        for col in range(len(tableData)):
            # Right-justify each string according to its column width
            rowStr += tableData[col][row].rjust(colWidths[col]) + ' '
        print(rowStr.rstrip())  # remove extra space at the end

if __name__ == "__main__":
    tableData = [['apples', 'oranges', 'cherries', 'banana'],
             ['Alice', 'Bob', 'Carol', 'David'],
             ['dogs', 'cats', 'moose', 'goose']]
    
    printTable(tableData)