

def isValidChessBoard(board):
    """
    Validate a chess board represented as a dictionary.

    Checks that the board has exactly one white king and one black king,
    no more than 16 pieces per side (max 8 pawns), valid piece names, and
    all pieces placed on valid squares from a1 to h8.
    """
    
    valid_pieces = {
        'pawn', 'knight', 'bishop', 'rook', 'queen', 'king'
    }

    white_pieces = 0
    black_pieces = 0
    white_kings = 0
    black_kings = 0
    pawn_count = {'w': 0, 'b': 0}

    for square, piece in board.items():
        # validate square
        if len(square) != 2:
            return False
        file, rank = square[0], square[1]
        if file not in 'abcdefgh' or rank not in '12345678':
            return False

        # validate piece format
        if len(piece) < 2:
            return False
        color = piece[0]
        name = piece[1:]

        if color not in ('w', 'b'):
            return False
        if name not in valid_pieces:
            return False

        # count pieces
        if color == 'w':
            white_pieces += 1
            if name == 'king':
                white_kings += 1
        else:
            black_pieces += 1
            if name == 'king':
                black_kings += 1

        if name == 'pawn':
            pawn_count[color] += 1
            if pawn_count[color] > 8:
                return False

    # final validations
    if white_kings != 1 or black_kings != 1:
        return False
    if white_pieces > 16 or black_pieces > 16:
        return False

    return True

if __name__ == "__main__":
    # valid board
    board1 = {
        'h1': 'bking',
        'c6': 'wqueen',
        'g2': 'bbishop',
        'h5': 'bqueen',
        'e3': 'wking'
    }

    # invalid boards
    board2 = {'e1': 'wking', 'e2': 'wking', 'e8': 'bking'}
    board3 = {'z9': 'wking', 'e8': 'bking'}

    print(isValidChessBoard(board1))  # True
    print(isValidChessBoard(board2))  # False
    print(isValidChessBoard(board3))  # False