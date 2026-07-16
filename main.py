import chess.pgn

pgn = open('pgnmentor.pgn')

first_game = chess.pgn.read_game(pgn)
second_game = chess.pgn.read_game(pgn)



board = chess.Board()




for square, piece in board.piece_map().items():
    if piece.color == chess.BLACK:
        print(chess.square_rank(square), chess.square_file(square), piece.piece_type)


# 1 is pawn, 2 is knight, 3 is bishop, 4 is rook, 5 is queen, 6 is king
# piece.piece_type - 1 = plane they are on
# for black pieces id += 6

