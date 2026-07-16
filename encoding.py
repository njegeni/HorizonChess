"""
NN cannot take chess boards as inputs, so we need to represent the games (each move) 
in terms of tensors.
The idea is that we use one plane for each variation so it would be this:
102 8x8 planes

planes 0-5 - side-to-move pieces, pawn knight bishop rook queen king
planes 6-11 - opponent pieces, pawn knight bishop rook queen king
plane 12 - side to move (always 1s after orientation; kept for spec)
plane 13 - side-to-move kingside castling right
plane 14 - side-to-move queenside castling right
plane 15 - opponent kingside castling right
plane 16 - opponent queenside castling right
plane 17 - en passant target square (1 at that square)
planes 18-29 - planes of s_t-1
plane 30-41 - planes of s_t-2
plane 42-53 - planes of s_t-3
plane 54-65 - planes of s_t-4
plane 66-77 - planes of s_t-5
plane 78-89 - planes of s_t-6
plane 90-101 - planes of s_t-7

"""

import numpy as np
import chess

def _fill_piece_planes(board: chess.Board, tensor: np.ndarray, base:int ):
    """
    piece.piece_type - 1 = plane they are on
    for black pieces id += 6
    """
    for square, piece in board.piece_map().items():
        row = chess.square_rank(square)
        col = chess.square_file(square)
        # 1 is pawn, 2 is knight, 3 is bishop, 4 is rook, 5 is queen, 6 is king
        idx = piece.piece_type - 1
        if piece.color == chess.BLACK:
            index += 6
        # tensor[plane, row, col]
        tensor[base + index, row, col] = 1.0

