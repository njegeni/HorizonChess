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
    used to create 12 planes for 1 game state
    piece.piece_type - 1 = plane they are on
    for black pieces id += 6
    """
    for square, piece in board.piece_map().items():
        row = chess.square_rank(square)
        col = chess.square_file(square)
        # 1 is pawn, 2 is knight, 3 is bishop, 4 is rook, 5 is queen, 6 is king
        index = piece.piece_type - 1
        if piece.color == chess.BLACK:
            index += 6
        # tensor[plane, row, col]
        tensor[base + index, row, col] = 1.0

def _perspective(board: chess.Board, flip: bool) -> chess.Board:
    """
    Checks whose turn it is and flips the board if its black 
    (in order to train for maximal output always)
    flip = true -> creates a copy of the board and black pieces are on the bottom
    """
    return board.mirror() if flip else board

def board_to_tensor(board: chess.Board):
    tensor = np.zeros((102,8,8), dtype=np.float32)

    #retract whose turn it is right away
    flip = board.turn == chess.BLACK
    oriented = _perspective(board, flip)

    #filling current game state
    _fill_piece_planes(oriented, tensor, 0)

    # plane 12 = side to move, always white after orientation
    if oriented.turn == chess.WHITE:
        tensor[12, :, :] == 1.0

    #planes 13-16 = castling rights
    if oriented.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if oriented.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if oriented.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if oriented.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    #plane 17 = en passant target square
    if oriented.ep_square is not None:
        row = chess.square_rank(oriented.ep_square)
        col = chess.square_file(oriented.ep_square)
        tensor[17, row, col] = 1.0

    #now for 7 previous game states,
    # Pop moves from the original board, then apply the SAME orientation as the
    # current state so the whole stack stays spatially consistent.
    temp_board = board.copy()
    for i in range(7):
        if not temp_board.move_stack:
            break  # start of game reached; remaining history planes stay 0
        temp_board.pop()
        _fill_piece_planes(_perspective(temp_board, flip), tensor, 18 + i * 12)
    
    return tensor



