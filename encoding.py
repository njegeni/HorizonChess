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
        tensor[12, :, :] = 1.0

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



#now the second part of the encoding, we need to implement move encoding/decoding
# so the NN can process our moves

# define direction/knight/unique move offsets for each piece type

#queen like moves are used for rook, bishop, queen, king and pawn
queen_directions = [
    (1, 0), (-1, 0), (0, 1), (0, -1),  # north south east west
    (1, 1), (-1, -1), (1, -1), (-1, 1)  # northeast northwest southeast southwest
]

#knight needs its own 8 unique moves
knight_directions = [
    (2, 1), (2, -1), (-2, 1), (-2, -1),
    (1, 2), (1, -2), (-1, 2), (-1, -2),
]

#underpromotion
underpromotion_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
underpromotion_directions = [(1,0), (1,1), (1,-1)]  # forward, capture-right, capture left


# The policy is an 8x8x73 = 4672 vector (AlphaZero layout):
#   planes 0-55  : queen-like moves, 8 directions x 7 distances (dir_idx * 7 + dist-1)
#   planes 56-63 : the 8 knight moves
#   planes 64-72 : underpromotions, 3 pieces x 3 directions (knight/bishop/rook)
# Queen-promotions are encoded as ordinary distance-1 queen moves.

def _move_to_plane(from_sq: int, to_sq: int, promotion):
    """
    Maps an already-oriented move to (plane, row, col).
    Side to move is always assumed to be white
    """
    fr, fc = chess.square_rank(from_sq), chess.square_file(from_sq)
    tr, tc = chess.square_rank(to_sq), chess.square_file(to_sq)
    dr, dc = tr - fr, tc - fc

    # underpromotion to knight/bishop/rook (queen promo falls through to queen move)
    if promotion in underpromotion_pieces:
        piece_idx = underpromotion_pieces.index(promotion)
        dir_idx = underpromotion_directions.index((dr, dc))
        plane = 64 + piece_idx * 3 + dir_idx
        return plane, fr, fc

    # knight move
    if (dr, dc) in knight_directions:
        plane = 56 + knight_directions.index((dr, dc))
        return plane, fr, fc

    # queen-like move: reduce to a unit direction + distance
    dist = max(abs(dr), abs(dc))
    unit = (dr // dist, dc // dist)  # exact since dr, dc are multiples of dist
    dir_idx = queen_directions.index(unit)
    plane = dir_idx * 7 + (dist - 1)
    return plane, fr, fc


def encode_move(move: chess.Move, board: chess.Board) -> int:
    """
    Encodes a move into a flat policy index in [0, 4671].
    Orientation matches board_to_tensor: if it's black's turn, mirror the
    squares vertically so the mover always faces up the board.
    Flat layout: plane * 64 + row * 8 + col.
    """
    flip = board.turn == chess.BLACK

    from_sq = move.from_square
    to_sq = move.to_square
    if flip:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    plane, row, col = _move_to_plane(from_sq, to_sq, move.promotion)
    return plane * 64 + row * 8 + col


#now that we imlpemented a encoding we need to imlpement the decoding, turning the policy output into an actual chess.Move

def _plane_to_move(plane: int, row: int, col: int):
    """
    Inverse of _move_to_plane. Given a (plane, row, col) in the oriented
    ('white to move') frame, returns (from_sq, to_sq, promotion).
    Promotion is only set for underpromotions; a queen-promotion looks like a
    plain queen move here and is resolved by the caller against the real board.
    """
    from_sq = chess.square(col, row)  # chess.square(file, rank)

    if plane < 56:  # queen-like move
        dir_idx = plane // 7
        dist = plane % 7 + 1
        dr, dc = queen_directions[dir_idx]
        tr, tc = row + dr * dist, col + dc * dist
        promotion = None
    elif plane < 64:  # knight move
        dr, dc = knight_directions[plane - 56]
        tr, tc = row + dr, col + dc
        promotion = None
    else:  # underpromotion
        idx = plane - 64
        piece_idx, dir_idx = idx // 3, idx % 3
        dr, dc = underpromotion_directions[dir_idx]
        tr, tc = row + dr, col + dc
        promotion = underpromotion_pieces[piece_idx]

    to_sq = chess.square(tc, tr)
    return from_sq, to_sq, promotion


def decode_move(index: int, board: chess.Board) -> chess.Move:
    "Decodes the move from the flat policy index to a chess.Move"
    flip = board.turn == chess.BLACK

    plane = index // 64
    sq = index % 64
    row, col = sq // 8, sq % 8

    from_sq, to_sq, promotion = _plane_to_move(plane, row, col)

    # mirror back to the real board if it was black's turn (self-inverse flip)
    if flip:
        from_sq = chess.square_mirror(from_sq)
        to_sq = chess.square_mirror(to_sq)

    # a queen-move plane onto the last rank by a pawn is an (auto-)queen promotion
    if promotion is None:
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in (0, 7):
            promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)

