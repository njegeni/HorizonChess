"""
Encoding PGN notation to tensors to be able to input them into NN

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


def _fill_piece_planes(board: chess.Board, tensor: np.ndarray, base: int):
    """Fill the 12 piece planes at tensor[base:base+12] for `board`.

    White pieces go to base..base+5, black pieces to base+6..base+11,
    ordered pawn, knight, bishop, rook, queen, king (piece_type - 1).
    Row = rank (0 = rank 1), column = file (0 = file a).
    """
    for square, piece in board.piece_map().items():
        row = chess.square_rank(square)
        col = chess.square_file(square)
        idx = piece.piece_type - 1
        if piece.color == chess.BLACK:
            idx += 6
        tensor[base + idx, row, col] = 1.0


def _orient(board: chess.Board, flip: bool) -> chess.Board:
    """Return the board oriented to the side to move.

    When `flip` is True (Black to move) the board is mirrored vertically with
    piece colors swapped, so the side to move always 'plays up the board' and
    its pieces occupy planes 0-5. `board.mirror()` returns a copy, so the input
    board is never mutated.
    """
    return board.mirror() if flip else board


def board_to_tensor(board: chess.Board):
    tensor = np.zeros((102, 8, 8), dtype=np.float32)

    # Orient everything to the current side to move.
    flip = board.turn == chess.BLACK
    oriented = _orient(board, flip)

    # --- current state s_t: planes 0-17 (from the oriented board) ---
    _fill_piece_planes(oriented, tensor, 0)

    # plane 12 - side to move (always White after orientation -> all 1s)
    if oriented.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # planes 13-16 - castling rights, side-to-move first, then opponent
    if oriented.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if oriented.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if oriented.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if oriented.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    # plane 17 - en passant target square (already transformed by mirror())
    if oriented.ep_square is not None:
        row = chess.square_rank(oriented.ep_square)
        col = chess.square_file(oriented.ep_square)
        tensor[17, row, col] = 1.0

    # -history s_{t-1} .. s_{t-7}: 12 piece planes each, from plane 18
    # Pop moves from the original board, then apply the SAME orientation as the
    # current state so the whole stack stays spatially consistent.
    temp_board = board.copy()
    for i in range(7):
        if not temp_board.move_stack:
            break  # start of game reached; remaining history planes stay 0
        temp_board.pop()
        _fill_piece_planes(_orient(temp_board, flip), tensor, 18 + i * 12)

    return tensor