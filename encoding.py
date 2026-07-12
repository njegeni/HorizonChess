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


"""
Move (policy) encoding — the output side of the network.

AlphaZero move representation: 73 planes x 8 x 8 = 4672 possible moves.
Each move is indexed by (plane, from_rank, from_file) and flattened as
    index = plane * 64 + from_rank * 8 + from_file
so the policy vector matches a conv policy head that outputs (73, 8, 8),
laid out the same channels-first way as the board tensor above.

Plane layout (all in the side-to-move oriented frame, so the mover always
"plays up the board" with increasing rank):
  planes  0-55 - queen-like moves: 8 directions x 7 distances (dir*7 + dist-1)
  planes 56-63 - the 8 knight moves
  planes 64-72 - underpromotions: 3 directions x 3 pieces (knight, bishop, rook)

Queen promotions are encoded as ordinary forward queen moves; only knight/
bishop/rook promotions use the underpromotion planes.
"""

POLICY_SIZE = 73 * 64  # 4672

# Queen-move directions as (d_rank, d_file), fixed order -> plane group.
_QUEEN_DIRS = [
    (1, 0),   # N
    (1, 1),   # NE
    (0, 1),   # E
    (-1, 1),  # SE
    (-1, 0),  # S
    (-1, -1),  # SW
    (0, -1),  # W
    (1, -1),  # NW
]

# Knight moves as (d_rank, d_file), fixed order -> plane 56 + index.
_KNIGHT_DIRS = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]

# Underpromotion pieces in plane order (knight, bishop, rook).
_UNDERPROMO_PIECES = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def _oriented_squares(move: chess.Move, flip: bool):
    """Return (from_sq, to_sq) mapped into the side-to-move oriented frame.

    Mirrors both squares vertically when Black is to move, matching the board
    orientation used by `board_to_tensor` (chess.square_mirror flips the rank).
    """
    if flip:
        return chess.square_mirror(move.from_square), chess.square_mirror(move.to_square)
    return move.from_square, move.to_square


def move_to_index(move: chess.Move, turn: bool) -> int:
    """Map a `chess.Move` to its policy index in [0, POLICY_SIZE).

    `turn` is the side to move (board.turn) so the move can be oriented the
    same way as the board tensor.
    """
    flip = turn == chess.BLACK
    from_sq, to_sq = _oriented_squares(move, flip)

    from_rank, from_file = chess.square_rank(from_sq), chess.square_file(from_sq)
    d_rank = chess.square_rank(to_sq) - from_rank
    d_file = chess.square_file(to_sq) - from_file

    # Underpromotion (knight/bishop/rook): 3 directions x 3 pieces.
    if move.promotion is not None and move.promotion != chess.QUEEN:
        dir_index = d_file + 1  # -1,0,1 -> 0,1,2 (capture-left, push, capture-right)
        piece_index = _UNDERPROMO_PIECES.index(move.promotion)
        plane = 64 + dir_index * 3 + piece_index
        return plane * 64 + from_rank * 8 + from_file

    # Knight move.
    if (d_rank, d_file) in _KNIGHT_DIRS:
        plane = 56 + _KNIGHT_DIRS.index((d_rank, d_file))
        return plane * 64 + from_rank * 8 + from_file

    # Queen-like move (includes queen promotions and normal king/pawn steps).
    step_rank = (d_rank > 0) - (d_rank < 0)
    step_file = (d_file > 0) - (d_file < 0)
    distance = max(abs(d_rank), abs(d_file))
    dir_index = _QUEEN_DIRS.index((step_rank, step_file))
    plane = dir_index * 7 + (distance - 1)
    return plane * 64 + from_rank * 8 + from_file


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Inverse of `move_to_index`: rebuild the `chess.Move` on `board`.

    `board` supplies the side to move (for un-orienting) and lets us detect
    when a queen-plane move is actually a queen promotion (pawn reaching the
    last rank).
    """
    plane, square = divmod(index, 64)
    from_rank, from_file = divmod(square, 8)

    if plane < 56:  # queen-like
        dir_index, dist = divmod(plane, 7)
        distance = dist + 1
        step_rank, step_file = _QUEEN_DIRS[dir_index]
        d_rank, d_file = step_rank * distance, step_file * distance
        promotion = None
    elif plane < 64:  # knight
        d_rank, d_file = _KNIGHT_DIRS[plane - 56]
        promotion = None
    else:  # underpromotion
        offset = plane - 64
        dir_index, piece_index = divmod(offset, 3)
        d_rank, d_file = 1, dir_index - 1
        promotion = _UNDERPROMO_PIECES[piece_index]

    to_rank, to_file = from_rank + d_rank, from_file + d_file
    from_sq = chess.square(from_file, from_rank)
    to_sq = chess.square(to_file, to_rank)

    # Un-orient back into the real board's frame.
    flip = board.turn == chess.BLACK
    if flip:
        from_sq, to_sq = chess.square_mirror(from_sq), chess.square_mirror(to_sq)

    # Queen-plane pawn move onto the back rank is a queen promotion.
    if promotion is None:
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in (0, 7):
            promotion = chess.QUEEN

    return chess.Move(from_sq, to_sq, promotion=promotion)